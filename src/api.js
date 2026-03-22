'use strict';
const express    = require('express');
const cors       = require('cors');
const {
  pool,
  getRounds, getStats, getStorageStats,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedAdvPreds, getLockedAdvPreds,
  saveLockedConsensusPreds, getLockedConsensusPreds,
  initWalletStorage, saveWallet, getWallets, deleteWallet,
} = require('./db');

const app  = express();
const PORT = process.env.PORT || 3001;

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '')
  .split(',').map(s => s.trim()).filter(Boolean);

app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (ALLOWED_ORIGINS.length === 0) return cb(null, true);
    if (ALLOWED_ORIGINS.some(o => origin.startsWith(o))) return cb(null, true);
    cb(new Error('Not allowed by CORS'));
  },
  credentials: true,
}));

app.use(express.json({ limit: '50kb' }));
app.use((req, res, next) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  next();
});

// ── Rate limiting ─────────────────────────────────────────────────────────────
// FIX: use a fixed-size LRU-style map (max 10k entries) to prevent unbounded
// memory growth. Cleanup interval now 60s instead of 300s.
const rateLimits  = new Map();
const RL_MAX_KEYS = 10000;

function rateLimit(maxPerMin) {
  return (req, res, next) => {
    const ip  = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || 'unknown';
    const key = `${ip}:${req.path}`;
    const now = Date.now();
    const win = rateLimits.get(key) || { count: 0, reset: now + 60000 };
    if (now > win.reset) { win.count = 0; win.reset = now + 60000; }
    win.count++;
    // FIX: evict oldest entry when map is at capacity to prevent unbounded growth
    if (!rateLimits.has(key) && rateLimits.size >= RL_MAX_KEYS) {
      rateLimits.delete(rateLimits.keys().next().value);
    }
    rateLimits.set(key, win);
    if (win.count > maxPerMin) return res.status(429).json({ ok: false, error: 'Too many requests' });
    next();
  };
}
// FIX: cleanup every 60s, not 300s — tighter memory bound
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of rateLimits) if (now > v.reset) rateLimits.delete(k);
}, 60000);

const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) console.warn('⚠️  ADMIN_SECRET not set');

function requireAdmin(req, res, next) {
  const secret = req.headers['x-admin-secret'];
  if (!ADMIN_SECRET || !secret || secret !== ADMIN_SECRET)
    return res.status(403).json({ ok: false, error: 'Forbidden' });
  next();
}

function getIP(req) {
  return req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.headers['x-real-ip'] || req.socket?.remoteAddress || 'unknown';
}

const APP_SECRET = process.env.APP_SECRET || process.env.ADMIN_SECRET;

// ── ROUNDS ────────────────────────────────────────────────────────────────────
app.get('/rounds', rateLimit(60), async (req, res) => {
  try {
    const limit      = Math.min(parseInt(req.query.limit  || '1000'), 100000);
    const offset     = parseInt(req.query.offset || '0');
    const since      = req.query.since ? Number(req.query.since) : null;
    const minRoundId = since && since > 0 ? since + 1 : null;
    const rounds = await getRounds({ limit, offset, from: req.query.from||null, to: req.query.to||null, minRoundId });
    res.json({ ok:true, count: rounds.length, rounds });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.get('/stats',         rateLimit(30), async (req,res) => { try { res.json({ ok:true, ...(await getStats()) }); } catch(e) { res.status(500).json({ ok:false, error:e.message }); } });
app.get('/storage-stats', rateLimit(20), async (req,res) => { try { res.json({ ok:true, ...(await getStorageStats()) }); } catch(e) { res.status(500).json({ ok:false, error:e.message }); } });
app.get('/health', (req,res) => res.json({ ok:true, ts:new Date().toISOString() }));

// ── PREDICTIONS ───────────────────────────────────────────────────────────────
const VALID_SOURCES = [
  'engine','pattern','consensus',
];

app.get('/predictions', rateLimit(300), async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit||'500'), 5000);
    const source = VALID_SOURCES.includes(req.query.source) ? req.query.source : null;
    const rows   = await getPredictions({ limit, target: req.query.target||null, source });
    res.json({ ok:true, count:rows.length, predictions:rows });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/predictions', rateLimit(600), async (req, res) => {
  try {
    const { target, minMult, outcome, lo, hi, hitRound, generation, source } = req.body;
    if (!target || !outcome || lo==null || hi==null)
      return res.status(400).json({ ok:false, error:'Missing required fields' });
    if (!['win','loss','early','retry-win','retry-loss'].includes(outcome))
      return res.status(400).json({ ok:false, error:'Invalid outcome' });
    if (typeof lo!=='number' || typeof hi!=='number' || hi<lo || !isFinite(lo) || !isFinite(hi))
      return res.status(400).json({ ok:false, error:'Invalid window' });
    // FIX: reject zero-width windows (lo===hi) — they represent no prediction
    if (lo === hi)
      return res.status(400).json({ ok:false, error:'Zero-width window' });
    const validSource = VALID_SOURCES.includes(source) ? source : 'engine';
    await savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation, source:validSource });
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/predictions', requireAdmin, async (req, res) => {
  try {
    await clearPredictions();
    require('./predictionEngine').resetEngineState();
    console.log('[api] predictions cleared + engine reset');
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── Batch history — no cache, always fresh from DB ───────────────────────────
app.get('/predictions-all', rateLimit(120), async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit || '500'), 2000);
    const [consensus,
           hlstm_xgb, htrans_lstm, htft, tft, nbeats, tcn,
           lgbm, gru, bilstm, stacking, sha512, ng_consensus] = await Promise.all([
      getPredictions({ limit, source: 'consensus' }),
      getPredictions({ limit, source: 'hlstm_xgb' }),
      getPredictions({ limit, source: 'htrans_lstm' }),
      getPredictions({ limit, source: 'htft' }),
      getPredictions({ limit, source: 'tft' }),
      getPredictions({ limit, source: 'nbeats' }),
      getPredictions({ limit, source: 'tcn' }),
      getPredictions({ limit, source: 'lgbm' }),
      getPredictions({ limit, source: 'gru' }),
      getPredictions({ limit, source: 'bilstm' }),
      getPredictions({ limit, source: 'stacking' }),
      getPredictions({ limit, source: 'sha512' }),
      getPredictions({ limit, source: 'ng_consensus' }),
    ]);
    res.json({ ok: true, consensus,
      hlstm_xgb, htrans_lstm, htft, tft, nbeats, tcn,
      lgbm, gru, bilstm, stacking, sha512, ng_consensus });
  } catch(e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── ENGINE LOCKED ─────────────────────────────────────────────────────────────
app.get('/locked', rateLimit(120), async (req,res) => {
  try { res.json({ ok:true, preds: await getLockedPreds() }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/locked', rateLimit(120), async (req, res) => {
  if (APP_SECRET && req.headers['x-app-secret'] !== APP_SECRET)
    return res.status(403).json({ ok:false, error:'Forbidden' });
  try {
    const { preds } = req.body;
    if (!preds || typeof preds!=='object') return res.status(400).json({ ok:false, error:'preds required' });
    await saveLockedPreds(preds);
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/locked', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM locked_preds');
    require('./predictionEngine').resetEngineState();
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── PATTERN LOCKED ────────────────────────────────────────────────────────────
app.get('/locked-pattern', rateLimit(120), async (req,res) => {
  try { res.json({ ok:true, preds: await getLockedPatternPreds() }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/locked-pattern', rateLimit(120), async (req, res) => {
  if (APP_SECRET && req.headers['x-app-secret'] !== APP_SECRET)
    return res.status(403).json({ ok:false, error:'Forbidden' });
  try {
    const { preds } = req.body;
    if (!preds || typeof preds!=='object') return res.status(400).json({ ok:false, error:'preds required' });
    await saveLockedPatternPreds(preds);
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/locked-pattern', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM locked_preds_pattern');
    require('./predictionEngine').resetEngineState();
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── ADVANCED ENGINE LOCKED PREDS ──────────────────────────────────────────────
app.get('/locked-adv', rateLimit(120), async (req, res) => {
  try {
    res.json({ ok: true, preds: await getLockedAdvPreds() });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/locked-adv', rateLimit(120), async (req, res) => {
  try {
    const { model, preds } = req.body;
    if (!model || !preds || typeof preds !== 'object')
      return res.status(400).json({ ok:false, error:'model and preds required' });
    // FIX: validate model name is a known engine — reject arbitrary strings
    const VALID_ADV_MODELS = [
      'consensus',
      'hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','stacking','sha512','ng_consensus',
    ];
    if (!VALID_ADV_MODELS.includes(model))
      return res.status(400).json({ ok:false, error:'Unknown model' });
    await saveLockedAdvPreds(model, preds);
    try { require('./advResolutionEngine').bustLockedCache(); } catch(_) {}
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/locked-adv', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM locked_preds_adv');
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── CONSENSUS MASTER SIGNAL LOCKED PREDS ─────────────────────────────────────
app.get('/locked-consensus', rateLimit(120), async (req, res) => {
  try {
    res.json({ ok: true, preds: await getLockedConsensusPreds() });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/locked-consensus', rateLimit(120), async (req, res) => {
  try {
    const { preds } = req.body;
    if (!preds || typeof preds !== 'object')
      return res.status(400).json({ ok:false, error:'preds required' });
    await saveLockedConsensusPreds(preds);
    try { require('./advResolutionEngine').bustLockedCache(); } catch(_) {}
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/locked-consensus', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM locked_preds_consensus');
    res.json({ ok:true });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── RESET ENGINE LOCKS ONLY ───────────────────────────────────────────────────
app.delete('/reset-locks', requireAdmin, async (req, res) => {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query('DELETE FROM locked_preds_adv');
    await client.query('DELETE FROM locked_preds_consensus');
    await client.query('DELETE FROM locked_preds');
    await client.query('DELETE FROM locked_preds_stat');
    // FIX: also clear pattern locks — they are engine locks, not history
    await client.query('DELETE FROM locked_preds_pattern');
    await client.query('COMMIT');
    // Bust NG compute engine in-memory window cache so fresh windows are locked next tick
    try { require('./ngComputeEngine').resetNgWindowsOnly(); } catch(_) {}
    try { require('./advResolutionEngine').bustLockedCache(); } catch(_) {}
    console.log('[api] /reset-locks — all locked windows cleared, predictions + rounds intact');
    res.json({ ok: true, message: 'Engine locks cleared — predictions and rounds preserved' });
  } catch(e) {
    await client.query('ROLLBACK');
    res.status(500).json({ ok: false, error: e.message });
  } finally {
    client.release();
  }
});

// ── CLEAR HISTORY ONLY ────────────────────────────────────────────────────────
app.delete('/clear-history', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM predictions');
    // FIX: also reset advResolution savedSet so server re-resolves all
    // existing locked windows after history is cleared
    require('./predictionEngine').resetEngineState();
    console.log('[api] /clear-history — predictions cleared, engine savedSets reset');
    res.json({ ok:true, message:'Prediction history cleared' });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── FULL RESET ────────────────────────────────────────────────────────────────
app.delete('/reset', requireAdmin, async (req, res) => {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query('DELETE FROM predictions');
    await client.query('DELETE FROM locked_preds');
    await client.query('DELETE FROM locked_preds_pattern');
    await client.query('DELETE FROM locked_preds_stat');
    await client.query('DELETE FROM locked_preds_adv');
    await client.query('DELETE FROM locked_preds_consensus');
    await client.query('COMMIT');
    require('./predictionEngine').resetEngineState();
    try { require('./advResolutionEngine').bustLockedCache(); } catch(_) {}
    console.log('[api] /reset — all prediction data cleared including adv engines + consensus');
    res.json({ ok:true, message:'All prediction data cleared and engine reset' });
  } catch(e) {
    await client.query('ROLLBACK');
    res.status(500).json({ ok:false, error:e.message });
  } finally {
    client.release();
  }
});

// ── ACCESS CODES ──────────────────────────────────────────────────────────────
app.post('/access/verify', rateLimit(20), async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) return res.json({ ok:false, reason:'no_code' });
    const row = await getAccessCode(code.trim());
    if (!row) return res.json({ ok:false, reason:'invalid' });
    if (new Date(row.expires_at) < new Date()) return res.json({ ok:false, reason:'expired' });
    const ip = getIP(req), sameIP = row.ip && row.ip === ip;
    if (row.use_count >= row.max_uses && !sameIP) return res.json({ ok:false, reason:'used_up' });
    if (!row.ip || (!sameIP && row.use_count < row.max_uses)) await updateAccessCodeIP(code.trim(), ip);
    res.json({ ok:true, expiresAt:row.expires_at });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/access/create', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const { code, expiresAt, note, maxUses } = req.body;
    if (!code || !expiresAt) return res.status(400).json({ ok:false, error:'code and expiresAt required' });
    res.json({ ok:true, row: await createAccessCode({ code, expiresAt, note, maxUses:maxUses||1 }) });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.get('/access/list', requireAdmin, rateLimit(20), async (req,res) => {
  try { res.json({ ok:true, codes: await getAllAccessCodes() }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/access/:id', requireAdmin, rateLimit(10), async (req,res) => {
  try { await deleteAccessCode(req.params.id); res.json({ ok:true }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

// ── WALLETS ───────────────────────────────────────────────────────────────────
app.get('/wallets', requireAdmin, rateLimit(20), async (req,res) => {
  try { res.json({ ok:true, wallets: await getWallets() }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.post('/wallets', requireAdmin, rateLimit(20), async (req, res) => {
  try {
    const { privateKey, rpcUrl, playerAccountPDA, pubkey } = req.body;
    if (!privateKey) return res.status(400).json({ ok:false, error:'privateKey required' });
    res.json({ ok:true, wallet: await saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey }) });
  } catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

app.delete('/wallets/:id', requireAdmin, rateLimit(10), async (req,res) => {
  try { await deleteWallet(req.params.id); res.json({ ok:true }); }
  catch(e) { res.status(500).json({ ok:false, error:e.message }); }
});

function startAPI() {
  initAccessCodes().catch(e => console.error('initAccessCodes error:', e.message));
  initWalletStorage().catch(e => console.error('initWalletStorage error:', e.message));
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI };