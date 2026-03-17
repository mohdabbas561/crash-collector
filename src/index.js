const express    = require('express');
const cors       = require('cors');
const {
  getRounds, getStats, getStorageStats,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  initWalletStorage, saveWallet, getWallets, deleteWallet,
} = require('./db');

// FIX: import resetEngineState so we can clear in-memory state after DB wipes
const { resetEngineState } = require('./predictionEngine');

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

const rateLimits = new Map();
function rateLimit(maxPerMin) {
  return (req, res, next) => {
    const ip  = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || 'unknown';
    const key = `${ip}:${req.path}`;
    const now = Date.now();
    const win = rateLimits.get(key) || { count: 0, reset: now + 60000 };
    if (now > win.reset) { win.count = 0; win.reset = now + 60000; }
    win.count++;
    rateLimits.set(key, win);
    if (win.count > maxPerMin) return res.status(429).json({ ok: false, error: 'Too many requests' });
    next();
  };
}
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of rateLimits) if (now > v.reset) rateLimits.delete(k);
}, 300000);

const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) console.warn('⚠️  ADMIN_SECRET not set');

function requireAdmin(req, res, next) {
  const secret = req.headers['x-admin-secret'];
  if (!ADMIN_SECRET || !secret || secret !== ADMIN_SECRET)
    return res.status(403).json({ ok: false, error: 'Forbidden' });
  next();
}

function getIP(req) {
  return req.headers['x-forwarded-for']?.split(',')[0]?.trim()
    || req.headers['x-real-ip']
    || req.socket?.remoteAddress
    || 'unknown';
}

const APP_SECRET = process.env.APP_SECRET || process.env.ADMIN_SECRET;

// ── ROUNDS ────────────────────────────────────────────────────────────────
app.get('/rounds', rateLimit(60), async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit  || '1000'), 10000);
    const offset = parseInt(req.query.offset || '0');
    const from   = req.query.from || null;
    const to     = req.query.to   || null;
    const rounds = await getRounds({ limit, offset, from, to });
    const since  = req.query.since ? Number(req.query.since) : 0;
    const result = since ? rounds.filter(r => r.roundId > since) : rounds;
    res.json({ ok: true, count: result.length, rounds: result });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.get('/stats',         rateLimit(30), async (req, res) => { try { res.json({ ok: true, ...(await getStats()) }); } catch (e) { res.status(500).json({ ok: false, error: e.message }); } });
app.get('/storage-stats', rateLimit(20), async (req, res) => { try { res.json({ ok: true, ...(await getStorageStats()) }); } catch (e) { res.status(500).json({ ok: false, error: e.message }); } });
app.get('/health', (req, res) => res.json({ ok: true, ts: new Date().toISOString() }));

// ── PREDICTIONS HISTORY ───────────────────────────────────────────────────
app.get('/predictions', rateLimit(60), async (req, res) => {
  try {
    const limit     = Math.min(parseInt(req.query.limit || '500'), 5000);
    const target    = req.query.target || null;
    const rawSource = req.query.source || null;
    // FIX: accept all valid sources including ens/geo/bay/km stat models
    const validSources = ['engine', 'pattern', 'ens', 'geo', 'bay', 'km'];
    const source = validSources.includes(rawSource) ? rawSource : null;
    const rows   = await getPredictions({ limit, target, source });
    res.json({ ok: true, count: rows.length, predictions: rows });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/predictions', rateLimit(120), async (req, res) => {
  try {
    const { target, minMult, outcome, lo, hi, hitRound, generation, source } = req.body;
    if (!target || !outcome || lo == null || hi == null)
      return res.status(400).json({ ok: false, error: 'Missing required fields' });
    const validOutcomes = ['win','loss','early','retry-win','retry-loss'];
    if (!validOutcomes.includes(outcome))
      return res.status(400).json({ ok: false, error: 'Invalid outcome' });
    if (typeof lo !== 'number' || typeof hi !== 'number' || hi < lo || !isFinite(lo) || !isFinite(hi))
      return res.status(400).json({ ok: false, error: 'Invalid window' });
    // FIX: accept ens/geo/bay/km as valid sources from frontend stat tracker
    const validSources = ['engine', 'pattern', 'ens', 'geo', 'bay', 'km'];
    const validSource = validSources.includes(source) ? source : 'engine';
    await savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation, source: validSource });
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// FIX: call resetEngineState() after clearing predictions so in-memory state
// is wiped and the next poll starts fresh from the (now-empty) DB.
app.delete('/predictions', requireAdmin, async (req, res) => {
  try {
    await clearPredictions();
    resetEngineState(); // FIX A+B+C: wipe in-memory lockedPreds, savedKeys, etc.
    console.log('[api] /predictions deleted + engine state reset');
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── ENGINE LOCKED PREDS ───────────────────────────────────────────────────
app.get('/locked', rateLimit(120), async (req, res) => {
  try { res.json({ ok: true, preds: await getLockedPreds() }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/locked', rateLimit(120), async (req, res) => {
  const token = req.headers['x-app-secret'];
  if (APP_SECRET && token !== APP_SECRET)
    return res.status(403).json({ ok: false, error: 'Forbidden' });
  try {
    const { preds } = req.body;
    if (!preds || typeof preds !== 'object')
      return res.status(400).json({ ok: false, error: 'preds object required' });
    await saveLockedPreds(preds);
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// FIX: add DELETE for locked preds — also resets engine state
app.delete('/locked', requireAdmin, async (req, res) => {
  try {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
    await pool.query('DELETE FROM locked_preds');
    await pool.end();
    resetEngineState(); // FIX: wipe stale in-memory windows
    console.log('[api] /locked deleted + engine state reset');
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── PATTERN LOCKED PREDS ──────────────────────────────────────────────────
app.get('/locked-pattern', rateLimit(120), async (req, res) => {
  try { res.json({ ok: true, preds: await getLockedPatternPreds() }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/locked-pattern', rateLimit(120), async (req, res) => {
  const token = req.headers['x-app-secret'];
  if (APP_SECRET && token !== APP_SECRET)
    return res.status(403).json({ ok: false, error: 'Forbidden' });
  try {
    const { preds } = req.body;
    if (!preds || typeof preds !== 'object')
      return res.status(400).json({ ok: false, error: 'preds object required' });
    await saveLockedPatternPreds(preds);
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// FIX: add DELETE for pattern locked preds
app.delete('/locked-pattern', requireAdmin, async (req, res) => {
  try {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
    await pool.query('DELETE FROM locked_preds_pattern');
    await pool.end();
    resetEngineState(); // FIX: wipe stale in-memory windows
    console.log('[api] /locked-pattern deleted + engine state reset');
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── ADMIN: full reset endpoint ────────────────────────────────────────────
// DELETE /reset — clears ALL prediction state (predictions + locked preds)
// and resets in-memory engine state in one shot.
app.delete('/reset', requireAdmin, async (req, res) => {
  try {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
    await pool.query('DELETE FROM predictions');
    await pool.query('DELETE FROM locked_preds');
    await pool.query('DELETE FROM locked_preds_pattern');
    await pool.end();
    resetEngineState();
    console.log('[api] /reset — all prediction tables cleared + engine state reset');
    res.json({ ok: true, message: 'All prediction data cleared and engine reset' });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── ACCESS CODES ──────────────────────────────────────────────────────────
app.post('/access/verify', rateLimit(20), async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) return res.json({ ok: false, reason: 'no_code' });
    const row = await getAccessCode(code.trim());
    if (!row) return res.json({ ok: false, reason: 'invalid' });
    if (new Date(row.expires_at) < new Date()) return res.json({ ok: false, reason: 'expired' });
    const ip     = getIP(req);
    const sameIP = row.ip && row.ip === ip;
    if (row.use_count >= row.max_uses && !sameIP)
      return res.json({ ok: false, reason: 'used_up' });
    if (!row.ip || (!sameIP && row.use_count < row.max_uses))
      await updateAccessCodeIP(code.trim(), ip);
    res.json({ ok: true, expiresAt: row.expires_at });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/access/create', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const { code, expiresAt, note, maxUses } = req.body;
    if (!code || !expiresAt)
      return res.status(400).json({ ok: false, error: 'code and expiresAt required' });
    const row = await createAccessCode({ code, expiresAt, note, maxUses: maxUses || 1 });
    res.json({ ok: true, row });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.get('/access/list', requireAdmin, rateLimit(20), async (req, res) => {
  try { res.json({ ok: true, codes: await getAllAccessCodes() }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.delete('/access/:id', requireAdmin, rateLimit(10), async (req, res) => {
  try { await deleteAccessCode(req.params.id); res.json({ ok: true }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── WALLET STORAGE ────────────────────────────────────────────────────────
app.get('/wallets', requireAdmin, rateLimit(20), async (req, res) => {
  try { res.json({ ok: true, wallets: await getWallets() }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/wallets', requireAdmin, rateLimit(20), async (req, res) => {
  try {
    const { privateKey, rpcUrl, playerAccountPDA, pubkey } = req.body;
    if (!privateKey) return res.status(400).json({ ok: false, error: 'privateKey required' });
    const row = await saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey });
    res.json({ ok: true, wallet: row });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.delete('/wallets/:id', requireAdmin, rateLimit(10), async (req, res) => {
  try { await deleteWallet(req.params.id); res.json({ ok: true }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

function startAPI() {
  initAccessCodes().catch(e => console.error('initAccessCodes error:', e.message));
  initWalletStorage().catch(e => console.error('initWalletStorage error:', e.message));
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));

  const { runPredictionEngine } = require('./predictionEngine');

// Start the prediction engine
runPredictionEngine();
setInterval(runPredictionEngine, 10000);

// Start the API server
startAPI();
}

module.exports = { startAPI };