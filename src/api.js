const express    = require('express');
const cors       = require('cors');
const {
  getRounds, getStats, getStorageStats,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
  saveLockedPreds, getLockedPreds,
  initWalletStorage, saveWallet, getWallets, deleteWallet,
} = require('./db');

const app  = express();
const PORT = process.env.PORT || 3001;

// ── SECURITY: Allowed frontend origins only ───────────────────────────────
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '')
  .split(',').map(s => s.trim()).filter(Boolean);

app.use(cors({
  origin: (origin, cb) => {
    // Allow Railway internal calls (no origin) and configured origins
    if (!origin) return cb(null, true);
    if (ALLOWED_ORIGINS.length === 0) return cb(null, true); // dev fallback
    if (ALLOWED_ORIGINS.some(o => origin.startsWith(o))) return cb(null, true);
    cb(new Error('Not allowed by CORS'));
  },
  credentials: true,
}));

app.use(express.json({ limit: '10kb' })); // limit request body size

// ── SECURITY: Simple in-memory rate limiter ───────────────────────────────
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
    if (win.count > maxPerMin) {
      return res.status(429).json({ ok: false, error: 'Too many requests' });
    }
    next();
  };
}
// Clean up old rate limit entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of rateLimits) if (now > v.reset) rateLimits.delete(k);
}, 300000);

// ── SECURITY: Admin middleware ────────────────────────────────────────────
const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) console.warn('⚠️  ADMIN_SECRET not set — admin endpoints unprotected!');

function requireAdmin(req, res, next) {
  const secret = req.headers['x-admin-secret'];
  if (!ADMIN_SECRET || !secret || secret !== ADMIN_SECRET) {
    return res.status(403).json({ ok: false, error: 'Forbidden' });
  }
  next();
}

function getIP(req) {
  return req.headers['x-forwarded-for']?.split(',')[0]?.trim()
    || req.headers['x-real-ip']
    || req.socket?.remoteAddress
    || 'unknown';
}

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

// ── PREDICTIONS ───────────────────────────────────────────────────────────
app.get('/predictions', rateLimit(30), async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit || '200'), 1000);
    const target = req.query.target || null;
    const rows   = await getPredictions({ limit, target });
    res.json({ ok: true, count: rows.length, predictions: rows });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/predictions', rateLimit(120), async (req, res) => {
  try {
    const { target, minMult, outcome, lo, hi, hitRound, generation } = req.body;
    if (!target || !outcome || lo == null || hi == null)
      return res.status(400).json({ ok: false, error: 'Missing required fields' });
    // Input validation
    const validOutcomes = ['win','loss','early','retry-win','retry-loss'];
    if (!validOutcomes.includes(outcome))
      return res.status(400).json({ ok: false, error: 'Invalid outcome' });
    if (typeof lo !== 'number' || typeof hi !== 'number' || hi < lo)
      return res.status(400).json({ ok: false, error: 'Invalid window' });
    await savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation });
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// PROTECTED — require admin to delete all history
app.delete('/predictions', requireAdmin, async (req, res) => {
  try { await clearPredictions(); res.json({ ok: true }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── LOCKED PREDICTIONS ────────────────────────────────────────────────────
app.get('/locked', rateLimit(60), async (req, res) => {
  try { res.json({ ok: true, preds: await getLockedPreds() }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// PROTECTED — only the app itself should write locks (use a shared app secret)
const APP_SECRET = process.env.APP_SECRET || process.env.ADMIN_SECRET;
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

// ── ACCESS CODES ──────────────────────────────────────────────────────────
app.post('/access/verify', rateLimit(20), async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) return res.json({ ok: false, reason: 'no_code' });
    const row = await getAccessCode(code.trim());
    if (!row) return res.json({ ok: false, reason: 'invalid' });
    if (new Date(row.expires_at) < new Date()) return res.json({ ok: false, reason: 'expired' });
    const ip      = getIP(req);
    const sameIP  = row.ip && row.ip === ip;
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

// ── WALLET STORAGE ───────────────────────────────────────────────────────────
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
}

module.exports = { startAPI };