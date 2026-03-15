const express = require('express');
const cors    = require('cors');
const {
  getRounds, getStats, getStorageStats,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
} = require('./db');

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Real IP behind Railway/proxy
function getIP(req) {
  return (
    req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
    req.headers['x-real-ip'] ||
    req.socket?.remoteAddress ||
    'unknown'
  );
}

// Admin secret middleware — reads from ADMIN_SECRET env var
// Falls back to 'iamnoob' if not set (for dev), but logs a warning
const ADMIN_SECRET = process.env.ADMIN_SECRET || 'iamnoob';
if (!process.env.ADMIN_SECRET) {
  console.warn('⚠️  ADMIN_SECRET env var not set — using default "iamnoob". Set it in Railway for security.');
}

function requireAdmin(req, res, next) {
  const secret = req.headers['x-admin-secret'];
  if (!secret || secret !== ADMIN_SECRET) {
    return res.status(403).json({ ok: false, error: 'Forbidden — invalid admin secret' });
  }
  next();
}

// ── ROUNDS ────────────────────────────────────────────────────────────────────
app.get('/rounds', async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit  || '1000'), 10000);
    const offset = parseInt(req.query.offset || '0');
    const from   = req.query.from || null;
    const to     = req.query.to   || null;
    const rounds = await getRounds({ limit, offset, from, to });
    const since  = req.query.since ? Number(req.query.since) : 0;
    const result = since ? rounds.filter(r => r.roundId > since) : rounds;
    res.json({ ok: true, count: result.length, rounds: result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/stats', async (req, res) => {
  try { res.json({ ok: true, ...(await getStats()) }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.get('/storage-stats', async (req, res) => {
  try { res.json({ ok: true, ...(await getStorageStats()) }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.get('/health', (req, res) => res.json({ ok: true, ts: new Date().toISOString() }));

// ── PREDICTIONS ───────────────────────────────────────────────────────────────
app.get('/predictions', async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit || '200'), 1000);
    const target = req.query.target || null;
    const rows   = await getPredictions({ limit, target });
    res.json({ ok: true, count: rows.length, predictions: rows });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post('/predictions', async (req, res) => {
  try {
    const { target, minMult, outcome, lo, hi, hitRound, generation } = req.body;
    if (!target || !outcome || lo == null || hi == null)
      return res.status(400).json({ ok: false, error: 'Missing required fields' });
    await savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation });
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.delete('/predictions', async (req, res) => {
  try { await clearPredictions(); res.json({ ok: true }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ── ACCESS CODES ──────────────────────────────────────────────────────────────

// Verify a code (public — called by frontend on every load)
app.post('/access/verify', async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) return res.json({ ok: false, reason: 'no_code' });

    const row = await getAccessCode(code.trim());
    if (!row) return res.json({ ok: false, reason: 'invalid' });
    if (new Date(row.expires_at) < new Date()) return res.json({ ok: false, reason: 'expired' });

    // Check max uses — but allow re-verification from same IP (page refreshes)
    const ip = getIP(req);
    const isSameIP = row.ip && row.ip === ip;
    if (row.use_count >= row.max_uses && !isSameIP) {
      return res.json({ ok: false, reason: 'used_up' });
    }

    // Only increment use count if this is a NEW ip (first activation or new user)
    if (!row.ip || (!isSameIP && row.use_count < row.max_uses)) {
      await updateAccessCodeIP(code.trim(), ip);
    }

    res.json({ ok: true, expiresAt: row.expires_at });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// Create / update a code (admin only)
app.post('/access/create', requireAdmin, async (req, res) => {
  try {
    const { code, expiresAt, note, maxUses } = req.body;
    if (!code || !expiresAt)
      return res.status(400).json({ ok: false, error: 'code and expiresAt required' });
    const row = await createAccessCode({ code, expiresAt, note, maxUses: maxUses || 1 });
    res.json({ ok: true, row });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// List all codes (admin)
app.get('/access/list', requireAdmin, async (req, res) => {
  try {
    const rows = await getAllAccessCodes();
    res.json({ ok: true, codes: rows });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// Delete a code (admin)
app.delete('/access/:id', requireAdmin, async (req, res) => {
  try {
    await deleteAccessCode(req.params.id);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

function startAPI() {
  initAccessCodes().catch(e => console.error('initAccessCodes error:', e.message));
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI };