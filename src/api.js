'use strict';
const express    = require('express');
const cors       = require('cors');
const { buildPredictionReport } = require('./predictionEngine');
const { computeLockedRangePredictions } = require('./lockedRangeEngine');
const {
  pool,
  getRounds, getStats, getStorageStats,
  getPredictions, savePrediction, clearPredictions, clearAllLocks,
  getLockedConsensusPreds, saveLockedConsensusPreds,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
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
    if (!rateLimits.has(key) && rateLimits.size >= RL_MAX_KEYS) {
      rateLimits.delete(rateLimits.keys().next().value);
    }
    rateLimits.set(key, win);
    if (win.count > maxPerMin) return res.status(429).json({ ok: false, error: 'Too many requests' });
    next();
  };
}
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

// prediction engine (history + clusters + pattern matching)
app.get('/predict', rateLimit(20), async (req, res) => {
  try {
    const requestedLimit = parseInt(req.query.limit || '25000', 10);
    const limit = Number.isFinite(requestedLimit)
      ? Math.min(Math.max(requestedLimit, 1000), 100000)
      : 25000;

    const rounds = await getRounds({ limit, order: 'ASC' });
    const report = buildPredictionReport(rounds);
    res.json({ ok: true, ...report });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// locked target windows (5x..1000x) persisted in DB
app.get('/predict/locked', rateLimit(20), async (req, res) => {
  try {
    const requestedLimit = parseInt(req.query.limit || '100000', 10);
    const limit = Number.isFinite(requestedLimit)
      ? Math.min(Math.max(requestedLimit, 2000), 100000)
      : 100000;

    const historyTargetRaw = String(req.query.historyTarget || '').trim();
    const historyTarget = historyTargetRaw
      ? (historyTargetRaw.toLowerCase().endsWith('x') ? historyTargetRaw.toLowerCase() : `${historyTargetRaw.toLowerCase()}x`)
      : null;

    const [rounds, locked, historyForCalibration] = await Promise.all([
      getRounds({ limit, order: 'ASC' }),
      getLockedConsensusPreds(),
      getPredictions({ limit: 600, source: 'range_lock_v1' }),
    ]);

    const engine = computeLockedRangePredictions(rounds, locked, {
      historyRows: historyForCalibration,
    });

    if (Object.keys(engine.locksToSave || {}).length) {
      await saveLockedConsensusPreds(engine.locksToSave);
    }

    let savedResolvedCount = 0;
    if (Array.isArray(engine.resolvedHistory) && engine.resolvedHistory.length) {
      for (const row of engine.resolvedHistory) {
        await savePrediction({
          target: row.target,
          minMult: row.minMult,
          outcome: row.outcome,
          lo: row.lo,
          hi: row.hi,
          hitRound: row.hitRound,
          generation: row.generation || 1,
          source: 'range_lock_v1',
          probW: null,
        });
        savedResolvedCount++;
      }
    }
    
    const fullHistory = await getPredictions({ limit: 600, source: 'range_lock_v1' });
    const history = historyTarget
      ? fullHistory.filter(h => String(h.target || '').toLowerCase() === historyTarget)
      : fullHistory;

    const summarize = (rows) => {
      const out = rows.reduce((acc, h) => {
        if (h.outcome === 'win') acc.win++;
        else if (h.outcome === 'early') acc.early++;
        else if (h.outcome === 'loss') acc.loss++;
        return acc;
      }, { win: 0, early: 0, loss: 0, total: rows.length });
      const base = out.win + out.loss;
      out.accuracy = base > 0 ? Number((out.win / base).toFixed(4)) : null;
      return out;
    };

    const summaryFromHistory = summarize(history);

    const byTarget = {};
    for (const t of ['5x', '10x', '20x', '50x', '100x', '500x', '1000x']) {
      byTarget[t] = summarize(fullHistory.filter(h => String(h.target || '').toLowerCase() === t));
    }

    res.json({
      ok: true,
      ...engine,
      history,
      historySummary: summaryFromHistory,
      historyByTarget: byTarget,
      historyFilter: historyTarget || 'all',
      historyStorage: 'postgres',
      savedResolvedCount,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── ACCESS CODES ──────────────────────────────────────────────────────────────
app.delete('/clear-history', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const result = await clearPredictions();
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.delete('/clear-locks', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const result = await clearAllLocks();
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

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
