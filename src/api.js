'use strict';
const express    = require('express');
const cors       = require('cors');
const { buildPredictionReport } = require('./predictionEngine');
const { computeLockedRangePredictions } = require('./lockedRangeEngine');
const {
  pool,
  getLatestRoundId, getRoundCount,
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

function toPositiveInt(value, fallback) {
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

const PREDICT_DEFAULT_LIMIT = toPositiveInt(process.env.PREDICT_DEFAULT_LIMIT, 12000);
const PREDICT_MAX_LIMIT = Math.max(PREDICT_DEFAULT_LIMIT, toPositiveInt(process.env.PREDICT_MAX_LIMIT, 25000));
const LOCKED_DEFAULT_LIMIT = toPositiveInt(process.env.LOCKED_DEFAULT_LIMIT, 15000);
const LOCKED_MAX_LIMIT = Math.max(LOCKED_DEFAULT_LIMIT, toPositiveInt(process.env.LOCKED_MAX_LIMIT, 25000));
const ROUNDS_DEFAULT_LIMIT = toPositiveInt(process.env.ROUNDS_DEFAULT_LIMIT, 5000);
const ROUNDS_MAX_LIMIT = Math.max(ROUNDS_DEFAULT_LIMIT, toPositiveInt(process.env.ROUNDS_MAX_LIMIT, 25000));
const PREDICT_CACHE_TTL_MS = toPositiveInt(process.env.PREDICT_CACHE_TTL_MS, 15000);
const LOCKED_CACHE_TTL_MS = toPositiveInt(process.env.LOCKED_CACHE_TTL_MS, 15000);
const LOCKED_USE_FULL_DATA = String(process.env.LOCKED_USE_FULL_DATA || 'true').trim().toLowerCase() !== 'false';
const LOCKED_BACKGROUND_INTERVAL_MS = toPositiveInt(process.env.LOCKED_BACKGROUND_INTERVAL_MS, 30000);

const predictCache = {
  asOfRound: null,
  limit: null,
  createdAt: 0,
  payload: null,
};
const lockedCache = {
  asOfRound: null,
  limit: null,
  createdAt: 0,
  basePayload: null,
};
let predictComputeInFlight = null;
let lockedComputeInFlight = null;
let lockedBackgroundTimer = null;
let dbState = {
  available: true,
  lastError: '',
  since: new Date().toISOString(),
};

function setDatabaseAvailability(available, errorMessage = '') {
  const nextAvailable = Boolean(available);
  const nextError = nextAvailable ? '' : String(errorMessage || 'Database unavailable');
  const changed = dbState.available !== nextAvailable || dbState.lastError !== nextError;
  if (changed) {
    dbState = {
      available: nextAvailable,
      lastError: nextError,
      since: new Date().toISOString(),
    };
    if (nextAvailable) {
      console.log('[db-state] ONLINE');
    } else {
      console.error('[db-state] OFFLINE:', nextError);
    }
  }
}

function requireDatabase(req, res, next) {
  if (dbState.available) return next();
  return res.status(503).json({
    ok: false,
    code: 'DB_UNAVAILABLE',
    error: 'Database temporarily unavailable',
    detail: dbState.lastError || 'Unknown database error',
    since: dbState.since,
  });
}

function invalidatePredictionCaches() {
  predictCache.asOfRound = null;
  predictCache.limit = null;
  predictCache.createdAt = 0;
  predictCache.payload = null;
  predictComputeInFlight = null;
  lockedCache.asOfRound = null;
  lockedCache.limit = null;
  lockedCache.createdAt = 0;
  lockedCache.basePayload = null;
  lockedComputeInFlight = null;
}

function normalizeHistoryTarget(raw) {
  const v = String(raw || '').trim().toLowerCase();
  if (!v) return null;
  return v.endsWith('x') ? v : `${v}x`;
}

function summarizeHistory(rows) {
  const out = (rows || []).reduce((acc, h) => {
    if (h.outcome === 'win') acc.win++;
    else if (h.outcome === 'early') acc.early++;
    else if (h.outcome === 'loss') acc.loss++;
    return acc;
  }, { win: 0, early: 0, loss: 0, total: (rows || []).length });
  const base = out.win + out.loss;
  out.accuracy = base > 0 ? Number((out.win / base).toFixed(4)) : null;
  return out;
}

function sameLock(a, b) {
  if (!a || !b) return false;
  return (
    Number(a.lo) === Number(b.lo) &&
    Number(a.hi) === Number(b.hi) &&
    Number(a.roundWhenMade ?? a.round_when_made) === Number(b.roundWhenMade ?? b.round_when_made) &&
    Number(a.generation || 1) === Number(b.generation || 1)
  );
}

function locksNeedSave(existing, next) {
  const keys = new Set([
    ...Object.keys(existing || {}),
    ...Object.keys(next || {}),
  ]);
  for (const k of keys) {
    if (!sameLock(existing?.[k], next?.[k])) return true;
  }
  return false;
}

function withHistoryFilter(basePayload, historyTarget) {
  const allRows = basePayload?.historyAll || [];
  const { historyAll, ...rest } = basePayload || {};
  const filtered = historyTarget
    ? allRows.filter(h => String(h.target || '').toLowerCase() === historyTarget)
    : allRows;
  return {
    ...rest,
    history: filtered,
    historySummary: summarizeHistory(filtered),
    historyFilter: historyTarget || 'all',
  };
}

function parseLockedLimit(rawLimit) {
  const raw = String(rawLimit ?? '').trim().toLowerCase();
  if (!raw) {
    if (LOCKED_USE_FULL_DATA) return { limit: null, limitKey: 'all' };
    return { limit: LOCKED_DEFAULT_LIMIT, limitKey: String(LOCKED_DEFAULT_LIMIT) };
  }
  if (raw === 'all' || raw === 'full' || raw === 'max') {
    return { limit: null, limitKey: 'all' };
  }
  const limit = clampInt(rawLimit, 2000, LOCKED_MAX_LIMIT, LOCKED_DEFAULT_LIMIT);
  return { limit, limitKey: String(limit) };
}

async function getRoundsForLockedPrediction(limit) {
  if (limit == null) {
    const total = await getRoundCount();
    if (!total) return [];
    return getRounds({ limit: total, order: 'ASC' });
  }
  return getRounds({ limit, order: 'DESC' });
}

async function computeAndPersistLockedPrediction({ latestRound, limit, limitKey }) {
  const [rounds, locked, historyRows] = await Promise.all([
    getRoundsForLockedPrediction(limit),
    getLockedConsensusPreds(),
    getPredictions({ limit: 3000, source: 'range_lock_v1' }),
  ]);

  const engine = computeLockedRangePredictions(rounds, locked, { historyRows });

  if (Object.keys(engine.locksToSave || {}).length && locksNeedSave(locked, engine.locksToSave)) {
    await saveLockedConsensusPreds(engine.locksToSave);
  }

  let savedResolvedCount = 0;
  if (Array.isArray(engine.resolvedHistory) && engine.resolvedHistory.length) {
    await Promise.all(engine.resolvedHistory.map((row) => savePrediction({
      target: row.target,
      minMult: row.minMult,
      outcome: row.outcome,
      lo: row.lo,
      hi: row.hi,
      hitRound: row.hitRound,
      generation: row.generation || 1,
      source: 'range_lock_v1',
      probW: row.confidence ?? null,
    })));
    savedResolvedCount = engine.resolvedHistory.length;
  }

  const fullHistory = savedResolvedCount > 0
    ? await getPredictions({ limit: 1200, source: 'range_lock_v1' })
    : historyRows.slice(0, 1200);

  const byTarget = {};
  for (const t of ['5x', '10x', '20x', '50x', '100x', '500x', '1000x']) {
    byTarget[t] = summarizeHistory(fullHistory.filter(h => String(h.target || '').toLowerCase() === t));
  }

  const basePayload = {
    ok: true,
    ...engine,
    historyAll: fullHistory,
    historyByTarget: byTarget,
    historyStorage: 'postgres',
    savedResolvedCount,
  };

  lockedCache.asOfRound = engine?.asOfRound ?? latestRound ?? null;
  lockedCache.limit = limitKey;
  lockedCache.createdAt = Date.now();
  lockedCache.basePayload = basePayload;
  return basePayload;
}

async function ensureLockedPredictionComputed({ latestRound, limit, limitKey }) {
  const sameInFlight = (
    lockedComputeInFlight &&
    lockedComputeInFlight.latestRound === latestRound &&
    lockedComputeInFlight.limitKey === limitKey
  );
  if (sameInFlight) return lockedComputeInFlight.promise;

  const promise = computeAndPersistLockedPrediction({ latestRound, limit, limitKey });
  lockedComputeInFlight = { latestRound, limitKey, promise };
  try {
    return await promise;
  } finally {
    if (lockedComputeInFlight?.promise === promise) {
      lockedComputeInFlight = null;
    }
  }
}

async function refreshLockedPredictionInBackground() {
  try {
    const latestRound = await getLatestRoundId();
    if (latestRound == null) return;
    const { limit, limitKey } = parseLockedLimit(null);
    const cacheFresh = (
      lockedCache.basePayload &&
      lockedCache.asOfRound != null &&
      lockedCache.asOfRound === latestRound &&
      lockedCache.limit === limitKey &&
      (Date.now() - lockedCache.createdAt) < LOCKED_CACHE_TTL_MS
    );
    if (cacheFresh) return;
    await ensureLockedPredictionComputed({ latestRound, limit, limitKey });
  } catch (e) {
    console.error('[locked-bg] refresh error:', e.message);
  }
}

app.get('/rounds', requireDatabase, rateLimit(60), async (req, res) => {
  try {
    const limit      = clampInt(req.query.limit, 100, ROUNDS_MAX_LIMIT, ROUNDS_DEFAULT_LIMIT);
    const offset     = parseInt(req.query.offset || '0');
    const since      = req.query.since ? Number(req.query.since) : null;
    const minRoundId = since && since > 0 ? since + 1 : null;
    const rounds = await getRounds({ limit, offset, from: req.query.from||null, to: req.query.to||null, minRoundId });
    res.json({ ok:true, count: rounds.length, rounds });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    console.error('[rounds] error:', e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.get('/stats', requireDatabase, rateLimit(30), async (req,res) => {
  try {
    res.json({ ok:true, ...(await getStats()) });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    console.error('[stats] error:', e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});
app.get('/storage-stats', requireDatabase, rateLimit(20), async (req,res) => {
  try {
    res.json({ ok:true, ...(await getStorageStats()) });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    console.error('[storage-stats] error:', e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});
app.get('/health', (req,res) => {
  res.json({
    ok: true,
    status: dbState.available ? 'online' : 'degraded',
    dbAvailable: dbState.available,
    dbError: dbState.lastError || null,
    dbSince: dbState.since,
    ts: new Date().toISOString(),
  });
});

app.get('/predict', requireDatabase, rateLimit(20), async (req, res) => {
  try {
    const limit = clampInt(req.query.limit, 1000, PREDICT_MAX_LIMIT, PREDICT_DEFAULT_LIMIT);

    const latestRound = await getLatestRoundId();
    const cacheFresh = (
      predictCache.payload &&
      predictCache.asOfRound != null &&
      predictCache.asOfRound === latestRound &&
      predictCache.limit === limit &&
      (Date.now() - predictCache.createdAt) < PREDICT_CACHE_TTL_MS
    );
    if (cacheFresh) {
      return res.json(predictCache.payload);
    }

    const sameInFlight = (
      predictComputeInFlight &&
      predictComputeInFlight.latestRound === latestRound &&
      predictComputeInFlight.limit === limit
    );
    if (sameInFlight) {
      const payload = await predictComputeInFlight.promise;
      return res.json(payload);
    }

    const promise = (async () => {
      const rounds = await getRounds({ limit, order: 'DESC' });
      const report = buildPredictionReport(rounds);
      const payload = { ok: true, ...report };
      predictCache.asOfRound = report?.asOfRound ?? latestRound ?? null;
      predictCache.limit = limit;
      predictCache.createdAt = Date.now();
      predictCache.payload = payload;
      return payload;
    })();

    predictComputeInFlight = { latestRound, limit, promise };
    try {
      const payload = await promise;
      return res.json(payload);
    } finally {
      if (predictComputeInFlight?.promise === promise) {
        predictComputeInFlight = null;
      }
    }
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/predict/locked', requireDatabase, rateLimit(20), async (req, res) => {
  try {
    const { limit, limitKey } = parseLockedLimit(req.query.limit);
    const historyTarget = normalizeHistoryTarget(req.query.historyTarget);

    const latestRound = await getLatestRoundId();
    const cacheFresh = (
      lockedCache.basePayload &&
      lockedCache.asOfRound != null &&
      lockedCache.asOfRound === latestRound &&
      lockedCache.limit === limitKey &&
      (Date.now() - lockedCache.createdAt) < LOCKED_CACHE_TTL_MS
    );

    if (cacheFresh) {
      return res.json(withHistoryFilter(lockedCache.basePayload, historyTarget));
    }

    const basePayload = await ensureLockedPredictionComputed({ latestRound, limit, limitKey });
    return res.json(withHistoryFilter(basePayload, historyTarget));
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

const clearHistoryHandler = async (req, res) => {
  try {
    const result = await clearPredictions();
    invalidatePredictionCaches();
    res.json({ ok: true, ...result });
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
};
app.delete('/clear-history', requireDatabase, requireAdmin, rateLimit(10), clearHistoryHandler);
app.post('/clear-history', requireDatabase, requireAdmin, rateLimit(10), clearHistoryHandler);

const clearLocksHandler = async (req, res) => {
  try {
    const result = await clearAllLocks();
    invalidatePredictionCaches();
    res.json({ ok: true, ...result });
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
};
app.delete('/clear-locks', requireDatabase, requireAdmin, rateLimit(10), clearLocksHandler);
app.post('/clear-locks', requireDatabase, requireAdmin, rateLimit(10), clearLocksHandler);

app.post('/access/verify', requireDatabase, rateLimit(20), async (req, res) => {
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
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.post('/access/create', requireDatabase, requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const { code, expiresAt, note, maxUses } = req.body;
    if (!code || !expiresAt) return res.status(400).json({ ok:false, error:'code and expiresAt required' });
    res.json({ ok:true, row: await createAccessCode({ code, expiresAt, note, maxUses:maxUses||1 }) });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.get('/access/list', requireDatabase, requireAdmin, rateLimit(20), async (req,res) => {
  try { res.json({ ok:true, codes: await getAllAccessCodes() }); }
  catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.delete('/access/:id', requireDatabase, requireAdmin, rateLimit(10), async (req,res) => {
  try { await deleteAccessCode(req.params.id); res.json({ ok:true }); }
  catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.get('/wallets', requireDatabase, requireAdmin, rateLimit(20), async (req,res) => {
  try { res.json({ ok:true, wallets: await getWallets() }); }
  catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.post('/wallets', requireDatabase, requireAdmin, rateLimit(20), async (req, res) => {
  try {
    const { privateKey, rpcUrl, playerAccountPDA, pubkey } = req.body;
    if (!privateKey) return res.status(400).json({ ok:false, error:'privateKey required' });
    res.json({ ok:true, wallet: await saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey }) });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.delete('/wallets/:id', requireDatabase, requireAdmin, rateLimit(10), async (req,res) => {
  try { await deleteWallet(req.params.id); res.json({ ok:true }); }
  catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

function startAPI() {
  initAccessCodes()
    .then(() => setDatabaseAvailability(true))
    .catch(e => {
      setDatabaseAvailability(false, e.message);
      console.error('initAccessCodes error:', e.message);
    });
  initWalletStorage().catch(e => {
    setDatabaseAvailability(false, e.message);
    console.error('initWalletStorage error:', e.message);
  });
  if (!lockedBackgroundTimer) {
    refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] warmup error:', e.message));
    lockedBackgroundTimer = setInterval(() => {
      refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] tick error:', e.message));
    }, LOCKED_BACKGROUND_INTERVAL_MS);
    console.log(`[locked-bg] running every ${LOCKED_BACKGROUND_INTERVAL_MS}ms (${LOCKED_USE_FULL_DATA ? 'full-data mode' : 'limited mode'})`);
  }
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI, setDatabaseAvailability };
