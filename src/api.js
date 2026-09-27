'use strict';

const express = require('express');
const cors = require('cors');
const { buildPredictionReport } = require('./predictionEngine');
const { computeLockedRangePredictions } = require('./lockedRangeEngine');
const {
  getLatestRoundId,
  getRounds,
  getStats,
  getStorageStats,
  getPredictions,
  savePrediction,
  clearPredictions,
  clearAllLocks,
  getLockedConsensusPreds,
  saveLockedConsensusPreds,
  initAccessCodes,
  createAccessCode,
  getAccessCode,
  updateAccessCodeIP,
  getAllAccessCodes,
  deleteAccessCode,
  initWalletStorage,
  saveWallet,
  getWallets,
  deleteWallet,
} = require('./db');

const app = express();
const PORT = process.env.PORT || 3001;

const LOCKED_SOURCE = 'range_lock_v1';
const TARGET_LABELS = ['5x', '10x', '20x', '50x', '100x', '500x', '1000x'];

const PREDICT_CACHE_TTL_MS = 15000;
const LOCKED_CACHE_TTL_MS = 15000;
const PREDICT_DEFAULT_LIMIT = 25000;
const PREDICT_MIN_LIMIT = 1000;
const PREDICT_MAX_LIMIT = 60000;
const LOCKED_DEFAULT_LIMIT = Number(process.env.LOCKED_DEFAULT_LIMIT || 60000);
const LOCKED_MIN_LIMIT = 2000;
const LOCKED_MAX_LIMIT = 60000;
const LOCKED_HISTORY_CALIB_LIMIT = 3000;
const LOCKED_HISTORY_PUBLIC_LIMIT = 1200;
const LOCKED_CATCHUP_MAX_STEPS = Number(process.env.LOCKED_CATCHUP_MAX_STEPS || 36);
const LOCKED_WORKER_INTERVAL_MS = Number(process.env.LOCKED_WORKER_INTERVAL_MS || 2500);

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (ALLOWED_ORIGINS.length === 0) return cb(null, true);
    if (ALLOWED_ORIGINS.some((o) => origin.startsWith(o))) return cb(null, true);
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

const rateLimits = new Map();
const RL_MAX_KEYS = 10000;

function rateLimit(maxPerMin) {
  return (req, res, next) => {
    const ip = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || 'unknown';
    const key = `${ip}:${req.path}`;
    const now = Date.now();
    const win = rateLimits.get(key) || { count: 0, reset: now + 60000 };
    if (now > win.reset) {
      win.count = 0;
      win.reset = now + 60000;
    }
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
  for (const [k, v] of rateLimits) {
    if (now > v.reset) rateLimits.delete(k);
  }
}, 60000);

const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) console.warn('ADMIN_SECRET not set');

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

function normalizeLimit(raw, fallback, min, max) {
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

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

const predictState = {
  inFlight: null,
};

const lockedWorkerState = {
  inFlight: null,
  lastProcessedRound: null,
  lastRunAt: 0,
  lastError: null,
};

let lockedWorkerTimer = null;

function invalidatePredictionCaches() {
  predictCache.asOfRound = null;
  predictCache.limit = null;
  predictCache.createdAt = 0;
  predictCache.payload = null;

  lockedCache.asOfRound = null;
  lockedCache.limit = null;
  lockedCache.createdAt = 0;
  lockedCache.basePayload = null;

  lockedWorkerState.lastProcessedRound = null;
  lockedWorkerState.lastError = null;
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

function buildHistoryByTarget(rows) {
  const byTarget = {};
  for (const t of TARGET_LABELS) {
    byTarget[t] = { win: 0, early: 0, loss: 0, total: 0, accuracy: null };
  }
  for (const row of rows || []) {
    const key = String(row.target || '').toLowerCase();
    if (!byTarget[key]) continue;
    byTarget[key].total++;
    if (row.outcome === 'win') byTarget[key].win++;
    else if (row.outcome === 'early') byTarget[key].early++;
    else if (row.outcome === 'loss') byTarget[key].loss++;
  }
  for (const t of TARGET_LABELS) {
    const b = byTarget[t];
    const denom = b.win + b.loss;
    b.accuracy = denom > 0 ? Number((b.win / denom).toFixed(4)) : null;
  }
  return byTarget;
}

function sameLock(a, b) {
  if (!a || !b) return false;
  return (
    Number(a.lo) === Number(b.lo)
    && Number(a.hi) === Number(b.hi)
    && Number(a.roundWhenMade ?? a.round_when_made) === Number(b.roundWhenMade ?? b.round_when_made)
    && Number(a.generation || 1) === Number(b.generation || 1)
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
    ? allRows.filter((h) => String(h.target || '').toLowerCase() === historyTarget)
    : allRows;
  return {
    ...rest,
    history: filtered,
    historySummary: summarizeHistory(filtered),
    historyFilter: historyTarget || 'all',
  };
}

function mapResolvedToHistoryRows(resolvedRows) {
  return (resolvedRows || []).map((row) => ({
    target: row.target,
    minMult: row.minMult,
    outcome: row.outcome,
    lo: row.lo,
    hi: row.hi,
    hitRound: row.hitRound ?? null,
    generation: row.generation || 1,
    source: LOCKED_SOURCE,
    probW: row.confidence ?? null,
    ts: Date.now(),
  }));
}

function mergeHistoryRows(currentRows, resolvedRows, maxLen = LOCKED_HISTORY_CALIB_LIMIT) {
  const prepended = mapResolvedToHistoryRows(resolvedRows);
  if (!prepended.length) return (currentRows || []).slice(0, maxLen);
  return [...prepended, ...(currentRows || [])].slice(0, maxLen);
}

async function persistResolvedHistory(resolvedRows) {
  if (!Array.isArray(resolvedRows) || !resolvedRows.length) return 0;
  await Promise.all(resolvedRows.map((row) => savePrediction({
    target: row.target,
    minMult: row.minMult,
    outcome: row.outcome,
    lo: row.lo,
    hi: row.hi,
    hitRound: row.hitRound,
    generation: row.generation || 1,
    source: LOCKED_SOURCE,
    probW: row.confidence ?? null,
  })));
  return resolvedRows.length;
}

async function computePredictPayload(limit, latestRoundHint = null) {
  const rounds = await getRounds({ limit, order: 'ASC' });
  const report = buildPredictionReport(rounds);
  const payload = { ok: true, ...report };
  predictCache.asOfRound = report?.asOfRound ?? latestRoundHint ?? null;
  predictCache.limit = limit;
  predictCache.createdAt = Date.now();
  predictCache.payload = payload;
  return payload;
}

async function refreshPredictCache({ limit, latestRoundHint = null, force = false }) {
  if (!force && predictState.inFlight) return predictState.inFlight;
  if (!force
    && predictCache.payload
    && predictCache.limit === limit
    && latestRoundHint != null
    && predictCache.asOfRound === latestRoundHint
  ) {
    return predictCache.payload;
  }

  const task = computePredictPayload(limit, latestRoundHint)
    .catch((err) => {
      console.error('[predict] refresh failed:', err.message);
      throw err;
    })
    .finally(() => {
      if (predictState.inFlight === task) predictState.inFlight = null;
    });

  predictState.inFlight = task;
  return task;
}

function triggerPredictRefresh(limit, latestRoundHint = null) {
  if (predictState.inFlight) return;
  refreshPredictCache({ limit, latestRoundHint, force: true }).catch(() => {});
}

async function computeLockedSnapshot(limit, latestRoundHint = null) {
  const [rounds, persistedLocks, persistedHistory] = await Promise.all([
    getRounds({ limit, order: 'ASC' }),
    getLockedConsensusPreds(),
    getPredictions({ limit: LOCKED_HISTORY_CALIB_LIMIT, source: LOCKED_SOURCE }),
  ]);

  if (!rounds.length) return null;

  const latestRound = Number(rounds[rounds.length - 1].roundId);
  const roundIndexById = new Map();
  for (let i = 0; i < rounds.length; i++) {
    roundIndexById.set(Number(rounds[i].roundId), i);
  }

  let historyRows = (persistedHistory || []).slice(0, LOCKED_HISTORY_CALIB_LIMIT);
  let workingLocks = persistedLocks || {};

  const lastProcessed = Number(lockedWorkerState.lastProcessedRound || 0);
  const delta = latestRound - lastProcessed;
  const canCatchUp = lastProcessed > 0 && delta > 0 && delta <= LOCKED_CATCHUP_MAX_STEPS;

  const stepRounds = canCatchUp
    ? rounds
      .map((r) => Number(r.roundId))
      .filter((rid) => rid > lastProcessed)
    : [latestRound];

  let finalEngine = null;
  const allResolved = [];

  for (const stepRoundId of stepRounds) {
    const stepIdx = roundIndexById.get(stepRoundId);
    if (!Number.isFinite(stepIdx)) continue;

    const roundsSlice = rounds.slice(0, stepIdx + 1);
    const engine = computeLockedRangePredictions(roundsSlice, workingLocks, { historyRows });
    finalEngine = engine;

    if (Array.isArray(engine.resolvedHistory) && engine.resolvedHistory.length) {
      allResolved.push(...engine.resolvedHistory);
      historyRows = mergeHistoryRows(historyRows, engine.resolvedHistory, LOCKED_HISTORY_CALIB_LIMIT);
    }

    if (engine.locksToSave && Object.keys(engine.locksToSave).length) {
      workingLocks = engine.locksToSave;
    }
  }

  if (!finalEngine) {
    finalEngine = computeLockedRangePredictions(rounds, workingLocks, { historyRows });
  }

  if (Object.keys(finalEngine.locksToSave || {}).length && locksNeedSave(persistedLocks, finalEngine.locksToSave)) {
    await saveLockedConsensusPreds(finalEngine.locksToSave);
  }

  const savedResolvedCount = await persistResolvedHistory(allResolved);

  const fullHistory = savedResolvedCount > 0
    ? await getPredictions({ limit: LOCKED_HISTORY_PUBLIC_LIMIT, source: LOCKED_SOURCE })
    : historyRows.slice(0, LOCKED_HISTORY_PUBLIC_LIMIT);

  const basePayload = {
    ok: true,
    ...finalEngine,
    historyAll: fullHistory,
    historyByTarget: buildHistoryByTarget(fullHistory),
    historyStorage: 'postgres',
    savedResolvedCount,
    worker: {
      intervalMs: LOCKED_WORKER_INTERVAL_MS,
      catchupSteps: stepRounds.length,
      catchupMode: canCatchUp,
      lastProcessedRound: latestRound,
      generatedAt: new Date().toISOString(),
    },
  };

  lockedCache.asOfRound = finalEngine?.asOfRound ?? latestRoundHint ?? latestRound;
  lockedCache.limit = limit;
  lockedCache.createdAt = Date.now();
  lockedCache.basePayload = basePayload;

  lockedWorkerState.lastProcessedRound = latestRound;
  lockedWorkerState.lastRunAt = Date.now();
  lockedWorkerState.lastError = null;

  return basePayload;
}

async function refreshLockedSnapshot({ limit, latestRoundHint = null, force = false }) {
  if (lockedWorkerState.inFlight) return lockedWorkerState.inFlight;

  if (!force
    && lockedCache.basePayload
    && latestRoundHint != null
    && lockedCache.asOfRound === latestRoundHint
    && lockedCache.limit >= limit
  ) {
    return lockedCache.basePayload;
  }

  const task = computeLockedSnapshot(limit, latestRoundHint)
    .catch((err) => {
      lockedWorkerState.lastError = err.message;
      console.error('[locked-worker] tick failed:', err.message);
      throw err;
    })
    .finally(() => {
      if (lockedWorkerState.inFlight === task) lockedWorkerState.inFlight = null;
    });

  lockedWorkerState.inFlight = task;
  return task;
}

async function runLockedWorkerTick() {
  const latestRound = await getLatestRoundId();
  if (!latestRound) return;

  const hasFreshCache = lockedCache.basePayload
    && lockedCache.asOfRound != null
    && Number(lockedCache.asOfRound) === Number(latestRound)
    && (Date.now() - lockedCache.createdAt) < LOCKED_CACHE_TTL_MS;

  if (hasFreshCache) return;

  await refreshLockedSnapshot({
    limit: Math.max(LOCKED_MIN_LIMIT, Math.min(LOCKED_DEFAULT_LIMIT, LOCKED_MAX_LIMIT)),
    latestRoundHint: latestRound,
    force: true,
  });
}

function startLockedWorker() {
  if (lockedWorkerTimer) return;
  runLockedWorkerTick().catch((err) => {
    console.error('[locked-worker] initial run failed:', err.message);
  });
  lockedWorkerTimer = setInterval(() => {
    runLockedWorkerTick().catch((err) => {
      console.error('[locked-worker] periodic run failed:', err.message);
    });
  }, Math.max(1000, LOCKED_WORKER_INTERVAL_MS));
}

app.get('/rounds', rateLimit(60), async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit || '1000', 10), 100000);
    const offset = parseInt(req.query.offset || '0', 10);
    const since = req.query.since ? Number(req.query.since) : null;
    const minRoundId = since && since > 0 ? since + 1 : null;
    const rounds = await getRounds({
      limit,
      offset,
      from: req.query.from || null,
      to: req.query.to || null,
      minRoundId,
    });
    res.json({ ok: true, count: rounds.length, rounds });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/stats', rateLimit(30), async (req, res) => {
  try {
    res.json({ ok: true, ...(await getStats()) });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/storage-stats', rateLimit(20), async (req, res) => {
  try {
    res.json({ ok: true, ...(await getStorageStats()) });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/health', (req, res) => {
  res.json({ ok: true, ts: new Date().toISOString() });
});

app.get('/predict', rateLimit(20), async (req, res) => {
  try {
    const limit = normalizeLimit(req.query.limit, PREDICT_DEFAULT_LIMIT, PREDICT_MIN_LIMIT, PREDICT_MAX_LIMIT);
    const latestRound = await getLatestRoundId();

    const cacheFresh = (
      predictCache.payload
      && predictCache.asOfRound != null
      && predictCache.asOfRound === latestRound
      && predictCache.limit === limit
      && (Date.now() - predictCache.createdAt) < PREDICT_CACHE_TTL_MS
    );
    if (cacheFresh) return res.json(predictCache.payload);

    if (predictCache.payload && predictCache.limit === limit) {
      triggerPredictRefresh(limit, latestRound);
      return res.json(predictCache.payload);
    }

    const payload = await refreshPredictCache({ limit, latestRoundHint: latestRound, force: true });
    return res.json(payload);
  } catch (e) {
    if (predictCache.payload) return res.json(predictCache.payload);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/predict/locked', rateLimit(20), async (req, res) => {
  try {
    const limit = normalizeLimit(req.query.limit, LOCKED_DEFAULT_LIMIT, LOCKED_MIN_LIMIT, LOCKED_MAX_LIMIT);
    const historyTarget = normalizeHistoryTarget(req.query.historyTarget);
    const latestRound = await getLatestRoundId();

    const fresh = (
      lockedCache.basePayload
      && lockedCache.asOfRound != null
      && Number(lockedCache.asOfRound) === Number(latestRound)
      && lockedCache.limit >= limit
      && (Date.now() - lockedCache.createdAt) < LOCKED_CACHE_TTL_MS
    );

    if (fresh) return res.json(withHistoryFilter(lockedCache.basePayload, historyTarget));

    if (!lockedWorkerState.inFlight) {
      await refreshLockedSnapshot({
        limit: Math.max(limit, LOCKED_DEFAULT_LIMIT),
        latestRoundHint: latestRound,
        force: true,
      });
    } else if (!lockedCache.basePayload) {
      await lockedWorkerState.inFlight;
    }

    if (lockedCache.basePayload) return res.json(withHistoryFilter(lockedCache.basePayload, historyTarget));

    const fallback = await computeLockedSnapshot(Math.max(limit, LOCKED_DEFAULT_LIMIT), latestRound);
    if (fallback) return res.json(withHistoryFilter(fallback, historyTarget));

    return res.status(503).json({ ok: false, error: 'No rounds available yet' });
  } catch (e) {
    if (lockedCache.basePayload) {
      return res.json({
        ...withHistoryFilter(lockedCache.basePayload, normalizeHistoryTarget(req.query.historyTarget)),
        stale: true,
        warning: e.message,
      });
    }
    return res.status(500).json({ ok: false, error: e.message });
  }
});

const clearHistoryHandler = async (req, res) => {
  try {
    const result = await clearPredictions();
    invalidatePredictionCaches();
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
};

app.delete('/clear-history', requireAdmin, rateLimit(10), clearHistoryHandler);
app.post('/clear-history', requireAdmin, rateLimit(10), clearHistoryHandler);

const clearLocksHandler = async (req, res) => {
  try {
    const result = await clearAllLocks();
    invalidatePredictionCaches();
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
};

app.delete('/clear-locks', requireAdmin, rateLimit(10), clearLocksHandler);
app.post('/clear-locks', requireAdmin, rateLimit(10), clearLocksHandler);

app.post('/access/verify', rateLimit(20), async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) return res.json({ ok: false, reason: 'no_code' });
    const row = await getAccessCode(code.trim());
    if (!row) return res.json({ ok: false, reason: 'invalid' });
    if (new Date(row.expires_at) < new Date()) return res.json({ ok: false, reason: 'expired' });
    const ip = getIP(req);
    const sameIP = row.ip && row.ip === ip;
    if (row.use_count >= row.max_uses && !sameIP) return res.json({ ok: false, reason: 'used_up' });
    if (!row.ip || (!sameIP && row.use_count < row.max_uses)) await updateAccessCodeIP(code.trim(), ip);
    return res.json({ ok: true, expiresAt: row.expires_at });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/access/create', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    const { code, expiresAt, note, maxUses } = req.body;
    if (!code || !expiresAt) {
      return res.status(400).json({ ok: false, error: 'code and expiresAt required' });
    }
    return res.json({ ok: true, row: await createAccessCode({ code, expiresAt, note, maxUses: maxUses || 1 }) });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/access/list', requireAdmin, rateLimit(20), async (req, res) => {
  try {
    return res.json({ ok: true, codes: await getAllAccessCodes() });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.delete('/access/:id', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    await deleteAccessCode(req.params.id);
    return res.json({ ok: true });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/wallets', requireAdmin, rateLimit(20), async (req, res) => {
  try {
    return res.json({ ok: true, wallets: await getWallets() });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/wallets', requireAdmin, rateLimit(20), async (req, res) => {
  try {
    const { privateKey, rpcUrl, playerAccountPDA, pubkey } = req.body;
    if (!privateKey) return res.status(400).json({ ok: false, error: 'privateKey required' });
    return res.json({ ok: true, wallet: await saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey }) });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.delete('/wallets/:id', requireAdmin, rateLimit(10), async (req, res) => {
  try {
    await deleteWallet(req.params.id);
    return res.json({ ok: true });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
});

function startAPI() {
  initAccessCodes().catch((e) => console.error('initAccessCodes error:', e.message));
  initWalletStorage().catch((e) => console.error('initWalletStorage error:', e.message));

  startLockedWorker();

  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI };
