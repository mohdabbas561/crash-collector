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

function normalizeSecretValue(raw) {
  const s = String(raw || '').trim();
  if (!s) return '';
  return s.replace(/^['"]+|['"]+$/g, '').trim();
}

const ADMIN_SECRET = normalizeSecretValue(process.env.ADMIN_SECRET);
if (!ADMIN_SECRET) console.warn('⚠️  ADMIN_SECRET not set');

function requireAdmin(req, res, next) {
  const headerSecret = normalizeSecretValue(req.headers['x-admin-secret']);
  const authHeader = String(req.headers.authorization || '');
  const bearerSecret = normalizeSecretValue(authHeader.replace(/^Bearer\s+/i, ''));
  const secret = headerSecret || bearerSecret;
  if (!ADMIN_SECRET) {
    return res.status(503).json({ ok: false, error: 'ADMIN_SECRET_NOT_SET' });
  }
  if (!secret) {
    return res.status(403).json({ ok: false, error: 'ADMIN_SECRET_MISSING' });
  }
  if (secret !== ADMIN_SECRET) {
    return res.status(403).json({ ok: false, error: 'ADMIN_SECRET_MISMATCH' });
  }
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
const DASHBOARD_DEFAULT_RECENT = toPositiveInt(process.env.DASHBOARD_DEFAULT_RECENT, 80);
const DASHBOARD_MAX_RECENT = Math.max(DASHBOARD_DEFAULT_RECENT, toPositiveInt(process.env.DASHBOARD_MAX_RECENT, 400));
const DASHBOARD_DELTA_MAX = Math.max(DASHBOARD_MAX_RECENT, toPositiveInt(process.env.DASHBOARD_DELTA_MAX, 2500));
const DASHBOARD_CACHE_TTL_MS = toPositiveInt(process.env.DASHBOARD_CACHE_TTL_MS, 10000);
const PREDICT_CACHE_TTL_MS = toPositiveInt(process.env.PREDICT_CACHE_TTL_MS, 25000);
const LOCKED_CACHE_TTL_MS = toPositiveInt(process.env.LOCKED_CACHE_TTL_MS, 25000);
const LOCKED_MIN_RECOMPUTE_MS = toPositiveInt(process.env.LOCKED_MIN_RECOMPUTE_MS, 8000);
const LOCKED_COMPUTE_TIMEOUT_MS = toPositiveInt(process.env.LOCKED_COMPUTE_TIMEOUT_MS, 25000);
const LOCKED_HISTORY_LIMIT = toPositiveInt(process.env.LOCKED_HISTORY_LIMIT, 1200);
const LOCKED_HISTORY_PREFETCH_LIMIT = Math.max(
  LOCKED_HISTORY_LIMIT,
  toPositiveInt(process.env.LOCKED_HISTORY_PREFETCH_LIMIT, 3000)
);
const RESPONSE_CACHE_ENABLED = String(process.env.RESPONSE_CACHE_ENABLED || 'false').trim().toLowerCase() === 'true';
const STRICT_FRESH_MODE = String(process.env.STRICT_FRESH_MODE || 'true').trim().toLowerCase() !== 'false';
const LOCKED_USE_FULL_DATA = String(process.env.LOCKED_USE_FULL_DATA || 'true').trim().toLowerCase() !== 'false';
const LOCKED_BACKGROUND_ENABLED = String(process.env.LOCKED_BACKGROUND_ENABLED || 'true').trim().toLowerCase() !== 'false';
const LOCKED_BACKGROUND_INTERVAL_MS = toPositiveInt(process.env.LOCKED_BACKGROUND_INTERVAL_MS, 30000);
const BOT_RPC_URL = String(
  process.env.BOT_RPC_URL ||
  process.env.CRASH_BOT_RPC_URL ||
  'https://mainnet.helius-rpc.com/?api-key=14a95398-c1a2-425f-aca6-dadc58b319c8'
).trim();
const BOT_PLAYER_ACCOUNT_PDA = String(
  process.env.BOT_PLAYER_ACCOUNT_PDA ||
  process.env.CRASH_BOT_PLAYER_ACCOUNT_PDA ||
  '7b1VfRjNoCn7gPEQ7HFAg8wtjaLkrVrZTQNvkEdmrwsj'
).trim();
const BOT_WALLET_AUTH_API = String(
  process.env.BOT_WALLET_AUTH_API ||
  'https://api.degencoinflip.com/v2'
).trim().replace(/\/+$/, '');
const BOT_CASHOUT_API = String(
  process.env.BOT_CASHOUT_API ||
  'https://crash-api.degencoinflip.com/api/status/set_cashout_multiplier'
).trim();
const HISTORY_TARGETS = ['5x', '10x', '20x', '50x', '100x', '500x', '1000x'];

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
const dashboardCache = {
  asOfRound: null,
  recentLimit: null,
  createdAt: 0,
  payload: null,
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
  // Do not hard-block requests based on stale health state.
  // Allow routes to attempt DB work and self-recover when DB comes back.
  return next();
}

function isLikelyDbError(err) {
  const code = String(err?.code || '').toUpperCase();
  if ([
    'ECONNREFUSED', 'ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN',
    '57P01', '57P02', '57P03', '08000', '08001', '08003', '08006',
    '53300', '53400', 'XX000',
  ].includes(code)) {
    return true;
  }
  const msg = String(err?.message || '').toLowerCase();
  return (
    msg.includes('connection terminated') ||
    msg.includes('connect econnrefused') ||
    msg.includes('remaining connection slots') ||
    msg.includes('database unavailable') ||
    msg.includes('the database system is starting up') ||
    msg.includes('timeout expired') ||
    msg.includes('project has exceeded the data transfer quota')
  );
}

function markDbHealthy() {
  if (!dbState.available) setDatabaseAvailability(true);
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
  dashboardCache.asOfRound = null;
  dashboardCache.recentLimit = null;
  dashboardCache.createdAt = 0;
  dashboardCache.payload = null;
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

function summarizeHistoryByTarget(rows) {
  const byTarget = {};
  for (const t of HISTORY_TARGETS) {
    byTarget[t] = summarizeHistory((rows || []).filter(h => String(h.target || '').toLowerCase() === t));
  }
  return byTarget;
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
  const allRows = Array.isArray(basePayload?.historyAll) ? basePayload.historyAll : [];
  const { historyAll, historyByTarget, ...rest } = basePayload || {};
  const filtered = historyTarget
    ? allRows.filter(h => String(h.target || '').toLowerCase() === historyTarget)
    : allRows;
  return {
    ...rest,
    historyByTarget: (historyByTarget && typeof historyByTarget === 'object')
      ? historyByTarget
      : summarizeHistoryByTarget(allRows),
    history: filtered,
    historySummary: summarizeHistory(filtered),
    historyFilter: historyTarget || 'all',
  };
}

async function attachLiveLockedHistory(basePayload) {
  if (!basePayload || typeof basePayload !== 'object') return basePayload;
  try {
    const liveHistory = await getPredictions({ limit: LOCKED_HISTORY_LIMIT, source: 'range_lock_v1' });
    markDbHealthy();
    return {
      ...basePayload,
      historyAll: liveHistory,
      historyByTarget: summarizeHistoryByTarget(liveHistory),
    };
  } catch (e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    return basePayload;
  }
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

function withTimeout(promise, timeoutMs, label) {
  const ms = Math.max(1000, Number(timeoutMs || 0));
  let timer = null;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${label || 'Operation'} timed out after ${ms}ms`));
    }, ms);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

async function readResponsePayload(res) {
  const raw = await res.text();
  try {
    return { raw, data: JSON.parse(raw) };
  } catch {
    return { raw, data: null };
  }
}

async function parseRoundsLimit(rawLimit) {
  const raw = String(rawLimit ?? '').trim().toLowerCase();
  if (raw === 'all' || raw === 'full' || raw === 'max') {
    const total = await getRoundCount();
    return Math.max(1, Number(total || 0));
  }
  return clampInt(rawLimit, 100, ROUNDS_MAX_LIMIT, ROUNDS_DEFAULT_LIMIT);
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
    getPredictions({ limit: LOCKED_HISTORY_PREFETCH_LIMIT, source: 'range_lock_v1' }),
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
    ? await getPredictions({ limit: LOCKED_HISTORY_LIMIT, source: 'range_lock_v1' })
    : historyRows.slice(0, LOCKED_HISTORY_LIMIT);

  const basePayload = {
    ok: true,
    ...engine,
    historyAll: fullHistory,
    historyByTarget: summarizeHistoryByTarget(fullHistory),
    historyStorage: 'postgres',
    savedResolvedCount,
  };

  // Always keep an in-memory same-round snapshot so repeated requests on the
  // same round do not re-run full engine compute. This is fresh-by-round and
  // does not serve stale data across new rounds.
  lockedCache.asOfRound = engine?.asOfRound ?? latestRound ?? null;
  lockedCache.limit = limitKey;
  lockedCache.createdAt = Date.now();
  lockedCache.basePayload = basePayload;
  return basePayload;
}

async function ensureLockedPredictionComputed({ latestRound, limit, limitKey }) {
  const sameInFlight = (
    lockedComputeInFlight &&
    lockedComputeInFlight.limitKey === limitKey
  );
  if (sameInFlight) return lockedComputeInFlight.promise;

  const promise = withTimeout(
    computeAndPersistLockedPrediction({ latestRound, limit, limitKey }),
    LOCKED_COMPUTE_TIMEOUT_MS,
    'Locked prediction compute'
  );
  lockedComputeInFlight = { latestRound, limitKey, promise };
  try {
    return await promise;
  } finally {
    if (lockedComputeInFlight?.promise === promise) {
      lockedComputeInFlight = null;
    }
  }
}

async function ensureLockedPredictionFresh({ limit, limitKey }) {
  const before = await getLatestRoundId();
  const first = await ensureLockedPredictionComputed({ latestRound: before, limit, limitKey });
  if (!STRICT_FRESH_MODE) return first;

  const after = await getLatestRoundId();
  if (after != null && before != null && after !== before) {
    return ensureLockedPredictionComputed({ latestRound: after, limit, limitKey });
  }
  return first;
}

async function ensurePredictFresh({ limit }) {
  const before = await getLatestRoundId();
  const rounds = await getRounds({ limit, order: 'DESC' });
  const report = buildPredictionReport(rounds);
  const firstPayload = { ok: true, ...report };
  if (!STRICT_FRESH_MODE) return firstPayload;

  const after = await getLatestRoundId();
  if (after != null && before != null && after !== before) {
    const roundsAgain = await getRounds({ limit, order: 'DESC' });
    const reportAgain = buildPredictionReport(roundsAgain);
    return { ok: true, ...reportAgain };
  }
  return firstPayload;
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
      lockedCache.limit === limitKey
    );
    if (cacheFresh) return;
    await ensureLockedPredictionComputed({ latestRound, limit, limitKey });
  } catch (e) {
    console.error('[locked-bg] refresh error:', e.message);
  }
}

app.get('/rounds', requireDatabase, rateLimit(60), async (req, res) => {
  try {
    const limit      = await parseRoundsLimit(req.query.limit);
    const offset     = parseInt(req.query.offset || '0');
    const since      = req.query.since ? Number(req.query.since) : null;
    const minRoundId = since && since > 0 ? since + 1 : null;
    const rounds = await getRounds({ limit, offset, from: req.query.from||null, to: req.query.to||null, minRoundId });
    markDbHealthy();
    res.json({ ok:true, count: rounds.length, rounds });
  } catch(e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    console.error('[rounds] error:', e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.get('/dashboard', requireDatabase, rateLimit(60), async (req, res) => {
  try {
    const recentLimit = clampInt(req.query.recentLimit, 30, DASHBOARD_MAX_RECENT, DASHBOARD_DEFAULT_RECENT);
    const since = req.query.since ? Number(req.query.since) : null;
    const minRoundId = Number.isFinite(since) && since > 0 ? since + 1 : null;
    const latestRound = await getLatestRoundId();

    if (!minRoundId) {
      const sameRoundFresh = (
        dashboardCache.payload &&
        dashboardCache.recentLimit === recentLimit &&
        dashboardCache.asOfRound != null &&
        latestRound != null &&
        dashboardCache.asOfRound === latestRound
      );
      const cacheFresh = sameRoundFresh && (
        !RESPONSE_CACHE_ENABLED ||
        (Date.now() - dashboardCache.createdAt) < DASHBOARD_CACHE_TTL_MS
      );
      if (cacheFresh) return res.json(dashboardCache.payload);
    }

    const [stats, recentRounds] = await Promise.all([
      getStats(),
      minRoundId
        ? getRounds({ limit: DASHBOARD_DELTA_MAX, minRoundId })
        : getRounds({ limit: recentLimit, order: 'DESC' }),
    ]);
    markDbHealthy();

    const payload = {
      ok: true,
      generatedAt: new Date().toISOString(),
      stats: {
        tracked: Number(stats?.tracked || 0),
        avg: stats?.avg ?? '0.00',
        highest: stats?.highest ?? '0.00',
        currentRound: Number(stats?.currentRound || 0),
        distribution: stats?.distribution || {},
      },
      gaps: stats?.gaps || {},
      distribution: stats?.distribution || {},
      recentRounds,
      count: recentRounds.length,
    };

    if (!minRoundId) {
      dashboardCache.asOfRound = latestRound ?? Number(stats?.currentRound || 0);
      dashboardCache.recentLimit = recentLimit;
      dashboardCache.createdAt = Date.now();
      dashboardCache.payload = payload;
    }

    res.json(payload);
  } catch (e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    console.error('[dashboard] error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/stats', requireDatabase, rateLimit(30), async (req,res) => {
  try {
    const data = await getStats();
    markDbHealthy();
    res.json({ ok:true, ...data });
  } catch(e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    console.error('[stats] error:', e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});
app.get('/storage-stats', requireDatabase, rateLimit(20), async (req,res) => {
  try {
    const data = await getStorageStats();
    markDbHealthy();
    res.json({ ok:true, ...data });
  } catch(e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
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

app.get('/bot/config', rateLimit(60), (req, res) => {
  res.json({
    ok: true,
    config: {
      rpcUrl: BOT_RPC_URL,
      playerAccountPDA: BOT_PLAYER_ACCOUNT_PDA,
    },
  });
});

app.get('/bot/site-auth/nonce/:walletId', rateLimit(30), async (req, res) => {
  try {
    const walletId = String(req.params.walletId || '').trim();
    if (!walletId) return res.status(400).json({ ok: false, error: 'walletId required' });

    const upstream = await fetch(`${BOT_WALLET_AUTH_API}/wallets/${encodeURIComponent(walletId)}/nonce`);
    const { raw, data } = await readResponsePayload(upstream);
    if (!upstream.ok) {
      return res.status(upstream.status).json({ ok: false, error: raw || `Nonce upstream error ${upstream.status}` });
    }
    res.json({ ok: true, ...(data && typeof data === 'object' ? data : { payload: raw }) });
  } catch (e) {
    console.error('[bot/site-auth/nonce] error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/bot/site-auth/authorize', express.json({ limit: '20kb' }), rateLimit(30), async (req, res) => {
  try {
    const walletId = String(req.body?.walletId || '').trim();
    const signature = String(req.body?.signature || '').trim();
    if (!walletId || !signature) {
      return res.status(400).json({ ok: false, error: 'walletId and signature required' });
    }

    const upstream = await fetch(`${BOT_WALLET_AUTH_API}/authorize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Signature-Encoding': 'base64',
      },
      body: JSON.stringify({ walletId, signature }),
    });
    const { raw, data } = await readResponsePayload(upstream);
    if (!upstream.ok) {
      return res.status(upstream.status).json({ ok: false, error: raw || `Authorize upstream error ${upstream.status}` });
    }
    res.json({ ok: true, ...(data && typeof data === 'object' ? data : { payload: raw }) });
  } catch (e) {
    console.error('[bot/site-auth/authorize] error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/predict', requireDatabase, rateLimit(20), async (req, res) => {
  try {
    const limit = clampInt(req.query.limit, 1000, PREDICT_MAX_LIMIT, PREDICT_DEFAULT_LIMIT);
    const latestRound = await getLatestRoundId();
    markDbHealthy();

    const sameRoundFresh = (
      predictCache.payload &&
      predictCache.asOfRound != null &&
      latestRound != null &&
      predictCache.asOfRound === latestRound &&
      predictCache.limit === limit
    );
    const cacheFresh = sameRoundFresh && (
      !RESPONSE_CACHE_ENABLED ||
      (Date.now() - predictCache.createdAt) < PREDICT_CACHE_TTL_MS
    );
    if (cacheFresh) {
      return res.json(predictCache.payload);
    }

    const sameInFlight = (
      predictComputeInFlight &&
      predictComputeInFlight.limit === limit
    );
    if (sameInFlight) {
      const payload = await predictComputeInFlight.promise;
      return res.json(payload);
    }

    const promise = (async () => {
      const payload = await ensurePredictFresh({ limit });
      if (!STRICT_FRESH_MODE) {
        const latestRound = payload?.asOfRound ?? await getLatestRoundId();
        predictCache.asOfRound = latestRound ?? null;
        predictCache.limit = limit;
        predictCache.createdAt = Date.now();
        predictCache.payload = payload;
      } else {
        predictCache.asOfRound = payload?.asOfRound ?? latestRound ?? null;
        predictCache.limit = limit;
        predictCache.createdAt = Date.now();
        predictCache.payload = payload;
      }
      return payload;
    })();

    predictComputeInFlight = { latestRound: null, limit, promise };
    try {
      const payload = await promise;
      return res.json(payload);
    } finally {
      if (predictComputeInFlight?.promise === promise) {
        predictComputeInFlight = null;
      }
    }
  } catch (e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/predict/locked', requireDatabase, rateLimit(20), async (req, res) => {
  try {
    const { limit, limitKey } = parseLockedLimit(req.query.limit);
    const historyTarget = normalizeHistoryTarget(req.query.historyTarget);
    const latestRound = await getLatestRoundId();

    markDbHealthy();
    const hasCacheForLimit = Boolean(lockedCache.basePayload && lockedCache.limit === limitKey);
    const sameRoundFresh = (
      hasCacheForLimit &&
      lockedCache.asOfRound != null &&
      latestRound != null &&
      lockedCache.asOfRound === latestRound
    );
    const cacheFresh = sameRoundFresh && (
      !RESPONSE_CACHE_ENABLED ||
      (Date.now() - lockedCache.createdAt) < LOCKED_CACHE_TTL_MS
    );

    if (cacheFresh) {
      const payload = await attachLiveLockedHistory(lockedCache.basePayload);
      return res.json(withHistoryFilter(payload, historyTarget));
    }

    if (!STRICT_FRESH_MODE && hasCacheForLimit) {
      // Non-strict mode: allow stale while async refresh.
      ensureLockedPredictionComputed({ latestRound, limit, limitKey })
        .catch((err) => console.error('[predict/locked] async refresh error:', err.message));
      const payload = await attachLiveLockedHistory(lockedCache.basePayload);
      return res.json(withHistoryFilter(payload, historyTarget));
    }

    const computedPayload = await ensureLockedPredictionFresh({ limit, limitKey });
    const basePayload = await attachLiveLockedHistory(computedPayload);
    markDbHealthy();
    return res.json(withHistoryFilter(basePayload, historyTarget));
  } catch (e) {
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
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
  try {
    const wallets = await getWallets();
    const safeWallets = (wallets || []).map((w) => ({
      id: w.id,
      rpc_url: w.rpc_url ?? null,
      player_account_pda: w.player_account_pda ?? null,
      pubkey: w.pubkey ?? null,
      updated_at: w.updated_at ?? null,
    }));
    res.json({ ok:true, wallets: safeWallets });
  }
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
  if (LOCKED_BACKGROUND_ENABLED && !lockedBackgroundTimer) {
    refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] warmup error:', e.message));
    lockedBackgroundTimer = setInterval(() => {
      refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] tick error:', e.message));
    }, LOCKED_BACKGROUND_INTERVAL_MS);
    console.log(`[locked-bg] running every ${LOCKED_BACKGROUND_INTERVAL_MS}ms (${LOCKED_USE_FULL_DATA ? 'full-data mode' : 'limited mode'}, strict=${STRICT_FRESH_MODE ? 'on' : 'off'}, cache=${RESPONSE_CACHE_ENABLED ? 'on' : 'off'})`);
  }
  if (STRICT_FRESH_MODE) {
    console.log('[strict-fresh] enabled (stale response cache disabled)');
  }
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI, setDatabaseAvailability };
