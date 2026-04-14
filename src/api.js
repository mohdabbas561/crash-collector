'use strict';
const http       = require('http');
const express    = require('express');
const cors       = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { buildPredictionReport } = require('./predictionEngine');
const { computeLockedRangePredictions } = require('./lockedRangeEngine');
const { ORACLE_TARGETS, normalizeRounds, computeOracleForecast, makeOracleLock } = require('./oracleEngine');
const {
  pool,
  getLatestRoundId, getRoundCount,
  getRounds, getStats, getStorageStats,
  getPredictions, savePrediction, clearPredictions, clearAllLocks,
  getOracleLocks, replaceOracleLocks,
  getLockedConsensusPreds, saveLockedConsensusPreds,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
  initWalletStorage, saveWallet, getWallets, getWalletByPubkey, deleteWallet,
  initSfbWalletStorage, saveSfbWallet,
} = require('./db');
 
const app  = express();
const PORT = process.env.PORT || 3001;
const SFB_PROXY_TARGET = String(process.env.SFB_PROXY_TARGET || 'https://sfb-api-service-mainnet.up.railway.app').trim().replace(/\/+$/, '');
const SFB_PROXY_ORIGIN = String(process.env.SFB_PROXY_ORIGIN || 'https://www.solanafatboys.com').trim().replace(/\/+$/, '');

function applySfbProxyHeaders(proxyReq) {
  proxyReq.setHeader('origin', SFB_PROXY_ORIGIN);
  proxyReq.setHeader('referer', `${SFB_PROXY_ORIGIN}/`);
}

const sfbApiProxy = createProxyMiddleware({
  target: SFB_PROXY_TARGET,
  changeOrigin: true,
  secure: true,
  xfwd: true,
  pathRewrite: { '^/sfb-api': '' },
  onProxyReq: applySfbProxyHeaders,
  onError(err, req, res) {
    if (res && !res.headersSent) {
      res.status(502).json({ ok: false, error: 'SFB API proxy unavailable' });
    }
  },
});

const sfbWsProxy = createProxyMiddleware({
  target: SFB_PROXY_TARGET,
  changeOrigin: true,
  secure: true,
  xfwd: true,
  ws: true,
  pathRewrite: { '^/sfb-ws': '/ws' },
  onProxyReq: applySfbProxyHeaders,
  onProxyReqWs: applySfbProxyHeaders,
  onError(err, req, res) {
    if (res && !res.headersSent) {
      res.status(502).json({ ok: false, error: 'SFB websocket proxy unavailable' });
    }
  },
});
 
const DEFAULT_ALLOWED_ORIGINS = [
  'http://localhost:3000',
  'http://127.0.0.1:3000',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
];
const ALLOWED_ORIGIN_RULES = Array.from(new Set([
  ...DEFAULT_ALLOWED_ORIGINS,
  ...(process.env.ALLOWED_ORIGINS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean),
]));
 
function normalizeOriginValue(value) {
  return String(value || '').trim().replace(/\/+$/, '');
}
 
function extractHost(value) {
  const normalized = normalizeOriginValue(value);
  if (!normalized) return '';
  try {
    return new URL(normalized).host.toLowerCase();
  } catch {
    return normalized.replace(/^https?:\/\//i, '').split('/')[0].toLowerCase();
  }
}
 
function originMatchesRule(origin, rule) {
  const normalizedOrigin = normalizeOriginValue(origin);
  const normalizedRule = normalizeOriginValue(rule);
  if (!normalizedRule) return false;
  if (normalizedRule === '*') return true;
  if (normalizedRule.endsWith('*')) {
    return normalizedOrigin.startsWith(normalizedRule.slice(0, -1));
  }
  if (normalizedRule.includes('://')) {
    return normalizedOrigin === normalizedRule;
  }
  return extractHost(normalizedOrigin) === extractHost(normalizedRule);
}
 
function isOriginAllowed(origin) {
  if (!origin) return true;
  if (ALLOWED_ORIGIN_RULES.length === 0) return true;
  return ALLOWED_ORIGIN_RULES.some(rule => originMatchesRule(origin, rule));
}
 
const corsOptionsDelegate = (req, cb) => {
  const requestOrigin = req.header('Origin');
  const allowed = isOriginAllowed(requestOrigin);
  cb(null, {
    origin: allowed ? true : false,
    credentials: true,
    methods: ['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'x-admin-secret',
      'x-access-code',
      'Cache-Control',
      'Pragma',
      'Expires',
    ],
    maxAge: 86400,
  });
};
 
app.use(cors(corsOptionsDelegate));
app.options('*', cors(corsOptionsDelegate));
app.use('/sfb-api', sfbApiProxy);
app.use('/sfb-ws', sfbWsProxy);
 
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
 
function extractAdminSecretFromRequest(req) {
  const headerSecret = normalizeSecretValue(req.headers['x-admin-secret']);
  const authHeader = String(req.headers.authorization || '');
  const bearerSecret = normalizeSecretValue(authHeader.replace(/^Bearer\s+/i, ''));
  return headerSecret || bearerSecret;
}
 
async function authorizeAccessCode(req) {
  const code = String(req.headers['x-access-code'] || req.body?.accessCode || '').trim();
  if (!code) return { ok: false, error: 'ACCESS_CODE_MISSING' };
  const row = await getAccessCode(code);
  if (!row) return { ok: false, error: 'ACCESS_CODE_INVALID' };
  if (new Date(row.expires_at) < new Date()) return { ok: false, error: 'ACCESS_CODE_EXPIRED' };
 
  const ip = getIP(req);
  const sameIP = row.ip && row.ip === ip;
  if (row.use_count >= row.max_uses && !sameIP) {
    return { ok: false, error: 'ACCESS_CODE_USED_UP' };
  }
  if (!row.ip || (!sameIP && row.use_count < row.max_uses)) {
    await updateAccessCodeIP(code, ip);
  }
 
  return { ok: true, code, row };
}
 
async function requireAdminOrAccess(req, res, next) {
  const secret = extractAdminSecretFromRequest(req);
  if (ADMIN_SECRET && secret && secret === ADMIN_SECRET) {
    req.authMode = 'admin';
    return next();
  }
  try {
    const access = await authorizeAccessCode(req);
    if (!access.ok) {
      const status = access.error === 'ACCESS_CODE_EXPIRED' || access.error === 'ACCESS_CODE_USED_UP' ? 403 : 401;
      return res.status(status).json({ ok: false, error: access.error });
    }
    req.authMode = 'access';
    req.accessCode = access.code;
    return next();
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    return res.status(500).json({ ok: false, error: e.message });
  }
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
const DASHBOARD_CACHE_TTL_MS = toPositiveInt(process.env.DASHBOARD_CACHE_TTL_MS, 30000);  // raised from 10000
const PREDICT_CACHE_TTL_MS = toPositiveInt(process.env.PREDICT_CACHE_TTL_MS, 60000);      // raised from 25000
const LOCKED_CACHE_TTL_MS = toPositiveInt(process.env.LOCKED_CACHE_TTL_MS, 60000);        // raised from 25000
const LOCKED_MIN_RECOMPUTE_MS = toPositiveInt(process.env.LOCKED_MIN_RECOMPUTE_MS, 8000);
const LOCKED_COMPUTE_TIMEOUT_MS = toPositiveInt(process.env.LOCKED_COMPUTE_TIMEOUT_MS, 25000);
const LOCKED_HISTORY_LIMIT = toPositiveInt(process.env.LOCKED_HISTORY_LIMIT, 1200);
const LOCKED_HISTORY_PREFETCH_LIMIT = Math.max(
  LOCKED_HISTORY_LIMIT,
  toPositiveInt(process.env.LOCKED_HISTORY_PREFETCH_LIMIT, 3000)
);
const LOCKED_CATCHUP_MAX_ITERS = toPositiveInt(process.env.LOCKED_CATCHUP_MAX_ITERS, 240);
const LOCKED_COMPUTE_BUDGET_MS = toPositiveInt(process.env.LOCKED_COMPUTE_BUDGET_MS, 2500);
const RESPONSE_CACHE_ENABLED = String(process.env.RESPONSE_CACHE_ENABLED || 'true').trim().toLowerCase() === 'true';
const STRICT_FRESH_MODE = String(process.env.STRICT_FRESH_MODE || 'false').trim().toLowerCase() !== 'false';
const LOCKED_USE_FULL_DATA = String(process.env.LOCKED_USE_FULL_DATA || 'true').trim().toLowerCase() !== 'false';
const LOCKED_BACKGROUND_ENABLED = String(process.env.LOCKED_BACKGROUND_ENABLED || 'true').trim().toLowerCase() !== 'false';
const LOCKED_BACKGROUND_INTERVAL_MS = toPositiveInt(process.env.LOCKED_BACKGROUND_INTERVAL_MS, 30000); // raised from 8000
const ORACLE_BACKGROUND_ENABLED = String(process.env.ORACLE_BACKGROUND_ENABLED || 'true').trim().toLowerCase() !== 'false';
const ORACLE_BACKGROUND_INTERVAL_MS = toPositiveInt(process.env.ORACLE_BACKGROUND_INTERVAL_MS, 30000);  // raised from 8000
function firstNonEmptyEnv(...values) {
  for (const raw of values) {
    const value = String(raw || '').trim();
    if (value) return value;
  }
  return '';
}
 
const BOT_RPC_URL = firstNonEmptyEnv(
  process.env.BOT_RPC_URL,
  process.env.CRASH_BOT_RPC_URL,
  process.env.RPC_URL,
  process.env.HELIUS_RPC,
  process.env.HELIUS_RPC_URL,
  process.env.MAINNET_RPC_URL,
  process.env.SOLANA_RPC_URL
);
const BOT_PLAYER_ACCOUNT_PDA = firstNonEmptyEnv(
  process.env.BOT_PLAYER_ACCOUNT_PDA,
  process.env.CRASH_BOT_PLAYER_ACCOUNT_PDA,
  process.env.BOT_PLAYER_PDA,
  process.env.PLAYER_ACCOUNT_PDA,
  process.env.PLAYER_PDA,
  process.env.CRASH_PLAYER_ACCOUNT_PDA,
  process.env.SOLANA_PLAYER_ACCOUNT_PDA
);
const BOT_WALLET_AUTH_API = String(
  process.env.BOT_WALLET_AUTH_API ||
  'https://api.degencoinflip.com/v2'
).trim().replace(/\/+$/, '');
const BOT_CASHOUT_API = String(
  process.env.BOT_CASHOUT_API ||
  'https://crash-api.degencoinflip.com/api/status/set_cashout_multiplier'
).trim();
const BOT_STOP_CASHOUT_API = String(
  process.env.BOT_STOP_CASHOUT_API ||
  'https://crash-api.degencoinflip.com/api/status/cashout'
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
const oraclePredictCache = {
  asOfRound: null,
  createdAt: 0,
  payload: null,
};
const dashboardCache = {
  asOfRound: null,
  recentLimit: null,
  createdAt: 0,
  payload: null,
};
let predictComputeInFlight = null;
let lockedComputeInFlight = null;
let oraclePredictInFlight = null;
const ORACLE_PREDICTION_SOURCE = 'oracle_v4';
let lockedBackgroundTimer = null;
let oracleBackgroundTimer = null;
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
  oraclePredictCache.asOfRound = null;
  oraclePredictCache.createdAt = 0;
  oraclePredictCache.payload = null;
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
 
  let engine = null;
  let liveLocks = locked || {};
  let liveHistoryRows = Array.isArray(historyRows) ? [...historyRows] : [];
  const resolvedRowsToPersist = [];
  const computeStartedAt = Date.now();
 
  // Catch-up mode: if rounds advanced while frontend/server was offline,
  // process multiple lock generations in one compute pass so history is not skipped.
  for (let iter = 0; iter < LOCKED_CATCHUP_MAX_ITERS; iter += 1) {
    engine = computeLockedRangePredictions(rounds, liveLocks, { historyRows: liveHistoryRows });
    const resolvedNow = Array.isArray(engine?.resolvedHistory) ? engine.resolvedHistory : [];
    const nextLocks = engine?.locksToSave || {};
    const lockDelta = locksNeedSave(liveLocks, nextLocks);
 
    if (resolvedNow.length) {
      resolvedRowsToPersist.push(...resolvedNow);
      // Feed freshly-resolved rows back into calibration immediately in this same tick.
      liveHistoryRows = [...resolvedNow, ...liveHistoryRows].slice(0, LOCKED_HISTORY_PREFETCH_LIMIT);
    }
 
    liveLocks = nextLocks;
 
    // Stop once no additional rows are being resolved in this pass.
    if (!resolvedNow.length) break;
    // Safety: if locks no longer move, avoid a pathological loop.
    if (!lockDelta) break;
    // Hard latency guard: keep API responses quick and continue catch-up on next tick.
    if ((Date.now() - computeStartedAt) >= LOCKED_COMPUTE_BUDGET_MS) break;
  }
 
  if (Object.keys(liveLocks || {}).length && locksNeedSave(locked, liveLocks)) {
    await saveLockedConsensusPreds(liveLocks);
  }
 
  let savedResolvedCount = 0;
  if (resolvedRowsToPersist.length) {
    const results = await Promise.allSettled(resolvedRowsToPersist.map((row) => savePrediction({
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
    const failures = results
      .map((r, idx) => ({ r, row: resolvedRowsToPersist[idx] }))
      .filter(({ r }) => r.status === 'rejected');
    if (failures.length) {
      for (const f of failures.slice(0, 5)) {
        console.error(
          '[predict/locked] savePrediction failed:',
          `target=${f.row?.target} lo=${f.row?.lo} hi=${f.row?.hi} gen=${f.row?.generation}`,
          String(f.r.reason?.message || f.r.reason || 'unknown')
        );
      }
      console.error(`[predict/locked] savePrediction failures: ${failures.length}/${results.length}`);
    }
    savedResolvedCount = results.length - failures.length;
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
 
function summarizeOracleHistoryByTarget(historyRows) {
  const summary = {};
  for (const target of ORACLE_TARGETS) {
    summary[target.label] = {
      wins: 0,
      early: 0,
      failed: 0,
      total: 0,
      counted: 0,
      winRate: null,
    };
  }
  for (const row of historyRows || []) {
    if (!summary[row.target]) continue;
    summary[row.target].total += 1;
    if (row.outcome === 'win') summary[row.target].wins += 1;
    else if (row.outcome === 'early') summary[row.target].early += 1;
    else summary[row.target].failed += 1;
  }
  for (const key of Object.keys(summary)) {
    const item = summary[key];
    item.counted = item.wins + item.failed;
    item.winRate = item.counted > 0 ? Math.round((item.wins / item.counted) * 100) : null;
  }
  return summary;
}
 
function getNextOracleGeneration(existingLock, forecast) {
  if (!existingLock) return 1;
  const existingGeneration = Number(existingLock.generation || 1);
  const existingLastHitId = Number(existingLock.lastHitId || 0);
  const forecastLastHitId = Number(forecast?.lastHit?.id || 0);
  if (forecastLastHitId > existingLastHitId) return 1;
  return existingGeneration + 1;
}
 
function advanceOracleForecastPastExistingWindow(existingLock, forecast) {
  if (!existingLock || !forecast || forecast.noData) return forecast;
  const sameHitChain = Number(existingLock.lastHitId || 0) === Number(forecast?.lastHit?.id || 0);
  if (!sameHitChain) return forecast;
  // Keep windows fixed to the forecast output. Sliding after a miss causes
  // confusing +1 drift for the user and makes lock spans feel unstable.
  return forecast;
}
 
function upperBoundRoundIndex(rounds, maxId) {
  let lo = 0;
  let hi = rounds.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (Number(rounds[mid]?.id || 0) <= maxId) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
 
function sliceRoundsUpTo(rounds, maxId) {
  if (!Array.isArray(rounds) || !rounds.length) return [];
  const end = upperBoundRoundIndex(rounds, maxId);
  return end > 0 ? rounds.slice(0, end) : [];
}
 
function findFirstOracleHitAfter(targetHits, afterId) {
  if (!Array.isArray(targetHits) || !targetHits.length) return null;
  let lo = 0;
  let hi = targetHits.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (Number(targetHits[mid]?.id || 0) <= afterId) lo = mid + 1;
    else hi = mid;
  }
  return targetHits[lo] || null;
}
 
function replayOracleTargetState({ rounds, nowId, target, existingLock, forecastOptions = {} }) {
  let liveForecast = computeOracleForecast(rounds, target, forecastOptions);
  let replayForecast = liveForecast;
  const resolvedRows = [];
  if (!liveForecast) return { forecast: null, activeLock: null, resolvedRows };
 
  const targetHits = rounds.filter((round) => round.val >= target.minVal);
  let activeLock = existingLock || null;
  let replayGuard = 0;
 
  while (activeLock && replayGuard < 64) {
    replayGuard += 1;
    const hit = findFirstOracleHitAfter(targetHits, Number(activeLock.lastHitId || 0));
    const hasFutureHit = Boolean(hit);
    const hitInsideOrBeforeWindow = hasFutureHit && Number(hit.id || 0) <= Number(activeLock.windowHi || 0);
    const expiredWithoutKnownHit = !hasFutureHit && nowId > Number(activeLock.windowHi || 0);
    const missedBeforeLaterHit = hasFutureHit && Number(hit.id || 0) > Number(activeLock.windowHi || 0);
 
    if (!hitInsideOrBeforeWindow && !expiredWithoutKnownHit && !missedBeforeLaterHit) {
      break;
    }
 
    let outcome = 'loss';
    let historyHitRound = null;
    let replayCutoffId = Number(activeLock.windowHi || nowId);
 
    if (hitInsideOrBeforeWindow) {
      historyHitRound = Number(hit.id || 0);
      replayCutoffId = historyHitRound;
      outcome = historyHitRound < Number(activeLock.windowLo || 0) ? 'early' : 'win';
    } else if (missedBeforeLaterHit) {
      historyHitRound = Number(hit.id || 0);
      // FIXED: Move replay to the hit point for fresh state computation.
      // No window sliding — next lock will be computed from scratch.
      replayCutoffId = historyHitRound;
      outcome = 'loss';
    }
 
    resolvedRows.push({
      target: target.label,
      minMult: target.minVal,
      outcome,
      lo: activeLock.windowLo,
      hi: activeLock.windowHi,
      hitRound: historyHitRound,
      generation: activeLock.generation || 1,
      source: ORACLE_PREDICTION_SOURCE,
      probW: activeLock.confidence != null ? Number(activeLock.confidence) / 100 : null,
      issueMode: activeLock.issueMode || replayForecast?.issueMode || liveForecast?.issueMode || null,
      regimeMode: activeLock.regimeMode || replayForecast?.regimeMode || liveForecast?.regimeMode || null,
    });
 
    const replayRounds = sliceRoundsUpTo(rounds, replayCutoffId);
    let nextForecast = computeOracleForecast(replayRounds, target, forecastOptions);
    if (!nextForecast || nextForecast.noData) {
      activeLock = null;
      replayForecast = nextForecast || replayForecast;
      break;
    }
    // FIXED: No window advancement on miss. Always compute fresh lock.
 
    const nextLock = nextForecast.issuePrediction
      ? {
          ...makeOracleLock(nextForecast, replayCutoffId),
          generation: getNextOracleGeneration(activeLock, nextForecast),
        }
      : null;
 
    replayForecast = nextForecast;
    activeLock = nextLock;
  }
 
  liveForecast = computeOracleForecast(rounds, target, forecastOptions) || replayForecast;
 
  if (!activeLock && liveForecast && !liveForecast.noData && liveForecast.issuePrediction) {
    activeLock = {
      ...makeOracleLock(liveForecast, nowId),
      generation: 1,
    };
  } else if (activeLock && liveForecast && !liveForecast.noData) {
    const sameHitChain = Number(activeLock.lastHitId || 0) === Number(liveForecast?.lastHit?.id || 0);
    if (!sameHitChain && Number(liveForecast?.lastHit?.id || 0) > 0) {
      activeLock = liveForecast.issuePrediction
        ? {
            ...makeOracleLock(liveForecast, nowId),
            generation: 1,
          }
        : null;
    }
  }
 
  return { forecast: liveForecast, activeLock, resolvedRows };
}
 
function buildOracleTargetPayload(forecast, activeLock, nowId) {
  const windowLo = activeLock?.windowLo ?? forecast.windowLo;
  const windowHi = activeLock?.windowHi ?? forecast.windowHi;
  const predictedRound = activeLock?.predictedRound ?? forecast.predictedRound;
  const roundsUntilWindowLo = Math.max(0, windowLo - nowId);
  const roundsUntilWindowHi = Math.max(0, windowHi - nowId);
  const inWindow = nowId >= windowLo && nowId <= windowHi;
  const lockConfidence = activeLock?.confidence ?? forecast.confidence;
  const liveConfidence = Number(forecast?.confidence ?? 0);
  const liveIssuePrediction = Boolean(forecast?.issuePrediction);
  const liveAvoidReason = forecast?.avoidReason || null;
  const lockDriftAlert = Boolean(activeLock) && inWindow && (
    !liveIssuePrediction ||
    (
      Number.isFinite(lockConfidence) &&
      Number.isFinite(liveConfidence) &&
      liveConfidence <= (lockConfidence - 12)
    )
  );
  const lockDriftReason = !liveIssuePrediction
    ? (liveAvoidReason || 'observe_only')
    : 'confidence_drop';
 
  return {
    ...forecast,
    activeLock: activeLock || null,
    predictedRound,
    windowLo,
    windowHi,
    roundsUntilWindowLo,
    roundsUntilWindowHi,
    inWindow,
    confidence: lockConfidence,
    liveConfidence,
    predBasis: activeLock?.predBasis ?? forecast.predBasis,
    predMethod: activeLock?.predMethod ?? forecast.predMethod,
    med: activeLock?.med ?? forecast.med,
    iqr: activeLock?.iqr ?? forecast.iqr,
    clusterCenter: activeLock?.clusterCenter ?? forecast.clusterCenter,
    issueMode: activeLock?.issueMode ?? forecast.issueMode ?? 'observe',
    regimeMode: activeLock?.regimeMode ?? forecast.regimeLabel ?? 'RANDOM',
    activePrediction: Boolean(activeLock),
    issuePrediction: Boolean(activeLock || forecast.issuePrediction),
    liveIssuePrediction,
    liveAvoidReason,
    lockDriftAlert,
    lockDriftReason,
    avoidReason: activeLock ? null : (forecast.avoidReason || null),
    // ── Oracle V4 signals ──
    whitePhase: forecast.whitePhase || 'NORMAL',
    whiteSignals: forecast.whiteSignals || {},
    b2bScore: Number(forecast.b2bScore || 0),
    b2bDetails: forecast.b2bDetails || {},
    regimeLabel: forecast.regimeLabel || 'RANDOM',
    regimeDetails: forecast.regimeDetails || {},
    ewmaSignal: forecast.ewmaSignal || {},
    markovProb: forecast.markovProb ?? null,
    patternSupport: forecast.patternSupport || {},
    layerBreakdown: forecast.layerBreakdown || [],
    ensembleP: forecast.ensembleP ?? null,
    baselineP: forecast.baselineP ?? null,
    ensembleEdge: forecast.ensembleEdge ?? null,
    ensembleEV: forecast.ensembleEV ?? null,
    rawConfidence: forecast.rawConfidence ?? null,
    pHit1: forecast.pHit1 ?? null,
    pHit5: forecast.pHit5 ?? null,
    pHitWindow: forecast.pHitWindow ?? null,
    droughtPct: forecast.droughtPct ?? null,
    engineVersion: forecast.engineVersion || 'oracle_v4',
  };
}
 
async function computeOraclePredictionPayload() {
  const latestRound = await getLatestRoundId();
  const totalRounds = latestRound == null ? 0 : await getRoundCount();
  const rawRounds = totalRounds > 0 ? await getRounds({ limit: totalRounds, order: 'ASC' }) : [];
  const rounds = normalizeRounds(rawRounds);
  const nowId = rounds.length ? rounds[rounds.length - 1].id : 0;
  const calibrationHistory = await getPredictions({ limit: 2000, source: ORACLE_PREDICTION_SOURCE });
  const calibrationByTarget = new Map();
  for (const row of calibrationHistory) {
    const key = String(row.target || '');
    if (!calibrationByTarget.has(key)) calibrationByTarget.set(key, []);
    calibrationByTarget.get(key).push(row);
  }
 
  const existingLocks = await getOracleLocks(ORACLE_PREDICTION_SOURCE);
  const existingLockMap = new Map(existingLocks.map((lock) => [lock.label, lock]));
  const resolvedRowsToPersist = [];
  const nextLocks = [];
  const forecasts = [];
 
  for (const target of ORACLE_TARGETS) {
    const existing = existingLockMap.get(target.label) || null;
    const replay = replayOracleTargetState({
      rounds,
      nowId,
      target,
      existingLock: existing,
      forecastOptions: {
        calibrationRows: calibrationByTarget.get(target.label) || [],
      },
    });
    if (!replay.forecast) continue;
    forecasts.push(replay.forecast);
    resolvedRowsToPersist.push(...replay.resolvedRows);
    if (replay.activeLock && !replay.forecast.noData) {
      nextLocks.push(replay.activeLock);
    }
  }
 
  if (resolvedRowsToPersist.length) {
    const results = await Promise.allSettled(resolvedRowsToPersist.map((row) => savePrediction(row)));
    const failures = results
      .map((result, index) => ({ result, row: resolvedRowsToPersist[index] }))
      .filter(({ result }) => result.status === 'rejected');
    if (failures.length) {
      for (const failure of failures.slice(0, 5)) {
        console.error(
          '[predict/oracle] savePrediction failed:',
          `target=${failure.row?.target} lo=${failure.row?.lo} hi=${failure.row?.hi}`,
          String(failure.result.reason?.message || failure.result.reason || 'unknown')
        );
      }
      console.error(`[predict/oracle] savePrediction failures: ${failures.length}/${results.length}`);
    }
  }
 
  await replaceOracleLocks(nextLocks, ORACLE_PREDICTION_SOURCE);
 
  const persistedHistory = await getPredictions({ limit: 500, source: ORACLE_PREDICTION_SOURCE });
  const history = persistedHistory.map((row) => ({
    ...row,
    result: row.outcome === 'win' ? 'WIN' : row.outcome === 'early' ? 'EARLY' : 'FAILED',
  }));
 
  const activeLockMap = new Map(nextLocks.map((lock) => [lock.label, lock]));
  const targets = forecasts.map((forecast) => buildOracleTargetPayload(
    forecast,
    activeLockMap.get(forecast.label),
    nowId
  ));
 
  // Compute global white phase from all forecasts
  const globalWhitePhases = forecasts.map(f => f.whitePhase || 'NORMAL');
  const globalWhiteActive = globalWhitePhases.filter(p => p === 'WHITE_ACTIVE').length;
  const globalWhiteEnding = globalWhitePhases.filter(p => p === 'WHITE_ENDING').length;
  const globalPreWhite = globalWhitePhases.filter(p => p === 'PRE_WHITE').length;
  const globalWhitePhase = globalWhiteActive > 3 ? 'WHITE_ACTIVE'
    : globalWhiteEnding > 2 ? 'WHITE_ENDING'
    : globalPreWhite > 3 ? 'PRE_WHITE'
    : 'NORMAL';

  // Dominant regime
  const regimeCounts = {};
  for (const f of forecasts) {
    const r = f.regimeLabel || 'RANDOM';
    regimeCounts[r] = (regimeCounts[r] || 0) + 1;
  }
  const dominantRegime = Object.keys(regimeCounts).reduce((a, b) => regimeCounts[a] >= regimeCounts[b] ? a : b, 'RANDOM');

  return {
    ok: true,
    generatedAt: new Date().toISOString(),
    asOfRound: nowId || null,
    roundsLoaded: rounds.length,
    activeLockCount: nextLocks.length,
    inWindowCount: targets.filter((target) => target.inWindow).length,
    targets,
    history,
    historyByTarget: summarizeOracleHistoryByTarget(persistedHistory),
    // ── Oracle V4 global signals ──
    globalWhitePhase,
    dominantRegime,
    engineVersion: 'oracle_v4',
  };
}
 
async function ensureOraclePredictionComputed({ latestRound }) {
  const sameInFlight = (
    oraclePredictInFlight &&
    oraclePredictInFlight.latestRound != null &&
    latestRound != null &&
    oraclePredictInFlight.latestRound === latestRound
  );
  if (sameInFlight) return oraclePredictInFlight.promise;
 
  const promise = computeOraclePredictionPayload();
  oraclePredictInFlight = { latestRound: latestRound ?? null, promise };
  try {
    const payload = await promise;
    oraclePredictCache.asOfRound = payload.asOfRound ?? latestRound ?? null;
    oraclePredictCache.createdAt = Date.now();
    oraclePredictCache.payload = payload;
    return payload;
  } finally {
    if (oraclePredictInFlight?.promise === promise) {
      oraclePredictInFlight = null;
    }
  }
}
 
async function refreshOraclePredictionInBackground() {
  try {
    const latestRound = await getLatestRoundId();
    if (latestRound == null) return;
    const cacheFresh = (
      oraclePredictCache.payload &&
      oraclePredictCache.asOfRound != null &&
      oraclePredictCache.asOfRound === latestRound
    );
    if (cacheFresh) return;
    await ensureOraclePredictionComputed({ latestRound });
  } catch (e) {
    console.error('[oracle-bg] refresh error:', e.message);
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
 
app.get('/predict/oracle', requireDatabase, rateLimit(20), async (req, res) => {
  try {
    const latestRound = await getLatestRoundId();
    markDbHealthy();
 
    const sameRoundFresh = (
      oraclePredictCache.payload &&
      oraclePredictCache.asOfRound != null &&
      latestRound != null &&
      oraclePredictCache.asOfRound === latestRound
    );
 
    if (sameRoundFresh) {
      return res.json(oraclePredictCache.payload);
    }
 
    const payload = await ensureOraclePredictionComputed({ latestRound });
    res.json(payload);
  } catch (e) {
    oraclePredictInFlight = null;
    if (isLikelyDbError(e)) setDatabaseAvailability(false, e.message);
    console.error('[predict/oracle] error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
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
 
async function resolveBotConfig() {
  if (BOT_RPC_URL && BOT_PLAYER_ACCOUNT_PDA) {
    return {
      rpcUrl: BOT_RPC_URL,
      playerAccountPDA: BOT_PLAYER_ACCOUNT_PDA,
      source: 'env',
    };
  }
 
  try {
    const wallets = await getWallets();
    const latest = (wallets || []).find((wallet) => (
      String(wallet?.rpc_url || '').trim() &&
      String(wallet?.player_account_pda || '').trim()
    ));
    if (latest) {
      return {
        rpcUrl: String(latest.rpc_url).trim(),
        playerAccountPDA: String(latest.player_account_pda).trim(),
        source: 'wallets',
      };
    }
  } catch (e) {
    console.error('[bot/config] wallet fallback error:', e.message);
  }
 
  return null;
}
 
app.get('/bot/config', rateLimit(60), async (req, res) => {
  const config = await resolveBotConfig();
  if (!config) {
    return res.status(503).json({
      ok: false,
      error: 'BOT_CONFIG_MISSING',
      message: 'Set BOT_RPC_URL + BOT_PLAYER_ACCOUNT_PDA (or save bot wallet config) in backend.',
    });
  }
  res.json({
    ok: true,
    config: {
      rpcUrl: config.rpcUrl,
      playerAccountPDA: config.playerAccountPDA,
      source: config.source,
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
 
app.post('/bot/site-cashout/multiplier', express.json({ limit: '20kb' }), rateLimit(40), async (req, res) => {
  try {
    const token = String(req.body?.token || '').trim();
    const multiplier = String(req.body?.multiplier || '').trim();
    if (!token || !multiplier) {
      return res.status(400).json({ ok: false, error: 'token and multiplier required' });
    }
 
    const upstream = await fetch(BOT_CASHOUT_API, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: token,
      },
      body: JSON.stringify({ multiplier }),
    });
    const { raw, data } = await readResponsePayload(upstream);
    if (!upstream.ok) {
      return res.status(upstream.status).json({ ok: false, error: raw || `Cashout upstream error ${upstream.status}` });
    }
    res.json({ ok: true, ...(data && typeof data === 'object' ? data : { payload: raw }) });
  } catch (e) {
    console.error('[bot/site-cashout/multiplier] error:', e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});
 
app.post('/bot/site-cashout/cashout', express.json({ limit: '20kb' }), rateLimit(40), async (req, res) => {
  try {
    const token = String(req.body?.token || '').trim();
    const mult = String(req.body?.mult || '').trim();
    if (!token || !mult) {
      return res.status(400).json({ ok: false, error: 'token and mult required' });
    }
 
    const upstream = await fetch(BOT_STOP_CASHOUT_API, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: token,
      },
      body: JSON.stringify({ message: 'STOP', mult }),
    });
    const { raw, data } = await readResponsePayload(upstream);
    if (!upstream.ok) {
      return res.status(upstream.status).json({ ok: false, error: raw || `Stop cashout upstream error ${upstream.status}` });
    }
    res.json({ ok: true, ...(data && typeof data === 'object' ? data : { payload: raw }) });
  } catch (e) {
    console.error('[bot/site-cashout/cashout] error:', e.message);
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
 
    let computedPayload;
    try {
      computedPayload = await ensureLockedPredictionFresh({ limit, limitKey });
    } catch (e) {
      const isTimeout = /timeout/i.test(String(e?.message || ''));
      if (isTimeout && hasCacheForLimit) {
        ensureLockedPredictionComputed({ latestRound, limit, limitKey })
          .catch((err) => console.error('[predict/locked] post-timeout async refresh error:', err.message));
        const stalePayload = await attachLiveLockedHistory(lockedCache.basePayload);
        return res.json(withHistoryFilter({
          ...stalePayload,
          warning: 'stale_snapshot_due_to_compute_timeout',
        }, historyTarget));
      }
      throw e;
    }
 
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
 
app.get('/wallets/by-pubkey/:pubkey', requireDatabase, requireAdminOrAccess, rateLimit(20), async (req, res) => {
  try {
    const pubkey = String(req.params.pubkey || '').trim();
    if (!pubkey) return res.status(400).json({ ok: false, error: 'pubkey required' });
    const wallet = await getWalletByPubkey(pubkey);
    if (!wallet) return res.status(404).json({ ok: false, error: 'wallet not found' });
    res.json({
      ok: true,
      wallet: {
        id: wallet?.id ?? null,
        private_key: wallet?.private_key ?? null,
        rpc_url: wallet?.rpc_url ?? null,
        player_account_pda: wallet?.player_account_pda ?? null,
        pubkey: wallet?.pubkey ?? null,
        updated_at: wallet?.updated_at ?? null,
      },
    });
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});
 
app.post('/wallets', requireDatabase, requireAdminOrAccess, rateLimit(20), async (req, res) => {
  try {
    const { privateKey, rpcUrl, playerAccountPDA, pubkey } = req.body;
    if (!privateKey) return res.status(400).json({ ok:false, error:'privateKey required' });
    const wallet = await saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey });
    const walletCount = (await getWallets())?.length || 0;
    res.json({
      ok: true,
      saveMode: 'append_v2',
      walletCount,
      wallet: {
        id: wallet?.id ?? null,
        rpc_url: wallet?.rpc_url ?? null,
        player_account_pda: wallet?.player_account_pda ?? null,
        pubkey: wallet?.pubkey ?? null,
        updated_at: wallet?.updated_at ?? null,
      },
    });
  } catch(e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok:false, error:e.message });
  }
});

app.post('/sfb-wallets', requireDatabase, rateLimit(30), async (req, res) => {
  try {
    const pubkey = String(req.body?.pubkey || '').trim();
    const balanceLamports = req.body?.balanceLamports;
    if (!pubkey) return res.status(400).json({ ok: false, error: 'pubkey required' });
    await saveSfbWallet({
      pubkey,
      balanceLamports,
      source: 'sfb-autobot',
    });
    res.json({ ok: true });
  } catch (e) {
    setDatabaseAvailability(false, e.message);
    res.status(500).json({ ok: false, error: e.message });
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
  initSfbWalletStorage().catch(e => {
    setDatabaseAvailability(false, e.message);
    console.error('initSfbWalletStorage error:', e.message);
  });
  if (LOCKED_BACKGROUND_ENABLED && !lockedBackgroundTimer) {
    refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] warmup error:', e.message));
    lockedBackgroundTimer = setInterval(() => {
      refreshLockedPredictionInBackground().catch(e => console.error('[locked-bg] tick error:', e.message));
    }, LOCKED_BACKGROUND_INTERVAL_MS);
    console.log(`[locked-bg] running every ${LOCKED_BACKGROUND_INTERVAL_MS}ms (${LOCKED_USE_FULL_DATA ? 'full-data mode' : 'limited mode'}, strict=${STRICT_FRESH_MODE ? 'on' : 'off'}, cache=${RESPONSE_CACHE_ENABLED ? 'on' : 'off'})`);
  }
  if (ORACLE_BACKGROUND_ENABLED && !oracleBackgroundTimer) {
    refreshOraclePredictionInBackground().catch(e => console.error('[oracle-bg] warmup error:', e.message));
    oracleBackgroundTimer = setInterval(() => {
      refreshOraclePredictionInBackground().catch(e => console.error('[oracle-bg] tick error:', e.message));
    }, ORACLE_BACKGROUND_INTERVAL_MS);
    console.log(`[oracle-bg] running every ${ORACLE_BACKGROUND_INTERVAL_MS}ms`);
  }
  if (STRICT_FRESH_MODE) {
    console.log('[strict-fresh] enabled (stale response cache disabled)');
  }
  const server = http.createServer(app);
  server.on('upgrade', sfbWsProxy.upgrade);
  server.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}
 
module.exports = { startAPI, setDatabaseAvailability };
