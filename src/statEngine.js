'use strict';
// statEngine.js — STANDALONE Statistical Engine (GEO / BAY / KM / ENS)
// ═══════════════════════════════════════════════════════════════════════════════
// REQUIREMENT: This engine is 100% independent of patternEngine.js.
// It shares ZERO code, helpers, constants, state, or modules with patternEngine.
// It may be run, modified, or removed without touching patternEngine.js.
//
// Architecture:
//   - Uses full 12k+ round dataset from DB every cycle
//   - GEO: Laplace-smoothed geometric probability
//   - BAY: Bayesian recency blend (max 5% weight)
//   - KM:  Kaplan-Meier empirical gap CDF
//   - ENS: BMA ensemble of geo+bay+km
//   - ENGINE: base engine (geo-style with timing)
//   - Stores ALL predictions and locked windows exclusively in DB
//   - No local storage, no per-user caching — DB is single source of truth
//   - Zero duplication: unique constraint (source, target, window_lo, window_hi)
// ═══════════════════════════════════════════════════════════════════════════════

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedPreds,        getLockedPreds,
  saveLockedStatPreds,    getLockedStatPreds,
} = require('./db');

const ENGINE_VERSION = 'v20-CLEAN-STAT';

console.log(`[${ENGINE_VERSION}] Loaded — geo/bay/km/ens only · pattern engine independent`);

// ── Targets — local, no shared import ────────────────────────────────────────
const TARGETS = [
  { label: '5x',    min: 5,    maxWidth: 3,  rare: false },
  { label: '10x',   min: 10,   maxWidth: 5,  rare: false },
  { label: '20x',   min: 20,   maxWidth: 7,  rare: false },
  { label: '50x',   min: 50,   maxWidth: 12, rare: false },
  { label: '100x',  min: 100,  maxWidth: 18, rare: true  },
  { label: '250x',  min: 250,  maxWidth: 25, rare: true  },
  { label: '500x',  min: 500,  maxWidth: 35, rare: true  },
  { label: '1000x', min: 1000, maxWidth: 50, rare: true  },
];

const STAT_MODELS = [
  { id: 'ens' },
  { id: 'geo' },
  { id: 'bay' },
  { id: 'km'  },
];

// ── Constants ─────────────────────────────────────────────────────────────────
const MIN_ROUNDS                    = 50;
const STALE_FORCE_REBUILD_THRESHOLD = 50000;
const WINDOW_LAMBDA                 = 0.008;

const REGIME_SHORT_WINDOW        = 30;
const REGIME_WIDTH_SCALE         = 0.10;
const REGIME_OFFSET_SCALE        = 0.08;
const REGIME_DECAY               = 0.06;
const REGIME_CUSUM_CLIP_HOT      = 1.2;
const REGIME_CUSUM_CLIP_COLD     = 0.8;
const REGIME_HYSTERESIS_REQUIRED = 3;
const REGIME_ACTIVATION_OUTCOMES = 80;
const REGIME_EARLY_HINT_SIGMA    = 2.5;
const REGIME_EARLY_HINT_MAX      = 0.06;
const REGIME_MAX_BLEND           = 0.06;

const CAL_BINS            = [0, 0.30, 0.45, 0.60, 0.75, 1.01];
const CAL_DECAY           = { normal: 0.05, rare: 0.02 };
const CAL_MIN_SAMPLES     = 20;
const CAL_WARMUP_OUTCOMES = 100;
const CAL_WARMUP_RAW_WEIGHT = 0.70;
const CAL_MAX_RATIO       = 1.12;
const CAL_MIN_RATIO       = 0.88;
const CAL_MAX_SHIFT       = 0.04;

const SIG_HARD_FLOOR      = 0.42;
const SIG_LOW_ZONE_TOP    = 0.46;
const SIG_LOW_ZONE_SPREAD = 0.08;
const SIG_LOW_ZONE_CONF   = 55;
const SIG_MODERATE_SPREAD = 0.12;
const SIG_MODERATE_CONF   = 50;
const SIG_STRONG_PROB     = 0.52;

const RARE_MIN_MULTIPLIER        = 100;
const RARE_TAIL_GEO_BLEND        = 0.40;
const RARE_TAIL_GEO_BLEND_SPARSE = 0.70;
const RARE_SPARSE_HITS           = 30;
const RARE_EV_THRESHOLD          = 0.80;
const RARE_TAIL_MIN              = 0.12;
const RARE_EXTREME_GAP_BOOST     = 0.15;
const RARE_EARLY_WINDOW_FRAC     = 0.60;
const RARE_LATE_WINDOW_FRAC      = 2.50;
const RARE_CAL_IMPACT_HITS       = 150;
const RARE_CAL_REDUCED_WEIGHT    = 0.30;
const RARE_PAYOUT = { '100x': 0.95, '250x': 0.92, '500x': 0.90, '1000x': 0.88 };

const TIMING_MAX_SHIFT_FACTOR     = 0.15;
const TIMING_GAP_CORRECTION_SCALE = 0.08;
const TIMING_CENTER_PULL_SCALE    = 0.10;
const TIMING_RECENT_WINDOW        = 20;
const TIMING_RECENT_SPIKE_THRESH  = 0.50;
const TIMING_RECENT_SPIKE_SHIFT   = 0.05;

const BMA_WARMUP_COUNT  = 80;
const BMA_WARMUP_PRIORS = [0.45, 0.35, 0.20];
const BAY_MAX_RECENCY_WEIGHT = 0.05;

// ── Engine IDs for this module only ──────────────────────────────────────────
const ENGINE_IDS = ['engine', 'ens', 'geo', 'bay', 'km'];

// ── Per-target timing feedback state ─────────────────────────────────────────
const timingState = {};
for (const t of TARGETS) {
  timingState[t.label] = { earlyCount: 0, totalCount: 0, earlyQueue: [], recentEarlyCount: 0 };
}

// ── Per-engine CUSUM / regime state ──────────────────────────────────────────
const engineCusumState = {};
for (const id of ENGINE_IDS) {
  engineCusumState[id] = {};
  for (const t of TARGETS) {
    engineCusumState[id][t.label] = {
      cusum: 0, regimeFactor: 0, ewmaRate: -1, count: 0,
      regimeLabel: 'neutral', hysteresisCount: 0, pendingLabel: 'neutral', confirmedFactor: 0,
    };
  }
}

// ── Per-engine state ──────────────────────────────────────────────────────────
const STATE = {
  engine:  { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ens:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  geo:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  bay:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  km:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
};

// ── Per-engine last-hit tracker ───────────────────────────────────────────────
const engineLastHit = {};
for (const id of Object.keys(STATE)) {
  engineLastHit[id] = {};
  for (const t of TARGETS) engineLastHit[id][t.label] = -1;
}

// ── Gap stats cache ───────────────────────────────────────────────────────────
const gapStatsCache = new Map();
let   cacheRoundId  = -1;

// ── Calibration state ─────────────────────────────────────────────────────────
const calibState = {};
for (const t of TARGETS) {
  calibState[t.label] = {};
  for (const m of STAT_MODELS) {
    calibState[t.label][m.id] = CAL_BINS.slice(0, -1).map((lo, i) => {
      const mid = (lo + CAL_BINS[i + 1]) / 2;
      return { ewmaAct: mid, ewmaPred: mid, count: 0 };
    });
  }
}

// ── Adaptive ensemble weights ─────────────────────────────────────────────────
const modelScores = {};
for (const t of TARGETS) {
  modelScores[t.label] = {};
  for (const m of STAT_MODELS) modelScores[t.label][m.id] = { ewma: 0.693, count: 0 };
}

// ── Validation metrics ────────────────────────────────────────────────────────
const valMetrics = {};
for (const t of TARGETS) {
  valMetrics[t.label] = {};
  for (const m of STAT_MODELS) {
    valMetrics[t.label][m.id] = {
      brierSum: 0, logLossSum: 0, count: 0,
      wins: 0, losses: 0, earlyCount: 0,
      takenWins: 0, takenTotal: 0,
      tradeCount: 0, totalWins: 0,
    };
  }
}

const takenTradesMetrics = {};
for (const t of TARGETS) {
  takenTradesMetrics[t.label] = { wins: 0, losses: 0, early: 0, total: 0 };
}

let initialised = false;

// ── In-memory rounds cache (local to this engine, never shared) ───────────────
let cachedRounds        = [];
let cachedRoundsLastId  = 0;

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetStatEngineState() {
  console.log('[statEngine] resetStatEngineState()');
  for (const id of Object.keys(STATE)) {
    STATE[id].lockedMap    = null;
    STATE[id].savedSet     = null;
    STATE[id].needsRebuild = true;
    STATE[id].lastRoundId  = 0;
  }
  for (const id of Object.keys(engineCusumState)) {
    for (const t of TARGETS) {
      engineCusumState[id][t.label] = {
        cusum: 0, regimeFactor: 0, ewmaRate: -1, count: 0,
        regimeLabel: 'neutral', hysteresisCount: 0, pendingLabel: 'neutral', confirmedFactor: 0,
      };
    }
  }
  gapStatsCache.clear();
  cacheRoundId = -1;
  initialised  = false;
  cachedRounds = [];
  cachedRoundsLastId = 0;
  for (const t of TARGETS) {
    timingState[t.label] = { earlyCount: 0, totalCount: 0, earlyQueue: [], recentEarlyCount: 0 };
    takenTradesMetrics[t.label] = { wins: 0, losses: 0, early: 0, total: 0 };
  }
  for (const id of Object.keys(engineLastHit)) {
    for (const t of TARGETS) engineLastHit[id][t.label] = -1;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function sigmoid(x)       { return 1 / (1 + Math.exp(-x)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function getEngineGapSinceLast(engineId, targetLabel, lastRoundId, sharedGapSinceLast) {
  const lastHit = engineLastHit[engineId]?.[targetLabel] ?? -1;
  if (lastHit < 0) return sharedGapSinceLast;
  return Math.max(0, lastRoundId - lastHit);
}

// ── Timing feedback ───────────────────────────────────────────────────────────
function recordTimingOutcome(targetLabel, isEarly) {
  const ts = timingState[targetLabel];
  if (!ts) return;
  ts.totalCount++;
  if (isEarly) ts.earlyCount++;
  const val = isEarly ? 1 : 0;
  ts.earlyQueue.push(val);
  ts.recentEarlyCount += val;
  if (ts.earlyQueue.length > TIMING_RECENT_WINDOW) ts.recentEarlyCount -= ts.earlyQueue.shift();
}

function getTimingParams(targetLabel) {
  const ts = timingState[targetLabel];
  if (!ts || ts.totalCount < 5) return { earlyRate: 0, recentEarlyRate: 0, timingShiftFactor: 0, hasData: false };
  const earlyRate       = ts.earlyCount / ts.totalCount;
  const recentTotal     = ts.earlyQueue.length;
  const recentEarlyRate = recentTotal > 0 ? ts.recentEarlyCount / recentTotal : earlyRate;
  return { earlyRate, recentEarlyRate, timingShiftFactor: clamp(earlyRate, 0, TIMING_MAX_SHIFT_FACTOR), hasData: true };
}

function applyTimingCorrection(expectedGap, effectiveWidth, targetLabel, maxWidth) {
  const { earlyRate, recentEarlyRate, timingShiftFactor, hasData } = getTimingParams(targetLabel);
  if (!hasData) {
    const low = Math.max(1, expectedGap - effectiveWidth);
    return { low, high: low + effectiveWidth - 1, effectiveWidth, timingShiftFactor: 0 };
  }
  const expectedGapCorrected = Math.max(1, Math.round(expectedGap * (1 - TIMING_GAP_CORRECTION_SCALE * earlyRate)));
  const center = Math.max(1, Math.round(expectedGapCorrected * (1 - TIMING_CENTER_PULL_SCALE * earlyRate)));
  let low  = Math.max(1, center - Math.floor(effectiveWidth / 2));
  let high = low + effectiveWidth - 1;
  if (recentEarlyRate > TIMING_RECENT_SPIKE_THRESH) {
    const spikeShift = Math.round(expectedGapCorrected * TIMING_RECENT_SPIKE_SHIFT);
    low  = Math.max(0, low - spikeShift);
    high = low + effectiveWidth - 1;
  }
  return { low, high, effectiveWidth, timingShiftFactor, expectedGapCorrected };
}

// ── Regime detection ──────────────────────────────────────────────────────────
function getDynamicRegimeParams(outcomeCount) {
  if (outcomeCount < REGIME_ACTIVATION_OUTCOMES) return { threshold: 9999, minFactor: 9999, active: false };
  const t = clamp((outcomeCount - REGIME_ACTIVATION_OUTCOMES) / (400 - REGIME_ACTIVATION_OUTCOMES), 0, 1);
  return { threshold: 2.4 - t * (2.4 - 1.65), minFactor: 0.45 - t * (0.45 - 0.18), active: true };
}

function updateEngineRegime(rounds, targetMin, targetLabel, engineId, gs, isRandomSignal) {
  const cs = engineCusumState[engineId]?.[targetLabel];
  if (!cs) return { regimeFactor: 0, regimeLabel: 'neutral', shortRate: 0, shortBaseline: 0, cusumNorm: 0, regimeConfidence: 0 };

  const n = rounds.length;
  const shortWindow = rounds.slice(Math.max(0, n - REGIME_SHORT_WINDOW));
  let shortHits = 0;
  for (const r of shortWindow) if (r.multiplier >= targetMin) shortHits++;
  const shortRate     = shortHits / Math.max(1, shortWindow.length);
  const shortBaseline = gs.hits / Math.max(1, gs.n);

  if (cs.ewmaRate < 0) cs.ewmaRate = shortRate;
  else cs.ewmaRate = (1 - REGIME_DECAY) * cs.ewmaRate + REGIME_DECAY * shortRate;
  cs.count++;

  const regimeConfidence = clamp((cs.count - REGIME_ACTIVATION_OUTCOMES) / 200, 0, 1);
  const { threshold, minFactor, active } = getDynamicRegimeParams(cs.count);

  if (!active) {
    const sigma        = Math.sqrt(Math.max(1e-9, shortBaseline * (1 - shortBaseline)));
    const deviationSig = sigma > 0 ? (shortRate - shortBaseline) / sigma : 0;
    let hintFactor = 0;
    if (Math.abs(deviationSig) >= REGIME_EARLY_HINT_SIGMA) {
      const rawHint = Math.sign(deviationSig) * REGIME_EARLY_HINT_MAX
        * clamp((Math.abs(deviationSig) - REGIME_EARLY_HINT_SIGMA) / REGIME_EARLY_HINT_SIGMA, 0, 1);
      hintFactor = rawHint * ((isRandomSignal === true) ? 0.30 : 0.60);
    }
    hintFactor = clamp(hintFactor, -REGIME_MAX_BLEND, REGIME_MAX_BLEND);
    cs.regimeFactor = cs.confirmedFactor = hintFactor;
    cs.regimeLabel  = hintFactor > 0.03 ? 'hot' : hintFactor < -0.03 ? 'cold' : 'neutral';
    return { regimeFactor: +hintFactor.toFixed(4), regimeLabel: cs.regimeLabel, shortRate, shortBaseline, cusumNorm: 0, regimeConfidence: 0, earlyHint: true };
  }

  cs.cusum += shortRate - shortBaseline;
  const sigma     = Math.sqrt(Math.max(1e-9, shortBaseline * (1 - shortBaseline)));
  const cusumNorm = cs.cusum / Math.max(sigma * Math.sqrt(REGIME_SHORT_WINDOW), 1e-6);
  const clipHot   = REGIME_CUSUM_CLIP_HOT  * sigma * Math.sqrt(REGIME_SHORT_WINDOW);
  const clipCold  = REGIME_CUSUM_CLIP_COLD * sigma * Math.sqrt(REGIME_SHORT_WINDOW);
  cs.cusum = clamp(cs.cusum, -clipCold, clipHot);

  const rawFactor    = Math.tanh(cusumNorm / (threshold * 1.5));
  const randomDampen = (isRandomSignal === true) ? 0.25 : 0.50;
  cs.regimeFactor    = clamp(
    (1 - REGIME_DECAY) * cs.regimeFactor + REGIME_DECAY * rawFactor * randomDampen,
    -REGIME_MAX_BLEND, REGIME_MAX_BLEND
  );

  const rawLabel = cs.regimeFactor > 0.03 ? 'hot' : cs.regimeFactor < -0.03 ? 'cold' : 'neutral';
  if (rawLabel === cs.pendingLabel) cs.hysteresisCount++;
  else { cs.pendingLabel = rawLabel; cs.hysteresisCount = 1; }
  if (cs.hysteresisCount >= REGIME_HYSTERESIS_REQUIRED) { cs.regimeLabel = cs.pendingLabel; cs.confirmedFactor = cs.regimeFactor; }

  const effectiveFactor = Math.abs(cs.confirmedFactor) >= minFactor ? cs.confirmedFactor : 0;
  return { regimeFactor: +effectiveFactor.toFixed(4), regimeLabel: cs.regimeLabel, shortRate: +shortRate.toFixed(4), shortBaseline: +shortBaseline.toFixed(4), cusumNorm: +cusumNorm.toFixed(3), regimeConfidence: +regimeConfidence.toFixed(3), earlyHint: false };
}

function applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regimeFactor) {
  if (regimeFactor === 0) return { low, high };
  const widthDelta  = clamp(-regimeFactor * REGIME_WIDTH_SCALE, -0.05, 0.08);
  const newWidth    = Math.max(1, Math.round(effectiveWidth * (1 + widthDelta)));
  const offsetShift = Math.round(regimeFactor * expectedGap * REGIME_OFFSET_SCALE);
  const newLow      = Math.max(0, low - offsetShift);
  return { low: newLow, high: newLow + newWidth - 1 };
}

function applyRegimeToP(baseP, shortRate, regimeFactor, blendStrength) {
  if (regimeFactor === 0) return baseP;
  const cappedFactor = clamp(regimeFactor, -REGIME_MAX_BLEND, REGIME_MAX_BLEND);
  const regimeBlend  = Math.abs(cappedFactor) * blendStrength;
  return Math.max(1e-6, Math.min(0.5, (1 - regimeBlend) * baseP + regimeBlend * shortRate));
}

// ── Gap stats cache ───────────────────────────────────────────────────────────
function getGapStats(rounds, targetMin, lastRoundId) {
  if (lastRoundId !== cacheRoundId) { gapStatsCache.clear(); cacheRoundId = lastRoundId; }
  const key = `${targetMin}`;
  if (gapStatsCache.has(key)) return gapStatsCache.get(key);
  const result = computeGapStats(rounds, targetMin);
  gapStatsCache.set(key, result);
  return result;
}

function computeGapStats(rounds, targetMin) {
  const n = rounds.length;
  let hits = 0, lastIdx = -1;
  const gaps = [];
  for (let i = 0; i < n; i++) {
    if (rounds[i].multiplier >= targetMin) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++;
    }
  }
  if (hits < 3) return null;

  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const pGlobal      = (hits + 1) / (n + 2);
  const r200hits     = rounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
  const pRecent      = (r200hits + 1) / 202;

  const p0 = hits / n;
  let cusum = 0, maxCusum = 0;
  const cusumWindow = rounds.slice(-150);
  for (const r of cusumWindow) {
    cusum += (r.multiplier >= targetMin ? 1 : 0) - p0;
    if (Math.abs(cusum) > maxCusum) maxCusum = Math.abs(cusum);
  }
  const sigma0       = Math.sqrt(Math.max(1e-9, p0 * (1 - p0)));
  const cusumNorm    = maxCusum / (sigma0 * Math.sqrt(cusumWindow.length));
  const rateShifted  = cusumNorm > 1.36;

  const sg = [...gaps].sort((a, b) => a - b);
  const m2 = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1 / pGlobal)
    : sg.length % 2 === 1 ? sg[m2] : (sg[m2 - 1] + sg[m2]) / 2;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap  = gaps.length > 0 ? gSum / gaps.length : 1 / pGlobal;
  const variance = gaps.length > 1 ? Math.max(0, gSS / gaps.length - meanGap ** 2) : meanGap * meanGap;
  const stdGap   = Math.sqrt(variance);
  const cv       = meanGap > 0 ? stdGap / meanGap : 1;

  const pctile = (frac) => sg.length === 0 ? meanGap : sg[Math.min(sg.length - 1, Math.floor(frac * sg.length))];
  const p75 = pctile(0.75), p90 = pctile(0.90), p95 = pctile(0.95);

  const hsm = halfSampleMode(sg, medianGap, cv);
  let modeWeight;
  if      (cv < 1.0) modeWeight = 0.50;
  else if (cv > 1.3) modeWeight = 0.10;
  else               modeWeight = 0.50 - (cv - 1.0) * (0.40 / 0.30);
  const kmExpectedGap = Math.max(1, Math.round(medianGap * (1 - modeWeight) + hsm * modeWeight));
  const kmCDF = buildKmCDF(sg);

  let currentStreak = 0;
  for (let i = n - 1; i >= 0; i--) { if (rounds[i].multiplier < targetMin) currentStreak++; else break; }

  let maxAC = 0;
  if (gaps.length >= 50) {
    let gVarSum = 0;
    for (const g of gaps) gVarSum += (g - meanGap) ** 2;
    if (gVarSum > 0) {
      for (let lag = 1; lag <= Math.min(5, gaps.length - 1); lag++) {
        let cov = 0;
        for (let i = lag; i < gaps.length; i++) cov += (gaps[i] - meanGap) * (gaps[i - lag] - meanGap);
        const ac = cov / gVarSum;
        if (Math.abs(ac) > Math.abs(maxAC)) maxAC = ac;
      }
    }
  }
  const isRandom = gaps.length >= 50 && Math.abs(maxAC) < 0.10;

  return { hits, n, pGlobal, pRecent, rateShifted, cusumNorm: +cusumNorm.toFixed(3), gapSinceLast, meanGap, medianGap, stdGap, cv, p75, p90, p95, sg, kmCDF, hsm, kmExpectedGap, currentStreak, maxAC: +maxAC.toFixed(4), isRandom };
}

function halfSampleMode(sg, medianGap, cv) {
  const n = sg.length;
  if (n < 8) return medianGap;
  if (cv > 2.0) return sg[Math.floor(n / 2)];
  const h = Math.max(2, Math.floor(n / 2));
  let bestRange = sg[sg.length - 1] - sg[0] + 1, bestStart = 0;
  for (let i = 0; i + h - 1 < n; i++) {
    const range = sg[i + h - 1] - sg[i];
    if (range < bestRange) { bestRange = range; bestStart = i; }
  }
  return (sg[bestStart] + sg[bestStart + h - 1]) / 2;
}

function buildKmCDF(sg) {
  if (sg.length < 5) return null;
  const m = sg.length, steps = [];
  let S = 1.0, i = 0;
  while (i < m) {
    const t = sg[i], nAtRisk = m - i;
    let d = 0;
    while (i < m && sg[i] === t) { d++; i++; }
    S *= (1 - d / nAtRisk);
    steps.push({ t, S: Math.max(0, S) });
  }
  return steps;
}

function kmCDFQuery(kmCDF, W) {
  if (!kmCDF || kmCDF.length === 0) return null;
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) { const mid = (lo + hi) >>> 1; if (kmCDF[mid].t <= W) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
  return Math.max(0, Math.min(1 - 1e-9, 1 - (idx >= 0 ? kmCDF[idx].S : 1.0)));
}

function kmWindowProb(kmCDF, lo, hi) {
  const cdfHi = kmCDFQuery(kmCDF, hi) ?? 0;
  const cdfLo = lo > 0 ? (kmCDFQuery(kmCDF, lo - 1) ?? 0) : 0;
  return Math.max(0, cdfHi - cdfLo);
}

// ── Window placement ──────────────────────────────────────────────────────────
function hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth) {
  const parametric = parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast);
  const km = kmCDF ? empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast) : parametric;
  const blendedLow  = Math.round(0.70 * parametric.low  + 0.30 * km.low);
  const blendedHigh = Math.round(0.70 * parametric.high + 0.30 * km.high);
  const { low, high, effectiveWidth: correctedWidth } = applyTimingCorrection(expectedGap, blendedHigh - blendedLow + 1, targetLabel ?? '', maxWidth ?? effectiveWidth);
  return { low, high, effectiveWidth: correctedWidth };
}

function parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) return { low: 1, high: effectiveWidth };
  const low = Math.max(1, (expectedGap - gapSinceLast) - effectiveWidth);
  return { low, high: low + effectiveWidth - 1 };
}

function empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) return { low: 1, high: effectiveWidth };
  const maxLow = Math.max(0, 3 * expectedGap - gapSinceLast - effectiveWidth);
  let bestScore = -Infinity, bestLow = 0;
  for (let low = 1; low <= maxLow; low++) {
    const absLo = gapSinceLast + low, absHi = gapSinceLast + low + effectiveWidth - 1;
    const score = kmWindowProb(kmCDF, absLo, absHi) - WINDOW_LAMBDA * effectiveWidth;
    if (score > bestScore) { bestScore = score; bestLow = low; }
    if (absLo > expectedGap * 3 && kmWindowProb(kmCDF, absLo, absHi) < 0.001) break;
  }
  return { low: bestLow, high: bestLow + effectiveWidth - 1 };
}

// ── Pareto tail correction ─────────────────────────────────────────────────────
function applyParetoCorrectedProbW(rawProbW, cv, expectedGap, maxWidth, isRare, hits) {
  if (!isRare || cv <= 1.1) return rawProbW;
  const alpha   = 1 / Math.max(0.1, cv * cv);
  const paretoP = 1 - Math.pow(1 / (1 + maxWidth / Math.max(1, expectedGap)), alpha);
  const expP    = 1 - Math.exp(-maxWidth / Math.max(1, expectedGap));
  const tailP   = paretoP * 0.60 + expP * 0.40;
  const blendW  = hits >= 120 ? Math.min(0.45, (cv - 1.0) * 0.70) : Math.min(0.30, (cv - 1.0) * 0.60);
  return Math.max(1e-6, Math.min(1 - 1e-6, (1 - blendW) * rawProbW + blendW * tailP));
}

// ── Calibration ───────────────────────────────────────────────────────────────
function getCalBinIdx(probW) {
  for (let i = 0; i < CAL_BINS.length - 1; i++) { if (probW < CAL_BINS[i + 1]) return i; }
  return CAL_BINS.length - 2;
}

function updateCalibration(targetLabel, modelId, predictedProbW, outcome) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return;
  const decay  = TARGETS.find(t => t.label === targetLabel)?.rare ? CAL_DECAY.rare : CAL_DECAY.normal;
  // FIXED: early treated as actual=0 (loss) for calibration purposes.
  // Early hits mean the engine predicted the WRONG time window — that is a calibration
  // failure. Treating as 0.5 softened probability estimates and masked the timing bias.
  const actual = outcome === 'win' ? 1 : 0;
  const bin    = bins[getCalBinIdx(predictedProbW)];
  bin.ewmaAct  = (1 - decay) * bin.ewmaAct  + decay * actual;
  bin.ewmaPred = (1 - decay) * bin.ewmaPred + decay * predictedProbW;
  bin.count    = Math.min(bin.count + 1, 500);
}

function getECE(targetLabel, modelId) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return null;
  let ece = 0, total = 0;
  for (const bin of bins) {
    if (bin.count < 2) continue;
    ece += Math.abs(bin.ewmaAct - bin.ewmaPred) * bin.count;
    total += bin.count;
  }
  return total > 0 ? ece / total : null;
}

function applyCalibrationRelaxed(probW, targetLabel, modelId, hits) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < CAL_MIN_SAMPLES) return probW;
  const ece    = getECE(targetLabel, modelId);
  const sparse = (hits ?? 0) < 200;
  const tight  = ece != null && ece < 0.02;
  const correctionWeight = (sparse || tight) ? 0.35 : 0.70;
  if (bin.ewmaPred < 1e-6) return probW;
  const ratio = bin.ewmaAct / bin.ewmaPred;
  if (Math.abs(ratio - 1) < 0.08) return probW;
  const corrected = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(CAL_MIN_RATIO, Math.min(CAL_MAX_RATIO, ratio))));
  const capped    = Math.min(corrected, probW + CAL_MAX_SHIFT);
  const warmupT   = clamp(bin.count / CAL_WARMUP_OUTCOMES, 0, 1);
  const rawWeight = CAL_WARMUP_RAW_WEIGHT * (1 - warmupT);
  const fullyCalibrated = probW * rawWeight + capped * (1 - rawWeight);
  return probW * (1 - correctionWeight) + fullyCalibrated * correctionWeight;
}

// ── Validation metrics ────────────────────────────────────────────────────────
function updateValidationMetrics(targetLabel, modelId, predictedProbW, outcome, recommendation) {
  const v = valMetrics[targetLabel]?.[modelId];
  if (!v) return;
  // FIXED: early treated as actual=0 for validation metrics (wrong timing = miss).
  const actual = outcome === 'win' ? 1 : 0;
  const p = Math.max(1e-7, Math.min(1 - 1e-7, predictedProbW));
  v.brierSum   += (actual - p) ** 2;
  v.logLossSum += -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
  v.count++;
  if (outcome === 'win') v.wins++; else if (outcome === 'early') v.earlyCount++; else v.losses++;
  if (recommendation === 'TAKE') { v.takenTotal++; v.tradeCount++; if (outcome === 'win') { v.takenWins++; v.totalWins++; } }
}

// ── BMA Ensemble ──────────────────────────────────────────────────────────────
function logLossVal(actual, probW) {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probW));
  return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
}

function updateModelScore(targetLabel, modelId, predictedProbW, outcome) {
  const s = modelScores[targetLabel]?.[modelId];
  if (!s) return;
  // FIXED: early treated as actual=0 for model scoring (wrong timing = miss).
  const actual = outcome === 'win' ? 1 : 0;
  const decay  = TARGETS.find(t => t.label === targetLabel)?.rare ? 0.015 : 0.03;
  s.ewma  = s.count === 0 ? logLossVal(actual, predictedProbW) : (1 - decay) * s.ewma + decay * logLossVal(actual, predictedProbW);
  s.count = Math.min(s.count + 1, 500);
}

function buildEnsemble(targetLabel, probGeo, probBay, probKm, cv) {
  const scores   = modelScores[targetLabel];
  const modelIds = ['geo', 'bay', 'km'];
  const probs    = [probGeo, probBay, probKm];
  const minCount = Math.min(...modelIds.map(id => scores[id]?.count ?? 0));

  let weights;
  if (minCount < BMA_WARMUP_COUNT) {
    const warmupT = clamp(minCount / BMA_WARMUP_COUNT, 0, 1);
    let cvPriors = cv > 1.5 ? [0.25, 0.30, 0.45] : cv < 1.0 ? [0.55, 0.30, 0.15] : [...BMA_WARMUP_PRIORS];
    const aw = modelIds.map(id => Math.exp(-(scores[id]?.count > 5 ? scores[id].ewma : 0.693) * 2));
    const awSum = aw.reduce((a, b) => a + b, 0);
    if (awSum > 0) for (let i = 0; i < aw.length; i++) aw[i] /= awSum;
    weights = cvPriors.map((p, i) => (1 - warmupT) * p + warmupT * aw[i]);
  } else {
    weights = modelIds.map(id => Math.exp(-(scores[id]?.count > 5 ? scores[id].ewma : 0.693) * 2));
  }

  const wSum = weights.reduce((a, b) => a + b, 0);
  if (wSum < 1e-9) return { ensProb: (probGeo + probBay + probKm) / 3, spread: 0, adjWeights: [1/3, 1/3, 1/3], modelDisagreementScore: 0 };
  for (let i = 0; i < weights.length; i++) weights[i] /= wSum;

  const wMean = probs.reduce((s, p, i) => s + weights[i] * p, 0);
  const wVar  = probs.reduce((s, p, i) => s + weights[i] * (p - wMean) ** 2, 0);
  const spread = Math.sqrt(wVar);

  const adjWeights = [...weights];
  for (let i = 0; i < probs.length; i++) if (Math.abs(probs[i] - wMean) > 0.12) adjWeights[i] *= 0.5;
  const adjSum = adjWeights.reduce((a, b) => a + b, 0);
  if (adjSum > 0) for (let i = 0; i < adjWeights.length; i++) adjWeights[i] /= adjSum;

  const ensProb = probs.reduce((s, p, i) => s + adjWeights[i] * p, 0);
  const modelDisagreementScore = +Math.max(Math.abs(probs[0]-probs[1]), Math.abs(probs[1]-probs[2]), Math.abs(probs[0]-probs[2])).toFixed(3);
  return { ensProb, spread, adjWeights, modelDisagreementScore };
}

// ── Beta confidence ───────────────────────────────────────────────────────────
function betaConf(probW, hits, spread, regimeConfidence, z) {
  const effectiveN = Math.min(hits, 300);
  const alpha = probW * effectiveN + 1, beta = (1 - probW) * effectiveN + 1;
  const postStd = Math.sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)));
  let c = probW * 100 - postStd * 120;
  if      (hits < 10) c -= 15;
  else if (hits < 25) c -= 7;
  else if (hits < 50) c -= 3;
  if (spread != null) c -= Math.min(15, Math.round(15 * spread / 0.5));
  const streakPenalty = (z != null && z > 3.0 && (regimeConfidence ?? 0) > 0.6) ? 5 : 0;
  c -= streakPenalty * (regimeConfidence ?? 0);
  return Math.max(20, Math.min(88, Math.round(c)));
}

// ── Signal decision ───────────────────────────────────────────────────────────
function computeSignalDecision(probW, confidence, spread, regimeFactor, regimeConfidence, isRandom, targetLabel, modelId, hits) {
  const spreadVal = spread ?? 0;
  const signalStrength = clamp(Math.round(
    clamp((probW - 0.35) / 0.30 * 100, 0, 100)       * 0.45
    + clamp((1 - spreadVal / 0.30) * 100, 0, 100)     * 0.30
    + clamp((confidence - 30) / 55 * 100, 0, 100)     * 0.25
  ), 0, 100);
  if (probW < SIG_HARD_FLOOR)   return buildSignalResult(probW, signalStrength, 'SKIP', 'FILTERED');
  if (probW >= SIG_STRONG_PROB) return buildSignalResult(probW, signalStrength, 'TAKE', 'STRONG_SIGNAL');
  if (probW < SIG_LOW_ZONE_TOP) {
    if (spreadVal < SIG_LOW_ZONE_SPREAD && confidence > SIG_LOW_ZONE_CONF)
      return buildSignalResult(probW, signalStrength, 'TAKE', 'MODERATE_SIGNAL');
    return buildSignalResult(probW, signalStrength, 'SKIP', 'FILTERED');
  }
  if (spreadVal < SIG_MODERATE_SPREAD && confidence > SIG_MODERATE_CONF)
    return buildSignalResult(probW, signalStrength, 'TAKE', 'MODERATE_SIGNAL');
  return buildSignalResult(probW, signalStrength, 'SKIP', 'FILTERED');
}

function buildSignalResult(probW, signalStrength, recommendation, decisionReason) {
  const ev = probW - (1 - probW);
  return {
    recommendation, decisionReason,
    finalProbUsed:  +probW.toFixed(4),
    amplifiedProbW: +probW.toFixed(4),
    rawProbW:       +probW.toFixed(4),
    signalStrength: clamp(signalStrength, 0, 100),
    aggressiveMode: false,
    ev:             +ev.toFixed(4),
    signalQuality:  clamp(Math.round(probW * 60 + signalStrength * 0.40), 0, 100),
    risk:           signalStrength >= 65 ? 'low' : signalStrength >= 45 ? 'medium' : 'high',
  };
}

function computeDecision(probW, confidence, spread) {
  const spreadVal = spread ?? 0, ev = probW - (1 - probW);
  const base = { ev: +ev.toFixed(4), aggressiveMode: false };
  if (probW < SIG_HARD_FLOOR)   return { ...base, signalQuality: 0,  risk: 'high',   recommendation: 'SKIP', decisionReason: 'FILTERED' };
  if (probW >= SIG_STRONG_PROB) return { ...base, signalQuality: 70, risk: 'low',    recommendation: 'TAKE', decisionReason: 'STRONG_SIGNAL' };
  if (probW < SIG_LOW_ZONE_TOP) {
    if (spreadVal < SIG_LOW_ZONE_SPREAD && confidence > SIG_LOW_ZONE_CONF)
      return { ...base, signalQuality: 50, risk: 'medium', recommendation: 'TAKE', decisionReason: 'MODERATE_SIGNAL' };
    return { ...base, signalQuality: 0, risk: 'high', recommendation: 'SKIP', decisionReason: 'FILTERED' };
  }
  if (spreadVal < SIG_MODERATE_SPREAD && confidence > SIG_MODERATE_CONF)
    return { ...base, signalQuality: 55, risk: 'medium', recommendation: 'TAKE', decisionReason: 'MODERATE_SIGNAL' };
  return { ...base, signalQuality: 0, risk: 'high', recommendation: 'SKIP', decisionReason: 'FILTERED' };
}

function getStreakInfo(gs, maxWidth, isRare) {
  const { gapSinceLast, meanGap, stdGap, p90, p95 } = gs;
  const z           = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;
  const confPenalty = Math.round(12 * sigmoid(z - 2.0));
  let streakStatus  = 'normal';
  if      (gapSinceLast >= p95) streakStatus = 'extreme';
  else if (gapSinceLast >= p90) streakStatus = 'severe';
  return { z: +z.toFixed(2), confPenalty, streakStatus, effectiveWidth: maxWidth };
}

// ── BUILD FUNCTIONS ───────────────────────────────────────────────────────────

function buildPrediction(rounds, targetMin, maxWidth, isRare, lastRoundId) {
  if (isRare && targetMin >= RARE_MIN_MULTIPLIER) {
    const gs = getGapStats(rounds, targetMin, lastRoundId);
    if (!gs) return null;
    const target = TARGETS.find(t => t.min === targetMin);
    return buildRare(gs, maxWidth, target?.label ?? '?', true, rounds, targetMin);
  }
  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const { hits, pGlobal, gapSinceLast, kmCDF, kmExpectedGap, cv } = gs;
  const target = TARGETS.find(t => t.min === targetMin), targetLabel = target?.label ?? '?';
  const regime  = updateEngineRegime(rounds, targetMin, targetLabel, 'engine', gs, gs.isRandom);
  const p       = applyRegimeToP(pGlobal, regime.shortRate, regime.regimeFactor, 0.10);
  const probW   = 1 - Math.pow(1 - p, maxWidth);
  const expectedGap = Math.max(1, Math.round((1 - p) / p));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare ?? false);
  const { low, high }              = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence                 = Math.max(20, betaConf(probW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision                   = computeDecision(probW, confidence, null);
  return { low: rLow, high: rHigh, expectedGap, opensIn: rLow, confidence, probW: +probW.toFixed(4), p: +p.toFixed(6), streakStatus, currentStreak: gs.currentStreak, z, gapSinceLast, hits, n: gs.n, regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, isRandom: gs.isRandom, maxAC: gs.maxAC, ...decision };
}

function buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, gapSinceLast: gsl, kmCDF, kmExpectedGap, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gsl;
  const regime       = updateEngineRegime(rounds, targetMin, targetLabel, 'geo', gs, gs.isRandom);
  const pGeo         = applyRegimeToP((hits + 1) / (n + 2), regime.shortRate, regime.regimeFactor, 0.10);
  const rawProbW     = 1 - Math.pow(1 - pGeo, maxWidth);
  const calibProbW   = applyCalibrationRelaxed(rawProbW, targetLabel, 'geo', hits);
  const expectedGap  = Math.max(1, Math.round((1 - pGeo) / pGeo));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high }              = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence                 = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision                   = computeDecision(calibProbW, confidence, null);
  return { low: rLow, high: rHigh, expectedGap, opensIn: rLow, confidence, probW: +calibProbW.toFixed(4), rawProbW: +rawProbW.toFixed(4), p: +pGeo.toFixed(6), streakStatus, currentStreak: gs.currentStreak, z, gapSinceLast, hits, n, rateShifted: gs.rateShifted, model: 'geo', regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, isRandom: gs.isRandom, maxAC: gs.maxAC, ...decision };
}

function buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, pRecent, rateShifted, kmCDF, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime       = updateEngineRegime(rounds, targetMin, targetLabel, 'bay', gs, gs.isRandom);
  const totalRecencyW = rateShifted ? BAY_MAX_RECENCY_WEIGHT : BAY_MAX_RECENCY_WEIGHT * 0.5;
  const pBay          = Math.max(1e-6, Math.min(0.5, (1 - totalRecencyW) * pGlobal + totalRecencyW * pRecent));
  const rawProbW      = 1 - Math.pow(1 - pBay, maxWidth);
  const calibProbW    = applyCalibrationRelaxed(rawProbW, targetLabel, 'bay', hits);
  const expectedGap   = Math.max(1, Math.round((1 - pBay) / pBay));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high }              = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence                 = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision                   = computeDecision(calibProbW, confidence, null);
  return { low: rLow, high: rHigh, expectedGap, opensIn: rLow, confidence, probW: +calibProbW.toFixed(4), rawProbW: +rawProbW.toFixed(4), p: +pBay.toFixed(6), streakStatus, currentStreak: gs.currentStreak, z, gapSinceLast, hits, n, rateShifted, model: 'bay', regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, isRandom: gs.isRandom, maxAC: gs.maxAC, ...decision };
}

function buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, kmCDF, kmExpectedGap, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  if (!kmCDF) return null;
  const regime    = updateEngineRegime(rounds, targetMin, targetLabel, 'km', gs, gs.isRandom);
  let rawProbW    = kmCDFQuery(kmCDF, maxWidth);
  if (rawProbW == null) return null;
  rawProbW        = Math.max(1e-6, Math.min(1 - 1e-6, applyParetoCorrectedProbW(rawProbW, cv, kmExpectedGap, maxWidth, isRare, hits)));
  const calibProbW  = applyCalibrationRelaxed(rawProbW, targetLabel, 'km', hits);
  const regimeExpGap = Math.max(1, kmExpectedGap + Math.round(-regime.regimeFactor * kmExpectedGap * 0.08));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high }              = hybridWindowPlacement(kmCDF, regimeExpGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, regimeExpGap, effectiveWidth, regime.regimeFactor);
  const confidence                 = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision                   = computeDecision(calibProbW, confidence, null);
  return { low: rLow, high: rHigh, expectedGap: kmExpectedGap, opensIn: rLow, confidence, probW: +calibProbW.toFixed(4), rawProbW: +rawProbW.toFixed(4), p: +rawProbW.toFixed(6), streakStatus, currentStreak: gs.currentStreak, z, gapSinceLast, hits, n, rateShifted: gs.rateShifted, model: 'km', regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, isRandom: gs.isRandom, maxAC: gs.maxAC, ...decision };
}

function buildEns(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, pRecent, rateShifted, kmCDF, kmExpectedGap, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime       = updateEngineRegime(rounds, targetMin, targetLabel, 'ens', gs, gs.isRandom);

  const pGeo = (hits + 1) / (n + 2);
  const totalRecencyW = rateShifted ? BAY_MAX_RECENCY_WEIGHT : BAY_MAX_RECENCY_WEIGHT * 0.5;
  const pBay = Math.max(1e-6, Math.min(0.5, (1 - totalRecencyW) * pGlobal + totalRecencyW * pRecent));

  const probGeo = applyCalibrationRelaxed(1 - Math.pow(1 - pGeo, maxWidth), targetLabel, 'geo', hits);
  const probBay = applyCalibrationRelaxed(1 - Math.pow(1 - pBay, maxWidth), targetLabel, 'bay', hits);
  let rawKmProb = kmCDF ? kmCDFQuery(kmCDF, maxWidth) : null;
  if (rawKmProb != null) rawKmProb = applyParetoCorrectedProbW(rawKmProb, cv, kmExpectedGap, maxWidth, isRare, hits);
  const probKm  = rawKmProb != null ? applyCalibrationRelaxed(Math.max(1e-6, Math.min(1 - 1e-6, rawKmProb)), targetLabel, 'km', hits) : probGeo;

  const { ensProb, spread, adjWeights, modelDisagreementScore } = buildEnsemble(targetLabel, probGeo, probBay, probKm, cv);
  const calibrated = applyCalibrationRelaxed(ensProb, targetLabel, 'ens', hits);

  const egGeo          = Math.max(1, Math.round((1 - pGeo) / pGeo));
  const egBay          = Math.max(1, Math.round((1 - pBay) / pBay));
  const ensExpectedGap = Math.max(1, Math.round(adjWeights[0] * egGeo + adjWeights[1] * egBay + adjWeights[2] * kmExpectedGap));
  const regimeExpGap   = Math.max(1, ensExpectedGap + Math.round(-regime.regimeFactor * ensExpectedGap * 0.05));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high }              = hybridWindowPlacement(kmCDF, regimeExpGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, regimeExpGap, effectiveWidth, regime.regimeFactor);

  const ensembleConfidence = clamp(100 - Math.round(spread * 100) - Math.round(modelDisagreementScore * 10), 20, 95);
  const confidence         = Math.max(20, betaConf(calibrated, hits, spread, regime.regimeConfidence, z) - confPenalty);
  const decision           = computeSignalDecision(calibrated, confidence, spread, regime.regimeFactor, regime.regimeConfidence, gs.isRandom, targetLabel, 'ens', hits);

  return { low: rLow, high: rHigh, expectedGap: ensExpectedGap, opensIn: rLow, confidence, probW: +calibrated.toFixed(4), rawProbW: +ensProb.toFixed(4), p: +pGeo.toFixed(6), streakStatus, currentStreak: gs.currentStreak, z, gapSinceLast, hits, n, rateShifted, model: 'ens', spread: +spread.toFixed(3), modelDisagreementScore, ensembleConfidence, regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, ensWeights: { geo: +adjWeights[0].toFixed(3), bay: +adjWeights[1].toFixed(3), km: +adjWeights[2].toFixed(3) }, isRandom: gs.isRandom, maxAC: gs.maxAC, ...decision };
}

// ── RARE ENGINE ───────────────────────────────────────────────────────────────
function paretoCDF(W, alpha, expectedGap) {
  const a = Math.max(1.05, alpha), scale = Math.max(1, expectedGap * (a - 1));
  return Math.max(0, Math.min(1 - 1e-9, 1 - Math.pow(1 + W / scale, -a)));
}
function paretoWindowProb(lo, hi, alpha, expectedGap) {
  return Math.max(0, paretoCDF(hi, alpha, expectedGap) - (lo > 0 ? paretoCDF(lo - 1, alpha, expectedGap) : 0));
}
function kmSurvival(kmCDF, t) {
  if (!kmCDF || !kmCDF.length) return 1;
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) { const mid = (lo + hi) >>> 1; if (kmCDF[mid].t <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
  return idx >= 0 ? Math.max(0, kmCDF[idx].S) : 1.0;
}
function kmWindowProbDirect(kmCDF, lo, hi) { return Math.max(0, (lo > 0 ? kmSurvival(kmCDF, lo - 1) : 1.0) - kmSurvival(kmCDF, hi)); }

function applyRareCalibration(probW, targetLabel, modelId, hits) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < CAL_MIN_SAMPLES) return probW;
  if (bin.ewmaPred < 1e-6) return probW;
  const ratio = bin.ewmaAct / bin.ewmaPred;
  if (Math.abs(ratio - 1) < 0.08) return probW;
  const corrected = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(CAL_MIN_RATIO, Math.min(CAL_MAX_RATIO, ratio))));
  const capped    = Math.min(corrected, probW + CAL_MAX_SHIFT);
  const calWeight = hits < RARE_CAL_IMPACT_HITS ? RARE_CAL_REDUCED_WEIGHT : 0.70;
  return probW * (1 - calWeight) + capped * calWeight;
}

function computeRareDecision(tailProbability, extremeGapScore, targetLabel, hits) {
  const payout  = RARE_PAYOUT[targetLabel] ?? 0.90;
  const rareEV  = tailProbability * payout;
  const confidence = Math.round(20 + 60 * clamp(hits / 80, 0, 1) * 0.50 + 60 * clamp(tailProbability / 0.35, 0, 1) * 0.50);
  const rareSignal = rareEV > RARE_EV_THRESHOLD || tailProbability > RARE_TAIL_MIN || extremeGapScore > 0;
  return { rareEV: +rareEV.toFixed(4), estimatedPayout: payout, rareSignal, recommendation: rareSignal ? 'TAKE' : 'SKIP', confidence: clamp(confidence, 20, 85), risk: rareEV > 1.20 ? 'low' : rareEV > 0.80 ? 'medium' : 'high', signalQuality: Math.round(clamp(rareEV * 60 + tailProbability * 40, 0, 100)), ev: +rareEV.toFixed(4) };
}

function buildRare(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, kmCDF, kmExpectedGap, cv, p95, meanGap, stdGap } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'ens', gs, gs.isRandom);
  const alpha  = 1 / Math.max(0.30, cv * cv);
  const pGeo   = (hits + 1) / (n + 2);
  const geoP   = 1 - Math.pow(1 - pGeo, maxWidth);
  const tailP_full = paretoCDF(maxWidth, alpha, kmExpectedGap);
  const kmP    = kmCDF ? kmWindowProbDirect(kmCDF, 0, maxWidth) : null;
  const sparse = hits < RARE_SPARSE_HITS;
  const geoWeight = sparse ? RARE_TAIL_GEO_BLEND_SPARSE : RARE_TAIL_GEO_BLEND;
  const tailEstimate = kmP != null ? tailP_full * 0.60 + kmP * 0.40 : tailP_full;
  let rawProbW = Math.max(1e-6, Math.min(1 - 1e-6, geoWeight * geoP + (1 - geoWeight) * tailEstimate));
  const calibProbW = applyRareCalibration(rawProbW, targetLabel, 'ens', hits);
  const z = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;
  const extremeGap = gapSinceLast > (p95 ?? meanGap * 2.5);
  const extremeGapScore = extremeGap ? +z.toFixed(2) : 0;
  let tailProbability = calibProbW;
  if (extremeGap) tailProbability = Math.min(0.95, tailProbability + RARE_EXTREME_GAP_BOOST);
  const earlyEnd = Math.max(1, Math.round(kmExpectedGap * RARE_EARLY_WINDOW_FRAC));
  const lateEnd  = Math.max(earlyEnd + 1, Math.round(kmExpectedGap * RARE_LATE_WINDOW_FRAC));
  const earlyWindow = { low: 1, high: earlyEnd + 1 }, lateWindow = { low: kmExpectedGap, high: lateEnd };
  const earlyTailP = kmCDF ? kmWindowProbDirect(kmCDF, earlyWindow.low, earlyWindow.high) : paretoWindowProb(earlyWindow.low, earlyWindow.high, alpha, kmExpectedGap);
  const lateTailP  = kmCDF ? kmWindowProbDirect(kmCDF, lateWindow.low, lateWindow.high)  : paretoWindowProb(lateWindow.low, lateWindow.high, alpha, kmExpectedGap);
  const regimeOffset = Math.round(-regime.regimeFactor * kmExpectedGap * 0.05);
  const baseWin      = earlyTailP >= lateTailP ? earlyWindow : lateWindow;
  const { low: tcLow, high: tcHigh } = applyTimingCorrection(kmExpectedGap, baseWin.high - baseWin.low + 1, targetLabel, maxWidth);
  const primaryLow  = Math.max(1, tcLow  - regimeOffset);
  const primaryHigh = Math.min(primaryLow + maxWidth - 1, Math.max(primaryLow, tcHigh - regimeOffset));
  const decision    = computeRareDecision(tailProbability, extremeGapScore, targetLabel, hits);
  return { low: primaryLow, high: primaryHigh, expectedGap: kmExpectedGap, opensIn: primaryLow, confidence: decision.confidence, probW: +calibProbW.toFixed(4), rawProbW: +rawProbW.toFixed(4), p: +pGeo.toFixed(6), streakStatus: extremeGapScore > 0 ? (z > 3 ? 'extreme' : 'severe') : 'normal', currentStreak: gs.currentStreak, z: +z.toFixed(2), gapSinceLast, hits, n, rateShifted: gs.rateShifted, model: 'rare', regime: regime.regimeLabel, regimeFactor: regime.regimeFactor, regimeConfidence: regime.regimeConfidence, isRandom: gs.isRandom, maxAC: gs.maxAC, tailProbability: +tailProbability.toFixed(4), extremeGapScore, alpha: +alpha.toFixed(3), earlyWindow: { ...earlyWindow, tailP: +earlyTailP.toFixed(4) }, lateWindow: { ...lateWindow, tailP: +lateTailP.toFixed(4) }, primaryWindow: earlyTailP >= lateTailP ? 'early' : 'late', ...decision };
}

function buildStatPrediction(rounds, targetMin, maxWidth, modelId, lastRoundId) {
  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const target      = TARGETS.find(t => t.min === targetMin);
  const targetLabel = target?.label ?? '?';
  const isRare      = target?.rare ?? false;
  const engineGap   = getEngineGapSinceLast(modelId, targetLabel, lastRoundId, gs.gapSinceLast);
  if (isRare && targetMin >= RARE_MIN_MULTIPLIER) return buildRare(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
  switch (modelId) {
    case 'geo': return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'bay': return buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'km':  return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'ens': return buildEns(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    default:    return null;
  }
}

// ── makeKey ───────────────────────────────────────────────────────────────────
function makeKey(source, target, lo, hi) { return `${source}-${target}-${Number(lo) || 0}-${Number(hi) || 0}`; }

// ── getStatus ─────────────────────────────────────────────────────────────────
function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow) return { status: 'miss', hitRound: null };
  let lo = 0, hi = sortedRounds.length - 1, startIdx = sortedRounds.length;
  while (lo <= hi) { const mid = (lo + hi) >>> 1; if (sortedRounds[mid].roundId >= anchorRound) { startIdx = mid; hi = mid - 1; } else lo = mid + 1; }
  // FIXED: earlyHit only within floor(maxWidth/2) rounds before window open.
  // Hits further back are treated as MISS (engine had wrong timing entirely).
  const target = TARGETS.find(t => t.min === pred.targetMin);
  const maxW   = target?.maxWidth ?? 5;
  const earlyFloor = absLow - Math.floor(maxW / 2); // earliest round that counts as early
  for (let i = startIdx; i < sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < absLow) {
      // Only mark as early if within tolerance window; otherwise ignore (let window expire as loss)
      if (r.roundId >= earlyFloor) return { status: 'early', hitRound: r.roundId };
      continue; // hit too far before window — not counted, keep scanning
    }
    return { status: 'hit', hitRound: r.roundId };
  }
  if (currentRoundId >= absHigh) return { status: 'miss', hitRound: null };
  if (currentRoundId >= absLow && currentRoundId < absHigh) return { status: 'active', hitRound: null };
  return { status: 'waiting', hitRound: null };
}

// ── saveOutcome — shared by both branches of processEngine ────────────────────
// FIX: was duplicated verbatim in isExpired/isStale branch AND active branch.
// DRY violation fixed: extract into a single helper called from both places.
async function saveOutcome(engineId, target, absLow, absHigh, status, existing, state) {
  const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
  const key     = makeKey(engineId, target.label, absLow, absHigh);
  if (state.savedSet.has(key)) return false;
  state.savedSet.add(key);
  try {
    await savePrediction({
      target: target.label, minMult: target.min, outcome,
      lo: absLow, hi: absHigh,
      hitRound:   status.hitRound || null,
      generation: existing.generation || 1,
      source:     engineId,
      probW:      existing.probW ?? null,
    });
    recordTimingOutcome(target.label, outcome === 'early');
    if ((outcome === 'win' || outcome === 'early') && status.hitRound && engineLastHit[engineId]) {
      engineLastHit[engineId][target.label] = status.hitRound;
    }
    if (STAT_MODELS.some(m => m.id === engineId) && existing.probW != null) {
      updateCalibration(target.label, engineId, existing.probW, outcome);
      updateModelScore(target.label, engineId, existing.probW, outcome);
      updateValidationMetrics(target.label, engineId, existing.probW, outcome, existing.recommendation ?? null);
    }
    const { earlyRate } = getTimingParams(target.label);
    console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${status.hitRound ? ` @#${status.hitRound}` : ''} earlyRate=${earlyRate.toFixed(2)}`);
    return true;
  } catch(e) {
    console.error(`[${engineId}] save fail:`, e.message);
    state.savedSet.delete(key); // remove from set so retry is possible next tick
    return false;
  }
}

// ── processEngine ─────────────────────────────────────────────────────────────
async function processEngine({ engineId, state, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;
  for (const target of TARGETS) {
    const existing = state.lockedMap[target.label];
    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin: target.min, anchorRound: lastRoundId, generation: 1, stale: false };
        anyChange = true;
        console.log(`[${engineId}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW ?? '—'} rec=${pred.recommendation ?? '—'}`);
      }
      continue;
    }
    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow || anchorRound === 0;
    const isExpired   = lastRoundId >= absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit', 'miss', 'early'].includes(status.status)) {
          await saveOutcome(engineId, target, absLow, absHigh, status, existing, state);
        }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin: target.min, anchorRound: lastRoundId, generation: (existing.generation || 1) + (isNonsense ? 0 : 1), stale: false };
        console.log(`[${engineId}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% rec=${pred.recommendation ?? '—'}`);
      } else { delete state.lockedMap[target.label]; console.warn(`[${engineId}] ${target.label} cleared — insufficient data`); }
      anyChange = true; state.needsRebuild = false; continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit', 'miss', 'early'].includes(status.status)) {
      await saveOutcome(engineId, target, absLow, absHigh, status, existing, state);
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin: target.min, anchorRound: lastRoundId, generation: (existing.generation || 1) + 1, stale: false };
        console.log(`[${engineId}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% rec=${pred.recommendation ?? '—'}`);
      } else { delete state.lockedMap[target.label]; }
      anyChange = true;
    }
  }
  return anyChange;
}

// ── buildSavePayload ──────────────────────────────────────────────────────────
function buildSavePayload(lockedMap) {
  const out = {};
  for (const [label, pred] of Object.entries(lockedMap)) {
    if (pred.stale) continue;
    const anchor = Number(pred.anchorRound);
    if (!Number.isFinite(anchor) || anchor === 0) continue;
    out[label] = {
      lo: anchor + (Number(pred.low) || 0), hi: anchor + (Number(pred.high) || 0),
      roundWhenMade: anchor, generation: pred.generation || 1,
      eta: {
        low: pred.low, high: pred.high, conf: pred.confidence,
        probW: pred.probW, rawProbW: pred.rawProbW ?? pred.probW,
        expectedGap: pred.expectedGap, opensIn: pred.opensIn,
        streakStatus: pred.streakStatus, currentStreak: pred.currentStreak,
        spread: pred.spread ?? null, regime: pred.regime ?? null,
        regimeFactor: pred.regimeFactor ?? null, regimeConfidence: pred.regimeConfidence ?? null,
        ensWeights: pred.ensWeights ?? null, isRandom: pred.isRandom ?? null,
        maxAC: pred.maxAC ?? null, ev: pred.ev ?? null,
        signalQuality: pred.signalQuality ?? null, risk: pred.risk ?? null,
        recommendation: pred.recommendation ?? null, decisionReason: pred.decisionReason ?? null,
        signalStrength: pred.signalStrength ?? null, finalProbUsed: pred.finalProbUsed ?? null,
        amplifiedProbW: pred.amplifiedProbW ?? null,
        tailProbability: pred.tailProbability ?? null, extremeGapScore: pred.extremeGapScore ?? null,
        rareEV: pred.rareEV ?? null, earlyWindow: pred.earlyWindow ?? null,
        lateWindow: pred.lateWindow ?? null, primaryWindow: pred.primaryWindow ?? null,
        alpha: pred.alpha ?? null, rareSignal: pred.rareSignal ?? null, n: pred.n ?? null,
      },
    };
  }
  return out;
}

function loadLockedMap(dbRows) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = TARGETS.find(t => t.label === label);
    if (!target) continue;
    const eta    = pred.eta || {};
    const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
    const low    = eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor);
    const high   = eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor);
    if (high - low + 1 > target.maxWidth) {
      console.log(`[statEngine] DISCARD stale wide window ${label}: ${high - low + 1}r > max ${target.maxWidth}r`);
      continue;
    }
    map[label] = {
      low, high, confidence: eta.conf ?? 50, probW: eta.probW ?? null, rawProbW: eta.rawProbW ?? null,
      expectedGap: eta.expectedGap ?? null, opensIn: eta.opensIn ?? null,
      streakStatus: eta.streakStatus ?? 'normal', currentStreak: eta.currentStreak ?? 0,
      spread: eta.spread ?? null, regime: eta.regime ?? null,
      regimeFactor: eta.regimeFactor ?? null, regimeConfidence: eta.regimeConfidence ?? null,
      ensWeights: eta.ensWeights ?? null, isRandom: eta.isRandom ?? null, maxAC: eta.maxAC ?? null,
      ev: eta.ev ?? null, signalQuality: eta.signalQuality ?? null, risk: eta.risk ?? null,
      recommendation: eta.recommendation ?? null, decisionReason: eta.decisionReason ?? null,
      signalStrength: eta.signalStrength ?? null, finalProbUsed: eta.finalProbUsed ?? null,
      amplifiedProbW: eta.amplifiedProbW ?? null, tailProbability: eta.tailProbability ?? null,
      extremeGapScore: eta.extremeGapScore ?? null, rareEV: eta.rareEV ?? null,
      earlyWindow: eta.earlyWindow ?? null, lateWindow: eta.lateWindow ?? null,
      primaryWindow: eta.primaryWindow ?? null, alpha: eta.alpha ?? null, rareSignal: eta.rareSignal ?? null,
      n: eta.n ?? null,
      targetMin: target.min, anchorRound: anchor, generation: pred.generation ?? 1, stale: false,
    };
  }
  return map;
}

// ── In-memory rounds cache ────────────────────────────────────────────────────
async function getStatRounds() {
  if (cachedRounds.length === 0) {
    const all = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRounds       = all;
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[statEngine] loaded ${cachedRounds.length} rounds`);
  } else {
    // FIX: fetch up to 5000 new rounds per cycle (was 500)
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Initialise ────────────────────────────────────────────────────────────────
async function initialise() {
  if (initialised) return;
  initialised = true;
  for (const id of Object.keys(STATE)) STATE[id].savedSet = new Set();

  try { STATE.engine.lockedMap = loadLockedMap(await getLockedPreds()); console.log(`[statEngine] loaded ${Object.keys(STATE.engine.lockedMap).length} engine preds`); }
  catch(e) { console.error('[statEngine] init:', e.message); STATE.engine.lockedMap = {}; }

  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      STATE[model.id].lockedMap = loadLockedMap(dbStats[model.id] || {});
      console.log(`[statEngine] loaded ${Object.keys(STATE[model.id].lockedMap).length} ${model.id} preds`);
    }
  } catch(e) { console.error('[statEngine] stat init:', e.message); for (const model of STAT_MODELS) STATE[model.id].lockedMap = {}; }

  try {
    const rows = await getPredictions({ limit: 10000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      // Skip pattern engine history — that belongs to patternEngine.js
      if (src === 'pattern') continue;
      const key = makeKey(src, r.target, r.lo, r.hi);
      if (STATE[src]?.savedSet) STATE[src].savedSet.add(key);
      if (r.outcome === 'win' && r.hitRound && engineLastHit[src]?.[r.target] !== undefined) {
        if (engineLastHit[src][r.target] < r.hitRound) engineLastHit[src][r.target] = r.hitRound;
      }
      if (r.probW != null && ['win', 'loss', 'early'].includes(r.outcome) && STAT_MODELS.some(m => m.id === src)) {
        try { updateCalibration(r.target, src, r.probW, r.outcome); updateModelScore(r.target, src, r.probW, r.outcome); } catch(_) {}
      }
    }
    console.log(`[statEngine] loaded ${rows.length} history keys (calibration pre-warmed)`);
  } catch(e) { console.error('[statEngine] history:', e.message); }

  for (const id of Object.keys(STATE)) STATE[id].needsRebuild = true;
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function runStatEngine() {
  try {
    await initialise();
    const rounds = await getStatRounds();
    if (rounds.length < MIN_ROUNDS) { console.log(`[statEngine] waiting (${rounds.length}/${MIN_ROUNDS})`); return; }
    const lastRoundId = rounds[rounds.length - 1].roundId;

    const allEngines = [
      {
        id: 'engine', state: STATE.engine,
        buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth, t.rare, lastRoundId),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedPreds(p); },
      },
      ...STAT_MODELS.map(model => ({
        id: model.id, state: STATE[model.id],
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth, model.id, lastRoundId),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedStatPreds(model.id, p); },
      })),
    ];

    const t0 = Date.now();
    for (const eng of allEngines) {
      if (!(lastRoundId > eng.state.lastRoundId || eng.state.needsRebuild)) continue;
      eng.state.needsRebuild = false;
      const changed = await processEngine({ engineId: eng.id, state: eng.state, sortedRounds: rounds, lastRoundId, buildFn: eng.buildFn });
      eng.state.lastRoundId = lastRoundId;
      if (changed) {
        const p = buildSavePayload(eng.state.lockedMap);
        try { await eng.saveFn(p); } catch(e) { console.error(`[${eng.id}] save:`, e.message); }
      }
    }
    console.log(`[statEngine ${ENGINE_VERSION}] done in ${Date.now() - t0}ms`);
  } catch(e) {
    console.error('[statEngine] Fatal:', e.message, e.stack);
  }
}

// ── Exports ───────────────────────────────────────────────────────────────────
function getLockedStatMap(modelId) { return STATE[modelId]?.lockedMap || {}; }

function getValidationMetrics() {
  const out = {};
  for (const t of TARGETS) {
    out[t.label] = {};
    for (const m of STAT_MODELS) {
      const v = valMetrics[t.label][m.id], ece = getECE(t.label, m.id), total = v.count;
      const winRateOnTaken = v.takenTotal > 0 ? +(v.takenWins / v.takenTotal).toFixed(4) : null;
      out[t.label][m.id] = {
        brier: total > 0 ? +(v.brierSum / total).toFixed(4) : null,
        logLoss: total > 0 ? +(v.logLossSum / total).toFixed(4) : null,
        ece: ece != null ? +ece.toFixed(4) : null,
        wins: v.wins, losses: v.losses, early: v.earlyCount, total,
        hitRate: total > 0 ? +((v.wins / total) * 100).toFixed(1) : null,
        effectiveAccuracy: total > 0 ? +((v.wins + 0.5 * v.earlyCount) / total * 100).toFixed(1) : null,
        earlyRate: total > 0 ? +((v.earlyCount / total) * 100).toFixed(1) : null,
        tradeCount: v.tradeCount, totalWins: v.totalWins, winRateOnTaken,
        balanceScore: (v.totalWins > 0 && winRateOnTaken != null) ? +(v.totalWins * winRateOnTaken).toFixed(2) : null,
      };
    }
  }
  return out;
}

module.exports = {
  runStatEngine,
  resetStatEngineState,
  getLockedStatMap,
  getValidationMetrics,
};