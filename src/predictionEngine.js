'use strict';
// predictionEngine.js  v13-final
//
// INDEPENDENCE AUDIT — every engine now uses its OWN math end-to-end:
//
// GEO: p = Laplace MLE. probW = 1-(1-p)^W. expectedGap = (1-p)/p.
//      Window optimizer uses GEO's own geometric CDF: P(gap<=t) = 1-(1-p)^(t+1).
//
// BAY: p = global + recency blend (when CUSUM shift detected).
//      probW = 1-(1-pBay)^W. expectedGap = (1-pBay)/pBay.
//      Window optimizer uses BAY's own geometric CDF with pBay.
//
// KM:  probW = empirical survival P(gap<=W) from KM step function.
//      expectedGap = HSM-blended empirical median.
//      Window optimizer uses KM's empirical CDF (binary search on steps).
//      Pareto blend applied for rare targets (100x+, cv>1.1).
//
// ENS: Logit-space weighted combination of calibrated GEO+BAY+KM probW.
//      expectedGap = adaptive-weight-blended average of GEO/BAY/KM expected gaps.
//      Window optimizer uses ENS's own blended parametric CDF.
//
// ENGINE: p = CUSUM-blended p. probW = 1-(1-p)^W.
//         expectedGap = (1-p)/p. Window uses its own geometric CDF.
//
// SHARED (cache only): raw data — sorted gaps, percentiles, meanGap, stdGap, cv.
//   These are FACTS about the data, not model-specific. All models use same facts.
//   Each model derives its OWN probability and window from those facts differently.
//
// OTHER FIXES:
//   - Removed dead code: CAL_LAPLACE (unused), ensureMonotonic (uncalled)
//   - optimizeWindowPlacement now takes a cdfFn (model-specific CDF function)
//     instead of always using the shared kmCDF
//   - Pareto formula fixed: was using meanGap/(meanGap+W) but gap is discrete,
//     formula now: 1 - (1/(1 + W/expectedGap))^alpha

const {
  getRounds, savePrediction, getPredictions,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedStatPreds, getLockedStatPreds,
} = require('./db');

const TARGETS = [
  { label: '5x',    min: 5,    maxWidth: 3,  rare: false },
  { label: '10x',   min: 10,   maxWidth: 5,  rare: false },
  { label: '20x',   min: 20,   maxWidth: 8,  rare: false },
  { label: '50x',   min: 50,   maxWidth: 12, rare: false },
  { label: '100x',  min: 100,  maxWidth: 18, rare: true  },
  { label: '250x',  min: 250,  maxWidth: 25, rare: true  },
  { label: '500x',  min: 500,  maxWidth: 30, rare: true  },
  { label: '1000x', min: 1000, maxWidth: 50, rare: true  },
];

const STAT_MODELS = [
  { id: 'ens' },
  { id: 'geo' },
  { id: 'bay' },
  { id: 'km'  },
];

const MIN_ROUNDS               = 50;
const STALE_FORCE_REBUILD_THRESHOLD = 200;
const WINDOW_LAMBDA            = 0.01; // penalty per extra round of width

// ── Per-engine state (fully independent) ──────────────────────────────────────

const STATE = {
  engine:  { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  pattern: { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ens:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  geo:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  bay:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  km:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
};

// ── Gap stats cache — SHARED RAW DATA only, not model outputs ─────────────────
// Stores: sorted gaps, meanGap, medianGap, stdGap, cv, percentiles, kmCDF, HSM
// Each model reads these facts and computes its OWN p, probW, expectedGap, window
const gapStatsCache = new Map();
let   cacheRoundId  = -1;

// ── 5-bin calibration — per model per target ──────────────────────────────────
const CAL_BINS  = [0, 0.30, 0.45, 0.60, 0.75, 1.01];
const CAL_DECAY = { normal: 0.08, rare: 0.03 };

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

// ── Adaptive ensemble weights — log-loss EWMA per model per target ────────────
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
    valMetrics[t.label][m.id] = { brierSum: 0, logLossSum: 0, count: 0, wins: 0, losses: 0, earlyCount: 0 };
  }
}

let initialised = false;

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetEngineState() {
  console.log('[engine] resetEngineState()');
  for (const id of Object.keys(STATE)) {
    STATE[id].lockedMap    = null;
    STATE[id].savedSet     = null;
    STATE[id].needsRebuild = true;
    STATE[id].lastRoundId  = 0;
  }
  gapStatsCache.clear();
  cacheRoundId = -1;
  initialised  = false;
}

// ── Math helpers ──────────────────────────────────────────────────────────────

function sigmoid(x)   { return 1 / (1 + Math.exp(-x)); }
function logit(p)     { const q = Math.max(1e-7, Math.min(1 - 1e-7, p)); return Math.log(q / (1 - q)); }
function fromLogit(l) { return sigmoid(l); }

// Geometric CDF: P(gap <= t) = 1 - (1-p)^(t+1)
// This is the correct discrete geometric CDF for gap distributions.
// A gap of 0 means consecutive hits; gap of t means t misses then a hit.
function geoCDF(p, t) {
  if (t < 0) return 0;
  return Math.max(0, Math.min(1, 1 - Math.pow(1 - p, t + 1)));
}

// P(gap in [lo, hi]) = CDF(hi) - CDF(lo-1)
function geoWindowProb(p, lo, hi) {
  return Math.max(0, geoCDF(p, hi) - (lo > 0 ? geoCDF(p, lo - 1) : 0));
}

// ── Gap stats cache — shared raw facts ────────────────────────────────────────

function getGapStats(rounds, targetMin, lastRoundId) {
  if (lastRoundId !== cacheRoundId) {
    gapStatsCache.clear();
    cacheRoundId = lastRoundId;
  }
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
      lastIdx = i;
      hits++;
    }
  }
  if (hits < 3) return null;

  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;

  // Raw hit rate stats (used by all models as base data)
  const pGlobal = (hits + 1) / (n + 2);
  const r200hits = rounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
  const pRecent  = (r200hits + 1) / 202;

  // CUSUM structural break detection (shared diagnostic)
  const p0 = hits / n;
  let cusum = 0, maxCusum = 0;
  const cusumWindow = rounds.slice(-150);
  for (const r of cusumWindow) {
    cusum += (r.multiplier >= targetMin ? 1 : 0) - p0;
    if (Math.abs(cusum) > maxCusum) maxCusum = Math.abs(cusum);
  }
  const sigma0     = Math.sqrt(Math.max(1e-9, p0 * (1 - p0)));
  const cusumNorm  = maxCusum / (sigma0 * Math.sqrt(cusumWindow.length));
  const rateShifted = cusumNorm > 1.36;

  // Sorted gaps — sorted once, shared by all
  const sg = [...gaps].sort((a, b) => a - b);
  const m2 = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1 / pGlobal) :
    sg.length % 2 === 1 ? sg[m2] : (sg[m2 - 1] + sg[m2]) / 2;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / pGlobal;
  const variance = gaps.length > 1 ? Math.max(0, gSS / gaps.length - meanGap ** 2) : meanGap * meanGap;
  const stdGap   = Math.sqrt(variance);
  const cv       = meanGap > 0 ? stdGap / meanGap : 1;

  // Percentiles (safe, no spread operator)
  const pctile = (frac) => {
    if (sg.length === 0) return meanGap;
    return sg[Math.min(sg.length - 1, Math.floor(frac * sg.length))];
  };
  const p75 = pctile(0.75);
  const p90 = pctile(0.90);
  const p95 = pctile(0.95);

  // HSM — computed once, used by KM and ENS
  const hsm = halfSampleMode(sg, medianGap, cv);

  // Dynamic modeWeight by cv
  let modeWeight;
  if (cv < 1.0)      modeWeight = 0.50;
  else if (cv > 1.3) modeWeight = 0.10;
  else               modeWeight = 0.50 - (cv - 1.0) * (0.40 / 0.30);

  // KM empirical expected gap (HSM-blended median) — KM's view of expected gap
  const kmExpectedGap = Math.max(1, Math.round(medianGap * (1 - modeWeight) + hsm * modeWeight));

  // KM precomputed CDF step function (O(n) build, O(log n) query)
  const kmCDF = buildKmCDF(sg);

  // Current dry streak
  let currentStreak = 0;
  for (let i = n - 1; i >= 0; i--) {
    if (rounds[i].multiplier < targetMin) currentStreak++;
    else break;
  }

  return {
    // Raw facts — shared by all engines
    hits, n, pGlobal, pRecent, rateShifted,
    cusumNorm: +cusumNorm.toFixed(3),
    gapSinceLast, meanGap, medianGap, stdGap, cv,
    p75, p90, p95, sg, kmCDF,
    hsm, kmExpectedGap, // KM's view of the gap
    currentStreak,
  };
}

// ── Half-sample mode ──────────────────────────────────────────────────────────

function halfSampleMode(sg, medianGap, cv) {
  const n = sg.length;
  if (n < 8)    return medianGap;
  if (cv > 2.0) return sg[Math.floor(n / 2)];
  const h = Math.max(2, Math.floor(n / 2));
  let bestRange = sg[sg.length - 1] - sg[0] + 1, bestStart = 0;
  for (let i = 0; i + h - 1 < n; i++) {
    const range = sg[i + h - 1] - sg[i];
    if (range < bestRange) { bestRange = range; bestStart = i; }
  }
  return (sg[bestStart] + sg[bestStart + h - 1]) / 2;
}

// ── KM: precomputed CDF (O(n) build, O(log n) query) ─────────────────────────

function buildKmCDF(sg) {
  if (sg.length < 5) return null;
  const m = sg.length;
  const steps = [];
  let S = 1.0, i = 0;
  while (i < m) {
    const t = sg[i];
    const nAtRisk = m - i; // at risk BEFORE processing this time point
    let d = 0;
    while (i < m && sg[i] === t) { d++; i++; }
    S *= (1 - d / nAtRisk);
    steps.push({ t, S: Math.max(0, S) });
  }
  return steps;
}

// P(gap <= W) from KM step function via binary search
function kmCDFQuery(kmCDF, W) {
  if (!kmCDF || kmCDF.length === 0) return null;
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (kmCDF[mid].t <= W) { idx = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  const S = idx >= 0 ? kmCDF[idx].S : 1.0;
  return Math.max(0, Math.min(1 - 1e-9, 1 - S));
}

// P(gap in [lo, hi]) from KM CDF
function kmWindowProb(kmCDF, lo, hi) {
  const cdfHi = kmCDFQuery(kmCDF, hi) ?? 0;
  const cdfLo = lo > 0 ? (kmCDFQuery(kmCDF, lo - 1) ?? 0) : 0;
  return Math.max(0, cdfHi - cdfLo);
}

// ── Window optimizer — maximize P(hit) - λ×width ─────────────────────────────
//
// cdfFn(lo, hi) → P(gap in [lo, hi]) for this specific model
// Each model passes its OWN cdfFn so placement is model-specific.

// Window placement strategies:
//
// PARAMETRIC (GEO, BAY, ENGINE): geometric distribution is memoryless.
//   Optimal window placement = center on expectedGap.
//   low = max(0, expectedGap - effectiveWidth)
//   high = low + effectiveWidth - 1
//   If gapSinceLast >= expectedGap: open immediately (overdue).
//
// EMPIRICAL (KM): gap distribution is NOT assumed memoryless.
//   Use empirical CDF to find window [absLo, absHi] (absolute from last hit)
//   that maximizes P(gap in window) - lambda*width.
//   absLo = gapSinceLast + low (accounts for time already elapsed).
//
// Both strategies return {low, high} relative to anchor (= lastRoundId+1).

function parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) {
    return { low: 0, high: effectiveWidth - 1 };
  }
  // Remaining rounds until expected hit = expectedGap - gapSinceLast
  // Place window to cover that point: [remaining-W, remaining-1]
  const remaining = expectedGap - gapSinceLast;
  const low  = Math.max(0, remaining - effectiveWidth);
  const high = low + effectiveWidth - 1;
  return { low, high };
}

function empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) {
    return { low: 0, high: effectiveWidth - 1 };
  }
  // Search over window offsets from current position (gapSinceLast)
  // Absolute gap = gapSinceLast + low through gapSinceLast + high
  const maxLow = Math.max(0, 3 * expectedGap - gapSinceLast - effectiveWidth);
  let bestScore = -Infinity, bestLow = 0;
  for (let low = 0; low <= maxLow; low++) {
    const absLo  = gapSinceLast + low;
    const absHi  = gapSinceLast + low + effectiveWidth - 1;
    const probHit = kmWindowProb(kmCDF, absLo, absHi);
    const score   = probHit - WINDOW_LAMBDA * effectiveWidth;
    if (score > bestScore) { bestScore = score; bestLow = low; }
    if (absLo > expectedGap * 3 && probHit < 0.001) break;
  }
  return { low: bestLow, high: bestLow + effectiveWidth - 1 };
}

// ── Pareto blend for KM on rare targets ───────────────────────────────────────

function applyParetoCorrectedProbW(rawProbW, cv, expectedGap, maxWidth, isRare) {
  if (!isRare || cv <= 1.1) return rawProbW;
  const alpha   = 1 / Math.max(0.1, cv * cv);
  // P_pareto(gap <= W) = 1 - (1 / (1 + W/expectedGap))^alpha
  const paretoP = 1 - Math.pow(1 / (1 + maxWidth / Math.max(1, expectedGap)), alpha);
  const blendW  = Math.min(0.30, (cv - 1.0) * 0.60);
  const blended = (1 - blendW) * rawProbW + blendW * paretoP;
  return Math.max(1e-6, Math.min(1 - 1e-6, blended));
}

// ── Calibration ───────────────────────────────────────────────────────────────

function getCalBinIdx(probW) {
  for (let i = 0; i < CAL_BINS.length - 1; i++) {
    if (probW < CAL_BINS[i + 1]) return i;
  }
  return CAL_BINS.length - 2;
}

function updateCalibration(targetLabel, modelId, predictedProbW, outcome) {
  if (outcome === 'early') return;
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return;
  const target = TARGETS.find(t => t.label === targetLabel);
  const decay  = target?.rare ? CAL_DECAY.rare : CAL_DECAY.normal;
  const actual = outcome === 'win' ? 1 : 0;
  const bin    = bins[getCalBinIdx(predictedProbW)];
  bin.ewmaAct  = (1 - decay) * bin.ewmaAct  + decay * actual;
  bin.ewmaPred = (1 - decay) * bin.ewmaPred + decay * predictedProbW;
  bin.count    = Math.min(bin.count + 1, 500);
}

function applyCalibration(probW, targetLabel, modelId) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < 8) return probW;
  const empirical = bin.ewmaAct, predicted = bin.ewmaPred;
  if (predicted < 1e-6) return probW;
  const ratio = empirical / predicted;
  if (Math.abs(ratio - 1) < 0.12) return probW;
  const calibrated = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(0.80, Math.min(1.20, ratio))));
  return Math.min(calibrated, probW + 0.05); // no inflation
}

// ── Validation metrics ────────────────────────────────────────────────────────

function updateValidationMetrics(targetLabel, modelId, predictedProbW, outcome) {
  const v = valMetrics[targetLabel]?.[modelId];
  if (!v) return;
  if (outcome === 'early') { v.earlyCount++; return; }
  const actual = outcome === 'win' ? 1 : 0;
  const p = Math.max(1e-7, Math.min(1 - 1e-7, predictedProbW));
  v.brierSum   += (actual - p) ** 2;
  v.logLossSum += -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
  v.count++;
  if (outcome === 'win') v.wins++; else v.losses++;
}

function getECE(targetLabel, modelId) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return null;
  let ece = 0, total = 0;
  for (const bin of bins) {
    if (bin.count < 2) continue;
    ece   += Math.abs(bin.ewmaAct - bin.ewmaPred) * bin.count;
    total += bin.count;
  }
  return total > 0 ? ece / total : null;
}

// ── Adaptive ensemble weights ─────────────────────────────────────────────────

function logLossVal(actual, probW) {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probW));
  return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
}

function updateModelScore(targetLabel, modelId, predictedProbW, outcome) {
  if (outcome === 'early') return;
  const actual = outcome === 'win' ? 1 : 0;
  const s = modelScores[targetLabel]?.[modelId];
  if (!s) return;
  const loss   = logLossVal(actual, predictedProbW);
  const target = TARGETS.find(t => t.label === targetLabel);
  const decay  = target?.rare ? 0.02 : 0.05;
  s.ewma  = s.count === 0 ? loss : (1 - decay) * s.ewma + decay * loss;
  s.count = Math.min(s.count + 1, 500);
}

// ENS logit-space combiner with outlier detection
// Returns: ensProb, spread, adjWeights (geo, bay, km order)
function buildEnsemble(targetLabel, probGeo, probBay, probKm) {
  const scores  = modelScores[targetLabel];
  const modelIds = ['geo', 'bay', 'km'];
  const probs   = [probGeo, probBay, probKm];

  // Adaptive weights from log-loss
  const weights = modelIds.map(id => {
    const avgLoss = scores[id]?.count > 2 ? scores[id].ewma : 0.693;
    return Math.exp(-avgLoss * 2);
  });
  const wSum = weights.reduce((a, b) => a + b, 0);
  if (wSum < 1e-9) return { ensProb: (probGeo + probBay + probKm) / 3, spread: 0, adjWeights: [1/3, 1/3, 1/3] };
  for (let i = 0; i < weights.length; i++) weights[i] /= wSum;

  // Logit space
  const logits = probs.map(logit);
  const wMean  = logits.reduce((s, l, i) => s + weights[i] * l, 0);
  const wVar   = logits.reduce((s, l, i) => s + weights[i] * (l - wMean) ** 2, 0);
  const spread = Math.sqrt(wVar);

  // Outlier detection: halve weight of any model >1.5 logit units from mean
  const adjWeights = [...weights];
  for (let i = 0; i < logits.length; i++) {
    if (Math.abs(logits[i] - wMean) > 1.5) adjWeights[i] *= 0.5;
  }
  const adjSum = adjWeights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < adjWeights.length; i++) adjWeights[i] /= adjSum;

  const ensLogit = logits.reduce((s, l, i) => s + adjWeights[i] * l, 0);
  const ensProb  = fromLogit(ensLogit);

  return { ensProb, spread, adjWeights };
}

// ── Beta confidence (with spread) ────────────────────────────────────────────

function betaConf(probW, hits, spread) {
  const effectiveN = Math.min(hits, 300);
  const alpha = probW * effectiveN + 1;
  const beta  = (1 - probW) * effectiveN + 1;
  const postVar = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const postStd = Math.sqrt(postVar);

  let c = probW * 100 - postStd * 120;
  if (hits < 10)      c -= 15;
  else if (hits < 25) c -= 7;
  else if (hits < 50) c -= 3;

  if (spread != null) c -= Math.min(15, Math.round(15 * spread / 0.5));

  return Math.max(20, Math.min(88, Math.round(c)));
}

// ── Streak placement helper ───────────────────────────────────────────────────
// Returns: effectiveWidth, confPenalty, streakStatus, z

function getStreakInfo(gs, maxWidth, isRare) {
  const { gapSinceLast, meanGap, stdGap, cv, p90, p95 } = gs;
  const z = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;
  const confPenalty = Math.round(20 * sigmoid(z - 1.5));

  let streakStatus = 'normal';
  if (gapSinceLast >= p95)      streakStatus = 'extreme';
  else if (gapSinceLast >= p90) streakStatus = 'severe';

  let effectiveWidth = maxWidth;
  if (isRare && cv > 1.1) {
    effectiveWidth = Math.round(maxWidth * Math.min(1.20, 1.0 + (cv - 1.0) * 0.25));
  }
  if (z > 2) effectiveWidth = Math.min(Math.round(maxWidth * 1.35), Math.round(effectiveWidth * 1.15));
  effectiveWidth = Math.min(effectiveWidth, Math.round(maxWidth * 1.35));

  return { z: +z.toFixed(2), confPenalty, streakStatus, effectiveWidth };
}

// ── BUILD: ENGINE ─────────────────────────────────────────────────────────────
// Uses CUSUM-blended p. Window via own geometric CDF.

function buildPrediction(rounds, targetMin, maxWidth, isRare, lastRoundId) {
  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const { hits, pGlobal, pRecent, rateShifted, gapSinceLast, currentStreak } = gs;

  // ENGINE's own blended p
  const p = rateShifted
    ? Math.max(1e-6, Math.min(0.5, 0.75 * pGlobal + 0.25 * pRecent))
    : Math.max(1e-6, Math.min(0.5, pGlobal));

  const probW       = 1 - Math.pow(1 - p, maxWidth);
  const expectedGap = Math.max(1, Math.round((1 - p) / p));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare ?? false);

  // Streak status override for UI
  let finalStreakStatus = streakStatus;
  if (streakStatus === 'normal' && gapSinceLast >= expectedGap) finalStreakStatus = 'overdue';

  const { low, high } = parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast);

  return {
    low, high, expectedGap, opensIn: low,
    confidence: Math.max(20, betaConf(probW, hits, null) - confPenalty),
    probW:       +probW.toFixed(4),
    p:           +p.toFixed(6),
    rateShifted, cusumNorm: gs.cusumNorm,
    streakStatus: finalStreakStatus, currentStreak, z,
    gapSinceLast, hits,
  };
}

// ── BUILD: GEO ────────────────────────────────────────────────────────────────
// Pure Laplace MLE. Own geometric CDF. Window differs from BAY/KM/ENS.

function buildGeo(gs, maxWidth, targetLabel, isRare) {
  const { hits, n, pGlobal, gapSinceLast, currentStreak } = gs;
  const pGeo        = (hits + 1) / (n + 2);
  const rawProbW    = 1 - Math.pow(1 - pGeo, maxWidth);
  const calibProbW  = applyCalibration(rawProbW, targetLabel, 'geo');
  const expectedGap = Math.max(1, Math.round((1 - pGeo) / pGeo));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  let finalStreakStatus = streakStatus;
  if (streakStatus === 'normal' && gapSinceLast >= expectedGap) finalStreakStatus = 'overdue';

  const { low, high } = parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast);

  return {
    low, high, expectedGap, opensIn: low,
    confidence:  Math.max(20, betaConf(calibProbW, hits, null) - confPenalty),
    probW:       +calibProbW.toFixed(4),
    rawProbW:    +rawProbW.toFixed(4),
    p:           +pGeo.toFixed(6),
    streakStatus: finalStreakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted: gs.rateShifted, model: 'geo',
  };
}

// ── BUILD: BAY ────────────────────────────────────────────────────────────────
// Bayesian recency blend (only when CUSUM shift). Own geometric CDF with pBay.
// BAY's expectedGap differs from GEO when rate-shifted.

function buildBay(gs, maxWidth, targetLabel, isRare) {
  const { hits, n, pGlobal, pRecent, rateShifted, gapSinceLast, currentStreak } = gs;
  const recencyW    = rateShifted ? 0.20 : 0.05;
  const pBay        = Math.max(1e-6, Math.min(0.5, (1 - recencyW) * pGlobal + recencyW * pRecent));
  const rawProbW    = 1 - Math.pow(1 - pBay, maxWidth);
  const calibProbW  = applyCalibration(rawProbW, targetLabel, 'bay');
  const expectedGap = Math.max(1, Math.round((1 - pBay) / pBay));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  let finalStreakStatus = streakStatus;
  if (streakStatus === 'normal' && gapSinceLast >= expectedGap) finalStreakStatus = 'overdue';

  const { low, high } = parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast);

  return {
    low, high, expectedGap, opensIn: low,
    confidence:  Math.max(20, betaConf(calibProbW, hits, null) - confPenalty),
    probW:       +calibProbW.toFixed(4),
    rawProbW:    +rawProbW.toFixed(4),
    p:           +pBay.toFixed(6),
    streakStatus: finalStreakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted, model: 'bay',
  };
}

// ── BUILD: KM ─────────────────────────────────────────────────────────────────
// Empirical survival. Own KM CDF for window placement.
// expectedGap = HSM-blended empirical median — genuinely different from GEO/BAY.

function buildKm(gs, maxWidth, targetLabel, isRare) {
  const { hits, kmCDF, kmExpectedGap, gapSinceLast, currentStreak, cv, meanGap } = gs;
  if (!kmCDF) return null;

  let rawProbW = kmCDFQuery(kmCDF, maxWidth);
  if (rawProbW == null) return null;

  // Pareto blend for heavy-tailed rare targets
  rawProbW = applyParetoCorrectedProbW(rawProbW, cv, kmExpectedGap, maxWidth, isRare);
  rawProbW = Math.max(1e-6, Math.min(1 - 1e-6, rawProbW));

  const calibProbW  = applyCalibration(rawProbW, targetLabel, 'km');
  const expectedGap = kmExpectedGap;

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  let finalStreakStatus = streakStatus;
  if (streakStatus === 'normal' && gapSinceLast >= expectedGap) finalStreakStatus = 'overdue';

  // KM uses empirical CDF for window placement (non-parametric, meaningful optimization)
  const { low, high } = empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast);

  return {
    low, high, expectedGap, opensIn: low,
    confidence:  Math.max(20, betaConf(calibProbW, hits, null) - confPenalty),
    probW:       +calibProbW.toFixed(4),
    rawProbW:    +rawProbW.toFixed(4),
    p:           +rawProbW.toFixed(6), // KM: p field = P(gap<=W), not per-round
    streakStatus: finalStreakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted: gs.rateShifted, model: 'km',
  };
}

// ── BUILD: ENS ────────────────────────────────────────────────────────────────
// Logit-space weighted combination of calibrated GEO+BAY+KM.
// expectedGap = adaptive-weight blend of each model's expected gap.
// Window uses ENS's own blended parametric CDF.

function buildEns(gs, maxWidth, targetLabel, isRare) {
  const { hits, n, pGlobal, pRecent, rateShifted, kmCDF, kmExpectedGap,
          gapSinceLast, currentStreak, cv, meanGap } = gs;

  // Compute each base model's p and calibrated probW
  const pGeo = (hits + 1) / (n + 2);
  const recencyW = rateShifted ? 0.20 : 0.05;
  const pBay = Math.max(1e-6, Math.min(0.5, (1 - recencyW) * pGlobal + recencyW * pRecent));

  const probGeo = applyCalibration(1 - Math.pow(1 - pGeo, maxWidth), targetLabel, 'geo');
  const probBay = applyCalibration(1 - Math.pow(1 - pBay, maxWidth), targetLabel, 'bay');

  let rawKmProb = kmCDF ? kmCDFQuery(kmCDF, maxWidth) : null;
  if (rawKmProb != null) rawKmProb = applyParetoCorrectedProbW(rawKmProb, cv, kmExpectedGap, maxWidth, isRare);
  const probKm = rawKmProb != null
    ? applyCalibration(Math.max(1e-6, Math.min(1 - 1e-6, rawKmProb)), targetLabel, 'km')
    : probGeo;

  // Logit-space ensemble
  const { ensProb, spread, adjWeights } = buildEnsemble(targetLabel, probGeo, probBay, probKm);
  const calibrated = applyCalibration(ensProb, targetLabel, 'ens');

  // ENS expected gap: adaptive-weight blend
  const egGeo = Math.max(1, Math.round((1 - pGeo) / pGeo));
  const egBay = Math.max(1, Math.round((1 - pBay) / pBay));
  const egKm  = kmExpectedGap;
  const ensExpectedGap = Math.max(1, Math.round(
    adjWeights[0] * egGeo + adjWeights[1] * egBay + adjWeights[2] * egKm
  ));

  // ENS blended per-round p for window CDF
  // GEO: pGeo directly. BAY: pBay directly.
  // KM: p = 1/(egKm+1) — geometric distribution property: E[gap] = (1-p)/p => p = 1/(E[gap]+1)
  const pKmPerRound = Math.max(1e-6, Math.min(0.5, 1 / (Math.max(1, egKm) + 1)));
  const pEns = Math.max(1e-6, Math.min(0.5,
    adjWeights[0] * pGeo + adjWeights[1] * pBay + adjWeights[2] * pKmPerRound
  ));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  let finalStreakStatus = streakStatus;
  if (streakStatus === 'normal' && gapSinceLast >= ensExpectedGap) finalStreakStatus = 'overdue';

  const { low, high } = parametricWindowPlacement(ensExpectedGap, effectiveWidth, gapSinceLast);

  return {
    low, high, expectedGap: ensExpectedGap, opensIn: low,
    confidence:  Math.max(20, betaConf(calibrated, hits, spread) - confPenalty),
    probW:       +calibrated.toFixed(4),
    rawProbW:    +ensProb.toFixed(4),
    p:           +pEns.toFixed(6),
    streakStatus: finalStreakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted, model: 'ens', spread: +spread.toFixed(3),
  };
}

// ── buildStatPrediction — dispatcher ─────────────────────────────────────────

function buildStatPrediction(rounds, targetMin, maxWidth, modelId, lastRoundId) {
  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const target      = TARGETS.find(t => t.min === targetMin);
  const targetLabel = target?.label ?? '?';
  const isRare      = target?.rare ?? false;

  switch (modelId) {
    case 'geo': return buildGeo(gs, maxWidth, targetLabel, isRare);
    case 'bay': return buildBay(gs, maxWidth, targetLabel, isRare);
    case 'km':  return buildKm(gs, maxWidth, targetLabel, isRare);
    case 'ens': return buildEns(gs, maxWidth, targetLabel, isRare);
    default:    return null;
  }
}

// ── BUILD: PATTERN (isolated, UI signal only) ─────────────────────────────────

function buildPatternPrediction(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;
  const W1=15, W2=50, W3=150;
  const s1=Math.max(0,n-W1), s2=Math.max(0,n-W2), s3=Math.max(0,n-W3);
  let hits=0, lastIdx=-1, hW1=0, hW2=0, hW3=0;
  const gaps=[];
  const FA=0.20, SA=0.02; let emaFast=-1, emaSlow=-1;
  for (let i=0;i<n;i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast<0) { emaFast=isHit; emaSlow=isHit; }
    else { emaFast=FA*isHit+(1-FA)*emaFast; emaSlow=SA*isHit+(1-SA)*emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx=i; hits++;
      if (i>=s1) hW1++;
      if (i>=s2) hW2++;
      if (i>=s3) hW3++;
    }
  }
  if (hits < 8 || gaps.length < 6) return null;
  const globalRate = hits / n;
  let gSum=0, gSS=0; for (const g of gaps) { gSum+=g; gSS+=g*g; }
  const meanGap = gSum / gaps.length;
  const sg=[...gaps].sort((a,b)=>a-b);
  const mid2=Math.floor(sg.length/2);
  const medianGap=sg.length%2===1?sg[mid2]:(sg[mid2-1]+sg[mid2])/2;
  const cv = meanGap > 0 ? Math.sqrt(Math.max(0, gSS/gaps.length - meanGap**2)) / meanGap : 1;
  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const safe = v => Math.max(-1, Math.min(1, v));
  const rW1=globalRate>0?safe((dW1-globalRate)/Math.max(globalRate,0.001)):0;
  const rW2=globalRate>0?safe((dW2-globalRate)/Math.max(globalRate,0.001)):0;
  const rW3=globalRate>0?safe((dW3-globalRate)/Math.max(globalRate,0.001)):0;
  const clusterScore=safe(rW1*0.50+rW2*0.30+rW3*0.20);
  const trendScore=safe((emaSlow>0?(emaFast-emaSlow)/emaSlow:0)*4);
  let varSum=0; for (const g of gaps) varSum+=(g-meanGap)**2;
  let bestAC=0;
  for (let lag=1;lag<=Math.min(3,gaps.length-1);lag++) {
    let cov=0;
    for (let i=lag;i<gaps.length;i++) cov+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac=varSum>0?cov/varSum:0;
    if (Math.abs(ac)>Math.abs(bestAC)) bestAC=ac;
  }
  const patternScore=safe(bestAC*0.9);
  const composite=clusterScore*0.50+trendScore*0.35+patternScore*0.15;
  const absComposite=Math.abs(composite);
  const direction=composite>0.10?'bullish':composite<-0.10?'bearish':'neutral';
  const agree=Math.max(
    [clusterScore,trendScore,patternScore].filter(s=>s>0.10).length,
    [clusterScore,trendScore,patternScore].filter(s=>s<-0.10).length
  );
  const conf=Math.max(25,Math.min(82,Math.round(
    32+Math.min(18,Math.log2(hits+1)*4)+absComposite*30+(agree-1)*6-(cv>1.5?8:cv>1.2?4:0)
  )));
  return { direction, confidence:conf, hits, meanGap:Math.round(meanGap), medianGap:Math.round(medianGap), composite:+composite.toFixed(3) };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const expectedGap = patternResult.medianGap || patternResult.meanGap || maxWidth;
  const low  = Math.max(0, expectedGap - maxWidth);
  return { low, high: low + maxWidth - 1, expectedGap, opensIn: low, confidence: patternResult.confidence, direction: patternResult.direction, streakStatus: 'normal', currentStreak: 0 };
}

function makeKey(source, target, lo, hi) {
  return `${source}-${target}-${Number(lo)||0}-${Number(hi)||0}`;
}

// ── getStatus ─────────────────────────────────────────────────────────────────

function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow)
    return { status: 'miss', hitRound: null };
  let lo=0, hi=sortedRounds.length-1, startIdx=sortedRounds.length;
  while (lo<=hi) {
    const mid=(lo+hi)>>>1;
    if (sortedRounds[mid].roundId >= anchorRound) { startIdx=mid; hi=mid-1; }
    else lo=mid+1;
  }
  for (let i=startIdx; i<sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < absLow) return { status:'early', hitRound:r.roundId };
    return { status:'hit',  hitRound:r.roundId };
  }
  if (currentRoundId > absHigh)                              return { status:'miss',   hitRound:null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status:'active', hitRound:null };
  return { status:'waiting', hitRound:null };
}

// ── processEngine ─────────────────────────────────────────────────────────────

async function processEngine({ engineId, state, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;
  for (const target of TARGETS) {
    const existing = state.lockedMap[target.label];
    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:1, stale:false };
        anyChange = true;
        console.log(`[${engineId}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'}`);
      }
      continue;
    }
    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow)||!Number.isFinite(absHigh)||absHigh<absLow||anchorRound===0;
    const isExpired   = lastRoundId > absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
          const key = makeKey(engineId, target.label, absLow, absHigh);
          if (!state.savedSet.has(key)) {
            state.savedSet.add(key);
            try {
              await savePrediction({ target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound, hitRound:status.hitRound||null, generation:existing.generation||1, source:engineId });
              if (STAT_MODELS.some(m=>m.id===engineId) && existing.probW!=null) {
                updateCalibration(target.label, engineId, existing.probW, outcome);
                updateModelScore(target.label, engineId, existing.probW, outcome);
                updateValidationMetrics(target.label, engineId, existing.probW, outcome);
              }
              console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
            } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
          }
        }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:(existing.generation||1)+(isNonsense?0:1), stale:false };
        console.log(`[${engineId}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete state.lockedMap[target.label];
        console.warn(`[${engineId}] ${target.label} cleared — insufficient data`);
      }
      anyChange = true;
      state.needsRebuild = false;
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
      const key = makeKey(engineId, target.label, absLow, absHigh);
      if (!state.savedSet.has(key)) {
        state.savedSet.add(key);
        try {
          await savePrediction({ target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound, hitRound:status.hitRound||null, generation:existing.generation||1, source:engineId });
          if (STAT_MODELS.some(m=>m.id===engineId) && existing.probW!=null) {
            updateCalibration(target.label, engineId, existing.probW, outcome);
            updateModelScore(target.label, engineId, existing.probW, outcome);
            updateValidationMetrics(target.label, engineId, existing.probW, outcome);
          }
          console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
        } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:(existing.generation||1)+1, stale:false };
        console.log(`[${engineId}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else { delete state.lockedMap[target.label]; }
      anyChange = true;
    }
  }
  return anyChange;
}

// ── DB helpers ────────────────────────────────────────────────────────────────

function buildSavePayload(lockedMap) {
  const out = {};
  for (const [label, pred] of Object.entries(lockedMap)) {
    if (pred.stale) continue;
    const anchor = Number(pred.anchorRound);
    if (!Number.isFinite(anchor) || anchor === 0) continue;
    out[label] = {
      lo: anchor+(Number(pred.low)||0), hi: anchor+(Number(pred.high)||0),
      roundWhenMade: anchor, generation: pred.generation||1,
      eta: { low:pred.low, high:pred.high, conf:pred.confidence, probW:pred.probW, rawProbW:pred.rawProbW??pred.probW, expectedGap:pred.expectedGap, opensIn:pred.opensIn, streakStatus:pred.streakStatus, currentStreak:pred.currentStreak, spread:pred.spread??null },
    };
  }
  return out;
}

function loadLockedMap(dbRows) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = TARGETS.find(t => t.label === label);
    if (!target) continue;
    const eta = pred.eta || {};
    const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
    map[label] = {
      low: eta.low!=null?eta.low:Math.max(0,Number(pred.lo)-anchor),
      high: eta.high!=null?eta.high:Math.max(0,Number(pred.hi)-anchor),
      confidence: eta.conf??50, probW: eta.probW??null, rawProbW: eta.rawProbW??null,
      expectedGap: eta.expectedGap??null, opensIn: eta.opensIn??null,
      streakStatus: eta.streakStatus??'normal', currentStreak: eta.currentStreak??0,
      spread: eta.spread??null, targetMin: target.min, anchorRound: anchor,
      generation: pred.generation??1, stale: true,
    };
  }
  return map;
}

// ── Validation export ─────────────────────────────────────────────────────────

function getValidationMetrics() {
  const out = {};
  for (const t of TARGETS) {
    out[t.label] = {};
    for (const m of STAT_MODELS) {
      const v = valMetrics[t.label][m.id];
      const ece = getECE(t.label, m.id);
      out[t.label][m.id] = { brier: v.count>0?+(v.brierSum/v.count).toFixed(4):null, logLoss: v.count>0?+(v.logLossSum/v.count).toFixed(4):null, ece: ece!=null?+ece.toFixed(4):null, wins:v.wins, losses:v.losses, early:v.earlyCount, total:v.count, hitRate:v.count>0?+((v.wins/v.count)*100).toFixed(1):null };
    }
  }
  return out;
}

// ── Initialise ────────────────────────────────────────────────────────────────

async function initialise() {
  if (initialised) return;
  initialised = true;
  for (const id of Object.keys(STATE)) STATE[id].savedSet = new Set();
  try { STATE.engine.lockedMap  = loadLockedMap(await getLockedPreds());      console.log(`[engine] loaded ${Object.keys(STATE.engine.lockedMap).length} engine preds`);  } catch(e) { console.error('[engine] init:',   e.message); STATE.engine.lockedMap  = {}; }
  try { STATE.pattern.lockedMap = loadLockedMap(await getLockedPatternPreds()); console.log(`[engine] loaded ${Object.keys(STATE.pattern.lockedMap).length} pattern preds`); } catch(e) { console.error('[engine] pattern:', e.message); STATE.pattern.lockedMap = {}; }
  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      STATE[model.id].lockedMap = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] loaded ${Object.keys(STATE[model.id].lockedMap).length} ${model.id} preds`);
    }
  } catch(e) { console.error('[engine] stat init:', e.message); for (const model of STAT_MODELS) STATE[model.id].lockedMap = {}; }
  try {
    const rows = await getPredictions({ limit: 10000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r.target, r.lo, r.hi);
      if (STATE[src]?.savedSet) STATE[src].savedSet.add(key);
    }
    console.log(`[engine] loaded ${rows.length} history keys`);
  } catch(e) { console.error('[engine] history:', e.message); }
  for (const id of Object.keys(STATE)) STATE[id].needsRebuild = true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function runPredictionEngine() {
  try {
    await initialise();
    const rounds = await getRounds({ limit: 5000, order: 'DESC' });
    if (rounds.length < MIN_ROUNDS) { console.log(`[engine] waiting (${rounds.length}/${MIN_ROUNDS})`); return; }
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    const allEngines = [
      { id:'engine',  state:STATE.engine,  buildFn:(t)=>buildPrediction(rounds,t.min,t.maxWidth,t.rare,lastRoundId),         saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedPreds(p);} },
      { id:'pattern', state:STATE.pattern, buildFn:(t)=>{const pp=buildPatternPrediction(rounds,t.min);return buildPatternWindow(pp,t.maxWidth);}, saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedPatternPreds(p);} },
      ...STAT_MODELS.map(model=>({ id:model.id, state:STATE[model.id], buildFn:(t)=>buildStatPrediction(rounds,t.min,t.maxWidth,model.id,lastRoundId), saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedStatPreds(model.id,p);} })),
    ];

    for (const eng of allEngines) {
      if (!(lastRoundId > eng.state.lastRoundId || eng.state.needsRebuild)) continue;
      eng.state.needsRebuild = false;
      const changed = await processEngine({ engineId:eng.id, state:eng.state, sortedRounds:rounds, lastRoundId, buildFn:eng.buildFn });
      eng.state.lastRoundId = lastRoundId;
      if (changed) { const p=buildSavePayload(eng.state.lockedMap); try{await eng.saveFn(p);}catch(e){console.error(`[${eng.id}] save:`,e.message);} }
    }
  } catch(e) { console.error('[predictionEngine] Fatal:', e.message, e.stack); }
}

function getLockedStatMap(modelId) { return STATE[modelId]?.lockedMap || {}; }

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap, getValidationMetrics };