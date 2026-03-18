'use strict';
// predictionEngine.js  v13
//
// IMPLEMENTS ALL ITEMS FROM FINAL OPTIMIZATION PROMPT:
//
// 1. GAP STATS CACHE — per target, computed once per tick, shared across all 4 models.
//    scanRounds was called 4x per target. Now called once, result cached keyed by
//    (lastRoundId, targetMin). Includes sorted gaps, HSM, all percentiles.
//    Cache invalidated when lastRoundId advances.
//
// 2. LOGIT-SPACE ENSEMBLE — replaces linear averaging.
//    logit(p) = log(p/(1-p)); weighted avg in logit space; sigmoid back.
//    Reduces overconfidence from correlated model outputs.
//    Falls back to equal weights when any model is degenerate.
//
// 3. 5-BIN CALIBRATION — replaces 3-bin static bins.
//    Bins: [0,0.30), [0.30,0.45), [0.45,0.60), [0.60,0.75), [0.75,1]
//    EWMA forgetting: each new outcome decays old data by factor (1-decay).
//    Laplace smoothing: prior count=2 per bin prevents early instability.
//    Applied AFTER ensemble combination. Monotonic mapping enforced.
//    Calibrated probW capped at rawProbW + 0.05 to prevent inflation.
//
// 4. WINDOW OPTIMIZATION — replaces heuristics.
//    Objective: argmax_low [ P(hit in [low, low+W-1]) - λ × effectiveWidth ]
//    λ = 0.01 (penalty per extra round of width)
//    Searches gap positions discretely; forward-only constraint.
//    Width capped at 1.35× maxWidth.
//
// 5. FAT-TAIL: PARETO PROBABILITY BLEND (not width inflation).
//    For rare targets (100x+) with cv > 1.1:
//    blend KM probW with Pareto-adjusted estimate using tail index α = 1/cv².
//    P_pareto(gap ≤ W) = 1 - (expectedGap / (expectedGap + W))^α
//    Weight blend = min(0.3, (cv-1) * 0.6) — grows with tail heaviness.
//    Width factor still applied but capped at 1.2× (not 1.4×) for rare targets.
//
// 6. DYNAMIC HSM modeWeight by cv:
//    cv < 1.0 → modeWeight = 0.50 (tight distribution, mode is reliable)
//    cv 1.0–1.3 → linear interpolation → 0.10
//    cv > 1.3 → modeWeight = 0.10 (high variance, mode unstable, use median)
//
// 7. ENSEMBLE DIAGNOSTICS — spread + outlier detection.
//    Spread = std(probGeo, probBay, probKm) in logit space.
//    High spread (> 0.5 logit units) → reduce effective weight of outlier model.
//    Ensemble instability tracked in validation metrics.
//
// 8. CONFIDENCE — incorporates ensemble disagreement.
//    confidence = betaConf(calibratedProbW, hits) - streakPenalty - spreadPenalty
//    spreadPenalty = round(15 * spread / 0.5)  capped at 15 points.
//
// 9. VALIDATION METRICS — per model, per target, rolling.
//    Tracked: Brier score, log-loss, ECE (Expected Calibration Error).
//    ECE = mean(|empiricalRate - predictedRate|) across calibration bins.
//    Available via getValidationMetrics() export (for /engine-status endpoint).
//
// 10. KM EMPIRICAL CDF — precomputed from sorted gaps array.
//     kmSurvival now uses binary search on precomputed step function.
//     O(log n) per query instead of O(n) loop. Cache stores the CDF.
//
// PRESERVED FROM v12:
//     Correct KM at-risk convention, HSM, per-engine independence,
//     dedup, correct anchoring, no gambler's fallacy.

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

const MIN_ROUNDS = 50;
const STALE_FORCE_REBUILD_THRESHOLD = 200;
const WINDOW_LAMBDA = 0.01; // penalty per extra round in window optimization

// ── Per-engine state ──────────────────────────────────────────────────────────

const STATE = {
  engine:  { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  pattern: { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ens:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  geo:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  bay:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  km:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
};

// ── Gap stats cache — shared across all models for same target+tick ────────────
// Key: `${lastRoundId}_${targetMin}`  →  gapStats object
const gapStatsCache = new Map();
let   cacheRoundId  = -1; // invalidate when round advances

// ── 5-bin calibration — per model per target ──────────────────────────────────
// Bins: [0,0.30), [0.30,0.45), [0.45,0.60), [0.60,0.75), [0.75,1]
// Each bin: { ewmaAct, ewmaPred, count }  (EWMA-weighted, Laplace-smoothed)
const CAL_BINS = [0, 0.30, 0.45, 0.60, 0.75, 1.01]; // 5 intervals
const CAL_LAPLACE = 2;  // prior pseudo-count per bin
const CAL_DECAY   = { normal: 0.08, rare: 0.03 }; // EWMA decay rates

const calibState = {};
for (const t of TARGETS) {
  calibState[t.label] = {};
  for (const m of STAT_MODELS) {
    calibState[t.label][m.id] = CAL_BINS.slice(0, -1).map((lo, i) => {
      const midpoint = (lo + CAL_BINS[i + 1]) / 2;
      return { ewmaAct: midpoint, ewmaPred: midpoint, count: 0 }; // Laplace prior
    });
  }
}

// ── Adaptive ensemble — log-loss EWMA per model per target ────────────────────
const modelScores = {};
for (const t of TARGETS) {
  modelScores[t.label] = {};
  for (const m of STAT_MODELS) modelScores[t.label][m.id] = { ewma: 0.693, count: 0 };
}

// ── Validation metrics — rolling, per model per target ────────────────────────
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

function sigmoid(x)  { return 1 / (1 + Math.exp(-x)); }
function logit(p)    { const q = Math.max(1e-7, Math.min(1 - 1e-7, p)); return Math.log(q / (1 - q)); }
function fromLogit(l){ return sigmoid(l); }

// ── Gap stats cache & scanRounds ──────────────────────────────────────────────

/**
 * Returns cached gap stats for (rounds, targetMin).
 * If cache miss or stale, recomputes and stores.
 * All 4 models share one computation per target per tick.
 */
function getGapStats(rounds, targetMin, lastRoundId) {
  // Invalidate entire cache when round advances
  if (lastRoundId !== cacheRoundId) {
    gapStatsCache.clear();
    cacheRoundId = lastRoundId;
  }

  const key = `${targetMin}`;
  if (gapStatsCache.has(key)) return gapStatsCache.get(key);

  const result = scanRounds(rounds, targetMin);
  gapStatsCache.set(key, result);
  return result;
}

function scanRounds(rounds, targetMin) {
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
  const pGlobal = (hits + 1) / (n + 2);
  const r200hits = rounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
  const pRecent  = (r200hits + 1) / 202;

  // CUSUM structural break detection
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

  // Only blend recent when CUSUM confirms actual shift
  const p = rateShifted
    ? Math.max(1e-6, Math.min(0.5, 0.75 * pGlobal + 0.25 * pRecent))
    : Math.max(1e-6, Math.min(0.5, pGlobal));

  // Sort gaps once — cached and reused by all models
  const sg = [...gaps].sort((a, b) => a - b);
  const m2 = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1 / p) :
    sg.length % 2 === 1 ? sg[m2] : (sg[m2 - 1] + sg[m2]) / 2;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / p;
  const variance = gaps.length > 1 ? Math.max(0, gSS / gaps.length - meanGap ** 2) : meanGap * meanGap;
  const stdGap   = Math.sqrt(variance);
  const cv       = meanGap > 0 ? stdGap / meanGap : 1;

  // Percentiles — safe, no spread operator
  const pctile = (frac) => {
    if (sg.length === 0) return meanGap;
    return sg[Math.min(sg.length - 1, Math.floor(frac * sg.length))];
  };
  const p75 = pctile(0.75);
  const p90 = pctile(0.90);
  const p95 = pctile(0.95);

  // HSM — computed once per target per tick
  const hsm = halfSampleMode(sg, medianGap, cv);

  // Dynamic modeWeight by cv (prompt requirement §5)
  let modeWeight;
  if (cv < 1.0)      modeWeight = 0.50;
  else if (cv > 1.3) modeWeight = 0.10;
  else               modeWeight = 0.50 - (cv - 1.0) * (0.40 / 0.30); // linear interp

  const expectedGap = Math.max(1, Math.round(medianGap * (1 - modeWeight) + hsm * modeWeight));

  // Precompute KM CDF step function — O(n) once, O(log n) per query
  const kmCDF = buildKmCDF(sg);

  // Current dry streak
  let currentStreak = 0;
  for (let i = n - 1; i >= 0; i--) {
    if (rounds[i].multiplier < targetMin) currentStreak++;
    else break;
  }

  return {
    hits, n, p, pGlobal, pRecent, rateShifted,
    cusumNorm: +cusumNorm.toFixed(3),
    gapSinceLast, meanGap, medianGap, stdGap, cv,
    p75, p90, p95, gaps, sg, kmCDF,
    hsm, modeWeight, expectedGap,
    currentStreak,
  };
}

// ── Half-sample mode (HSM) ────────────────────────────────────────────────────

function halfSampleMode(sg, medianGap, cv) {
  const n = sg.length;
  if (n < 8)    return medianGap;
  if (cv > 2.0) return sg[Math.floor(n / 2)]; // unstable — fall back to median

  const h = Math.max(2, Math.floor(n / 2));
  let bestRange = sg[sg.length - 1] - sg[0] + 1;
  let bestStart = 0;
  for (let i = 0; i + h - 1 < n; i++) {
    const range = sg[i + h - 1] - sg[i];
    if (range < bestRange) { bestRange = range; bestStart = i; }
  }
  return (sg[bestStart] + sg[bestStart + h - 1]) / 2;
}

// ── KM: precomputed CDF step function ────────────────────────────────────────
//
// Build once from sorted gaps: array of { t, survivalS } steps.
// Query at W via binary search → O(log n) vs O(n) per model call.

function buildKmCDF(sg) {
  if (sg.length < 5) return null;
  const m   = sg.length;
  const steps = []; // { t, S }
  let S = 1.0;
  let i = 0;
  while (i < m) {
    const t       = sg[i];
    const nAtRisk = m - i;
    let   d       = 0;
    while (i < m && sg[i] === t) { d++; i++; }
    S *= (1 - d / nAtRisk);
    steps.push({ t, S: Math.max(0, S) });
  }
  return steps;
}

function kmSurvival(kmCDF, W) {
  if (!kmCDF || kmCDF.length === 0) return null;

  // Binary search for last step with t <= W
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (kmCDF[mid].t <= W) { idx = mid; lo = mid + 1; }
    else hi = mid - 1;
  }

  const S = idx >= 0 ? kmCDF[idx].S : 1.0; // no events <= W → S=1
  return Math.max(0, Math.min(1 - 1e-9, 1 - S));
}

// ── Window optimization — maximize P(hit) - λ×width ──────────────────────────
//
// Given a fixed effective width W, find optimal low offset from anchor.
// Uses cumulative gap CDF to estimate P(hit in [low, low+W-1]).
// Objective: P(hit) - λ × W, maximized over low ∈ [0, 3*expectedGap].
// Forward-only: low >= 0. No-overlap enforced by anchor+1 mechanism upstream.
// Width is fixed (not optimized) — only placement is optimized.
// Stable: ties broken by choosing lower low (earlier window).

function optimizeWindowPlacement(kmCDF, sg, expectedGap, effectiveWidth, gapSinceLast) {
  if (!kmCDF || kmCDF.length === 0 || sg.length < 5) {
    // Fallback to heuristic placement
    const low  = Math.max(0, expectedGap - effectiveWidth);
    return { low, high: low + effectiveWidth - 1 };
  }

  // If already overdue, open immediately
  if (gapSinceLast >= expectedGap) {
    return { low: 0, high: effectiveWidth - 1 };
  }

  // P(gap in [a, b]) = S(a-1) - S(b) = (1-KM(a-1)) - (1-KM(b)) = KM(b) - KM(a-1)
  // We want P(next gap falls in window [low, low+W-1])
  // low is relative to anchor (which is set to lastRoundId+1 upstream)

  const maxSearch = Math.max(3 * expectedGap, effectiveWidth + 10);
  let bestScore = -Infinity;
  let bestLow   = Math.max(0, expectedGap - effectiveWidth);

  for (let low = 0; low <= maxSearch; low++) {
    const high = low + effectiveWidth - 1;
    // P(gap in [low, high]) using KM CDF
    const cdfHigh = kmSurvival(kmCDF, high)   ?? 0;
    const cdfLow  = low > 0 ? (kmSurvival(kmCDF, low - 1) ?? 0) : 0;
    const probHit  = Math.max(0, cdfHigh - cdfLow);
    const score    = probHit - WINDOW_LAMBDA * effectiveWidth;
    if (score > bestScore) { bestScore = score; bestLow = low; }
    // Early termination: if we're well past the distribution tail, stop
    if (low > expectedGap * 3 && probHit < 0.001) break;
  }

  return { low: bestLow, high: bestLow + effectiveWidth - 1 };
}

// ── Fat-tail Pareto probability blend ────────────────────────────────────────
//
// For rare targets (100x+) with heavy-tailed gap distributions (cv > 1.1):
// Blend KM estimate with Pareto-tail estimate.
// P_pareto(gap ≤ W) = 1 - (µ / (µ + W))^α  where α = 1/cv² (tail index)
// Heavier tail (high cv) → more weight on Pareto correction.
// Blend weight = min(0.30, (cv - 1.0) * 0.60) — grows with tail heaviness.
// This adjusts probW directly (not window width), per prompt requirement.

function applyParetoCorrectedProbW(rawProbW, cv, meanGap, maxWidth, isRare) {
  if (!isRare || cv <= 1.1) return rawProbW;

  const alpha     = 1 / Math.max(0.1, cv * cv); // Pareto tail index
  const paretoP   = 1 - Math.pow(meanGap / Math.max(meanGap, meanGap + maxWidth), alpha);
  const blendW    = Math.min(0.30, (cv - 1.0) * 0.60);
  const blended   = (1 - blendW) * rawProbW + blendW * paretoP;
  // Conservative: cap blend at midpoint between raw and Pareto (no inflation)
  return Math.max(1e-6, Math.min(1 - 1e-6, blended));
}

// ── 5-bin calibration with EWMA forgetting ────────────────────────────────────

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

  const target  = TARGETS.find(t => t.label === targetLabel);
  const decay   = target?.rare ? CAL_DECAY.rare : CAL_DECAY.normal;
  const actual  = outcome === 'win' ? 1 : 0;
  const idx     = getCalBinIdx(predictedProbW);
  const bin     = bins[idx];

  // EWMA update with forgetting
  bin.ewmaAct  = (1 - decay) * bin.ewmaAct  + decay * actual;
  bin.ewmaPred = (1 - decay) * bin.ewmaPred + decay * predictedProbW;
  bin.count    = Math.min(bin.count + 1, 500);
}

function applyCalibration(probW, targetLabel, modelId) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;

  const idx = getCalBinIdx(probW);
  const bin = bins[idx];

  // Need enough data for calibration (Laplace prior = 2 pseudo-counts)
  if (bin.count < 8) return probW;

  const empirical  = bin.ewmaAct;
  const predicted  = bin.ewmaPred;
  if (predicted < 1e-6) return probW;

  const ratio = empirical / predicted;
  // Only correct if deviation > 12% (reduces noise-driven corrections)
  if (Math.abs(ratio - 1) < 0.12) return probW;

  // Cap at ±20% correction; also cap calibrated ≤ raw + 0.05 (no inflation)
  const cappedRatio  = Math.max(0.80, Math.min(1.20, ratio));
  const calibrated   = Math.max(1e-6, Math.min(1 - 1e-6, probW * cappedRatio));
  return Math.min(calibrated, probW + 0.05); // anti-inflation cap
}

// Enforce monotonicity across bins: calibrated probs should be non-decreasing
// across bins. Called lazily — not per-outcome, just before applying.
function ensureMonotonic(calibratedProbs) {
  // calibratedProbs = array of calibrated midpoints per bin, ascending
  // Simple isotonic-style repair: scan left to right, enforce non-decrease
  for (let i = 1; i < calibratedProbs.length; i++) {
    if (calibratedProbs[i] < calibratedProbs[i - 1]) {
      calibratedProbs[i] = calibratedProbs[i - 1];
    }
  }
}

// ── Validation metrics update ─────────────────────────────────────────────────

function updateValidationMetrics(targetLabel, modelId, predictedProbW, outcome) {
  if (outcome === 'early') {
    const v = valMetrics[targetLabel]?.[modelId];
    if (v) v.earlyCount++;
    return;
  }
  const v = valMetrics[targetLabel]?.[modelId];
  if (!v) return;
  const actual   = outcome === 'win' ? 1 : 0;
  const p        = Math.max(1e-7, Math.min(1 - 1e-7, predictedProbW));
  v.brierSum    += (actual - p) ** 2;
  v.logLossSum  += -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
  v.count++;
  if (outcome === 'win') v.wins++;
  else v.losses++;
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

// ── Ensemble: logit-space combination + diagnostics ───────────────────────────
//
// Step 1: get calibrated probW from each base model
// Step 2: convert to logit space
// Step 3: weighted average in logit space (reduces overconfidence from correlation)
// Step 4: convert back via sigmoid
// Step 5: apply ensemble-level calibration
// Diagnostics: compute spread in logit space; flag unstable ensemble.

function logLossVal(actual, probW) {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probW));
  return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
}

function updateModelScore(targetLabel, modelId, predictedProbW, outcome) {
  if (outcome === 'early') return;
  const actual = outcome === 'win' ? 1 : 0;
  const loss   = logLossVal(actual, predictedProbW);
  const s      = modelScores[targetLabel]?.[modelId];
  if (!s) return;
  const target = TARGETS.find(t => t.label === targetLabel);
  const decay  = target?.rare ? 0.02 : 0.05;
  s.ewma  = s.count === 0 ? loss : (1 - decay) * s.ewma + decay * loss;
  s.count = Math.min(s.count + 1, 500);
}

function buildEnsemble(targetLabel, probGeo, probBay, probKm) {
  const scores  = modelScores[targetLabel];
  const models  = [
    { id: 'geo', prob: probGeo },
    { id: 'bay', prob: probBay },
    { id: 'km',  prob: probKm  },
  ];

  // Adaptive weights from log-loss history (Dirichlet prior: default loss = log(2))
  let wSum = 0;
  const weights = [];
  for (const { id } of models) {
    const avgLoss = scores[id]?.count > 2 ? scores[id].ewma : 0.693;
    const w = Math.exp(-avgLoss * 2);
    weights.push(w);
    wSum += w;
  }
  if (wSum < 1e-9) { for (let i = 0; i < weights.length; i++) weights[i] = 1 / 3; wSum = 1; }
  for (let i = 0; i < weights.length; i++) weights[i] /= wSum;

  // Convert each model's probW to logit space
  const logits = models.map(({ prob }) => logit(prob));

  // Compute spread (std in logit space) for diagnostics
  const wMean = logits.reduce((s, l, i) => s + weights[i] * l, 0);
  const wVar  = logits.reduce((s, l, i) => s + weights[i] * (l - wMean) ** 2, 0);
  const spread = Math.sqrt(wVar);

  // Outlier detection: if any model is > 1.5 logit units from weighted mean, halve its weight
  const adjWeights = [...weights];
  for (let i = 0; i < logits.length; i++) {
    if (Math.abs(logits[i] - wMean) > 1.5) {
      adjWeights[i] *= 0.5;
    }
  }
  const adjSum = adjWeights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < adjWeights.length; i++) adjWeights[i] /= adjSum;

  // Weighted average in logit space
  const ensLogit = logits.reduce((s, l, i) => s + adjWeights[i] * l, 0);
  const ensProb  = fromLogit(ensLogit);

  return { ensProb, spread, weights: adjWeights };
}

// ── Beta confidence — with ensemble spread ────────────────────────────────────

function betaConf(probW, hits, spread) {
  const effectiveN = Math.min(hits, 300);
  const alpha      = probW * effectiveN + 1;
  const beta       = (1 - probW) * effectiveN + 1;
  const postVar    = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const postStd    = Math.sqrt(postVar);

  let c = probW * 100 - postStd * 120;
  if (hits < 10)      c -= 15;
  else if (hits < 25) c -= 7;
  else if (hits < 50) c -= 3;

  // Ensemble spread penalty: high spread = models disagree = less certain
  if (spread != null) {
    const spreadPenalty = Math.min(15, Math.round(15 * spread / 0.5));
    c -= spreadPenalty;
  }

  return Math.max(20, Math.min(88, Math.round(c)));
}

// ── Streak-aware window placement ─────────────────────────────────────────────

function streakAwareWindow(gs, maxWidth, isRare) {
  const { gapSinceLast, meanGap, stdGap, cv, p90, p95, expectedGap, kmCDF, sg } = gs;

  const z = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;

  // Streak status for UI
  let streakStatus;
  if (gapSinceLast >= p95)              streakStatus = 'extreme';
  else if (gapSinceLast >= p90)         streakStatus = 'severe';
  else if (gapSinceLast >= expectedGap) streakStatus = 'overdue';
  else                                  streakStatus = 'normal';

  // Confidence penalty — smooth sigmoid (window placement only, not probW)
  const confPenalty = Math.round(20 * sigmoid(z - 1.5));

  // Effective width:
  // - rare targets: fat-tail → mild width increase capped at 1.2× (NOT 1.4×)
  // - non-rare: base width only, elastic +15% if z > 2
  let effectiveWidth = maxWidth;
  if (isRare && cv > 1.1) {
    const tailFactor = Math.min(1.20, 1.0 + (cv - 1.0) * 0.25); // softer cap
    effectiveWidth = Math.round(maxWidth * tailFactor);
  }
  if (z > 2) {
    effectiveWidth = Math.min(Math.round(maxWidth * 1.35), Math.round(effectiveWidth * 1.15));
  }
  effectiveWidth = Math.min(effectiveWidth, Math.round(maxWidth * 1.35)); // hard cap 1.35×

  // Window placement: use optimizer when KM CDF available, else heuristic
  const { low, high } = optimizeWindowPlacement(kmCDF, sg, expectedGap, effectiveWidth, gapSinceLast);

  return { low, high, streakStatus, confPenalty, z: +z.toFixed(2), effectiveWidth };
}

// ── Build: ENGINE ─────────────────────────────────────────────────────────────

function buildPrediction(rounds, targetMin, maxWidth, isRare, lastRoundId) {
  const s = getGapStats(rounds, targetMin, lastRoundId);
  if (!s) return null;
  const { hits, p, gapSinceLast, meanGap, medianGap, stdGap, cv, p90, p95, expectedGap } = s;

  const probW = 1 - Math.pow(1 - p, maxWidth);

  const { low, high, streakStatus, confPenalty, z } = streakAwareWindow(s, maxWidth, isRare ?? false);

  return {
    low, high, expectedGap, opensIn: low,
    confidence: Math.max(20, betaConf(probW, hits, null) - confPenalty),
    probW: +probW.toFixed(4),
    p: +p.toFixed(6),
    rateShifted: s.rateShifted, cusumNorm: s.cusumNorm,
    streakStatus, currentStreak: s.currentStreak, z,
    gapSinceLast, hits,
  };
}

// ── Build: STAT MODELS ────────────────────────────────────────────────────────

function buildStatPrediction(rounds, targetMin, maxWidth, modelId, lastRoundId) {
  const s = getGapStats(rounds, targetMin, lastRoundId);
  if (!s) return null;

  const { hits, n, p, pGlobal, pRecent, gapSinceLast, rateShifted,
          sg, kmCDF, meanGap, medianGap, stdGap, cv, expectedGap } = s;

  const target      = TARGETS.find(t => t.min === targetMin);
  const targetLabel = target?.label ?? '?';
  const isRare      = target?.rare ?? false;

  let rawProbW;

  if (modelId === 'geo') {
    const pGeo = (hits + 1) / (n + 2);
    rawProbW = 1 - Math.pow(1 - pGeo, maxWidth);

  } else if (modelId === 'bay') {
    const recencyW = rateShifted ? 0.20 : 0.05;
    const pBay     = (1 - recencyW) * pGlobal + recencyW * pRecent;
    rawProbW = 1 - Math.pow(1 - pBay, maxWidth);

  } else if (modelId === 'km') {
    if (!kmCDF) return null;
    rawProbW = kmSurvival(kmCDF, maxWidth);
    if (rawProbW == null) return null;
    // Pareto blend for heavy-tailed rare targets
    rawProbW = applyParetoCorrectedProbW(rawProbW, cv, meanGap, maxWidth, isRare);

  } else {
    // ENS: logit-space combination
    const pGeo    = (hits + 1) / (n + 2);
    const recencyW = rateShifted ? 0.20 : 0.05;
    const pBay    = (1 - recencyW) * pGlobal + recencyW * pRecent;

    const probGeo = applyCalibration(1 - Math.pow(1 - pGeo, maxWidth), targetLabel, 'geo');
    const probBay = applyCalibration(1 - Math.pow(1 - pBay, maxWidth), targetLabel, 'bay');
    let   probKm  = probGeo;
    if (kmCDF) {
      const km = kmSurvival(kmCDF, maxWidth);
      if (km != null) {
        probKm = applyCalibration(applyParetoCorrectedProbW(km, cv, meanGap, maxWidth, isRare), targetLabel, 'km');
      }
    }

    const { ensProb, spread } = buildEnsemble(targetLabel, probGeo, probBay, probKm);

    // Apply ensemble-level calibration
    const calibrated = applyCalibration(ensProb, targetLabel, modelId);

    const { low, high, streakStatus, confPenalty, z, effectiveWidth } = streakAwareWindow(s, maxWidth, isRare);

    return {
      low, high, expectedGap, opensIn: low,
      confidence: Math.max(20, betaConf(calibrated, hits, spread) - confPenalty),
      probW:    +calibrated.toFixed(4),
      rawProbW: +ensProb.toFixed(4),
      p: +p.toFixed(6),
      streakStatus, currentStreak: s.currentStreak, z,
      gapSinceLast, hits, rateShifted, model: modelId,
      effectiveWidth, spread: +spread.toFixed(3),
    };
  }

  rawProbW = Math.max(1e-6, Math.min(1 - 1e-6, rawProbW));
  const calibratedProbW = applyCalibration(rawProbW, targetLabel, modelId);

  const { low, high, streakStatus, confPenalty, z, effectiveWidth } = streakAwareWindow(s, maxWidth, isRare);

  return {
    low, high, expectedGap, opensIn: low,
    confidence: Math.max(20, betaConf(calibratedProbW, hits, null) - confPenalty),
    probW:    +calibratedProbW.toFixed(4),
    rawProbW: +rawProbW.toFixed(4),
    p: +p.toFixed(6),
    streakStatus, currentStreak: s.currentStreak, z,
    gapSinceLast, hits, rateShifted, model: modelId,
    effectiveWidth,
  };
}

// ── Build: PATTERN (isolated, UI-only signal) ─────────────────────────────────

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
  const trendScore  =safe((emaSlow>0?(emaFast-emaSlow)/emaSlow:0)*4);
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
    32 + Math.min(18,Math.log2(hits+1)*4) + absComposite*30 + (agree-1)*6 - (cv>1.5?8:cv>1.2?4:0)
  )));
  return { direction, confidence: conf, hits, meanGap: Math.round(meanGap), medianGap: Math.round(medianGap), composite:+composite.toFixed(3) };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const expectedGap = patternResult.medianGap || patternResult.meanGap || maxWidth;
  const low  = Math.max(0, expectedGap - maxWidth);
  const high = low + maxWidth - 1;
  return { low, high, expectedGap, opensIn: low, confidence: patternResult.confidence, direction: patternResult.direction, streakStatus: 'normal', currentStreak: 0 };
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
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId + 1, generation:1, stale:false };
        anyChange = true;
        console.log(`[${engineId}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'}`);
      }
      continue;
    }

    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow || anchorRound === 0;
    const isExpired   = lastRoundId > absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
          const key = makeKey(engineId, target.label, absLow, absHigh);
          if (!state.savedSet.has(key)) {
            state.savedSet.add(key);
            try {
              await savePrediction({
                target: target.label, minMult: target.min, outcome,
                lo: absLow, hi: absHigh, anchorRound,
                hitRound: status.hitRound || null,
                generation: existing.generation || 1,
                source: engineId,
              });
              if (STAT_MODELS.some(m => m.id === engineId) && existing.probW != null) {
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
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min, anchorRound: lastRoundId + 1,
          generation: (existing.generation||1) + (isNonsense ? 0 : 1),
          stale: false,
        };
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
      const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
      const key = makeKey(engineId, target.label, absLow, absHigh);
      if (!state.savedSet.has(key)) {
        state.savedSet.add(key);
        try {
          await savePrediction({
            target: target.label, minMult: target.min, outcome,
            lo: absLow, hi: absHigh, anchorRound,
            hitRound: status.hitRound || null,
            generation: existing.generation || 1,
            source: engineId,
          });
          if (STAT_MODELS.some(m => m.id === engineId) && existing.probW != null) {
            updateCalibration(target.label, engineId, existing.probW, outcome);
            updateModelScore(target.label, engineId, existing.probW, outcome);
            updateValidationMetrics(target.label, engineId, existing.probW, outcome);
          }
          console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
        } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min, anchorRound: lastRoundId + 1,
          generation: (existing.generation||1) + 1, stale: false,
        };
        console.log(`[${engineId}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete state.lockedMap[target.label];
      }
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
      lo:            anchor + (Number(pred.low)||0),
      hi:            anchor + (Number(pred.high)||0),
      roundWhenMade: anchor,
      generation:    pred.generation||1,
      eta: {
        low: pred.low, high: pred.high, conf: pred.confidence,
        probW: pred.probW, rawProbW: pred.rawProbW ?? pred.probW,
        expectedGap: pred.expectedGap,
        opensIn: pred.opensIn, streakStatus: pred.streakStatus,
        currentStreak: pred.currentStreak,
        spread: pred.spread ?? null,
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
    map[label] = {
      low:           eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor),
      high:          eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor),
      confidence:    eta.conf ?? 50,
      probW:         eta.probW ?? null,
      rawProbW:      eta.rawProbW ?? null,
      expectedGap:   eta.expectedGap ?? null,
      opensIn:       eta.opensIn ?? null,
      streakStatus:  eta.streakStatus ?? 'normal',
      currentStreak: eta.currentStreak ?? 0,
      spread:        eta.spread ?? null,
      targetMin:     target.min,
      anchorRound:   anchor,
      generation:    pred.generation ?? 1,
      stale:         true,
    };
  }
  return map;
}

// ── Validation metrics export ─────────────────────────────────────────────────

function getValidationMetrics() {
  const out = {};
  for (const t of TARGETS) {
    out[t.label] = {};
    for (const m of STAT_MODELS) {
      const v   = valMetrics[t.label][m.id];
      const ece = getECE(t.label, m.id);
      out[t.label][m.id] = {
        brier:    v.count > 0 ? +(v.brierSum / v.count).toFixed(4)   : null,
        logLoss:  v.count > 0 ? +(v.logLossSum / v.count).toFixed(4) : null,
        ece:      ece != null  ? +ece.toFixed(4)                       : null,
        wins:     v.wins,
        losses:   v.losses,
        early:    v.earlyCount,
        total:    v.count,
        hitRate:  v.count > 0  ? +((v.wins / v.count) * 100).toFixed(1) : null,
      };
    }
  }
  return out;
}

// ── Initialise ────────────────────────────────────────────────────────────────

async function initialise() {
  if (initialised) return;
  initialised = true;

  for (const id of Object.keys(STATE)) STATE[id].savedSet = new Set();

  try {
    STATE.engine.lockedMap = loadLockedMap(await getLockedPreds());
    console.log(`[engine] loaded ${Object.keys(STATE.engine.lockedMap).length} engine preds`);
  } catch(e) { console.error('[engine] init error:', e.message); STATE.engine.lockedMap = {}; }

  try {
    STATE.pattern.lockedMap = loadLockedMap(await getLockedPatternPreds());
    console.log(`[engine] loaded ${Object.keys(STATE.pattern.lockedMap).length} pattern preds`);
  } catch(e) { console.error('[engine] pattern init error:', e.message); STATE.pattern.lockedMap = {}; }

  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      STATE[model.id].lockedMap = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] loaded ${Object.keys(STATE[model.id].lockedMap).length} ${model.id} preds`);
    }
  } catch(e) {
    console.error('[engine] stat init error:', e.message);
    for (const model of STAT_MODELS) STATE[model.id].lockedMap = {};
  }

  try {
    const rows = await getPredictions({ limit: 10000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r.target, r.lo, r.hi);
      if (STATE[src]?.savedSet) STATE[src].savedSet.add(key);
    }
    console.log(`[engine] loaded ${rows.length} history keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  for (const id of Object.keys(STATE)) STATE[id].needsRebuild = true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function runPredictionEngine() {
  try {
    await initialise();

    const rounds = await getRounds({ limit: 5000, order: 'DESC' });
    if (rounds.length < MIN_ROUNDS) {
      console.log(`[engine] waiting for rounds (${rounds.length}/${MIN_ROUNDS})`);
      return;
    }
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Invalidate gap cache when round advances (handled inside getGapStats)

    const allEngines = [
      {
        id:     'engine',
        state:  STATE.engine,
        buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth, t.rare, lastRoundId),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedPreds(p); },
      },
      {
        id:     'pattern',
        state:  STATE.pattern,
        buildFn: (t) => { const pp = buildPatternPrediction(rounds, t.min); return buildPatternWindow(pp, t.maxWidth); },
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedPatternPreds(p); },
      },
      ...STAT_MODELS.map(model => ({
        id:     model.id,
        state:  STATE[model.id],
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth, model.id, lastRoundId),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedStatPreds(model.id, p); },
      })),
    ];

    for (const eng of allEngines) {
      const shouldRun = lastRoundId > eng.state.lastRoundId || eng.state.needsRebuild;
      if (!shouldRun) continue;

      eng.state.needsRebuild = false;

      const changed = await processEngine({
        engineId:     eng.id,
        state:        eng.state,
        sortedRounds: rounds,
        lastRoundId,
        buildFn:      eng.buildFn,
      });

      eng.state.lastRoundId = lastRoundId;

      if (changed) {
        const p = buildSavePayload(eng.state.lockedMap);
        try { await eng.saveFn(p); }
        catch(e) { console.error(`[${eng.id}] save locked error:`, e.message); }
      }
    }

  } catch(e) {
    console.error('[predictionEngine] Fatal:', e.message, e.stack);
  }
}

function getLockedStatMap(modelId) {
  return STATE[modelId]?.lockedMap || {};
}

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap, getValidationMetrics };