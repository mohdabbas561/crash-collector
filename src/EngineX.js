'use strict';

/*
  Edge-first crash engine.
  Statistical assumptions:
  1) Fair baseline for threshold target x is P(hit next round) = 1 / x.
  2) We detect edge as p_model - p_baseline.
  3) Decision layer uses EV = p * x - 1 and only activates when EV exceeds threshold.
  4) Model blending is inverse-error weighted by rolling Brier score, not sample size.
  5) White-cluster effect is intentionally weak and only uses (reboundProb - continueProb).
  6) Walk-forward evaluation is used for reliability/backtest to avoid future leakage.
*/

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const FIXED_WINDOW_SPAN = Object.freeze({
  5: 3,
  10: 6,
  20: 10,
  50: 18,
  100: 27,
  500: 50,
  1000: 75,
});
const REPORT_THRESHOLDS = [2, 5, 10, 25, 50];

const DEFAULT_CONFIG = Object.freeze({
  // Minimum history before we trust adaptive models.
  minTrainingRounds: 240,

  // Feature windows for KNN representation.
  featureShortWindow: 24,
  featureLongWindow: 96,
  featureVolWindow: 64,

  // KNN search scope.
  knnLookback: 1800,
  knnMaxCandidates: 900,
  knnKMin: 24,
  knnKMax: 120,

  // Hazard model uses rates first; gap only as weak, bounded modifier.
  hazardRecentWindow: 120,
  // Hard-capped by design to avoid gambler's-fallacy style overreaction to gap.
  hazardGapInfluence: 0.05,
  hazardMinEvalCount: 80,
  hazardDisableTolerance: 0.0,

  // White cluster thresholds from data quantiles.
  whiteCutQuantile: 0.40,
  reboundCutQuantile: 0.75,

  // White effect must remain weak.
  whiteModifierStrength: 0.08,

  // Rolling Brier memory.
  brierWindow: 600,

  // Blend constraints.
  maxHazardBlend: 0.25,
  minBaselineBlend: 0.15,
  modelErrorEps: 1e-6,
  modelDisagreementHigh: 0.22,
  modelDisagreementHard: 0.40,

  // EV gating.
  evThreshold: 0.0,
  clusteredEvTolerance: 0.03,

  // Entropy/random regime detection.
  entropyRecentWindow: 300,
  entropyStep: 12,
  entropyQuantile: 0.35,
  entropyAutoDisable: true,
  entropySpikeStdMult: 1.5,
  entropyTrendPenaltyScale: 4.0,

  // Walk-forward range used both for reliability and backtest.
  walkForwardWindow: 3000,

  // Edge validation (predicted vs realized EV consistency).
  edgeValidationWindow: 220,
  edgeValidationMinTrades: 20,

  // Contextual KNN.
  // Feature vector order:
  // [prev, prev2, shortMean, longMean, vol, lowRateShort, lowRateLong, hitRateLong, gapNorm, whiteRunNorm]
  featureWeights: [0.45, 0.35, 0.70, 0.65, 0.45, 0.22, 0.25, 1.00, 0.08, 0.12],
  knnTemporalDecayRounds: 360,
  knnDistanceSlack: 1.6,
  knnMinReliability: 0.12,

  // B2B exploit is intentionally tiny.
  b2bMomentumWindowShort: 8,
  b2bMomentumWindowLong: 180,
  b2bBoostCap: 0.05,
  b2bNearWindow: 5,

  // Pre-condition detector windows/thresholds (all thresholding is quantile-based).
  // These implement state detection, not hard outcome prediction.
  preconditionWhiteWindow: 36,
  preconditionVolShortWindow: 24,
  preconditionVolLongWindow: 96,
  preconditionCalibrationMin: 220,
  whiteRegimeThresholdQuantile: 0.72,
  releaseRunThresholdQuantile: 0.80,
  releaseThresholdQuantile: 0.70,
  momentumThresholdQuantile: 0.72,

  // Risk sizing.
  maxRiskFraction: 0.15,

  // Performance dashboard.
  performanceWindow: 160,

  // Regime thresholds.
  regimeLagTrendThreshold: 0.10,
  regimeDriftTrendThreshold: 0.05,
  regimeVolatilityShiftThreshold: 0.20,
  regimeClusterThreshold: 0.18,

  // Edge validation confidence scaling.
  edgeValidationFloor: 0.25,
  edgeValidationCeil: 1.2,

  // Execution mode toggle:
  // true => always emit lock windows (quick prediction mode),
  // false => strict pre-condition gate can keep targets idle.
  alwaysEmitLocks: true,
});

const BUCKETS = [
  { id: 'micro', label: 'Micro', min: 1, max: 1.99, color: '#ff4560' },
  { id: 'low', label: 'Low', min: 2, max: 4.99, color: '#ffd84d' },
  { id: 'mid', label: 'Mid', min: 5, max: 9.99, color: '#00ff88' },
  { id: 'high', label: 'High', min: 10, max: 24.99, color: '#00d4ff' },
  { id: 'moon', label: 'Moon', min: 25, max: Number.POSITIVE_INFINITY, color: '#c084fc' },
];

function clamp(v, lo, hi) {
  const n = Number(v);
  if (!Number.isFinite(n)) return lo;
  if (n < lo) return lo;
  if (n > hi) return hi;
  return n;
}

function roundNum(v, digits = 6) {
  const n = Number(v);
  return Number.isFinite(n) ? Number(n.toFixed(digits)) : null;
}

function mean(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  let s = 0;
  for (let i = 0; i < arr.length; i += 1) s += Number(arr[i]) || 0;
  return s / arr.length;
}

function variance(arr, avg) {
  if (!Array.isArray(arr) || arr.length < 2) return null;
  const m = Number.isFinite(avg) ? avg : mean(arr);
  if (m === null) return null;
  let s = 0;
  for (let i = 0; i < arr.length; i += 1) {
    const d = (Number(arr[i]) || 0) - m;
    s += d * d;
  }
  return s / arr.length;
}

function stddev(arr, avg) {
  const v = variance(arr, avg);
  return v === null ? null : Math.sqrt(v);
}

function sortedCopy(arr) {
  return [...arr].sort((a, b) => a - b);
}

function quantileFromSorted(sorted, q) {
  if (!sorted.length) return null;
  const qq = clamp(q, 0, 1);
  const idx = (sorted.length - 1) * qq;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const t = idx - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}

function quantile(arr, q) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  return quantileFromSorted(sortedCopy(arr), q);
}

function lowerBound(arr, v) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function sigmoid(z) {
  const x = clamp(z, -30, 30);
  return 1 / (1 + Math.exp(-x));
}

function tanh(z) {
  const x = clamp(z, -30, 30);
  const e2 = Math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

function normalizeRounds(rounds) {
  const clean = (rounds || [])
    .map((r) => ({
      roundId: Number(r.roundId),
      multiplier: Number(r.multiplier),
      timestamp: Number(r.timestamp) || 0,
    }))
    .filter((r) => Number.isFinite(r.roundId) && Number.isFinite(r.multiplier) && r.multiplier > 0)
    .sort((a, b) => a.roundId - b.roundId);

  const dedup = [];
  let last = null;
  for (let i = 0; i < clean.length; i += 1) {
    const r = clean[i];
    if (r.roundId === last) dedup[dedup.length - 1] = r;
    else {
      dedup.push(r);
      last = r.roundId;
    }
  }
  return dedup;
}

function normalizeLockInput(input) {
  if (!input) return null;
  return {
    lo: Number(input.lo),
    hi: Number(input.hi),
    roundWhenMade: Number(input.roundWhenMade ?? input.round_when_made),
    generation: Number(input.generation || 1),
    suspended: Boolean(input.suspended ?? input.eta?.suspended),
    confidence: Number(input.confidence ?? input.eta?.aiConfidence ?? 0),
    eta: input.eta || null,
  };
}

function buildPrefix(values) {
  const out = new Array(values.length + 1).fill(0);
  for (let i = 0; i < values.length; i += 1) out[i + 1] = out[i] + (Number(values[i]) || 0);
  return out;
}

function windowSum(prefix, left, right) {
  const l = Math.max(0, left);
  const r = Math.max(l, right);
  return prefix[r + 1] - prefix[l];
}

function windowMean(prefix, left, right) {
  if (right < left) return null;
  const n = right - left + 1;
  if (n <= 0) return null;
  return windowSum(prefix, left, right) / n;
}

function computeGapSeries(hits) {
  const n = hits.length;
  const gap = new Array(n).fill(0);
  const hitIdx = [];
  let lastHit = -1;
  for (let i = 0; i < n; i += 1) {
    if (hits[i]) {
      lastHit = i;
      hitIdx.push(i);
      gap[i] = 0;
    } else {
      gap[i] = lastHit >= 0 ? i - lastHit : i + 1;
    }
  }
  const intervals = [];
  for (let i = 1; i < hitIdx.length; i += 1) intervals.push(hitIdx[i] - hitIdx[i - 1]);
  const intervalPrefix = buildPrefix(intervals);
  const intervalSqPrefix = buildPrefix(intervals.map((v) => v * v));
  return { gap, hitIdx, intervals, intervalPrefix, intervalSqPrefix };
}

function buildWhiteFlags(multipliers, whiteCut) {
  const flags = new Array(multipliers.length).fill(0);
  const run = new Array(multipliers.length).fill(0);
  for (let i = 0; i < multipliers.length; i += 1) {
    flags[i] = multipliers[i] < whiteCut ? 1 : 0;
    run[i] = flags[i] ? (i > 0 ? run[i - 1] + 1 : 1) : 0;
  }
  return { flags, run, prefix: buildPrefix(flags) };
}

function histogram(values, bins) {
  const counts = new Array(Math.max(1, bins.length - 1)).fill(0);
  if (!values.length) return counts;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    let idx = counts.length - 1;
    for (let j = 0; j < bins.length - 1; j += 1) {
      if (v >= bins[j] && v < bins[j + 1]) {
        idx = j;
        break;
      }
    }
    counts[idx] += 1;
  }
  return counts;
}

function normalizeProb(counts) {
  const total = counts.reduce((s, x) => s + x, 0);
  if (total <= 0) return counts.map(() => 0);
  return counts.map((x) => x / total);
}

function klDiv(p, q) {
  let s = 0;
  for (let i = 0; i < p.length; i += 1) {
    const pi = p[i];
    const qi = q[i];
    if (pi <= 0) continue;
    const qSafe = qi <= 0 ? 1e-12 : qi;
    s += pi * Math.log(pi / qSafe);
  }
  return s;
}

function jensenShannon(p, q) {
  if (!p.length || p.length !== q.length) return 0;
  const m = p.map((x, i) => 0.5 * (x + q[i]));
  return 0.5 * klDiv(p, m) + 0.5 * klDiv(q, m);
}

function lag1Correlation(series, start, end) {
  const s = Math.max(1, start);
  const e = Math.max(s, end);
  const x = [];
  const y = [];
  for (let i = s; i <= e; i += 1) {
    x.push(series[i - 1] || 0);
    y.push(series[i] || 0);
  }
  if (x.length < 3) return 0;
  const mx = mean(x);
  const my = mean(y);
  if (mx === null || my === null) return 0;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < x.length; i += 1) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  if (dx2 <= 0 || dy2 <= 0) return 0;
  return num / Math.sqrt(dx2 * dy2);
}

function buildEntropyBins(multipliers) {
  const qs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97];
  const points = [1];
  for (let i = 0; i < qs.length; i += 1) {
    const qv = quantile(multipliers, qs[i]);
    if (Number.isFinite(qv)) points.push(qv);
  }
  const maxV = quantile(multipliers, 0.999);
  if (Number.isFinite(maxV)) points.push(maxV * 1.05);
  points.push(Number.POSITIVE_INFINITY);
  const uniq = [];
  for (let i = 0; i < points.length; i += 1) {
    const v = points[i];
    if (!Number.isFinite(v) && v !== Number.POSITIVE_INFINITY) continue;
    if (!uniq.length || v > uniq[uniq.length - 1]) uniq.push(v);
  }
  if (uniq.length < 4) return [1, 2, 3, 5, 10, Number.POSITIVE_INFINITY];
  uniq[0] = 1;
  uniq[uniq.length - 1] = Number.POSITIVE_INFINITY;
  return uniq;
}

function buildGlobalState(cleanRounds, cfg) {
  const multipliers = cleanRounds.map((r) => Number(r.multiplier));
  const roundIds = cleanRounds.map((r) => Number(r.roundId));
  const n = cleanRounds.length;
  const whiteCutRaw = quantile(multipliers, cfg.whiteCutQuantile);
  const reboundCutRaw = quantile(multipliers, cfg.reboundCutQuantile);
  const whiteCut = clamp(whiteCutRaw, 1.6, 4.0);
  const reboundCut = clamp(Math.max(reboundCutRaw, whiteCut + 0.4), whiteCut + 0.4, Math.max(whiteCut + 0.4, 1000));
  const logM = multipliers.map((m) => Math.log(Math.max(1, m)));
  const logPrefix = buildPrefix(logM);
  const logSqPrefix = buildPrefix(logM.map((v) => v * v));
  const white = buildWhiteFlags(multipliers, whiteCut);
  const bins = buildEntropyBins(multipliers);
  return {
    rounds: cleanRounds,
    multipliers,
    roundIds,
    n,
    whiteCut,
    reboundCut,
    whiteFlags: white.flags,
    whiteRun: white.run,
    whitePrefix: white.prefix,
    logM,
    logPrefix,
    logSqPrefix,
    entropyBins: bins,
  };
}

function makeTargetState(state, target) {
  const hits = state.multipliers.map((m) => (m >= target ? 1 : 0));
  const hitPrefix = buildPrefix(hits);
  const gap = computeGapSeries(hits);
  return {
    target,
    hits,
    hitPrefix,
    gap: gap.gap,
    hitIdx: gap.hitIdx,
    intervals: gap.intervals,
    intervalPrefix: gap.intervalPrefix,
    intervalSqPrefix: gap.intervalSqPrefix,
    featureCache: new Map(),
  };
}

function rateInWindow(prefix, idx, window) {
  const right = idx;
  const left = Math.max(0, right - window + 1);
  const n = right - left + 1;
  if (n <= 0) return 0;
  return windowSum(prefix, left, right) / n;
}

function safeRateInWindow(prefix, idx, window) {
  const i = Number(idx);
  if (!Number.isFinite(i) || i < 0) return 0;
  return rateInWindow(prefix, i, window);
}

function volatilityContextAt(state, idx, shortW, longW) {
  const i = clamp(idx, 0, state.n - 1);
  const sw = Math.max(6, Math.round(shortW));
  const lw = Math.max(sw + 4, Math.round(longW));
  const shortL = Math.max(0, i - sw + 1);
  const longL = Math.max(0, i - lw + 1);
  const shortN = i - shortL + 1;
  const longN = i - longL + 1;
  const shortMean = shortN > 0 ? (windowSum(state.logPrefix, shortL, i) / shortN) : 0;
  const longMean = longN > 0 ? (windowSum(state.logPrefix, longL, i) / longN) : shortMean;
  const shortSq = shortN > 0 ? (windowSum(state.logSqPrefix, shortL, i) / shortN) : 0;
  const longSq = longN > 0 ? (windowSum(state.logSqPrefix, longL, i) / longN) : 0;
  const shortStd = Math.sqrt(Math.max(0, shortSq - (shortMean * shortMean)));
  const longStd = Math.sqrt(Math.max(0, longSq - (longMean * longMean)));
  const denom = Math.max(1e-6, longStd);
  const expansion = clamp((shortStd - longStd) / denom, 0, 1);
  const lowVolatility = clamp(1 - expansion, 0, 1);
  return {
    shortStd,
    longStd,
    expansion,
    lowVolatility,
  };
}

function calibratePreconditionThresholds(state, targetState, cfg, calibrationEnd) {
  const end = Math.min(state.n - 1, Math.max(0, Math.round(calibrationEnd)));
  const minNeeded = Math.max(cfg.preconditionCalibrationMin, cfg.featureLongWindow);
  if (end < minNeeded) {
    return {
      sample: 0,
      whiteRegimeThreshold: 0.62,
      releaseThreshold: Number.POSITIVE_INFINITY,
      momentumThreshold: Number.POSITIVE_INFINITY,
      releaseRunThreshold: Math.max(2, Math.round(quantile(state.whiteRun.filter((x) => x > 0), 0.80) || 2)),
    };
  }

  const whiteWindow = Math.max(8, cfg.preconditionWhiteWindow);
  const volShort = Math.max(6, cfg.preconditionVolShortWindow);
  const volLong = Math.max(volShort + 4, cfg.preconditionVolLongWindow);
  const whiteRuns = [];
  for (let i = 1; i <= end; i += 1) {
    const run = Number(state.whiteRun[i] || 0);
    if (run > 0) whiteRuns.push(run);
  }
  const runQ = quantile(whiteRuns, cfg.releaseRunThresholdQuantile);
  const releaseRunThreshold = Math.max(2, Math.round(Number.isFinite(runQ) ? runQ : 2));

  const whiteScores = [];
  const releaseScores = [];
  const momentumScores = [];
  for (let i = 1; i <= end; i += 1) {
    const volCtx = volatilityContextAt(state, i, volShort, volLong);
    const currentWhiteRate = safeRateInWindow(state.whitePrefix, i, whiteWindow);
    const previousWhiteRate = safeRateInWindow(state.whitePrefix, i - whiteWindow, whiteWindow);
    const whiteRun = Number(state.whiteRun[i] || 0);
    const whiteRunNormalized = clamp(whiteRun / Math.max(1, releaseRunThreshold), 0, 1);
    const whiteRegimeScore = (
      currentWhiteRate * 0.60 +
      whiteRunNormalized * 0.25 +
      volCtx.lowVolatility * 0.15
    );

    const releaseScore = (
      (whiteRun > releaseRunThreshold ? 1 : 0) *
      volCtx.expansion *
      Math.max(0, previousWhiteRate - currentWhiteRate)
    );

    const recentHighHitRate = safeRateInWindow(targetState.hitPrefix, i, cfg.b2bMomentumWindowShort);
    const longTermHighHitRate = safeRateInWindow(targetState.hitPrefix, i, cfg.b2bMomentumWindowLong);
    const momentumScore = (
      (recentHighHitRate - longTermHighHitRate) * 0.70 +
      volCtx.expansion * 0.30
    );

    whiteScores.push(whiteRegimeScore);
    if (releaseScore > 0) releaseScores.push(releaseScore);
    momentumScores.push(momentumScore);
  }

  const whiteRegimeThresholdQ = quantile(whiteScores, cfg.whiteRegimeThresholdQuantile);
  const releaseThresholdQ = quantile(releaseScores, cfg.releaseThresholdQuantile);
  const momentumThresholdQ = quantile(momentumScores, cfg.momentumThresholdQuantile);
  const whiteFallback = clamp(
    (Number(mean(whiteScores) || 0.5) + (Number(stddev(whiteScores) || 0) * 0.20)),
    0.45,
    0.90
  );

  return {
    sample: whiteScores.length,
    whiteRegimeThreshold: Number.isFinite(whiteRegimeThresholdQ) ? whiteRegimeThresholdQ : whiteFallback,
    releaseThreshold: Number.isFinite(releaseThresholdQ) ? Math.max(0, releaseThresholdQ) : Number.POSITIVE_INFINITY,
    momentumThreshold: Number.isFinite(momentumThresholdQ) ? momentumThresholdQ : Number.POSITIVE_INFINITY,
    releaseRunThreshold,
  };
}

function classifyPreconditionState({
  whiteDominant,
  releasePhase,
  momentumPhase,
  entropyBlocked,
  whiteRegimeScore,
  whiteRegimeThreshold,
  releaseScore,
  releaseThreshold,
  momentumScore,
  momentumThreshold,
}) {
  if (whiteDominant) {
    return {
      state: 'WHITE_DOMINANT',
      explanation: `White regime score ${roundNum(whiteRegimeScore, 4)} > ${roundNum(whiteRegimeThreshold, 4)}.`,
    };
  }
  if (releasePhase) {
    return {
      state: 'RELEASE_PHASE',
      explanation: `Release score ${roundNum(releaseScore, 5)} > ${roundNum(releaseThreshold, 5)}.`,
    };
  }
  if (momentumPhase) {
    return {
      state: 'MOMENTUM',
      explanation: `Momentum score ${roundNum(momentumScore, 5)} > ${roundNum(momentumThreshold, 5)}.`,
    };
  }
  if (entropyBlocked) {
    return {
      state: 'NEUTRAL',
      explanation: 'Entropy is random-like; pre-decision filter blocked.',
    };
  }
  return {
    state: 'NEUTRAL',
    explanation: 'No release or momentum pre-condition detected.',
  };
}

function featureAtIndex(state, targetState, idx, cfg) {
  if (targetState.featureCache.has(idx)) return targetState.featureCache.get(idx);
  const n = state.n;
  const i = clamp(idx, 0, n - 1);
  const shortW = cfg.featureShortWindow;
  const longW = cfg.featureLongWindow;
  const volW = cfg.featureVolWindow;

  const shortL = Math.max(0, i - shortW + 1);
  const longL = Math.max(0, i - longW + 1);
  const volL = Math.max(0, i - volW + 1);

  const shortLogMean = windowMean(state.logPrefix, shortL, i) ?? 0;
  const longLogMean = windowMean(state.logPrefix, longL, i) ?? shortLogMean;
  const volN = i - volL + 1;
  const volMean = windowMean(state.logPrefix, volL, i) ?? shortLogMean;
  const volSq = volN > 0 ? windowSum(state.logSqPrefix, volL, i) / volN : 0;
  const vol = Math.sqrt(Math.max(0, volSq - volMean * volMean));

  const lowRateShort = (windowSum(state.whitePrefix, shortL, i) || 0) / Math.max(1, i - shortL + 1);
  const lowRateLong = (windowSum(state.whitePrefix, longL, i) || 0) / Math.max(1, i - longL + 1);
  const hitRateLong = rateInWindow(targetState.hitPrefix, i, longW);
  const gapNorm = Math.min(1, (targetState.gap[i] || 0) / Math.max(10, longW));
  const whiteRunNorm = Math.min(1, (state.whiteRun[i] || 0) / 20);
  const prev = Math.log(Math.max(1, state.multipliers[i] || 1));
  const prev2 = Math.log(Math.max(1, state.multipliers[i - 1] || state.multipliers[i] || 1));

  const f = [
    prev,
    prev2,
    shortLogMean,
    longLogMean,
    vol,
    lowRateShort,
    lowRateLong,
    hitRateLong,
    gapNorm,
    whiteRunNorm,
  ];
  targetState.featureCache.set(idx, f);
  return f;
}

function distance(a, b, weights) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = (a[i] || 0) - (b[i] || 0);
    const w = Number(weights?.[i]);
    const ww = Number.isFinite(w) && w > 0 ? w : 1;
    s += ww * d * d;
  }
  return Math.sqrt(s);
}

function hazardPredictAt(state, targetState, idx, cfg) {
  if (idx < 1) return null;
  const hitsSoFar = targetState.hitPrefix[idx + 1];
  if (hitsSoFar < 3) return null;

  const globalRate = hitsSoFar / (idx + 1);
  const recentRate = rateInWindow(targetState.hitPrefix, idx, cfg.hazardRecentWindow);
  let p = 0.65 * recentRate + 0.35 * globalRate;

  const m = lowerBound(targetState.hitIdx, idx + 1);
  if (m >= 2) {
    const cnt = m - 1;
    const sum = targetState.intervalPrefix[cnt];
    const sumSq = targetState.intervalSqPrefix[cnt];
    const gapMean = sum / cnt;
    const gapVar = Math.max(0, (sumSq / cnt) - (gapMean * gapMean));
    const gapStd = Math.sqrt(gapVar) || Math.max(1, gapMean * 0.35);
    const currentGap = targetState.gap[idx] || 0;
    const z = (currentGap - gapMean) / Math.max(1e-6, gapStd);
    const lift = tanh(z);
    p *= (1 + Math.min(cfg.hazardGapInfluence, 0.05) * lift);
  }

  return clamp(p, 0, 1);
}

function knnPredictAt(state, targetState, idx, cfg) {
  if (idx < cfg.minTrainingRounds) return null;
  const query = featureAtIndex(state, targetState, idx, cfg);
  const left = Math.max(cfg.minTrainingRounds - 1, idx - cfg.knnLookback);
  const right = idx - 1;
  if (right < left) return null;

  const range = right - left + 1;
  const stride = Math.max(1, Math.ceil(range / cfg.knnMaxCandidates));
  const pairs = [];
  for (let j = left; j <= right; j += stride) {
    if (j + 1 >= state.n) break;
    const candidate = featureAtIndex(state, targetState, j, cfg);
    const d = distance(query, candidate, cfg.featureWeights);
    const y = targetState.hits[j + 1] ? 1 : 0;
    const age = Math.max(0, idx - j);
    const tDecay = Math.exp(-age / Math.max(1, cfg.knnTemporalDecayRounds));
    pairs.push({ d, y, tDecay, age });
  }
  if (pairs.length < cfg.knnKMin) return null;

  pairs.sort((a, b) => a.d - b.d);
  const k = clamp(Math.round(Math.sqrt(pairs.length)), cfg.knnKMin, Math.min(cfg.knnKMax, pairs.length));
  let wSum = 0;
  let ySum = 0;
  let dSum = 0;
  for (let i = 0; i < k; i += 1) {
    const item = pairs[i];
    const w = (1 / Math.max(1e-6, item.d)) * item.tDecay;
    wSum += w;
    ySum += w * item.y;
    dSum += item.d;
  }
  if (wSum <= 0) return null;
  const p = clamp(ySum / wSum, 0, 1);
  const avgD = dSum / k;
  const dQ25 = quantile(pairs.map((x) => x.d), 0.25);
  const distanceThreshold = Number.isFinite(dQ25) ? (dQ25 * cfg.knnDistanceSlack) : null;
  const similarityPass = Number.isFinite(distanceThreshold) ? (avgD <= distanceThreshold) : true;
  const similarityScore = Number.isFinite(distanceThreshold)
    ? clamp(1 - (avgD / Math.max(1e-6, distanceThreshold)), 0, 1)
    : 0.5;
  const sampleScore = clamp(pairs.length / Math.max(cfg.knnKMax, 1), 0, 1);
  const support = 1 / (1 + avgD);
  const reliability = clamp(0.45 * support + 0.35 * similarityScore + 0.20 * sampleScore, 0, 1);
  if (!similarityPass || reliability < cfg.knnMinReliability) return null;
  return {
    p,
    support: clamp(support, 0, 1),
    reliability: roundNum(reliability, 6),
    similarityScore: roundNum(similarityScore, 6),
    avgDistance: roundNum(avgD, 6),
    distanceThreshold: roundNum(distanceThreshold, 6),
    k,
    candidates: pairs.length,
  };
}

function makeRollingBrier(maxLen) {
  const q = [];
  let sum = 0;
  return {
    add(v) {
      const n = Number(v);
      if (!Number.isFinite(n)) return;
      q.push(n);
      sum += n;
      if (q.length > maxLen) {
        const old = q.shift();
        sum -= old;
      }
    },
    mean() {
      if (!q.length) return null;
      return sum / q.length;
    },
    count() {
      return q.length;
    },
  };
}

function blendWeightsFromBrier(brier, cfg, availability, tuning = {}) {
  const eps = cfg.modelErrorEps;
  const prior = 0.25; // Neutral Bernoulli uncertainty baseline.
  const vals = {
    baseline: Number.isFinite(brier.baseline) ? brier.baseline : prior,
    hazard: Number.isFinite(brier.hazard) ? brier.hazard : prior,
    knn: Number.isFinite(brier.knn) ? brier.knn : prior,
  };

  const w = {
    baseline: 1 / Math.max(eps, vals.baseline),
    hazard: availability.hazard ? (1 / Math.max(eps, vals.hazard)) : 0,
    knn: availability.knn ? (1 / Math.max(eps, vals.knn)) : 0,
  };

  let total = w.baseline + w.hazard + w.knn;
  if (total <= 0) return { baseline: 1, hazard: 0, knn: 0 };
  w.baseline /= total;
  w.hazard /= total;
  w.knn /= total;

  // Hazard is only retained if it has proven value.
  if (tuning.hazardDisabled) {
    w.hazard = 0;
    total = w.baseline + w.knn;
    if (total <= 0) return { baseline: 1, hazard: 0, knn: 0 };
    w.baseline /= total;
    w.knn /= total;
    return { baseline: w.baseline, hazard: 0, knn: w.knn };
  }

  if (w.hazard > cfg.maxHazardBlend) {
    const excess = w.hazard - cfg.maxHazardBlend;
    w.hazard = cfg.maxHazardBlend;
    const redistributeBase = w.baseline + w.knn;
    if (redistributeBase > 0) {
      w.baseline += excess * (w.baseline / redistributeBase);
      w.knn += excess * (w.knn / redistributeBase);
    } else {
      w.baseline += excess;
    }
  }

  if (w.baseline < cfg.minBaselineBlend) {
    const need = cfg.minBaselineBlend - w.baseline;
    w.baseline = cfg.minBaselineBlend;
    const donor = w.hazard + w.knn;
    if (donor > 0) {
      w.hazard -= need * (w.hazard / donor);
      w.knn -= need * (w.knn / donor);
    }
  }

  total = w.baseline + w.hazard + w.knn;
  if (total <= 0) return { baseline: 1, hazard: 0, knn: 0 };
  w.baseline /= total;
  w.hazard /= total;
  w.knn /= total;

  // Contextual KNN reliability only nudges blend weight; it never dominates.
  if (Number.isFinite(tuning.knnReliability)) {
    const reliability = clamp(tuning.knnReliability, 0, 1);
    w.knn *= reliability;
  }

  // Trending regime can modestly favor KNN if it remains reliable.
  if (Number.isFinite(tuning.trendingBoost) && tuning.trendingBoost > 0) {
    w.knn *= (1 + clamp(tuning.trendingBoost, 0, 0.20));
  }

  total = w.baseline + w.hazard + w.knn;
  if (total <= 0) return { baseline: 1, hazard: 0, knn: 0 };
  w.baseline /= total;
  w.hazard /= total;
  w.knn /= total;
  return w;
}

function createWhiteTransitionStats(maxRun = 40) {
  return {
    maxRun,
    obs: new Array(maxRun + 1).fill(0),
    cont: new Array(maxRun + 1).fill(0),
    reb: new Array(maxRun + 1).fill(0),
    totalObs: 0,
    totalCont: 0,
    totalReb: 0,
  };
}

function whiteUpdate(stats, runLen, isContinue, isRebound) {
  if (!runLen || runLen <= 0) return;
  const r = Math.min(stats.maxRun, Math.max(1, Math.round(runLen)));
  stats.obs[r] += 1;
  stats.totalObs += 1;
  if (isContinue) {
    stats.cont[r] += 1;
    stats.totalCont += 1;
  }
  if (isRebound) {
    stats.reb[r] += 1;
    stats.totalReb += 1;
  }
}

function whiteEstimate(stats, runLen) {
  const r = Math.min(stats.maxRun, Math.max(1, Math.round(runLen || 0)));
  const localObs = stats.obs[r] || 0;
  const globalCont = stats.totalObs > 0 ? stats.totalCont / stats.totalObs : null;
  const globalReb = stats.totalObs > 0 ? stats.totalReb / stats.totalObs : null;
  if (localObs <= 0) {
    return {
      continueProb: globalCont,
      reboundProb: globalReb,
      sample: stats.totalObs,
      reliable: stats.totalObs >= 20,
    };
  }
  return {
    continueProb: stats.cont[r] / localObs,
    reboundProb: stats.reb[r] / localObs,
    sample: localObs,
    reliable: localObs >= 8,
  };
}

function deriveEntropyThresholds(state, targetState, cfg, trainEnd) {
  const w = cfg.entropyRecentWindow;
  const step = Math.max(1, cfg.entropyStep);
  const base = 1 / targetState.target;
  const jsSeries = [];
  const driftSeries = [];
  const lagSeries = [];

  if (trainEnd < (2 * w)) return { js: null, drift: null, lag: null, sample: 0 };

  for (let end = (2 * w); end <= trainEnd; end += step) {
    const recent = state.multipliers.slice(end - w + 1, end + 1);
    const prev = state.multipliers.slice(end - (2 * w) + 1, end - w + 1);
    if (!recent.length || !prev.length) continue;
    const pr = normalizeProb(histogram(recent, state.entropyBins));
    const pp = normalizeProb(histogram(prev, state.entropyBins));
    jsSeries.push(jensenShannon(pr, pp));
    driftSeries.push(Math.abs(rateInWindow(targetState.hitPrefix, end, w) - base));
    lagSeries.push(Math.abs(lag1Correlation(targetState.hits, end - w + 1, end)));
  }

  if (!jsSeries.length) return { js: null, drift: null, lag: null, sample: 0 };
  return {
    js: quantile(jsSeries, cfg.entropyQuantile),
    drift: quantile(driftSeries, cfg.entropyQuantile),
    lag: quantile(lagSeries, cfg.entropyQuantile),
    sample: jsSeries.length,
  };
}

function entropyAtIndex(state, targetState, idx, thresholds, cfg) {
  const w = cfg.entropyRecentWindow;
  if (idx < (2 * w)) {
    return {
      js: null,
      drift: null,
      lagCorr: null,
      randomLike: false,
      disabled: false,
      thresholdSample: thresholds.sample || 0,
    };
  }

  const recent = state.multipliers.slice(idx - w + 1, idx + 1);
  const prev = state.multipliers.slice(idx - (2 * w) + 1, idx - w + 1);
  const pr = normalizeProb(histogram(recent, state.entropyBins));
  const pp = normalizeProb(histogram(prev, state.entropyBins));
  const js = jensenShannon(pr, pp);
  const drift = Math.abs(rateInWindow(targetState.hitPrefix, idx, w) - (1 / targetState.target));
  const lagCorr = Math.abs(lag1Correlation(targetState.hits, idx - w + 1, idx));

  const hasThresholds = Number.isFinite(thresholds.js) && Number.isFinite(thresholds.drift) && Number.isFinite(thresholds.lag);
  const randomLike = hasThresholds
    ? (js <= thresholds.js && drift <= thresholds.drift && lagCorr <= thresholds.lag)
    : false;

  return {
    js: roundNum(js, 8),
    drift: roundNum(drift, 8),
    lagCorr: roundNum(lagCorr, 8),
    randomLike,
    disabled: Boolean(cfg.entropyAutoDisable && randomLike),
    thresholdSample: thresholds.sample || 0,
  };
}

function confidenceFromComponents(p, baseline, weights, entropyInfo, extras = {}) {
  const edge = Math.abs((p ?? baseline ?? 0) - (baseline ?? 0));
  const edgeScore = clamp(edge / Math.max(0.01, baseline || 0.01), 0, 1);
  const modelBalance = clamp((weights.knn || 0) + (weights.hazard || (0.5 * (weights.baseline || 0))), 0, 1);
  const agreement = clamp(Number(extras.modelAgreement ?? 0.5), 0, 1);
  const edgeConfidenceScore = clamp(Number(extras.edgeConfidenceScore ?? 1), 0, 1.2);
  const entropyTrend = Number(extras.entropyTrend || 0);
  const disagreement = clamp(Number(extras.modelDisagreement || 0), 0, 1);
  const disagreementPenalty = clamp(1 - disagreement, 0.45, 1);
  const entropyPenalty = entropyInfo?.disabled ? 0.35 : 1.0;
  const entropyNoisePenalty = entropyInfo?.randomLike ? 0.65 : 1.0;
  const entropySpikePenalty = extras.entropySpike ? 0.70 : 1.0;
  const entropyTrendPenalty = entropyTrend > 0
    ? clamp(1 - (entropyTrend * 4), 0.45, 1)
    : 1;
  const core = (
    0.35 * edgeScore +
    0.25 * modelBalance +
    0.25 * agreement +
    0.15 * clamp(edgeConfidenceScore, 0, 1)
  );
  return clamp(core * disagreementPenalty * entropyPenalty * entropyNoisePenalty * entropySpikePenalty * entropyTrendPenalty, 0, 1);
}

function evaluateExistingLock(lock, hitRoundIds, currentRound) {
  if (!lock) return { resolved: false, status: 'idle', outcome: null, hitRound: null };
  const lifecycle = String(lock?.eta?.lockStatus || '').toUpperCase();
  if (lifecycle === 'IDLE' || Boolean(lock?.suspended)) {
    return { resolved: false, status: 'idle', outcome: null, hitRound: null };
  }
  const lo = Number(lock.lo);
  const hi = Number(lock.hi);
  const madeRaw = Number(lock.roundWhenMade ?? (lo - 1));
  const made = Number.isFinite(madeRaw) ? madeRaw : (Number.isFinite(lo) ? lo - 1 : currentRound);

  const lb = lowerBound(hitRoundIds, made + 1);
  const ub = lowerBound(hitRoundIds, currentRound + 1);
  const firstHit = lb < ub ? hitRoundIds[lb] : null;

  if (firstHit !== null) {
    if (firstHit < lo) return { resolved: true, status: 'resolved', outcome: 'early', hitRound: firstHit };
    if (firstHit <= hi) return { resolved: true, status: 'resolved', outcome: 'win', hitRound: firstHit };
  }
  // Close unresolved window immediately at hi boundary so history persists without a one-round lag.
  if (currentRound >= hi) return { resolved: true, status: 'resolved', outcome: 'loss', hitRound: null };
  if (currentRound >= lo) return { resolved: false, status: 'window-open', outcome: null, hitRound: null };
  return { resolved: false, status: 'locked', outcome: null, hitRound: null };
}

function confidenceBand(score) {
  const s = clamp(score ?? 0, 0, 1);
  if (s >= 0.8) return 'VERY HIGH';
  if (s >= 0.6) return 'HIGH';
  if (s >= 0.35) return 'MED';
  if (s >= 0.15) return 'LOW';
  return 'NONE';
}

function buildUiTarget(target, lock, status, currentRound, previousOutcome, timingHint = null) {
  const lo = Number(lock?.lo);
  const hi = Number(lock?.hi);
  const eta = lock?.eta || {};
  const aheadLo = Number.isFinite(lo) ? Math.max(0, lo - currentRound) : null;
  const aheadHi = Number.isFinite(hi) ? Math.max(0, hi - currentRound) : null;
  const lockCreatedAtRound = Number(lock?.roundWhenMade ?? eta.lockCreatedAtRound ?? currentRound);
  const roundsSinceLock = Math.max(0, currentRound - lockCreatedAtRound);
  const lockStatus = (
    status === 'window-open' ? 'ACTIVE'
      : status === 'locked' ? 'LOCKED'
        : status === 'resolved' ? 'RESOLVED'
          : 'IDLE'
  );
  return {
    target,
    targetLabel: `${target}x`,
    // Keep legacy status values for existing frontend mapping.
    status: status === 'idle' ? 'waiting' : status,
    lockStatus,
    confidence: clamp(Number(lock?.confidence ?? eta.aiConfidence ?? 0), 0, 1),
    confidenceBand: confidenceBand(lock?.confidence ?? eta.aiConfidence ?? 0),
    state: eta.preconditionState ?? 'NEUTRAL',
    explanation: eta.preconditionExplanation ?? '',
    window: {
      lo: Number.isFinite(lo) ? lo : null,
      hi: Number.isFinite(hi) ? hi : null,
      span: (!Number.isFinite(lo) || !Number.isFinite(hi)) ? null : Math.max(1, hi - lo + 1),
      roundsUntilWindow: aheadLo,
      roundsLeftInWindow: aheadHi,
      aheadLo,
      aheadHi,
    },
    signals: {
      pHit1: Number(eta.pHit1 ?? 0),
      pHitSoon: Number(eta.pHitSoon ?? eta.pHit1 ?? 0),
      quickHit: Number(eta.quickHit ?? eta.pHit1 ?? 0),
      whiteClusterRisk: eta.whiteClusterRisk ?? null,
      whiteClusterRelease: eta.whiteClusterRelease ?? null,
      whiteClusterRun: Number(eta.whiteClusterRun ?? 0),
      b2bImmRate: eta.b2bImmRate ?? null,
      b2bNearRate: eta.b2bNearRate ?? null,
      hazardP1: eta.hazardP1 ?? null,
      knnP1: eta.knnP1 ?? null,
      knnSupport: eta.knnSupport ?? null,
      baseRate: eta.baseRate ?? null,
      edge: eta.edge ?? null,
      ev: eta.ev ?? null,
      entropyRandomLike: eta.entropyRandomLike ?? null,
      preconditionState: eta.preconditionState ?? 'NEUTRAL',
      preconditionExplanation: eta.preconditionExplanation ?? '',
      preconditionPass: Boolean(eta.preconditionPass),
      whiteRegimeScore: eta.whiteRegimeScore ?? null,
      releaseScore: eta.releaseScore ?? null,
      momentumScore: eta.momentumScore ?? null,
    },
    debug: {
      lockCreatedAtRound,
      lockStatus,
      isMutable: false,
      roundsSinceLock,
      pFinal: Number(eta.pFinal ?? eta.pHit1 ?? 0),
      edge: Number(eta.edge ?? 0),
      ev: Number(eta.ev ?? 0),
      confidence: Number(lock?.confidence ?? eta.aiConfidence ?? 0),
      edgeConfidenceScore: Number(eta.edgeConfidenceScore ?? 0),
      entropyRandomLike: Boolean(eta.entropyRandomLike),
      entropyDisabled: Boolean(eta.entropyDisabled),
      modelWeights: {
        baseline: Number(eta?.blend?.baseline ?? 0),
        hazard: Number(eta?.blend?.hazard ?? 0),
        knn: Number(eta?.blend?.knn ?? 0),
      },
    },
    previousOutcome: previousOutcome || null,
    timingHint: timingHint || null,
  };
}

function makeRollingMeanStd(maxLen) {
  const q = [];
  let sum = 0;
  let sumSq = 0;
  return {
    add(v) {
      const n = Number(v);
      if (!Number.isFinite(n)) return;
      q.push(n);
      sum += n;
      sumSq += n * n;
      if (q.length > maxLen) {
        const old = q.shift();
        sum -= old;
        sumSq -= old * old;
      }
    },
    mean() {
      if (!q.length) return null;
      return sum / q.length;
    },
    std() {
      if (q.length < 2) return null;
      const m = sum / q.length;
      return Math.sqrt(Math.max(0, (sumSq / q.length) - (m * m)));
    },
    count() {
      return q.length;
    },
    values() {
      return q.slice();
    },
  };
}

function makeRollingEdgeValidation(maxLen) {
  const errors = [];
  const predicted = [];
  const realized = [];
  let sumErr = 0;
  let sumPred = 0;
  let sumReal = 0;
  let negCount = 0;
  return {
    add(predEV, realEV) {
      const p = Number(predEV);
      const r = Number(realEV);
      if (!Number.isFinite(p) || !Number.isFinite(r)) return;
      const e = r - p;
      errors.push(e);
      predicted.push(p);
      realized.push(r);
      sumErr += e;
      sumPred += p;
      sumReal += r;
      if (e < 0) negCount += 1;
      if (errors.length > maxLen) {
        const oldE = errors.shift();
        const oldP = predicted.shift();
        const oldR = realized.shift();
        sumErr -= oldE;
        sumPred -= oldP;
        sumReal -= oldR;
        if (oldE < 0) negCount -= 1;
      }
    },
    metrics() {
      const n = errors.length;
      if (!n) {
        return {
          count: 0,
          predictedEV: null,
          realizedEV: null,
          edgeError: null,
          negativeErrorRatio: null,
          score: 1,
        };
      }
      const meanErr = sumErr / n;
      const meanPred = sumPred / n;
      const meanReal = sumReal / n;
      const negRatio = negCount / n;
      const scale = Math.max(0.20, Math.abs(meanPred) + 0.20);
      let score = 1 + (meanErr / scale);
      if (negRatio > 0.55) {
        score *= clamp(1 - (negRatio - 0.55) * 0.8, 0.4, 1);
      }
      score = clamp(score, 0, 1);
      return {
        count: n,
        predictedEV: meanPred,
        realizedEV: meanReal,
        edgeError: meanErr,
        negativeErrorRatio: negRatio,
        score,
      };
    },
  };
}

function kellyFraction(p, target, confidence, maxRisk) {
  const pp = clamp(Number(p), 0, 1);
  const t = Number(target);
  if (!Number.isFinite(t) || t <= 1) return 0;
  const raw = (pp * t - 1) / (t - 1);
  return clamp(raw * clamp(Number(confidence), 0, 1), 0, maxRisk);
}

function sharpeLike(values) {
  if (!Array.isArray(values) || values.length < 2) return null;
  const m = mean(values);
  const s = stddev(values, m);
  if (!Number.isFinite(m) || !Number.isFinite(s) || s <= 0) return null;
  return m / s;
}

function classifyRegime(ctx, cfg) {
  const random = Boolean(ctx.entropy?.randomLike);
  if (random) {
    return { label: 'RANDOM', randomLike: true, trendingBias: 0, windowMult: 1.2, evAdjust: 0 };
  }
  const volSpike = Number(ctx.volShift || 0);
  const lag = Math.abs(Number(ctx.lagCorr || 0));
  const drift = Math.abs(Number(ctx.hitRateDrift || 0));
  const clusterScore = Number(ctx.clusterScore || 0);
  const entropyTrend = Number(ctx.entropyTrend || 0);

  if (volSpike > cfg.regimeVolatilityShiftThreshold || entropyTrend > 0.02) {
    return { label: 'VOLATILE', randomLike: false, trendingBias: -0.02, windowMult: 1.25, evAdjust: 0 };
  }
  if (clusterScore > cfg.regimeClusterThreshold) {
    return {
      label: 'CLUSTERED',
      randomLike: false,
      trendingBias: 0.0,
      windowMult: 1.12,
      evAdjust: -Math.abs(cfg.clusteredEvTolerance || 0),
    };
  }
  if (lag > cfg.regimeLagTrendThreshold || drift > cfg.regimeDriftTrendThreshold) {
    return { label: 'TRENDING', randomLike: false, trendingBias: 0.08, windowMult: 1.0, evAdjust: 0 };
  }
  return { label: 'RANDOM', randomLike: true, trendingBias: 0, windowMult: 1.15, evAdjust: 0 };
}

function geometricQuantile(p, q) {
  const pp = clamp(p, 1e-6, 1 - 1e-6);
  const qq = clamp(q, 1e-6, 1 - 1e-6);
  return Math.max(1, Math.ceil(Math.log(1 - qq) / Math.log(1 - pp)));
}

function estimateAheadBand(state, targetResult) {
  const live = targetResult.live;
  const ts = targetResult.targetState;
  const intervals = ts.intervals || [];
  const gapNow = Number(ts.gap[state.n - 1] || 0);
  const p = clamp(Number(live.pAdj || live.baselineP || (1 / targetResult.target)), 1e-6, 0.999);

  const geoLo = geometricQuantile(p, 0.20);
  const geoHi = geometricQuantile(p, 0.55);

  let lo = geoLo;
  let hi = Math.max(lo, geoHi);
  let q20 = null;
  let q50 = null;
  let q55 = null;
  let q75 = null;
  let q90 = null;
  let q95 = null;

  if (intervals.length >= 20) {
    q20 = quantile(intervals, 0.20);
    q50 = quantile(intervals, 0.50);
    q55 = quantile(intervals, 0.55);
    q75 = quantile(intervals, 0.75);
    q90 = quantile(intervals, 0.90);
    q95 = quantile(intervals, 0.95);
    lo = Math.max(1, Math.round(0.50 * geoLo + 0.50 * q20));
    hi = Math.max(lo, Math.round(0.45 * geoHi + 0.55 * q55));
  } else {
    q50 = quantile(intervals, 0.50);
    q75 = quantile(intervals, 0.75);
    q90 = quantile(intervals, 0.90);
    q95 = quantile(intervals, 0.95);
  }

  const ref = Number.isFinite(q50) ? q50 : Math.max(1, geoLo);
  const overdue = clamp((gapNow - ref) / Math.max(1, ref), -0.8, 1.5);
  // Keep gap influence weak to avoid gambler's-fallacy style drift.
  lo = Math.max(1, Math.round(lo * (1 - 0.08 * overdue)));
  hi = Math.max(lo, Math.round(hi * (1 - 0.06 * overdue)));

  const edgeNorm = clamp(
    (Number(live.edge || 0)) / Math.max(1e-6, Number(live.baselineP || (1 / targetResult.target))),
    -1,
    2
  );
  lo = Math.max(1, Math.round(lo * (1 - 0.12 * edgeNorm)));
  hi = Math.max(lo, Math.round(hi * (1 - 0.08 * edgeNorm)));

  if (live.entropy?.randomLike) {
    lo = Math.max(1, Math.round(lo * 1.10));
    hi = Math.max(lo, Math.round(hi * 1.30));
  }

  const intervalCap = Number.isFinite(q95)
    ? Math.max(hi, Math.round(q95 * 2.5))
    : Math.max(hi, Math.round(geoHi * 1.8));
  const absoluteCap = Math.max(40, intervalCap);
  hi = Math.min(absoluteCap, hi);
  lo = Math.min(lo, hi);

  return {
    aheadLo: Math.max(1, lo),
    aheadHi: Math.max(lo, hi),
    gapNow,
    q20: roundNum(q20, 3),
    q50: roundNum(q50, 3),
    q75: roundNum(q75, 3),
    q90: roundNum(q90, 3),
    q95: roundNum(q95, 3),
  };
}

function runTargetWalkForward(state, target, cfg) {
  const targetState = makeTargetState(state, target);
  const n = state.n;
  const evalStartBase = Math.max(
    cfg.minTrainingRounds,
    cfg.featureLongWindow + 2,
    cfg.entropyRecentWindow * 2
  );
  const evalStart = Math.max(evalStartBase, n - cfg.walkForwardWindow - 1);
  const evalEnd = n - 2;

  const brierHaz = makeRollingBrier(cfg.brierWindow);
  const brierKnn = makeRollingBrier(cfg.brierWindow);
  const brierBase = makeRollingBrier(cfg.brierWindow);
  const entropyBaseline = makeRollingMeanStd(Math.max(40, Math.floor(cfg.entropyRecentWindow / Math.max(1, cfg.entropyStep))));
  const edgeValidation = makeRollingEdgeValidation(cfg.edgeValidationWindow);

  const whiteStats = createWhiteTransitionStats();
  for (let j = 0; j < Math.max(0, evalStart - 1); j += 1) {
    const runLen = state.whiteRun[j] || 0;
    const nextIsWhite = Boolean(state.whiteFlags[j + 1]);
    const nextIsRebound = Number(state.multipliers[j + 1]) >= state.reboundCut;
    whiteUpdate(whiteStats, runLen, nextIsWhite, nextIsRebound);
  }

  const entropyThresholds = deriveEntropyThresholds(state, targetState, cfg, Math.max(evalStart, n - 2));
  const preconditionThresholds = calibratePreconditionThresholds(
    state,
    targetState,
    cfg,
    Math.max(cfg.minTrainingRounds, evalStart - 1)
  );

  let bets = 0;
  let wins = 0;
  let pnl = 0;
  let peak = 0;
  let maxDrawdown = 0;
  let evSum = 0;
  const rollingReturns = [];
  let edgePredCount = 0;
  let edgePredCorrect = 0;
  let hazardDisabled = false;

  function pushRollingReturn(v) {
    rollingReturns.push(v);
    if (rollingReturns.length > cfg.performanceWindow) rollingReturns.shift();
  }

  function maxDrawdownOf(values) {
    if (!Array.isArray(values) || !values.length) return null;
    let eq = 0;
    let peakEq = 0;
    let dd = 0;
    for (let i = 0; i < values.length; i += 1) {
      eq += values[i];
      peakEq = Math.max(peakEq, eq);
      dd = Math.max(dd, peakEq - eq);
    }
    return dd;
  }

  function predictAt(idx) {
    const baselineP = 1 / target;
    const entropy = entropyAtIndex(state, targetState, idx, entropyThresholds, cfg);
    const entropyMean = entropyBaseline.mean();
    const entropyStd = entropyBaseline.std();
    const entropyTrend = (Number.isFinite(entropy.js) && Number.isFinite(entropyMean))
      ? (entropy.js - entropyMean)
      : 0;
    const entropySpike = (
      Number.isFinite(entropyTrend) &&
      Number.isFinite(entropyStd) &&
      entropyStd > 0 &&
      entropyTrend > (entropyStd * cfg.entropySpikeStdMult)
    );

    const shortVolW = Math.max(12, Math.floor(cfg.featureVolWindow / 2));
    const longVolW = Math.max(shortVolW + 4, cfg.featureVolWindow);
    const shortL = Math.max(0, idx - shortVolW + 1);
    const longL = Math.max(0, idx - longVolW + 1);
    const shortN = idx - shortL + 1;
    const longN = idx - longL + 1;
    const shortMean = shortN > 0 ? (windowSum(state.logPrefix, shortL, idx) / shortN) : 0;
    const longMean = longN > 0 ? (windowSum(state.logPrefix, longL, idx) / longN) : shortMean;
    const shortSq = shortN > 0 ? (windowSum(state.logSqPrefix, shortL, idx) / shortN) : 0;
    const longSq = longN > 0 ? (windowSum(state.logSqPrefix, longL, idx) / longN) : 0;
    const shortStd = Math.sqrt(Math.max(0, shortSq - (shortMean * shortMean)));
    const longStd = Math.sqrt(Math.max(0, longSq - (longMean * longMean)));
    const volShift = longStd > 1e-6 ? ((shortStd - longStd) / longStd) : 0;

    // --- Pre-condition detectors (state detection, not direct outcome prediction) ---
    const whiteWindow = Math.max(8, cfg.preconditionWhiteWindow);
    const preVol = volatilityContextAt(
      state,
      idx,
      cfg.preconditionVolShortWindow,
      cfg.preconditionVolLongWindow
    );
    const currentWhiteRate = safeRateInWindow(state.whitePrefix, idx, whiteWindow);
    const previousWhiteRate = safeRateInWindow(state.whitePrefix, idx - whiteWindow, whiteWindow);
    const whiteRunNow = Number(state.whiteRun[idx] || 0);
    const whiteRunNormalized = clamp(
      whiteRunNow / Math.max(1, Number(preconditionThresholds.releaseRunThreshold || 1)),
      0,
      1
    );
    const whiteRegimeScore = (
      currentWhiteRate * 0.60 +
      whiteRunNormalized * 0.25 +
      preVol.lowVolatility * 0.15
    );
    const whiteDominant = whiteRegimeScore > Number(preconditionThresholds.whiteRegimeThreshold || Number.POSITIVE_INFINITY);

    const releaseScore = (
      (whiteRunNow > Number(preconditionThresholds.releaseRunThreshold || Number.POSITIVE_INFINITY) ? 1 : 0) *
      preVol.expansion *
      Math.max(0, previousWhiteRate - currentWhiteRate)
    );
    const releasePhase = releaseScore > Number(preconditionThresholds.releaseThreshold || Number.POSITIVE_INFINITY);

    const recentHitRate = rateInWindow(targetState.hitPrefix, idx, cfg.b2bMomentumWindowShort);
    const longHitRate = rateInWindow(targetState.hitPrefix, idx, cfg.b2bMomentumWindowLong);
    const hitRateDrift = recentHitRate - longHitRate;

    const momentumScore = (
      (recentHitRate - longHitRate) * 0.70 +
      preVol.expansion * 0.30
    );
    const momentumPhase = (
      momentumScore > Number(preconditionThresholds.momentumThreshold || Number.POSITIVE_INFINITY) &&
      !entropy.randomLike &&
      !entropy.disabled
    );

    const entropyBlocked = Boolean(entropy.randomLike || entropy.disabled);
    const signalBlocked = !releasePhase && !momentumPhase;
    const preconditionPass = !whiteDominant && !entropyBlocked && !signalBlocked;
    const precondition = classifyPreconditionState({
      whiteDominant,
      releasePhase,
      momentumPhase,
      entropyBlocked,
      whiteRegimeScore,
      whiteRegimeThreshold: preconditionThresholds.whiteRegimeThreshold,
      releaseScore,
      releaseThreshold: preconditionThresholds.releaseThreshold,
      momentumScore,
      momentumThreshold: preconditionThresholds.momentumThreshold,
    });

    const brierMeans = {
      baseline: brierBase.mean(),
      hazard: brierHaz.mean(),
      knn: brierKnn.mean(),
    };
    const hazardEligible = (
      brierHaz.count() >= cfg.hazardMinEvalCount &&
      Number.isFinite(brierMeans.hazard) &&
      Number.isFinite(brierMeans.baseline)
    );
    if (hazardEligible) {
      const tol = Math.max(0, cfg.hazardDisableTolerance);
      if (brierMeans.hazard >= (brierMeans.baseline + tol)) {
        hazardDisabled = true;
      } else if (brierMeans.hazard < brierMeans.baseline) {
        // Re-enable hazard when rolling evidence says it adds value again.
        // This keeps all engines participating when they are actually useful.
        hazardDisabled = false;
      }
    }

    const pHaz = hazardDisabled ? null : hazardPredictAt(state, targetState, idx, cfg);
    const knn = knnPredictAt(state, targetState, idx, cfg);
    const pKnn = knn?.p ?? null;
    const availability = {
      // Enforce reliability minimum before a model can influence blending.
      hazard: Number.isFinite(pHaz) && brierHaz.count() >= cfg.hazardMinEvalCount && !hazardDisabled,
      knn: Number.isFinite(pKnn) && brierKnn.count() >= cfg.knnKMin && Number(knn?.reliability ?? 0) >= cfg.knnMinReliability,
    };

    const recentWhiteRate = (windowSum(
      state.whitePrefix,
      Math.max(0, idx - cfg.b2bMomentumWindowShort + 1),
      idx
    ) || 0) / Math.max(1, Math.min(cfg.b2bMomentumWindowShort, idx + 1));
    const longWhiteRate = (windowSum(
      state.whitePrefix,
      Math.max(0, idx - cfg.b2bMomentumWindowLong + 1),
      idx
    ) || 0) / Math.max(1, Math.min(cfg.b2bMomentumWindowLong, idx + 1));
    const clusterScore = Math.max(0, recentWhiteRate - longWhiteRate);

    const regime = classifyRegime({
      entropy,
      lagCorr: Number(entropy.lagCorr || 0),
      volShift,
      hitRateDrift,
      clusterScore,
      entropyTrend,
    }, cfg);

    const weights = blendWeightsFromBrier(brierMeans, cfg, availability, {
      hazardDisabled,
      knnReliability: Number(knn?.reliability ?? 0),
      trendingBoost: regime.label === 'TRENDING' ? regime.trendingBias : 0,
    });
    const h = Number.isFinite(pHaz) ? pHaz : baselineP;
    const k = Number.isFinite(pKnn) ? pKnn : baselineP;
    let pBlend = clamp(
      weights.baseline * baselineP +
      weights.hazard * h +
      weights.knn * k,
      0,
      1
    );

    const modelDisagreement = (
      Number.isFinite(pHaz) && Number.isFinite(pKnn)
        ? Math.abs(pHaz - pKnn)
        : Math.max(Math.abs(h - baselineP), Math.abs(k - baselineP))
    );
    if (modelDisagreement > cfg.modelDisagreementHigh) {
      const denom = Math.max(0.01, cfg.modelDisagreementHard - cfg.modelDisagreementHigh);
      const shrink = clamp(1 - ((modelDisagreement - cfg.modelDisagreementHigh) / denom), 0.50, 1);
      pBlend = baselineP + ((pBlend - baselineP) * shrink);
    }

    const whiteEst = whiteEstimate(whiteStats, whiteRunNow);
    const continueProb = Number.isFinite(whiteEst.continueProb) ? whiteEst.continueProb : 0;
    const reboundProb = Number.isFinite(whiteEst.reboundProb) ? whiteEst.reboundProb : 0;
    const whiteDelta = clamp(reboundProb - continueProb, -1, 1);
    const whiteStrength = Math.min(Number(cfg.whiteModifierStrength || 0), 0.08);
    let pAdjModel = clamp(pBlend * (1 + whiteStrength * whiteDelta), 0, 1);

    const b2bImm = rateInWindow(targetState.hitPrefix, idx, 2);
    const b2bNear = rateInWindow(targetState.hitPrefix, idx, cfg.b2bNearWindow);
    const b2bNearRate = rateInWindow(targetState.hitPrefix, idx, cfg.b2bMomentumWindowShort);
    const b2bLongRate = rateInWindow(targetState.hitPrefix, idx, cfg.b2bMomentumWindowLong);
    const b2bSampleReady = (idx + 1) >= Math.max(32, cfg.b2bMomentumWindowShort * 2);
    const b2bMomentum = b2bSampleReady ? clamp(b2bNearRate - b2bLongRate, -1, 1) : 0;
    if (b2bMomentum > 0 && !entropy.randomLike) {
      pAdjModel = clamp(
        pAdjModel * (1 + (Math.min(cfg.b2bBoostCap, 0.05) * b2bMomentum)),
        0,
        1
      );
    }

    const edgeVal = edgeValidation.metrics();
    const edgeConfidenceScore = edgeVal.count >= cfg.edgeValidationMinTrades
      ? clamp(edgeVal.score, 0, 1)
      : 1;
    const pAdj = preconditionPass ? pAdjModel : baselineP;
    const pFinal = preconditionPass ? clamp(pAdj * edgeConfidenceScore, 0, 1) : baselineP;

    const edge = pFinal - baselineP;
    const ev = (pFinal * target) - 1;
    const evThreshold = (cfg.evThreshold || 0) + (regime.label === 'CLUSTERED' ? regime.evAdjust : 0);
    const actionable = preconditionPass && !entropySpike && ev > evThreshold;

    const modelAgreement = clamp(
      1 - (modelDisagreement / Math.max(1e-6, cfg.modelDisagreementHard)),
      0,
      1
    );
    const confidenceRaw = confidenceFromComponents(pFinal, baselineP, weights, entropy, {
      modelAgreement,
      edgeConfidenceScore,
      entropyTrend,
      modelDisagreement,
      entropySpike,
    });
    const confidence = preconditionPass ? confidenceRaw : clamp(confidenceRaw * 0.45, 0, 0.45);
    const recommendedBetFraction = kellyFraction(pFinal, target, confidence, cfg.maxRiskFraction);

    return {
      baselineP,
      pHaz,
      pKnn,
      pBlend,
      pFinal,
      pAdj,
      edge,
      ev,
      actionable,
      entropy,
      entropyTrend,
      entropySpike,
      regime,
      confidence,
      recommendedBetFraction,
      weights,
      knnSupport: knn?.support ?? null,
      knnReliability: knn?.reliability ?? null,
      knnSimilarity: knn?.similarityScore ?? null,
      modelDisagreement: roundNum(modelDisagreement, 6),
      modelAgreement: roundNum(modelAgreement, 6),
      edgeConfidenceScore: roundNum(edgeConfidenceScore, 6),
      edgeValidation: edgeVal,
      whiteRun: whiteRunNow,
      whiteContinue: whiteEst.continueProb,
      whiteRebound: whiteEst.reboundProb,
      whiteSample: whiteEst.sample,
      currentWhiteRate,
      previousWhiteRate,
      whiteRegimeScore: roundNum(whiteRegimeScore, 6),
      whiteRegimeThreshold: roundNum(preconditionThresholds.whiteRegimeThreshold, 6),
      releaseScore: roundNum(releaseScore, 6),
      releaseThreshold: roundNum(preconditionThresholds.releaseThreshold, 6),
      momentumScore: roundNum(momentumScore, 6),
      momentumThreshold: roundNum(preconditionThresholds.momentumThreshold, 6),
      releaseRunThreshold: Number(preconditionThresholds.releaseRunThreshold || 0),
      volatilityExpansion: roundNum(preVol.expansion, 6),
      lowVolatility: roundNum(preVol.lowVolatility, 6),
      preconditionState: precondition.state,
      preconditionExplanation: precondition.explanation,
      preconditionPass,
      preconditionBlockers: {
        whiteDominant,
        entropyBlocked,
        signalMissing: signalBlocked,
      },
      b2bImm,
      b2bNear,
      b2bNearRate,
      b2bLongRate,
      b2bMomentum,
    };
  }

  if (evalEnd >= evalStart) {
    for (let i = evalStart; i <= evalEnd; i += 1) {
      const pred = predictAt(i);
      const y = targetState.hits[i + 1] ? 1 : 0;

      brierBase.add((pred.baselineP - y) ** 2);
      if (Number.isFinite(pred.pHaz)) brierHaz.add((pred.pHaz - y) ** 2);
      if (Number.isFinite(pred.pKnn)) brierKnn.add((pred.pKnn - y) ** 2);
      if (Number.isFinite(pred.entropy.js)) entropyBaseline.add(pred.entropy.js);

      if (pred.actionable) {
        bets += 1;
        wins += y;
        const realizedEV = y ? (target - 1) : -1;
        pnl += realizedEV;
        peak = Math.max(peak, pnl);
        maxDrawdown = Math.max(maxDrawdown, peak - pnl);
        evSum += pred.ev;
        pushRollingReturn(realizedEV);
        edgeValidation.add(pred.ev, realizedEV);
        edgePredCount += 1;
        const predPositive = pred.ev >= 0;
        const realPositive = realizedEV >= 0;
        if (predPositive === realPositive) edgePredCorrect += 1;
      }

      const runLen = state.whiteRun[i] || 0;
      const nextWhite = Boolean(state.whiteFlags[i + 1]);
      const nextRebound = Number(state.multipliers[i + 1]) >= state.reboundCut;
      whiteUpdate(whiteStats, runLen, nextWhite, nextRebound);
    }
  }

  const liveIdx = n - 1;
  const live = predictAt(liveIdx);
  const empiricalHitRate = evalEnd >= evalStart
    ? rateInWindow(targetState.hitPrefix, evalEnd + 1, evalEnd - evalStart + 1)
    : null;
  const edgeValidationMetrics = edgeValidation.metrics();
  const rollingEV = mean(rollingReturns);
  const rollingWinRate = rollingReturns.length
    ? (rollingReturns.filter((v) => v > 0).length / rollingReturns.length)
    : null;
  const rollingDrawdown = maxDrawdownOf(rollingReturns);
  const rollingSharpe = sharpeLike(rollingReturns);
  const edgeAccuracy = edgePredCount > 0 ? (edgePredCorrect / edgePredCount) : null;

  return {
    target,
    targetState,
    live,
    brier: {
      baseline: brierBase.mean(),
      hazard: brierHaz.mean(),
      knn: brierKnn.mean(),
      baselineCount: brierBase.count(),
      hazardCount: brierHaz.count(),
      knnCount: brierKnn.count(),
    },
    backtest: {
      rounds: Math.max(0, evalEnd - evalStart + 1),
      bets,
      wins,
      losses: Math.max(0, bets - wins),
      winRate: bets > 0 ? wins / bets : null,
      avgEV: bets > 0 ? evSum / bets : null,
      totalEV: evSum,
      maxDrawdown,
      randomBaseline: {
        empiricalHitRate: empiricalHitRate,
        evPerBet: Number.isFinite(empiricalHitRate) ? (empiricalHitRate * target) - 1 : null,
        expectedTotalEVAtSameBets: (Number.isFinite(empiricalHitRate) && bets > 0)
          ? (((empiricalHitRate * target) - 1) * bets)
          : null,
      },
      performance: {
        rollingEV: roundNum(rollingEV, 6),
        rollingWinRate: roundNum(rollingWinRate, 6),
        rollingMaxDrawdown: roundNum(rollingDrawdown, 6),
        sharpeLike: roundNum(rollingSharpe, 6),
        edgeAccuracy: roundNum(edgeAccuracy, 6),
      },
      edgeValidation: {
        count: edgeValidationMetrics.count,
        predictedEV: roundNum(edgeValidationMetrics.predictedEV, 6),
        realizedEV: roundNum(edgeValidationMetrics.realizedEV, 6),
        edgeError: roundNum(edgeValidationMetrics.edgeError, 6),
        negativeErrorRatio: roundNum(edgeValidationMetrics.negativeErrorRatio, 6),
        edgeConfidenceScore: roundNum(edgeValidationMetrics.score, 6),
      },
      walkForward: true,
      noFutureLeakage: true,
    },
    entropyThresholds,
    performanceDashboard: {
      rollingEV: roundNum(rollingEV, 6),
      winRate: roundNum(rollingWinRate, 6),
      maxDrawdown: roundNum(rollingDrawdown, 6),
      sharpeLike: roundNum(rollingSharpe, 6),
      edgeAccuracy: roundNum(edgeAccuracy, 6),
    },
  };
}

function normalizePressure(score, threshold) {
  const s = Number(score);
  const t = Number(threshold);
  if (!Number.isFinite(s) || !Number.isFinite(t)) return 0;
  const denom = Math.max(1e-6, Math.abs(t));
  return clamp((s - t) / denom, 0, 3);
}

function adjustAheadByRegime(baseAheadLo, fixedSpan, live, target = 0) {
  const base = Math.max(1, Number(baseAheadLo || 1));
  const span = Math.max(1, Number(fixedSpan || 1));
  const t = Number(target || 0);
  const preState = String(live?.preconditionState || 'NEUTRAL').toUpperCase();
  const whiteRisk = clamp(Number(live?.whiteContinue || 0), 0, 1);
  const whiteRelease = clamp(Number(live?.whiteRebound || 0), 0, 1);
  const b2bMomentum = clamp(Number(live?.b2bMomentum || 0), -1, 1);
  const pHitSoon = clamp(Number(live?.pHitSoon ?? live?.pFinal ?? live?.pAdj ?? 0), 0, 1);
  const baselineP = clamp(Number(live?.baselineP ?? 0), 0, 1);
  const soonPressure = clamp(pHitSoon - baselineP, -1, 1);

  const whitePressure = normalizePressure(live?.whiteRegimeScore, live?.whiteRegimeThreshold);
  const releasePressure = normalizePressure(live?.releaseScore, live?.releaseThreshold);
  const momentumPressure = normalizePressure(live?.momentumScore, live?.momentumThreshold);

  const whiteDelta = clamp(whiteRisk - whiteRelease, 0, 1);
  const softWhiteCaution = clamp(
    (whiteDelta * 1.25) + (Math.max(0, -soonPressure) * 1.10),
    0,
    1.8
  );
  const downtrendPressure = clamp(
    (softWhiteCaution * 0.75) +
    (Math.max(0, -soonPressure) * 1.05) +
    (Number(live?.entropy?.randomLike) ? 0.35 : 0),
    0,
    2
  );
  const signalQuality = clamp(
    (
      clamp(Number(live?.edgeConfidenceScore ?? 0.5), 0, 1) * 0.45 +
      clamp(Number(live?.modelAgreement ?? 0.5), 0, 1) * 0.30 +
      clamp(Number(live?.confidence ?? 0.5), 0, 1) * 0.25
    ),
    0,
    1
  );
  // Target-aware normalization: near-term pressure should impact low targets more than moon targets.
  const soonBoost = (
    t <= 20 ? Math.max(0, soonPressure)
    : t <= 100 ? Math.max(0, soonPressure * 0.40)
    : Math.max(0, soonPressure * 0.15)
  );
  const b2bBoost = (
    t <= 20 ? Math.max(0, b2bMomentum)
    : t <= 100 ? Math.max(0, b2bMomentum * 0.55)
    : Math.max(0, b2bMomentum * 0.25)
  );
  const releaseMomentum = Math.max(
    releasePressure,
    momentumPressure,
    clamp(whiteRelease - whiteRisk, 0, 1),
    b2bBoost,
    soonBoost
  );

  let delayBoost = 0;
  if (whitePressure > 0 || whiteDelta > 0 || preState === 'WHITE_DOMINANT') {
    const delaySpanScale = (
      t <= 10 ? 0.72
        : t <= 20 ? 0.62
          : t <= 100 ? 0.46
            : 0.32
    );
    const whiteDrive = clamp(
      ((whitePressure * 0.50) + (whiteDelta * 0.80) + (softWhiteCaution * 0.35)) * (0.55 + (signalQuality * 0.45)),
      0,
      1.6
    );
    delayBoost = Math.round((span * delaySpanScale) * whiteDrive);
  }
  if (downtrendPressure > 0.15) {
    const downtrendScale = (
      t <= 10 ? 0.58
        : t <= 20 ? 0.48
          : t <= 100 ? 0.36
            : 0.24
    );
    delayBoost += Math.round((span * downtrendScale) * clamp(downtrendPressure, 0, 1.8));
  }
  let nearPull = 0;
  if (releaseMomentum > 0 || preState === 'RELEASE_PHASE' || preState === 'MOMENTUM') {
    const releaseDrive = clamp(
      ((releaseMomentum * 0.85) + (Math.max(0, b2bMomentum) * 0.45)) * (0.55 + (signalQuality * 0.45)),
      0,
      1.8
    );
    const nearPullRaw = Math.round((span * 0.52) * releaseDrive);
    // Prevent collapse-to-1 across all targets.
    const maxPullCap = Math.max(
      1,
      Math.round(base * (
        t <= 20 ? 0.70
        : t <= 100 ? 0.55
        : 0.40
      ))
    );
    nearPull = Math.min(nearPullRaw, maxPullCap);
    // Guard against bullish bias during low/downtrend pressure:
    // keep early pull small when white/downtrend evidence is elevated.
    const pullSuppression = clamp(downtrendPressure * 0.55, 0, 0.85);
    nearPull = Math.max(0, Math.round(nearPull * (1 - pullSuppression)));
  }

  let adjusted = Math.max(1, base + delayBoost - nearPull);

  if (preState === 'WHITE_DOMINANT') {
    // Extend on white regime with target-aware floor so low targets do not reopen too early.
    const whiteFloorScale = (
      t <= 10 ? 0.75
        : t <= 20 ? 0.68
          : t <= 100 ? 0.52
            : 0.35
    );
    adjusted = Math.max(adjusted, Math.max(2, Math.round(base + (span * whiteFloorScale))));
  }
  // Soft delay floor even in NEUTRAL if white/downtrend pressure is building.
  if (preState === 'NEUTRAL' && t <= 100 && softWhiteCaution >= 0.22) {
    const neutralWhiteFloor = (
      t <= 10 ? 0.55
        : t <= 20 ? 0.46
          : 0.34
    );
    adjusted = Math.max(adjusted, Math.max(2, Math.round(base + (span * neutralWhiteFloor))));
  }
  if (preState === 'RELEASE_PHASE' || preState === 'MOMENTUM') {
    // Pull closer during release/momentum so b2b/high phases are not missed.
    adjusted = Math.min(adjusted, Math.max(1, Math.round(base * 0.72)));
  }

  // Keep target spacing sane: do not let higher targets collapse to 1 round.
  const baseFloorRatio = (
    t <= 20 ? 0.18
    : t <= 100 ? 0.28
    : 0.38
  );
  const releaseFloorRatio = (
    preState === 'RELEASE_PHASE' || preState === 'MOMENTUM'
      ? baseFloorRatio * 0.75
      : baseFloorRatio
  );
  const minAhead = Math.max(1, Math.round(base * releaseFloorRatio));
  adjusted = Math.max(minAhead, adjusted);

  return {
    adjustedAheadLo: adjusted,
    whitePressure: roundNum(whitePressure, 6),
    releaseMomentum: roundNum(releaseMomentum, 6),
    soonPressure: roundNum(soonPressure, 6),
    b2bMomentum: roundNum(b2bMomentum, 6),
    delayBoost,
    nearPull,
    preState,
  };
}

function buildTimingHint(lock, currentRound, targetResult, state) {
  if (!lock || !targetResult) return null;
  const lockStatus = String(lock?.eta?.lockStatus || '').toUpperCase();
  if (lockStatus === 'IDLE' || Boolean(lock?.suspended)) return null;

  const target = Number(targetResult?.target || 0);
  const fixedSpan = Number(FIXED_WINDOW_SPAN[target] || 3);
  const live = targetResult.live;
  const regimeLabel = String(live?.regime?.label || 'NEUTRAL').toUpperCase();
  const preconditionState = String(live?.preconditionState || 'NEUTRAL').toUpperCase();
  const band = estimateAheadBand(state, targetResult);
  const regimeWindowMult = clamp(Number(live?.regime?.windowMult ?? 1), 1, 1.35);
  const baseAheadLo = Math.max(1, Math.round(band.aheadLo * regimeWindowMult));
  const aheadAdjust = adjustAheadByRegime(baseAheadLo, fixedSpan, live, target);
  const dynamicAheadLo = Math.max(1, Number(aheadAdjust.adjustedAheadLo || 1));

  const lockedLo = Number(lock.lo);
  if (!Number.isFinite(lockedLo)) return null;

  // Hint reliability: if model alignment/edge validation is weak, avoid aggressive early/later calls.
  const edgeConfidenceScore = clamp(Number(live?.edgeConfidenceScore ?? 0.5), 0, 1);
  const modelAgreement = clamp(Number(live?.modelAgreement ?? 0.5), 0, 1);
  const lockConfidence = clamp(Number(live?.confidence ?? lock?.confidence ?? 0.5), 0, 1);
  const hintReliability = clamp(
    (edgeConfidenceScore * 0.45) +
    (modelAgreement * 0.25) +
    (lockConfidence * 0.30),
    0,
    1
  );
  const edgeAccuracy = clamp(Number(targetResult?.backtest?.performance?.edgeAccuracy ?? 0.5), 0, 1);
  const directionalReliability = clamp(
    (hintReliability * 0.65) + (edgeAccuracy * 0.35),
    0,
    1
  );

  // Neutralization layer:
  // - Reduce delay bias when near-term hit pressure/momentum is rising.
  // - Keep delay/early hints conservative for lower targets to avoid over-strict guidance.
  const pHitSoon = clamp(Number(live?.pHitSoon ?? live?.pFinal ?? live?.pAdj ?? 0), 0, 1);
  const baselineP = clamp(Number(live?.baselineP ?? (target > 0 ? (1 / target) : 0)), 0, 1);
  const soonPressure = clamp(pHitSoon - baselineP, -1, 1);
  const b2bMomentum = clamp(Number(live?.b2bMomentum ?? 0), -1, 1);
  const whiteRisk = clamp(Number(live?.whiteContinue ?? 0), 0, 1);
  const whiteRelease = clamp(Number(live?.whiteRebound ?? 0), 0, 1);

  const delayBias = (
    clamp(Number(aheadAdjust.whitePressure || 0), 0, 3) * 1.20 +
    Math.max(0, whiteRisk - whiteRelease) * 2.00
  );
  const earlyBias = (
    clamp(Number(aheadAdjust.releaseMomentum || 0), 0, 3) * 0.90 +
    Math.max(0, b2bMomentum) * 1.20 +
    Math.max(0, soonPressure) * 4.00
  );

  let calibratedAheadLo = dynamicAheadLo + Math.round(delayBias - earlyBias);
  if (target <= 20) {
    // Lower targets are very reactive; avoid stale/strict delay bias.
    calibratedAheadLo -= Math.round(Math.max(0, soonPressure) * 3);
  }
  calibratedAheadLo = Math.max(1, calibratedAheadLo);

  const suggestedLoRaw = currentRound + calibratedAheadLo;
  const rawDelta = Math.round(suggestedLoRaw - lockedLo);
  const scaledDelta = Math.round(rawDelta * (0.40 + (hintReliability * 0.60)));
  let deltaRounds = scaledDelta;

  let trend = 'on_track';
  // Conservative thresholds (especially for low targets) to prevent noisy false alerts.
  const reliabilityPenalty = Math.max(0, 0.45 - directionalReliability);
  const reliabilityBoost = Math.round(reliabilityPenalty * 2);
  const trendThreshold = (
    target <= 20 ? 2
    : target <= 100 ? 2
    : 3
  ) + (hintReliability < 0.5 ? 1 : 0) + reliabilityBoost;

  if (deltaRounds <= -trendThreshold) trend = 'earlier';
  if (deltaRounds >= trendThreshold) trend = 'later';

  const strongEarlierEvidence = (
    soonPressure > (target <= 100 ? 0.035 : 0.025) ||
    Math.max(0, b2bMomentum) > (target <= 100 ? 0.06 : 0.04) ||
    Number(aheadAdjust.releaseMomentum || 0) > 0.45
  );
  const strongLaterEvidence = (
    Number(aheadAdjust.whitePressure || 0) > 0.30 &&
    (whiteRisk - whiteRelease) > 0.04
  );

  // Regime-aware directional nudges:
  // if white pressure is genuinely dominating, force a bounded delay hint;
  // if release/momentum is genuinely dominating, force a bounded early hint.
  const whiteDominantNow = preconditionState === 'WHITE_DOMINANT';
  const releaseOrMomentumNow = preconditionState === 'RELEASE_PHASE' || preconditionState === 'MOMENTUM';
  const whiteDrive = (
    (Number(aheadAdjust.whitePressure || 0) * 0.60) +
    (Math.max(0, whiteRisk - whiteRelease) * 1.10) -
    (Math.max(0, soonPressure) * 0.70) -
    (Math.max(0, b2bMomentum) * 0.45)
  );
  const releaseDrive = (
    (Number(aheadAdjust.releaseMomentum || 0) * 0.75) +
    (Math.max(0, soonPressure) * 2.20) +
    (Math.max(0, b2bMomentum) * 1.10) -
    (Math.max(0, whiteRisk - whiteRelease) * 0.75)
  );
  const delayMinShift = (
    target <= 20 ? 2
      : target <= 100 ? 3
        : 6
  );
  const earlyMinShift = (
    target <= 20 ? 1
      : target <= 100 ? 2
        : 4
  );
  if (whiteDominantNow && whiteDrive >= 0.20 && trend !== 'earlier') {
    trend = 'later';
    deltaRounds = Math.max(deltaRounds, delayMinShift);
  } else if (releaseOrMomentumNow && releaseDrive >= 0.18 && trend !== 'later') {
    trend = 'earlier';
    deltaRounds = Math.min(deltaRounds, -earlyMinShift);
  }

  // Low-quality signal gate: avoid directional hints unless evidence is strong.
  if (directionalReliability < 0.35 && !(strongEarlierEvidence || strongLaterEvidence)) {
    trend = 'on_track';
  }

  // If near-term pressure is strong for low targets, suppress "later" hints.
  if (target <= 20 && trend === 'later' && soonPressure > 0.06) {
    trend = 'on_track';
  }
  if (target <= 10 && trend === 'later' && Math.max(0, b2bMomentum) > 0.08) {
    trend = 'on_track';
  }

  // Watch layer: show momentum/pressure drift even when not strong enough for hard earlier/later.
  // This keeps banner informative between hard directional flips.
  if (trend === 'on_track') {
    const whiteWatchScore = clamp(
      (Math.max(0, whiteRisk - whiteRelease) * 1.25) +
      (Number(aheadAdjust.whitePressure || 0) * 0.35) +
      (Math.max(0, -soonPressure) * 0.95),
      0,
      3
    );
    const releaseWatchScore = clamp(
      (Math.max(0, soonPressure) * 2.0) +
      (Math.max(0, b2bMomentum) * 1.1) +
      (Number(aheadAdjust.releaseMomentum || 0) * 0.65),
      0,
      3
    );
    const whiteWatch = whiteWatchScore >= 0.08;
    const releaseWatch = releaseWatchScore >= 0.08;

    if (whiteWatch && releaseWatch) {
      if ((whiteWatchScore - releaseWatchScore) >= 0.03) {
        trend = 'later_watch';
        deltaRounds = Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.35)));
      } else if ((releaseWatchScore - whiteWatchScore) >= 0.03) {
        trend = 'earlier_watch';
        deltaRounds = -Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.35)));
      } else if (Math.max(whiteWatchScore, releaseWatchScore) >= 0.20) {
        if (whiteWatchScore >= releaseWatchScore) {
          trend = 'later_watch';
          deltaRounds = Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.30)));
        } else {
          trend = 'earlier_watch';
          deltaRounds = -Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.30)));
        }
      }
    } else if (whiteWatch) {
      trend = 'later_watch';
      deltaRounds = Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.35)));
    } else if (releaseWatch) {
      trend = 'earlier_watch';
      deltaRounds = -Math.max(1, Math.round(Math.max(1, Math.abs(rawDelta) * 0.35)));
    }
  }
  if (trend === 'on_track' && Math.abs(rawDelta) >= 2 && directionalReliability >= 0.28) {
    trend = rawDelta > 0 ? 'later_watch' : 'earlier_watch';
    deltaRounds = rawDelta > 0
      ? Math.max(1, Math.round(Math.abs(rawDelta) * 0.30))
      : -Math.max(1, Math.round(Math.abs(rawDelta) * 0.30));
  }

  const absDelta = Math.abs(deltaRounds);

  let severity = 'low';
  if (absDelta >= 8) severity = 'high';
  else if (absDelta >= 4) severity = 'med';

  const suggestedLo = lockedLo + deltaRounds;
  const suggestedHi = suggestedLo + fixedSpan - 1;
  const suggestedAheadLo = Math.max(1, suggestedLo - currentRound);

  let message = `${target}x no change (as per lock).`;
  if (trend === 'earlier') {
    message = `${target}x may come earlier by ~${absDelta} rounds (around +${suggestedAheadLo}).`;
  } else if (trend === 'earlier_watch') {
    message = `${target}x early pressure building (watch ~${absDelta} rounds sooner, around +${suggestedAheadLo}).`;
  } else if (trend === 'later') {
    message = `${target}x may come later by ~${absDelta} rounds (around +${suggestedAheadLo}).`;
  } else if (trend === 'later_watch') {
    message = `${target}x delay pressure building (watch ~${absDelta} rounds later, around +${suggestedAheadLo}).`;
  } else if (preconditionState === 'RELEASE_PHASE' || preconditionState === 'MOMENTUM') {
    message = `${target}x no shift yet (as per lock), but ${preconditionState === 'MOMENTUM' ? 'momentum' : 'release'} is building.`;
  } else if (preconditionState === 'WHITE_DOMINANT') {
    message = `${target}x no shift yet (as per lock), white pressure is elevated.`;
  } else if (regimeLabel === 'TRENDING') {
    message = `${target}x no change (as per lock), trend regime is stable.`;
  } else if (regimeLabel === 'VOLATILE') {
    message = `${target}x no change (as per lock), volatility regime is elevated.`;
  } else if (regimeLabel === 'CLUSTERED') {
    message = `${target}x no change (as per lock), clustered flow is active.`;
  } else if (regimeLabel === 'RANDOM') {
    message = `${target}x no change (as per lock), random/noise regime is active.`;
  } else if (Math.abs(soonPressure) >= 0.02) {
    message = `${target}x no change (as per lock), near-hit pressure is slightly ${soonPressure > 0 ? 'up' : 'down'}.`;
  }

  const reasonParts = [];
  reasonParts.push(`regime ${regimeLabel}`);
  reasonParts.push(`state ${preconditionState}`);
  if (trend === 'earlier' || trend === 'earlier_watch') {
    if (soonPressure > 0.04) reasonParts.push(`near-hit pressure above baseline (${roundNum(soonPressure, 3)})`);
    if (b2bMomentum > 0.05) reasonParts.push(`b2b momentum rising (${roundNum(b2bMomentum, 3)})`);
    if (Number(aheadAdjust.releaseMomentum || 0) > 0.35) {
      reasonParts.push(`release momentum active (${roundNum(aheadAdjust.releaseMomentum, 3)})`);
    }
  } else if (trend === 'later' || trend === 'later_watch') {
    if (Number(aheadAdjust.whitePressure || 0) > 0.15) {
      reasonParts.push(`white pressure elevated (${roundNum(aheadAdjust.whitePressure, 3)})`);
    }
    if ((whiteRisk - whiteRelease) > 0.02) {
      reasonParts.push(`white continue > release (${roundNum(whiteRisk - whiteRelease, 3)})`);
    }
    if (soonPressure < -0.03) {
      reasonParts.push(`near-hit pressure below baseline (${roundNum(soonPressure, 3)})`);
    }
  } else {
    reasonParts.push(`pressures are balanced`);
    reasonParts.push(`no strong early/delay edge`);
    if (Math.abs(soonPressure) >= 0.02) {
      reasonParts.push(`near-hit pressure ${soonPressure > 0 ? 'above' : 'below'} baseline (${roundNum(soonPressure, 3)})`);
    }
    if (Math.abs(b2bMomentum) >= 0.04) {
      reasonParts.push(`b2b momentum ${b2bMomentum > 0 ? 'up' : 'down'} (${roundNum(b2bMomentum, 3)})`);
    }
    if (Number(aheadAdjust.whitePressure || 0) >= 0.12 || Math.abs(whiteRisk - whiteRelease) >= 0.04) {
      reasonParts.push(
        `white pressure ${roundNum(Number(aheadAdjust.whitePressure || 0), 3)} | continue-release ${roundNum(whiteRisk - whiteRelease, 3)}`
      );
    }
  }
  reasonParts.push(`signal quality ${Math.round(directionalReliability * 100)}%`);
  const reason = reasonParts.join(' | ');

  return {
    trend,
    severity,
    deltaRounds,
    suggestedAheadLo,
    suggestedLo,
    suggestedHi,
    whitePressure: aheadAdjust.whitePressure,
    releaseMomentum: aheadAdjust.releaseMomentum,
    hintReliability: roundNum(hintReliability, 6),
    directionalReliability: roundNum(directionalReliability, 6),
    soonPressure: roundNum(soonPressure, 6),
    b2bMomentum: roundNum(b2bMomentum, 6),
    message,
    reason,
  };
}

function buildLockFromLive(state, targetResult, currentRound, generation, cfg, options = {}) {
  const forceLock = Boolean(options?.forceLock);
  const previousOutcome = options?.previousOutcome || null;
  const live = targetResult.live;
  const band = estimateAheadBand(state, targetResult);
  const regimeWindowMult = clamp(Number(live?.regime?.windowMult ?? 1), 1, 1.35);
  const fixedSpan = Number(FIXED_WINDOW_SPAN[targetResult.target] || 3);
  const baseAheadLo = Math.max(1, Math.round(band.aheadLo * regimeWindowMult));
  const aheadAdjust = adjustAheadByRegime(baseAheadLo, fixedSpan, live, targetResult.target);
  let aheadLo = aheadAdjust.adjustedAheadLo;
  const preState = String(live?.preconditionState || 'NEUTRAL').toUpperCase();
  const regimeLabel = String(live?.regime?.label || 'RANDOM').toUpperCase();
  const whiteRisk = clamp(Number(live?.whiteContinue || 0), 0, 1);
  const whiteRelease = clamp(Number(live?.whiteRebound || 0), 0, 1);
  const whiteDelta = Math.max(0, whiteRisk - whiteRelease);
  const pHitSoon = clamp(Number(live?.pHitSoon ?? live?.pFinal ?? live?.pAdj ?? 0), 0, 1);
  const baselineP = clamp(Number(live?.baselineP ?? (targetResult?.target > 0 ? (1 / targetResult.target) : 0)), 0, 1);
  const soonPressure = pHitSoon - baselineP;
  const entropyRandomLike = Boolean(live?.entropy?.randomLike || live?.entropy?.disabled);
  const lowTrendStrength = clamp(
    (Number(aheadAdjust?.whitePressure || 0) * 0.60) +
    (whiteDelta * 1.15) +
    (entropyRandomLike ? 0.40 : 0) +
    (regimeLabel === 'RANDOM' ? 0.35 : 0) +
    (regimeLabel === 'VOLATILE' ? 0.18 : 0) +
    (preState === 'WHITE_DOMINANT' ? 0.45 : 0) +
    (Math.max(0, -soonPressure) * 1.20),
    0,
    3
  );
  const lowTrendMode = (
    preState === 'WHITE_DOMINANT' ||
    regimeLabel === 'RANDOM' ||
    entropyRandomLike ||
    whiteDelta > 0.06 ||
    lowTrendStrength >= 0.45
  );
  if (previousOutcome && String(previousOutcome.outcome || '').toLowerCase() === 'early') {
    const prevLo = Number(previousOutcome.lo);
    const prevHit = Number(previousOutcome.hitRound);
    if (Number.isFinite(prevLo) && Number.isFinite(prevHit)) {
      const earlyBy = Math.max(1, prevLo - prevHit);
      const quality = clamp(Number(live?.edgeConfidenceScore ?? 0.5), 0, 1);
      const pull = Math.round(Math.min(earlyBy, fixedSpan * 2) * (0.45 + (quality * 0.35)));
      const adjustedPull = lowTrendMode
        ? Math.max(0, Math.round(Math.max(1, pull) * clamp(1 - (lowTrendStrength * 0.35), 0.15, 0.75)))
        : Math.max(1, pull);
      aheadLo = Math.max(1, aheadLo - adjustedPull);
    } else {
      const fallbackPull = Math.max(1, Math.round(fixedSpan * 0.35));
      const adjustedFallbackPull = lowTrendMode
        ? Math.max(0, Math.round(fallbackPull * clamp(1 - (lowTrendStrength * 0.30), 0.20, 0.80)))
        : fallbackPull;
      aheadLo = Math.max(1, aheadLo - adjustedFallbackPull);
    }
    if (lowTrendMode) {
      // Prevent instant re-open after EARLY during low/downtrend pressure.
      aheadLo += Math.max(1, Math.round(fixedSpan * 0.22));
    }
  } else if (previousOutcome && String(previousOutcome.outcome || '').toLowerCase() === 'loss') {
    // After LOSS, apply cooldown in low/downtrend regimes so locks do not reopen too aggressively.
    if (lowTrendMode) {
      const t = Number(targetResult?.target || 0);
      const baseCooldown = (
        t <= 10 ? Math.max(3, Math.round(fixedSpan * 0.95))
          : t <= 20 ? Math.max(3, Math.round(fixedSpan * 0.80))
            : t <= 100 ? Math.max(4, Math.round(fixedSpan * 0.62))
              : Math.max(6, Math.round(fixedSpan * 0.45))
      );
      const scaledCooldown = Math.round(baseCooldown * clamp(0.55 + (lowTrendStrength * 0.35), 0.55, 1.65));
      const cooldown = Math.max(2, scaledCooldown);
      aheadLo += cooldown;
    }
  } else if (previousOutcome && String(previousOutcome.outcome || '').toLowerCase() === 'win') {
    if (lowTrendMode) {
      // Even after WIN, avoid reopening too tightly while low/downtrend pressure persists.
      const softCooldown = Math.max(1, Math.round(fixedSpan * clamp(0.16 + (lowTrendStrength * 0.08), 0.16, 0.40)));
      aheadLo += softCooldown;
    }
  }
  const aheadHi = Math.max(aheadLo, aheadLo + fixedSpan - 1);
  const lo = currentRound + aheadLo;
  const hi = lo + fixedSpan - 1;
  const p = Number.isFinite(live.pFinal) ? live.pFinal : live.pAdj;
  const pSoon = clamp(1 - Math.pow(1 - p, 3), 0, 1);
  const allowForcedLock = Boolean(forceLock);
  const shouldOpen = Boolean(live?.actionable || allowForcedLock);
  const suspended = !shouldOpen;

  return {
    lo,
    hi,
    roundWhenMade: currentRound,
    generation,
    suspended,
    confidence: live.confidence,
    eta: {
      modelVersion: 'EngineX-v1-edge',
      baselineFormula: 'P(x)=1/x',
      pHit1: roundNum(p, 6),
      pHitSoon: roundNum(pSoon, 6),
      quickHit: roundNum(p, 6),
      pAdj: roundNum(live.pAdj, 6),
      pFinal: roundNum(live.pFinal, 6),
      edge: roundNum(live.edge, 6),
      ev: roundNum(live.ev, 6),
      targetEVThreshold: roundNum(cfg.evThreshold, 6),
      baseRate: roundNum(live.baselineP, 6),
      hazardP1: roundNum(live.pHaz, 6),
      knnP1: roundNum(live.pKnn, 6),
      knnSupport: roundNum(live.knnSupport, 6),
      knnReliabilityScore: roundNum(live.knnReliability, 6),
      knnSimilarityScore: roundNum(live.knnSimilarity, 6),
      blend: {
        baseline: roundNum(live.weights.baseline, 6),
        hazard: roundNum(live.weights.hazard, 6),
        knn: roundNum(live.weights.knn, 6),
      },
      whiteClusterRun: Number(live.whiteRun || 0),
      whiteClusterRisk: roundNum(live.whiteContinue, 6),
      whiteClusterRelease: roundNum(live.whiteRebound, 6),
      whiteClusterDelta: roundNum((live.whiteRebound ?? 0) - (live.whiteContinue ?? 0), 6),
      whiteClusterSample: Number(live.whiteSample || 0),
      preconditionState: String(live.preconditionState || 'NEUTRAL'),
      preconditionExplanation: String(live.preconditionExplanation || ''),
      preconditionPass: Boolean(live.preconditionPass),
      whiteRegimeScore: roundNum(live.whiteRegimeScore, 6),
      whiteRegimeThreshold: roundNum(live.whiteRegimeThreshold, 6),
      releaseScore: roundNum(live.releaseScore, 6),
      releaseThreshold: roundNum(live.releaseThreshold, 6),
      momentumScore: roundNum(live.momentumScore, 6),
      momentumThreshold: roundNum(live.momentumThreshold, 6),
      releaseRunThreshold: Number(live.releaseRunThreshold || 0),
      volatilityExpansion: roundNum(live.volatilityExpansion, 6),
      lowVolatility: roundNum(live.lowVolatility, 6),
      b2bImmRate: roundNum(live.b2bImm, 6),
      b2bNearRate: roundNum(live.b2bNear, 6),
      b2bMomentum: roundNum(live.b2bMomentum, 6),
      entropyRandomLike: Boolean(live.entropy.randomLike),
      entropyDisabled: Boolean(live.entropy.disabled),
      entropyJs: live.entropy.js,
      entropyDrift: live.entropy.drift,
      entropyLag: live.entropy.lagCorr,
      entropyTrend: roundNum(live.entropyTrend, 8),
      entropySpike: Boolean(live.entropySpike),
      entropyThresholdSample: Number(live.entropy.thresholdSample || 0),
      regime: live.regime?.label || 'RANDOM',
      regimeWindowMult: roundNum(regimeWindowMult, 4),
      aiConfidence: roundNum(live.confidence, 6),
      edgeConfidenceScore: roundNum(live.edgeConfidenceScore, 6),
      modelAgreement: roundNum(live.modelAgreement, 6),
      modelDisagreement: roundNum(live.modelDisagreement, 6),
      recommendedBetFraction: roundNum(live.recommendedBetFraction, 6),
      baseAheadLo: roundNum(baseAheadLo, 3),
      regimeAdjustedAheadLo: roundNum(aheadLo, 3),
      whitePressure: aheadAdjust.whitePressure,
      releaseMomentumPressure: aheadAdjust.releaseMomentum,
      delayBoostRounds: Number(aheadAdjust.delayBoost || 0),
      nearPullRounds: Number(aheadAdjust.nearPull || 0),
      aheadLo,
      aheadHi,
      q25: band.q20,
      q50: band.q50,
      q75: band.q75,
      q90: band.q90,
      q95: band.q95,
      gapNow: band.gapNow,
      lockCreatedAtRound: Number(currentRound),
      lockStatus: suspended ? 'IDLE' : 'LOCKED',
      isMutable: false,
      roundsSinceLock: 0,
      suspended,
      suspendReason: suspended
        ? (!live.actionable ? 'ev_below_threshold' : null)
        : null,
      forcedLock: forceLock,
      walkForwardNoLeakage: true,
    },
  };
}

function buildIdleLock(state, targetResult, currentRound, generation, cfg) {
  const live = targetResult.live;
  const band = estimateAheadBand(state, targetResult);
  const regimeWindowMult = clamp(Number(live?.regime?.windowMult ?? 1), 1, 1.35);
  const fixedSpan = Number(FIXED_WINDOW_SPAN[targetResult.target] || 3);
  const baseAheadLo = Math.max(1, Math.round(band.aheadLo * regimeWindowMult));
  const aheadAdjust = adjustAheadByRegime(baseAheadLo, fixedSpan, live, targetResult.target);
  const aheadLo = aheadAdjust.adjustedAheadLo;
  const aheadHi = Math.max(aheadLo, aheadLo + fixedSpan - 1);
  const lo = currentRound + aheadLo;
  const hi = lo + fixedSpan - 1;
  const baselineP = 1 / targetResult.target;
  return {
    lo,
    hi,
    roundWhenMade: currentRound,
    generation,
    suspended: true,
    confidence: Number(live?.confidence || 0),
    eta: {
      modelVersion: 'EngineX-v1-edge',
      baselineFormula: 'P(x)=1/x',
      pHit1: roundNum(live?.pFinal ?? live?.pAdj ?? baselineP, 6),
      pHitSoon: roundNum(live?.pFinal ?? live?.pAdj ?? baselineP, 6),
      quickHit: roundNum(live?.pFinal ?? live?.pAdj ?? baselineP, 6),
      edge: roundNum(live?.edge, 6),
      ev: roundNum(live?.ev, 6),
      baseRate: roundNum(baselineP, 6),
      aiConfidence: roundNum(live?.confidence, 6),
      edgeConfidenceScore: roundNum(live?.edgeConfidenceScore, 6),
      preconditionState: String(live?.preconditionState || 'NEUTRAL'),
      preconditionExplanation: String(live?.preconditionExplanation || 'No pre-condition edge detected.'),
      preconditionPass: Boolean(live?.preconditionPass),
      whiteRegimeScore: roundNum(live?.whiteRegimeScore, 6),
      whiteRegimeThreshold: roundNum(live?.whiteRegimeThreshold, 6),
      releaseScore: roundNum(live?.releaseScore, 6),
      releaseThreshold: roundNum(live?.releaseThreshold, 6),
      momentumScore: roundNum(live?.momentumScore, 6),
      momentumThreshold: roundNum(live?.momentumThreshold, 6),
      baseAheadLo: roundNum(baseAheadLo, 3),
      regimeAdjustedAheadLo: roundNum(aheadLo, 3),
      whitePressure: aheadAdjust.whitePressure,
      releaseMomentumPressure: aheadAdjust.releaseMomentum,
      delayBoostRounds: Number(aheadAdjust.delayBoost || 0),
      nearPullRounds: Number(aheadAdjust.nearPull || 0),
      aheadLo,
      aheadHi,
      lockCreatedAtRound: Number(currentRound),
      lockStatus: 'IDLE',
      isMutable: false,
      roundsSinceLock: 0,
      suspended: true,
      suspendReason: 'no_actionable_edge',
      walkForwardNoLeakage: true,
    },
  };
}

function summarizeBacktests(results) {
  const rows = [];
  let totalBets = 0;
  let totalWins = 0;
  let totalEv = 0;
  let maxDd = 0;
  let randomExpected = 0;
  for (let i = 0; i < results.length; i += 1) {
    const r = results[i];
    rows.push({
      target: r.target,
      bets: r.backtest.bets,
      wins: r.backtest.wins,
      losses: r.backtest.losses,
      winRate: roundNum(r.backtest.winRate, 6),
      avgEV: roundNum(r.backtest.avgEV, 6),
      totalEV: roundNum(r.backtest.totalEV, 6),
      maxDrawdown: roundNum(r.backtest.maxDrawdown, 6),
      randomEVPerBet: roundNum(r.backtest.randomBaseline.evPerBet, 6),
      randomExpectedTotalEVAtSameBets: roundNum(r.backtest.randomBaseline.expectedTotalEVAtSameBets, 6),
      walkForward: true,
      noFutureLeakage: true,
    });
    totalBets += r.backtest.bets;
    totalWins += r.backtest.wins;
    totalEv += r.backtest.totalEV || 0;
    maxDd = Math.max(maxDd, r.backtest.maxDrawdown || 0);
    randomExpected += r.backtest.randomBaseline.expectedTotalEVAtSameBets || 0;
  }
  return {
    perTarget: rows,
    aggregate: {
      bets: totalBets,
      wins: totalWins,
      losses: Math.max(0, totalBets - totalWins),
      winRate: totalBets > 0 ? roundNum(totalWins / totalBets, 6) : null,
      totalEV: roundNum(totalEv, 6),
      avgEV: totalBets > 0 ? roundNum(totalEv / totalBets, 6) : null,
      maxDrawdown: roundNum(maxDd, 6),
      randomExpectedTotalEVAtSameBets: roundNum(randomExpected, 6),
      walkForward: true,
      noFutureLeakage: true,
    },
  };
}

function summarizePerformanceDashboard(results) {
  const perTarget = [];
  const evSeries = [];
  const wrSeries = [];
  const ddSeries = [];
  const sharpeSeries = [];
  const edgeAccSeries = [];
  for (let i = 0; i < results.length; i += 1) {
    const r = results[i];
    const perf = r?.backtest?.performance || {};
    perTarget.push({
      target: r.target,
      rollingEV: roundNum(perf.rollingEV, 6),
      winRate: roundNum(perf.rollingWinRate, 6),
      maxDrawdown: roundNum(perf.rollingMaxDrawdown, 6),
      sharpeLike: roundNum(perf.sharpeLike, 6),
      edgeAccuracy: roundNum(perf.edgeAccuracy, 6),
      edgeConfidenceScore: roundNum(r?.live?.edgeConfidenceScore, 6),
      recommendedBetFraction: roundNum(r?.live?.recommendedBetFraction, 6),
      regime: r?.live?.regime?.label || 'RANDOM',
    });
    if (Number.isFinite(perf.rollingEV)) evSeries.push(perf.rollingEV);
    if (Number.isFinite(perf.rollingWinRate)) wrSeries.push(perf.rollingWinRate);
    if (Number.isFinite(perf.rollingMaxDrawdown)) ddSeries.push(perf.rollingMaxDrawdown);
    if (Number.isFinite(perf.sharpeLike)) sharpeSeries.push(perf.sharpeLike);
    if (Number.isFinite(perf.edgeAccuracy)) edgeAccSeries.push(perf.edgeAccuracy);
  }
  return {
    perTarget,
    aggregate: {
      rollingEV: roundNum(mean(evSeries), 6),
      winRate: roundNum(mean(wrSeries), 6),
      maxDrawdown: roundNum(mean(ddSeries), 6),
      sharpeLike: roundNum(mean(sharpeSeries), 6),
      edgeAccuracy: roundNum(mean(edgeAccSeries), 6),
    },
  };
}

function computeLockedRangePredictions(rounds, existingLocksRaw = {}, options = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...(options.config || {}) };
  const cleanRounds = normalizeRounds(rounds);

  if (!cleanRounds.length) {
    return {
      model: 'EngineX-v1-edge',
      generatedAt: new Date().toISOString(),
      asOfRound: null,
      sampleSize: 0,
      targets: [],
      locksToSave: {},
      resolvedHistory: [],
      whiteCluster: null,
      summary: { waiting: 0, windowOpen: 0, relocked: 0, sampleSize: 0 },
      settings: {
        mode: 'precondition-edge-detection',
        baselineFormula: 'P(x)=1/x',
        evThreshold: cfg.evThreshold,
      },
    };
  }

  const state = buildGlobalState(cleanRounds, cfg);
  const currentRound = cleanRounds[cleanRounds.length - 1].roundId;
  const targetResultCache = new Map();
  const hitRoundCache = new Map();
  const getTargetResult = (target) => {
    const key = Number(target);
    if (!targetResultCache.has(key)) {
      targetResultCache.set(key, runTargetWalkForward(state, key, cfg));
    }
    return targetResultCache.get(key);
  };
  const getHitRoundIds = (target) => {
    const key = Number(target);
    if (!hitRoundCache.has(key)) {
      const ids = [];
      for (let ix = 0; ix < state.n; ix += 1) {
        if (state.multipliers[ix] >= key) ids.push(state.roundIds[ix]);
      }
      hitRoundCache.set(key, ids);
    }
    return hitRoundCache.get(key);
  };

  const locksToSave = {};
  const resolvedHistory = [];
  const targetsOut = [];
  const timingAlerts = [];
  let waitingCount = 0;
  let openCount = 0;
  let relockedCount = 0;

  for (let i = 0; i < TARGETS.length; i += 1) {
    const target = TARGETS[i];
    const key = String(target);
    const existing = normalizeLockInput(existingLocksRaw[key]);
    const hitRoundIds = getHitRoundIds(target);
    const evalExisting = evaluateExistingLock(existing, hitRoundIds, currentRound);
    const expectedSpan = Number(FIXED_WINDOW_SPAN[target] || 3);
    const existingSpan = existing
      ? Math.max(1, Number(existing.hi) - Number(existing.lo) + 1)
      : null;
    const spanMismatch = existing && Number.isFinite(existingSpan) && existingSpan !== expectedSpan;

    let previousOutcome = null;
    let lockToUse = existing;
    let status = evalExisting.status;
    let liveResult = null;
    let timingHint = null;

    // STRICT LOCK LIFECYCLE:
    // If an existing lock is still active, freeze it completely (no mutation, no drift).
    if (existing && !evalExisting.resolved && evalExisting.status !== 'idle' && !spanMismatch) {
      lockToUse = existing;
      status = evalExisting.status;
      liveResult = getTargetResult(target);
      timingHint = buildTimingHint(existing, currentRound, liveResult, state);
    } else {
      // Recompute target model only when we are about to create a new decision
      // (fresh target OR after resolution OR idle monitor update).
      liveResult = getTargetResult(target);
      const existingWasIdle = existing
        ? (
            String(existing?.eta?.lockStatus || '').toUpperCase() === 'IDLE' ||
            Boolean(existing?.suspended)
          )
        : false;
      if (existing && evalExisting.resolved) {
        previousOutcome = existingWasIdle ? null : {
          outcome: evalExisting.outcome,
          hitRound: evalExisting.hitRound,
          lo: existing.lo,
          hi: existing.hi,
          generation: existing.generation,
        };
        if (!existingWasIdle) {
          resolvedHistory.push({
            target: `${target}x`,
            minMult: Number(target),
            outcome: evalExisting.outcome,
            lo: Number(existing.lo),
            hi: Number(existing.hi),
            hitRound: evalExisting.hitRound,
            generation: Number(existing.generation || 1),
            probW: existing?.eta?.pHit1 ?? null,
            confidence: existing?.confidence ?? existing?.eta?.aiConfidence ?? null,
            roundWhenMade: Number(existing.roundWhenMade),
          });
        }
      }
      const generation = existing ? Number(existing.generation || 1) + 1 : 1;

      // Freeze IDLE locks while signal remains non-actionable to prevent +1 sliding drift.
      if (existingWasIdle && !evalExisting.resolved && !spanMismatch && !liveResult.live.actionable && !cfg.alwaysEmitLocks) {
        lockToUse = existing;
        status = 'idle';
      } else if (liveResult.live.actionable || cfg.alwaysEmitLocks) {
        lockToUse = buildLockFromLive(state, liveResult, currentRound, generation, cfg, {
          forceLock: Boolean(cfg.alwaysEmitLocks),
          previousOutcome,
        });
        status = lockToUse?.suspended ? 'idle' : 'locked';
        relockedCount += 1;
      } else {
        lockToUse = buildIdleLock(state, liveResult, currentRound, generation, cfg);
        status = 'idle';
      }

      if (!timingHint && liveResult && lockToUse && !lockToUse.suspended) {
        timingHint = buildTimingHint(lockToUse, currentRound, liveResult, state);
      }
    }

    if (status === 'locked' && Number(lockToUse.lo) <= currentRound) {
      status = 'window-open';
    }
    if (status === 'idle') waitingCount += 1;
    if (status === 'window-open') openCount += 1;

    locksToSave[key] = {
      lo: Number(lockToUse.lo),
      hi: Number(lockToUse.hi),
      roundWhenMade: Number(lockToUse.roundWhenMade),
      generation: Number(lockToUse.generation || 1),
      suspended: Boolean(lockToUse.suspended),
      confidence: Number(lockToUse.confidence || 0),
      eta: lockToUse.eta || null,
    };

    if (timingHint && String(timingHint.trend).toLowerCase() !== 'on_track') {
      timingAlerts.push({
        target,
        targetLabel: `${target}x`,
        trend: timingHint.trend,
        severity: timingHint.severity,
        deltaRounds: timingHint.deltaRounds,
        message: timingHint.message,
        reason: timingHint.reason,
        suggestedLo: timingHint.suggestedLo,
        suggestedHi: timingHint.suggestedHi,
      });
    }

    targetsOut.push(buildUiTarget(target, lockToUse, status, currentRound, previousOutcome, timingHint));
  }

  targetsOut.sort((a, b) => a.target - b.target);
  const preconditionStates = targetsOut.reduce((acc, t) => {
    const st = String(t?.signals?.preconditionState || 'NEUTRAL');
    acc[st] = (acc[st] || 0) + 1;
    return acc;
  }, {});
  const whiteRunNow = state.whiteRun[state.n - 1] || 0;
  const computedResults = TARGETS.map((t) => targetResultCache.get(Number(t))).filter(Boolean);
  const whiteGlobalContinue = mean(computedResults.map((r) => r.live.whiteContinue).filter((v) => Number.isFinite(v)));
  const whiteGlobalRebound = mean(computedResults.map((r) => r.live.whiteRebound).filter((v) => Number.isFinite(v)));
  const randomSignals = computedResults.map((r) => Boolean(r.live.entropy.randomLike));
  const performanceDashboard = computedResults.length
    ? summarizePerformanceDashboard(computedResults)
    : { perTarget: [], aggregate: { rollingEV: null, winRate: null, maxDrawdown: null, sharpeLike: null, edgeAccuracy: null } };
  const backtestSummary = computedResults.length
    ? summarizeBacktests(computedResults)
    : { perTarget: [], aggregate: { bets: 0, wins: 0, losses: 0, winRate: null, totalEV: null, avgEV: null, maxDrawdown: null, randomExpectedTotalEVAtSameBets: null, walkForward: true, noFutureLeakage: true } };

  return {
    model: 'EngineX-v1-edge',
    generatedAt: new Date().toISOString(),
    asOfRound: currentRound,
    sampleSize: state.n,
    targets: targetsOut,
    timingAlerts,
    locksToSave,
    resolvedHistory,
    whiteCluster: {
      activeRun: whiteRunNow,
      cut: roundNum(state.whiteCut, 4),
      reboundCut: roundNum(state.reboundCut, 4),
      continueProb: roundNum(whiteGlobalContinue, 6),
      reboundProb: roundNum(whiteGlobalRebound, 6),
      delta: roundNum((whiteGlobalRebound ?? 0) - (whiteGlobalContinue ?? 0), 6),
      weakModifierOnly: true,
    },
    summary: {
      waiting: waitingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: state.n,
      randomLikeTargets: randomSignals.filter(Boolean).length,
      preconditionStates,
    },
    backtest: backtestSummary,
    performanceDashboard,
    settings: {
      modelVersion: 'EngineX-v1-edge',
      mode: 'precondition-edge-detection',
      baselineFormula: 'P(x)=1/x',
      noFixedWindows: false,
      fixedWindowSpans: FIXED_WINDOW_SPAN,
      alwaysEmitLocks: Boolean(cfg.alwaysEmitLocks),
      immutableLocks: true,
      relockOnlyAfterResolved: true,
      perRoundPrediction: false,
      strictRealtimeRecalcOnOutcomeOnly: true,
      hazardAsWeakSignal: true,
      gapPressurePrimarySignal: false,
      blendByInverseBrier: true,
      entropyAutoDisable: Boolean(cfg.entropyAutoDisable),
      evThreshold: roundNum(cfg.evThreshold, 6),
      walkForwardValidation: true,
      noFutureLeakage: true,
    },
  };
}

function buildThresholdSnapshot(cleanRounds, threshold) {
  const n = cleanRounds.length;
  if (!n) {
    return { target: threshold, p1: null, baseline: roundNum(1 / threshold, 6), edge: null, ev: null };
  }
  const m = cleanRounds.map((r) => Number(r.multiplier));
  const hit = m.map((x) => (x >= threshold ? 1 : 0));
  const prefix = buildPrefix(hit);
  const recentWindow = Math.min(800, Math.max(100, Math.floor(n * 0.25)));
  const idx = n - 1;
  const recentRate = rateInWindow(prefix, idx, recentWindow);
  const baseline = 1 / threshold;
  const edge = recentRate - baseline;
  const ev = (recentRate * threshold) - 1;
  return {
    target: threshold,
    p1: roundNum(recentRate, 6),
    baseline: roundNum(baseline, 6),
    edge: roundNum(edge, 6),
    ev: roundNum(ev, 6),
    expectedGap: recentRate > 0 ? roundNum(1 / recentRate, 3) : null,
  };
}

function buildPredictionReport(rounds, options = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...(options.config || {}) };
  const cleanRounds = normalizeRounds(rounds);
  if (!cleanRounds.length) {
    return {
      model: 'EngineX-v1-edge-report',
      generatedAt: new Date().toISOString(),
      asOfRound: null,
      sampleSize: 0,
      bucketProbabilities: BUCKETS.map((b) => ({ ...b, probability: null })),
      predictedBucket: null,
      targetProbabilities: REPORT_THRESHOLDS.map((t) => ({
        target: t,
        p1: null,
        baseline: roundNum(1 / t, 6),
        edge: null,
        ev: null,
      })),
      whiteCluster: null,
      edgeSummary: null,
      backtest: null,
    };
  }

  const multipliers = cleanRounds.map((r) => Number(r.multiplier));
  const counts = new Array(BUCKETS.length).fill(0);
  for (let i = 0; i < multipliers.length; i += 1) {
    const m = multipliers[i];
    for (let j = 0; j < BUCKETS.length; j += 1) {
      const b = BUCKETS[j];
      if (m >= b.min && m <= b.max) {
        counts[j] += 1;
        break;
      }
    }
  }
  const total = multipliers.length;
  const bucketProbabilities = BUCKETS.map((b, i) => ({
    ...b,
    count: counts[i],
    probability: roundNum(counts[i] / Math.max(1, total), 6),
  }));
  const topIdx = counts.indexOf(Math.max(...counts));

  const targetProbabilities = REPORT_THRESHOLDS.map((t) => buildThresholdSnapshot(cleanRounds, t));
  const edgeSummary = targetProbabilities.map((x) => ({
    target: x.target,
    edge: x.edge,
    ev: x.ev,
    actionable: Number.isFinite(x.ev) ? (x.ev > cfg.evThreshold) : false,
  }));

  const state = buildGlobalState(cleanRounds, cfg);
  const whiteRunNow = state.whiteRun[state.n - 1] || 0;
  const whiteSeries = state.whiteRun.filter((x) => x > 0);
  const backtest = backtestEdgeEngine(cleanRounds, { config: cfg });

  return {
    model: 'EngineX-v1-edge-report',
    generatedAt: new Date().toISOString(),
    asOfRound: cleanRounds[cleanRounds.length - 1].roundId,
    sampleSize: cleanRounds.length,
    expectedMean: roundNum(mean(multipliers), 4),
    expectedMedian: roundNum(quantile(multipliers, 0.5), 4),
    expectedP75: roundNum(quantile(multipliers, 0.75), 4),
    expectedP90: roundNum(quantile(multipliers, 0.90), 4),
    bucketProbabilities,
    predictedBucket: bucketProbabilities[topIdx],
    targetProbabilities,
    whiteCluster: {
      activeRun: whiteRunNow,
      cut: roundNum(state.whiteCut, 4),
      reboundCut: roundNum(state.reboundCut, 4),
      runQ85: roundNum(quantile(whiteSeries, 0.85), 3),
      runQ95: roundNum(quantile(whiteSeries, 0.95), 3),
    },
    edgeSummary,
    backtest,
    diagnostics: {
      baselineFormula: 'P(x)=1/x',
      mode: 'precondition-edge-detection',
      noFixedWindows: false,
      fixedWindowSpans: FIXED_WINDOW_SPAN,
      walkForwardValidation: true,
      noFutureLeakage: true,
    },
  };
}

function backtestEdgeEngine(rounds, options = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...(options.config || {}) };
  const clean = normalizeRounds(rounds);
  if (!clean.length) {
    return {
      perTarget: [],
      aggregate: {
        bets: 0,
        wins: 0,
        losses: 0,
        winRate: null,
        totalEV: null,
        avgEV: null,
        maxDrawdown: null,
        randomExpectedTotalEVAtSameBets: null,
        walkForward: true,
        noFutureLeakage: true,
      },
    };
  }
  const state = buildGlobalState(clean, cfg);
  const targetResults = TARGETS.map((t) => runTargetWalkForward(state, t, cfg));
  return summarizeBacktests(targetResults);
}

module.exports = {
  TARGETS,
  BUCKETS,
  DEFAULT_CONFIG,
  computeLockedRangePredictions,
  buildPredictionReport,
  backtestEdgeEngine,
  _internal: {
    normalizeRounds,
    buildGlobalState,
    makeTargetState,
    hazardPredictAt,
    knnPredictAt,
    blendWeightsFromBrier,
    backtestEdgeEngine,
  },
};
