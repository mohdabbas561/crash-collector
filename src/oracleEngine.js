'use strict';

const ORACLE_TARGETS = Object.freeze([
  { label: '5x', minVal: 5, color: '#00ff88', window: 3, scanN: 120, minHits: 5 },
  { label: '10x', minVal: 10, color: '#00d4ff', window: 5, scanN: 100, minHits: 4 },
  { label: '15x', minVal: 15, color: '#ff6b9d', window: 7, scanN: 80, minHits: 3 },
  { label: '30x', minVal: 30, color: '#ff9f43', window: 12, scanN: 60, minHits: 2 },
  { label: '50x', minVal: 50, color: '#4db8ff', window: 18, scanN: 50, minHits: 2 },
  { label: '100x', minVal: 100, color: '#39ff8a', window: 25, scanN: 40, minHits: 2 },
  { label: '200x', minVal: 200, color: '#c77dff', window: 35, scanN: 30, minHits: 2 },
  { label: '500x', minVal: 500, color: '#ff4da6', window: 50, scanN: 24, minHits: 1 },
  { label: '1000x', minVal: 1000, color: '#7aa2ff', window: 75, scanN: 20, minHits: 1 },
]);

const REGIME_DRIFT_THRESHOLD = 0.35;
const MIN_FORECAST_GAPS = 8;
const MIN_KM_GAPS = 20;
const MIN_HIGH_QUANTILE_GAPS = 30;
const MIN_EXTREME_QUANTILE_GAPS = 60;
const MIN_BUCKET_CALIBRATION = 8;
const MIN_GLOBAL_CALIBRATION = 20;
const CALIBRATION_RECENT_LIMIT = 240;
const WHITE_CLUSTER_HARD_MAX = 1.25;
const WHITE_CLUSTER_SOFT_MAX = 1.6;

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

// Robust linear-interpolation quantile. We reuse this everywhere so the engine
// stays interpretable and deterministic.
function quantile(sorted, p) {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + ((sorted[hi] - sorted[lo]) * (idx - lo));
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function meanLog(values) {
  if (!values.length) return 0;
  return mean(values.map((value) => Math.log(Math.max(1.0001, value))));
}

function maxOrZero(values) {
  return values.length ? Math.max(...values) : 0;
}

function weightedPct(values, predicate) {
  if (!values.length) return 0;
  let hits = 0;
  let total = 0;
  const denom = Math.max(1, values.length - 1);
  for (let i = 0; i < values.length; i += 1) {
    const weight = 0.65 + ((i / denom) * 1.35);
    total += weight;
    if (predicate(values[i])) hits += weight;
  }
  return total > 0 ? (hits / total) * 100 : 0;
}

function computeTrendPercent(values) {
  if (values.length < 4) return 0;
  const logs = values.map((value) => Math.log(Math.max(1.0001, value)));
  const n = logs.length;
  const meanX = (n - 1) / 2;
  const meanY = mean(logs);
  let numer = 0;
  let denom = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = i - meanX;
    numer += dx * (logs[i] - meanY);
    denom += dx * dx;
  }
  if (denom <= 0 || meanY === 0) return 0;
  const slope = numer / denom;
  return ((slope * (n - 1)) / Math.max(0.0001, Math.abs(meanY))) * 100;
}

function buildSegmentPeaks(values, segments = 4) {
  if (!values.length) return [];
  const size = Math.max(1, Math.ceil(values.length / segments));
  const peaks = [];
  for (let start = 0; start < values.length; start += size) {
    peaks.push(maxOrZero(values.slice(start, start + size)));
  }
  return peaks.filter((value) => value > 0);
}

function normalizeRounds(rounds) {
  const mapped = new Map();
  for (const round of rounds || []) {
    const id = Number(round?.roundId ?? round?.id);
    const val = Number.parseFloat(round?.multiplier ?? round?.val);
    if (!Number.isFinite(id) || !Number.isFinite(val) || val <= 0) continue;
    mapped.set(id, { id, val });
  }
  return [...mapped.values()].sort((a, b) => a.id - b.id);
}

function trimSorted(sorted, trimRatio) {
  if (!sorted.length) return [];
  const trimCount = Math.floor(sorted.length * trimRatio);
  if (trimCount <= 0 || (trimCount * 2) >= sorted.length) return sorted.slice();
  return sorted.slice(trimCount, sorted.length - trimCount);
}

function tukeyFilterSorted(sorted) {
  if (sorted.length < 8) return sorted.slice();
  const q1 = quantile(sorted, 25);
  const q3 = quantile(sorted, 75);
  const iqr = Math.max(1, q3 - q1);
  const lo = q1 - (1.5 * iqr);
  const hi = q3 + (1.5 * iqr);
  const filtered = sorted.filter((gap) => gap >= lo && gap <= hi);
  return filtered.length >= Math.max(5, Math.floor(sorted.length * 0.6))
    ? filtered
    : sorted.slice();
}

// Robust stats are taken from the selected regime only. We never blend recent
// and full data once regime selection is made.
function buildRobustStats(sourceSorted) {
  const filtered = tukeyFilterSorted(sourceSorted);
  const trimmed = trimSorted(filtered, filtered.length >= 10 ? 0.1 : 0);
  const working = trimmed.length ? trimmed : filtered;
  const q25 = quantile(working, 25);
  const q50 = quantile(working, 50);
  const q75 = quantile(working, 75);
  const iqr = Math.max(1, q75 - q25);
  const p10 = working.length >= 10 ? quantile(working, 10) : working[0];
  const p90 = working.length >= MIN_HIGH_QUANTILE_GAPS ? quantile(working, 90) : q75;
  const p99 = working.length >= MIN_EXTREME_QUANTILE_GAPS ? quantile(working, 99) : p90;
  return {
    filtered,
    trimmed: working,
    min: working[0],
    max: working[working.length - 1],
    p10: Math.round(p10),
    p25: Math.round(q25),
    med: Math.round(q50),
    p75: Math.round(q75),
    p90: Math.round(p90),
    p99: Math.round(p99),
    iqr: Math.max(1, Math.round(iqr)),
    avg: Math.round(mean(working)),
  };
}

function deriveBinWidth(sorted) {
  if (sorted.length < 3) return 1;
  const q25 = quantile(sorted, 25);
  const q75 = quantile(sorted, 75);
  const iqr = Math.max(1, q75 - q25);
  const fdWidth = Math.max(1, Math.round((2 * iqr) / Math.cbrt(sorted.length)));
  const spread = Math.max(1, sorted[sorted.length - 1] - sorted[0]);
  const sqrtWidth = Math.max(1, Math.round(spread / Math.max(2, Math.sqrt(sorted.length))));
  return Math.max(1, Math.min(fdWidth, sqrtWidth));
}

// Histogram clustering is more stable than KDE here. It is discrete, supports
// multiple peaks, and lets us measure support directly.
function buildHistogramClusters(sorted) {
  if (!sorted.length) {
    return {
      binWidth: 1,
      primary: null,
      secondary: null,
      bins: [],
    };
  }

  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const binWidth = deriveBinWidth(sorted);
  const binCount = Math.max(1, Math.floor((max - min) / binWidth) + 1);
  const bins = Array.from({ length: binCount }, (_, index) => ({
    index,
    lo: min + (index * binWidth),
    hi: min + ((index + 1) * binWidth) - 1,
    count: 0,
    sum: 0,
  }));

  for (const gap of sorted) {
    const idx = clampNumber(Math.floor((gap - min) / binWidth), 0, binCount - 1);
    bins[idx].count += 1;
    bins[idx].sum += gap;
  }

  const smoothed = bins.map((bin, index) => {
    const prev = bins[index - 1]?.count || 0;
    const next = bins[index + 1]?.count || 0;
    return bin.count + ((prev + next) * 0.55);
  });

  const peaks = [];
  for (let i = 0; i < bins.length; i += 1) {
    const left = smoothed[i - 1] ?? -Infinity;
    const right = smoothed[i + 1] ?? -Infinity;
    if (smoothed[i] >= left && smoothed[i] >= right && bins[i].count > 0) {
      peaks.push({ index: i, strength: smoothed[i] });
    }
  }

  peaks.sort((a, b) => b.strength - a.strength);

  function expandCluster(peakIndex) {
    const peakStrength = smoothed[peakIndex];
    const threshold = peakStrength * 0.42;
    let lo = peakIndex;
    let hi = peakIndex;
    while (lo > 0 && smoothed[lo - 1] >= threshold) lo -= 1;
    while (hi < bins.length - 1 && smoothed[hi + 1] >= threshold) hi += 1;
    const clusterBins = bins.slice(lo, hi + 1);
    const supportCount = clusterBins.reduce((sum, bin) => sum + bin.count, 0);
    const weightedSum = clusterBins.reduce((sum, bin) => sum + bin.sum, 0);
    return {
      lo: clusterBins[0].lo,
      hi: clusterBins[clusterBins.length - 1].hi,
      center: supportCount > 0 ? Math.round(weightedSum / supportCount) : Math.round((clusterBins[0].lo + clusterBins[clusterBins.length - 1].hi) / 2),
      supportCount,
      supportPct: supportCount / sorted.length,
      peakStrength,
    };
  }

  const primary = peaks.length ? expandCluster(peaks[0].index) : null;
  let secondary = null;
  for (const peak of peaks.slice(1)) {
    const candidate = expandCluster(peak.index);
    if (
      primary &&
      Math.abs(candidate.center - primary.center) >= binWidth &&
      candidate.supportPct >= Math.max(0.12, primary.supportPct * 0.45)
    ) {
      secondary = candidate;
      break;
    }
  }

  return { binWidth, bins, primary, secondary };
}

// In an orderless gap model, a shuffled-gap null is identical. The honest null
// comparison is: "how often would a random window of the same size hit?".
function computeChanceWindowRate(sorted, width) {
  if (!sorted.length) return 0;
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  if (max <= min) return 100;
  const startMin = min;
  const startMax = Math.max(min, max - width + 1);
  const totalWindows = Math.max(1, startMax - startMin + 1);
  let coveredStarts = 0;

  // Exact mean occupancy across all integer-aligned windows of this width.
  for (const gap of sorted) {
    const loStart = Math.max(startMin, gap - width + 1);
    const hiStart = Math.min(startMax, gap);
    if (loStart <= hiStart) {
      coveredStarts += (hiStart - loStart + 1);
    }
  }

  return (coveredStarts / (totalWindows * sorted.length)) * 100;
}

function computeConditionalExpectedGap(survivors, roundsSince) {
  if (!survivors.length) return null;
  const filtered = tukeyFilterSorted(survivors);
  const trimmed = trimSorted(filtered, filtered.length >= 10 ? 0.1 : 0);
  const base = trimmed.length ? trimmed : filtered;
  const expectedGap = Math.round(mean(base));
  return Math.max(expectedGap, roundsSince + 1);
}

function getIssueThresholds(target) {
  if (target.minVal <= 10) {
    return { minProbability: 22, minLift: 0, minClusterSupport: 8, readinessFactor: 0.4 };
  }
  if (target.minVal <= 30) {
    return { minProbability: 18, minLift: 0, minClusterSupport: 8, readinessFactor: 0.45 };
  }
  if (target.minVal <= 100) {
    return { minProbability: 12, minLift: 0, minClusterSupport: 6, readinessFactor: 0.5 };
  }
  if (target.minVal <= 200) {
    return { minProbability: 9, minLift: 0, minClusterSupport: 6, readinessFactor: 0.55 };
  }
  if (target.minVal <= 500) {
    return { minProbability: 7, minLift: 0, minClusterSupport: 5, readinessFactor: 0.6 };
  }
  return { minProbability: 5, minLift: 0, minClusterSupport: 4, readinessFactor: 0.65 };
}

function getSampleRequirements(target) {
  if (target.minVal <= 10) {
    return { minForecastGaps: 8, minKMGaps: 16 };
  }
  if (target.minVal <= 30) {
    return { minForecastGaps: 7, minKMGaps: 14 };
  }
  if (target.minVal <= 100) {
    return { minForecastGaps: 6, minKMGaps: 12 };
  }
  if (target.minVal <= 200) {
    return { minForecastGaps: 5, minKMGaps: 10 };
  }
  if (target.minVal <= 500) {
    return { minForecastGaps: 4, minKMGaps: 8 };
  }
  return { minForecastGaps: 4, minKMGaps: 6 };
}

function getPatternThresholds(target) {
  if (target.minVal <= 10) {
    return {
      hardWhitePctThreshold: 84,
      softWhitePctThreshold: 94,
      lowPressureFactor: 0.48,
      previewFactor: 0.84,
      lowPressurePctThreshold: 72,
      tailLowPressurePctThreshold: 62,
      hardStreakThreshold: 5,
      softStreakThreshold: 8,
      downtrendThreshold: -26,
      tailPeakRatio: 0.68,
      b2bBlockRepeatMax: 12,
      b2bFriendlyRepeatMin: 28,
      b2bMomentumMin: 6,
      recentHitFactor: 0.45,
      minRecentHitsForFriendly: 2,
    };
  }
  if (target.minVal <= 30) {
    return {
      hardWhitePctThreshold: 74,
      softWhitePctThreshold: 88,
      lowPressureFactor: 0.26,
      previewFactor: 0.56,
      lowPressurePctThreshold: 66,
      tailLowPressurePctThreshold: 56,
      hardStreakThreshold: 4,
      softStreakThreshold: 6,
      downtrendThreshold: -22,
      tailPeakRatio: 0.72,
      b2bBlockRepeatMax: 16,
      b2bFriendlyRepeatMin: 26,
      b2bMomentumMin: 7,
      recentHitFactor: 0.38,
      minRecentHitsForFriendly: 2,
    };
  }
  if (target.minVal <= 100) {
    return {
      hardWhitePctThreshold: 64,
      softWhitePctThreshold: 82,
      lowPressureFactor: 0.18,
      previewFactor: 0.36,
      lowPressurePctThreshold: 60,
      tailLowPressurePctThreshold: 52,
      hardStreakThreshold: 4,
      softStreakThreshold: 6,
      downtrendThreshold: -18,
      tailPeakRatio: 0.78,
      b2bBlockRepeatMax: 18,
      b2bFriendlyRepeatMin: 22,
      b2bMomentumMin: 8,
      recentHitFactor: 0.3,
      minRecentHitsForFriendly: 1,
    };
  }
  if (target.minVal <= 500) {
    return {
      hardWhitePctThreshold: 54,
      softWhitePctThreshold: 74,
      lowPressureFactor: 0.08,
      previewFactor: 0.18,
      lowPressurePctThreshold: 54,
      tailLowPressurePctThreshold: 48,
      hardStreakThreshold: 3,
      softStreakThreshold: 5,
      downtrendThreshold: -15,
      tailPeakRatio: 0.82,
      b2bBlockRepeatMax: 20,
      b2bFriendlyRepeatMin: 16,
      b2bMomentumMin: 9,
      recentHitFactor: 0.22,
      minRecentHitsForFriendly: 1,
    };
  }
  return {
    hardWhitePctThreshold: 48,
    softWhitePctThreshold: 68,
    lowPressureFactor: 0.05,
    previewFactor: 0.1,
    lowPressurePctThreshold: 48,
    tailLowPressurePctThreshold: 44,
    hardStreakThreshold: 3,
    softStreakThreshold: 4,
    downtrendThreshold: -12,
    tailPeakRatio: 0.86,
    b2bBlockRepeatMax: 24,
    b2bFriendlyRepeatMin: 12,
    b2bMomentumMin: 10,
    recentHitFactor: 0.18,
    minRecentHitsForFriendly: 1,
  };
}

function getNearHitFactor(target) {
  if (target.minVal <= 10) return 0.7;
  if (target.minVal <= 30) return 0.58;
  if (target.minVal <= 100) return 0.46;
  if (target.minVal <= 500) return 0.34;
  return 0.28;
}

function computeRecentPatternDiagnostics(rounds, target, roundsSince, allGapsRaw, selectedStats) {
  const thresholds = getPatternThresholds(target);
  const lookback = clampNumber(Math.max(target.window * 8, 12), 12, 96);
  const recentRounds = rounds.slice(-lookback);
  const values = recentRounds.map((round) => round.val);
  const nearHitMin = Math.max(1.2, target.minVal * getNearHitFactor(target));
  const lowPressureMax = Math.max(WHITE_CLUSTER_SOFT_MAX, target.minVal * thresholds.lowPressureFactor);
  const previewMin = Math.max(
    lowPressureMax * (target.minVal >= 100 ? 1.8 : 1.45),
    target.minVal * thresholds.previewFactor
  );
  if (!values.length) {
    return {
      lookback,
      hardWhitePct: 0,
      softWhitePct: 0,
      lowPressurePct: 0,
      maxHardWhiteStreak: 0,
      maxSoftWhiteStreak: 0,
      maxLowPressureStreak: 0,
      recentTargetHits: 0,
      recentNearHits: 0,
      recentPreviewHits: 0,
      tailTargetHits: 0,
      tailNearHits: 0,
      tailPreviewHits: 0,
      headPreviewHits: 0,
      weightedHardWhitePct: 0,
      weightedSoftWhitePct: 0,
      weightedLowPressurePct: 0,
      headSoftWhitePct: 0,
      tailSoftWhitePct: 0,
      headLowPressurePct: 0,
      tailLowPressurePct: 0,
      endingHardWhiteStreak: 0,
      endingSoftWhiteStreak: 0,
      endingLowPressureStreak: 0,
      lowPressureCluster: false,
      compressionSupport: false,
      emergingWhiteRisk: false,
      releaseWatch: false,
      whiteRelease: false,
      reboundSupport: false,
      transitionSupportScore: 0,
      transitionReady: false,
      patternScore: 0,
      supportScore: 0,
      riskScore: 0,
      downtrend: false,
      downtrendEarly: false,
      downtrendPct: 0,
      shortRepeatRate: 0,
      localShortRepeatRate: 0,
      shortRepeatMomentum: 0,
      recentHitThreshold: 0,
      b2bRiskScore: 0,
      b2bSupportScore: 0,
      whiteRiskScore: 0,
      downtrendRiskScore: 0,
      b2bBlocked: false,
      b2bFriendly: false,
      recentHitTooSoon: false,
      previewMin,
    };
  }

  let hardWhiteCount = 0;
  let softWhiteCount = 0;
  let lowPressureCount = 0;
  let hardWhiteStreak = 0;
  let softWhiteStreak = 0;
  let lowPressureStreak = 0;
  let maxHardWhiteStreak = 0;
  let maxSoftWhiteStreak = 0;
  let maxLowPressureStreak = 0;

  for (const value of values) {
    if (value <= WHITE_CLUSTER_HARD_MAX) {
      hardWhiteCount += 1;
      hardWhiteStreak += 1;
    } else {
      hardWhiteStreak = 0;
    }

    if (value <= WHITE_CLUSTER_SOFT_MAX) {
      softWhiteCount += 1;
      softWhiteStreak += 1;
    } else {
      softWhiteStreak = 0;
    }

    if (value <= lowPressureMax) {
      lowPressureCount += 1;
      lowPressureStreak += 1;
    } else {
      lowPressureStreak = 0;
    }

    maxHardWhiteStreak = Math.max(maxHardWhiteStreak, hardWhiteStreak);
    maxSoftWhiteStreak = Math.max(maxSoftWhiteStreak, softWhiteStreak);
    maxLowPressureStreak = Math.max(maxLowPressureStreak, lowPressureStreak);
  }
  const endingHardWhiteStreak = hardWhiteStreak;
  const endingSoftWhiteStreak = softWhiteStreak;
  const endingLowPressureStreak = lowPressureStreak;

  const split = Math.max(1, Math.floor(values.length / 2));
  const firstHalf = values.slice(0, split);
  const secondHalf = values.slice(split);
  const last4 = values.slice(-Math.min(4, values.length));
  const prev4 = values.slice(-Math.min(8, values.length), -Math.min(4, values.length));
  const tailWindowSize = Math.max(4, Math.min(values.length, Math.max(target.window * 2, 6)));
  const tailValues = values.slice(-tailWindowSize);
  const headValues = values.slice(0, Math.max(1, values.length - tailWindowSize));
  const headLogMean = meanLog(firstHalf);
  const tailLogMean = meanLog(secondHalf);
  const trendDelta = headLogMean > 0 ? ((tailLogMean - headLogMean) / headLogMean) : 0;
  const trendPct = computeTrendPercent(values);
  const headPeak = maxOrZero(firstHalf);
  const tailPeak = maxOrZero(secondHalf);
  const recentTargetHits = values.filter((value) => value >= target.minVal).length;
  const recentNearHits = values.filter((value) => value >= nearHitMin).length;
  const recentPreviewHits = values.filter((value) => value >= previewMin).length;
  const tailTargetHits = tailValues.filter((value) => value >= target.minVal).length;
  const tailNearHits = tailValues.filter((value) => value >= nearHitMin).length;
  const tailPreviewHits = tailValues.filter((value) => value >= previewMin).length;
  const headPreviewHits = headValues.filter((value) => value >= previewMin).length;
  const weightedHardWhitePct = weightedPct(values, (value) => value <= WHITE_CLUSTER_HARD_MAX);
  const weightedSoftWhitePct = weightedPct(values, (value) => value <= WHITE_CLUSTER_SOFT_MAX);
  const weightedLowPressurePct = weightedPct(values, (value) => value <= lowPressureMax);
  const headSoftWhitePct = headValues.length
    ? (headValues.filter((value) => value <= WHITE_CLUSTER_SOFT_MAX).length / headValues.length) * 100
    : 0;
  const tailSoftWhitePct = tailValues.length
    ? (tailValues.filter((value) => value <= WHITE_CLUSTER_SOFT_MAX).length / tailValues.length) * 100
    : 0;
  const headLowPressurePct = headValues.length
    ? (headValues.filter((value) => value <= lowPressureMax).length / headValues.length) * 100
    : 0;
  const tailLowPressurePct = tailValues.length
    ? (tailValues.filter((value) => value <= lowPressureMax).length / tailValues.length) * 100
    : 0;
  const last4SoftWhitePct = last4.length
    ? (last4.filter((value) => value <= WHITE_CLUSTER_SOFT_MAX).length / last4.length) * 100
    : 0;
  const last4LowPressurePct = last4.length
    ? (last4.filter((value) => value <= lowPressureMax).length / last4.length) * 100
    : 0;
  const segmentCount = clampNumber(
    Math.round(values.length / Math.max(4, Math.min(18, target.window))),
    3,
    6
  );
  const segmentPeaks = buildSegmentPeaks(values, segmentCount);
  let lowerHighCount = 0;
  let risingHighCount = 0;
  for (let i = 1; i < segmentPeaks.length; i += 1) {
    const declineFactor = target.minVal <= 30 ? 0.9 : target.minVal <= 100 ? 0.87 : 0.84;
    const riseFactor = target.minVal <= 30 ? 1.07 : target.minVal <= 100 ? 1.1 : 1.14;
    if (segmentPeaks[i] <= segmentPeaks[i - 1] * declineFactor) lowerHighCount += 1;
    if (segmentPeaks[i] >= segmentPeaks[i - 1] * riseFactor) risingHighCount += 1;
  }

  const hardWhitePct = (hardWhiteCount / values.length) * 100;
  const softWhitePct = (softWhiteCount / values.length) * 100;
  const lowPressurePct = (lowPressureCount / values.length) * 100;
  const earlyDowntrendPct = thresholds.downtrendThreshold * 0.72;
  const downtrend = (
    trendDelta <= (thresholds.downtrendThreshold / 100) &&
    tailPeak <= (headPeak * thresholds.tailPeakRatio)
  );
  const lowerHighRequirement = target.minVal <= 30 ? 2 : 3;
  const downtrendEarly = (
    trendPct <= earlyDowntrendPct ||
    (
      lowerHighCount >= lowerHighRequirement &&
      tailPeak <= Math.max(nearHitMin, headPeak * (thresholds.tailPeakRatio + 0.08))
    ) ||
    (
      prev4.length > 0 &&
      last4.length > 0 &&
      meanLog(last4) < meanLog(prev4) * 0.92 &&
      tailNearHits === 0 &&
      tailLowPressurePct >= thresholds.tailLowPressurePctThreshold
    )
  );

  const shortRepeatWindow = Math.max(2, Math.min(target.window, 6));
  const shortRepeatRate = allGapsRaw.length
    ? allGapsRaw.filter((gap) => gap <= shortRepeatWindow).length / allGapsRaw.length
    : 0;
  const localGapSample = allGapsRaw.slice(-Math.min(Math.max(8, target.window * 2), 24));
  const localShortRepeatRate = localGapSample.length
    ? localGapSample.filter((gap) => gap <= shortRepeatWindow).length / localGapSample.length
    : shortRepeatRate;
  const olderGapSample = allGapsRaw.slice(
    Math.max(0, allGapsRaw.length - (localGapSample.length * 2)),
    Math.max(0, allGapsRaw.length - localGapSample.length)
  );
  const olderShortRepeatRate = olderGapSample.length
    ? olderGapSample.filter((gap) => gap <= shortRepeatWindow).length / olderGapSample.length
    : shortRepeatRate;
  const shortRepeatMomentum = (localShortRepeatRate - olderShortRepeatRate) * 100;
  const localGapSorted = [...localGapSample].sort((a, b) => a - b);
  const localGapP25 = localGapSorted.length >= 4
    ? Math.round(quantile(localGapSorted, 25))
    : shortRepeatWindow;
  const recentHitThreshold = Math.max(1, Math.min(
    target.window,
    localGapP25,
    Math.max(1, Math.round(selectedStats.med * thresholds.recentHitFactor))
  ));
  const recentHitTooSoon = roundsSince <= recentHitThreshold;
  const lowPressureCluster = (
    weightedLowPressurePct >= thresholds.lowPressurePctThreshold &&
    (
      tailLowPressurePct >= thresholds.tailLowPressurePctThreshold ||
      endingLowPressureStreak >= Math.max(2, Math.ceil(target.window * 0.4))
    )
  );
  const whiteCluster = (
    hardWhitePct >= thresholds.hardWhitePctThreshold ||
    softWhitePct >= thresholds.softWhitePctThreshold ||
    maxHardWhiteStreak >= thresholds.hardStreakThreshold ||
    maxSoftWhiteStreak >= thresholds.softStreakThreshold ||
    (
      target.minVal >= 30 &&
      lowPressureCluster &&
      maxLowPressureStreak >= Math.max(3, Math.ceil(target.window * 0.45))
    )
  );
  const emergingWhiteRisk = !whiteCluster && (
    weightedSoftWhitePct >= Math.max(54, thresholds.softWhitePctThreshold - 12) ||
    weightedHardWhitePct >= Math.max(28, thresholds.hardWhitePctThreshold - 12) ||
    (tailSoftWhitePct - headSoftWhitePct) >= 16 ||
    weightedLowPressurePct >= Math.max(46, thresholds.lowPressurePctThreshold - 10) ||
    (tailLowPressurePct - headLowPressurePct) >= 12 ||
    last4SoftWhitePct >= Math.max(50, thresholds.softWhitePctThreshold - 22) ||
    last4LowPressurePct >= Math.max(50, thresholds.tailLowPressurePctThreshold)
  ) && tailNearHits === 0 && tailTargetHits === 0 && tailPreviewHits === 0;
  const releaseWatch = (
    headLowPressurePct >= Math.max(42, thresholds.lowPressurePctThreshold - 14) &&
    tailLowPressurePct <= Math.max(44, thresholds.tailLowPressurePctThreshold - 4) &&
    (headLowPressurePct - tailLowPressurePct) >= 10 &&
    (
      trendPct > (thresholds.downtrendThreshold * 0.9) ||
      risingHighCount > 0 ||
      tailPreviewHits > 0 ||
      tailPeak >= previewMin
    )
  );
  const whiteRelease = releaseWatch && (
    tailNearHits > 0 ||
    tailTargetHits > 0 ||
    tailPreviewHits > 0 ||
    (last4.some((value) => value >= nearHitMin) && risingHighCount > 0)
  );
  const previewHitFloor = Math.max(1, Math.ceil(tailWindowSize * (target.minVal <= 30 ? 0.18 : target.minVal <= 100 ? 0.14 : 0.1)));
  const compressionSupport = (
    target.minVal >= 30 &&
    tailLowPressurePct >= Math.max(36, thresholds.tailLowPressurePctThreshold - 8) &&
    !downtrend &&
    !downtrendEarly &&
    (
      risingHighCount > 0 ||
      trendPct > (thresholds.downtrendThreshold * 0.45) ||
      tailPreviewHits >= previewHitFloor
    ) &&
    (
      tailPeak >= Math.max(lowPressureMax * 1.55, previewMin) ||
      last4.some((value) => value >= Math.max(lowPressureMax * 1.35, previewMin * 0.82))
    ) &&
    (
      tailPreviewHits > 0 ||
      tailNearHits > 0 ||
      tailTargetHits > 0 ||
      last4LowPressurePct < tailLowPressurePct ||
      tailPeak >= Math.max(headPeak * 0.92, previewMin)
    )
  );
  const reboundSupport = (
    whiteRelease ||
    compressionSupport ||
    (releaseWatch && risingHighCount > 0) ||
    (tailPreviewHits >= previewHitFloor) ||
    (tailNearHits >= Math.max(1, Math.ceil(tailWindowSize * 0.18))) ||
    (
      tailPeak >= Math.max(lowPressureMax * 1.12, previewMin, headPeak * 0.9) &&
      trendDelta >= -0.04 &&
      risingHighCount > 0
    )
  );
  const b2bFriendly = (
    recentHitTooSoon &&
    (
      localShortRepeatRate >= (thresholds.b2bFriendlyRepeatMin / 100) ||
      shortRepeatMomentum >= thresholds.b2bMomentumMin ||
      recentTargetHits >= thresholds.minRecentHitsForFriendly ||
      tailTargetHits > 0 ||
      (target.minVal <= 30 && tailPreviewHits > 0) ||
      (tailNearHits >= Math.max(1, Math.ceil(tailWindowSize * 0.2)) && !downtrendEarly) ||
      reboundSupport
    )
  );
  const b2bBlocked = (
    recentHitTooSoon &&
    !b2bFriendly &&
    localShortRepeatRate < (thresholds.b2bBlockRepeatMax / 100) &&
    shortRepeatMomentum < thresholds.b2bMomentumMin &&
    tailNearHits === 0 &&
    tailTargetHits === 0 &&
    !reboundSupport
  );

  let whiteRiskScore = 0;
  if (whiteCluster) whiteRiskScore += 24;
  if (lowPressureCluster) whiteRiskScore += 12;
  if (emergingWhiteRisk) whiteRiskScore += 18;
  whiteRiskScore += clampNumber((weightedSoftWhitePct - Math.max(40, thresholds.softWhitePctThreshold - 24)) * 0.35, 0, 16);
  whiteRiskScore += clampNumber((weightedHardWhitePct - Math.max(14, thresholds.hardWhitePctThreshold - 22)) * 0.45, 0, 14);
  whiteRiskScore += clampNumber((weightedLowPressurePct - Math.max(34, thresholds.lowPressurePctThreshold - 18)) * 0.3, 0, 16);
  whiteRiskScore += clampNumber((tailSoftWhitePct - headSoftWhitePct) * 0.35, 0, 12);
  whiteRiskScore += clampNumber((tailLowPressurePct - headLowPressurePct) * 0.32, 0, 12);
  whiteRiskScore += clampNumber((endingLowPressureStreak - Math.max(1, Math.ceil(target.window * 0.25))) * 3, 0, 12);
  if (whiteRelease) whiteRiskScore -= 12;
  if (releaseWatch) whiteRiskScore -= 6;
  if (compressionSupport) whiteRiskScore -= 8;
  if (reboundSupport) whiteRiskScore -= 6;
  whiteRiskScore -= clampNumber(tailPreviewHits * (target.minVal <= 30 ? 4 : target.minVal <= 100 ? 3 : 2), 0, 10);
  whiteRiskScore = Math.round(clampNumber(whiteRiskScore, 0, 80));

  let downtrendRiskScore = 0;
  if (downtrendEarly) downtrendRiskScore += 12;
  if (downtrend) downtrendRiskScore += 16;
  downtrendRiskScore += clampNumber((Math.abs(Math.min(0, trendPct)) - Math.abs(earlyDowntrendPct)) * 0.4, 0, 12);
  downtrendRiskScore += clampNumber(
    lowerHighCount * (target.minVal <= 30 ? 3 : target.minVal <= 100 ? 2.5 : 2),
    0,
    12
  );
  if (risingHighCount > 0) downtrendRiskScore -= 6;
  if (reboundSupport) downtrendRiskScore -= 5;
  if (tailPreviewHits > 0) downtrendRiskScore -= Math.min(6, tailPreviewHits * 2);
  downtrendRiskScore = Math.round(clampNumber(downtrendRiskScore, 0, 70));

  let b2bSupportScore = 0;
  if (recentHitTooSoon) {
    b2bSupportScore += clampNumber((localShortRepeatRate * 100) - thresholds.b2bFriendlyRepeatMin, 0, 18);
    b2bSupportScore += clampNumber(shortRepeatMomentum - thresholds.b2bMomentumMin + 3, 0, 12);
    if (recentTargetHits >= thresholds.minRecentHitsForFriendly) b2bSupportScore += 10;
    if (tailTargetHits > 0) b2bSupportScore += 10;
    if (tailNearHits >= Math.max(1, Math.ceil(tailWindowSize * 0.2))) b2bSupportScore += 8;
    if (target.minVal <= 30 && tailPreviewHits > 0) b2bSupportScore += 6;
    if (reboundSupport) b2bSupportScore += 6;
  }
  b2bSupportScore = Math.round(clampNumber(b2bSupportScore, 0, 60));

  let b2bRiskScore = 0;
  if (recentHitTooSoon) {
    if (localShortRepeatRate < (thresholds.b2bBlockRepeatMax / 100)) {
      b2bRiskScore += clampNumber(((thresholds.b2bBlockRepeatMax / 100) - localShortRepeatRate) * 100, 0, 18);
    }
    if (tailNearHits === 0 && tailTargetHits === 0) b2bRiskScore += 10;
    if (
      tailSoftWhitePct >= Math.max(50, thresholds.softWhitePctThreshold - 18) ||
      tailLowPressurePct >= thresholds.tailLowPressurePctThreshold
    ) b2bRiskScore += 10;
    if (tailPreviewHits > 0) b2bRiskScore -= Math.min(6, tailPreviewHits * 2);
    if (downtrendEarly) b2bRiskScore += 8;
  }
  if (b2bSupportScore > 0) b2bRiskScore -= Math.round(Math.min(10, b2bSupportScore * 0.35));
  b2bRiskScore = Math.round(clampNumber(b2bRiskScore, 0, 60));

  let transitionSupportScore = 0;
  if (releaseWatch) transitionSupportScore += 8;
  if (whiteRelease) transitionSupportScore += 16;
  if (compressionSupport) transitionSupportScore += target.minVal >= 100 ? 18 : 12;
  transitionSupportScore += clampNumber((tailPreviewHits - headPreviewHits) * 4, 0, 16);
  transitionSupportScore += clampNumber(tailPreviewHits * (target.minVal <= 30 ? 4 : target.minVal <= 100 ? 3 : 2), 0, 12);
  transitionSupportScore += clampNumber(risingHighCount * 4, 0, 12);
  transitionSupportScore += clampNumber((headLowPressurePct - tailLowPressurePct) * 0.22, 0, 10);
  if (trendPct > (thresholds.downtrendThreshold * 0.35)) transitionSupportScore += 6;
  transitionSupportScore = Math.round(clampNumber(transitionSupportScore, 0, 80));
  const transitionReady = (
    whiteRelease ||
    compressionSupport ||
    (releaseWatch && (tailPreviewHits > 0 || risingHighCount > 0)) ||
    transitionSupportScore >= (target.minVal <= 30 ? 16 : target.minVal <= 100 ? 14 : 12)
  );

  const supportScore = Math.round(clampNumber(
    Math.min(20, recentTargetHits * 6) +
    Math.min(14, Math.max(0, recentNearHits - recentTargetHits) * 3) +
    Math.min(14, Math.max(0, recentPreviewHits - recentNearHits) * (target.minVal <= 30 ? 3.5 : target.minVal <= 100 ? 3 : 2.5)) +
    Math.min(18, tailTargetHits * 8) +
    Math.min(12, Math.max(0, tailNearHits - tailTargetHits) * 2.5) +
    Math.min(14, Math.max(0, tailPreviewHits - tailNearHits) * (target.minVal <= 30 ? 4 : target.minVal <= 100 ? 3.5 : 3)) +
    (releaseWatch ? 10 : 0) +
    (whiteRelease ? 18 : 0) +
    (reboundSupport ? 12 : 0) +
    (trendPct > 4 ? 8 : 0) +
    Math.min(20, b2bSupportScore) +
    Math.min(20, transitionSupportScore * 0.45),
    0,
    100
  ));
  const riskScore = Math.round(clampNumber(
    whiteRiskScore +
    downtrendRiskScore +
    b2bRiskScore,
    0,
    100
  ));
  const patternScore = Math.round(clampNumber(supportScore - (riskScore * 0.75), -40, 100));

  return {
    lookback,
    hardWhitePct: Number(hardWhitePct.toFixed(1)),
    softWhitePct: Number(softWhitePct.toFixed(1)),
    lowPressurePct: Number(lowPressurePct.toFixed(1)),
    weightedHardWhitePct: Number(weightedHardWhitePct.toFixed(1)),
    weightedSoftWhitePct: Number(weightedSoftWhitePct.toFixed(1)),
    weightedLowPressurePct: Number(weightedLowPressurePct.toFixed(1)),
    maxHardWhiteStreak,
    maxSoftWhiteStreak,
    maxLowPressureStreak,
    endingHardWhiteStreak,
    endingSoftWhiteStreak,
    endingLowPressureStreak,
    recentTargetHits,
    recentNearHits,
    recentPreviewHits,
    tailTargetHits,
    tailNearHits,
    tailPreviewHits,
    headPreviewHits,
    whiteCluster,
    lowPressureCluster,
    compressionSupport,
    emergingWhiteRisk,
    headSoftWhitePct: Number(headSoftWhitePct.toFixed(1)),
    tailSoftWhitePct: Number(tailSoftWhitePct.toFixed(1)),
    headLowPressurePct: Number(headLowPressurePct.toFixed(1)),
    tailLowPressurePct: Number(tailLowPressurePct.toFixed(1)),
    releaseWatch,
    whiteRelease,
    reboundSupport,
    transitionSupportScore,
    transitionReady,
    patternScore,
    supportScore,
    riskScore,
    downtrend,
    downtrendEarly,
    downtrendPct: Number(trendPct.toFixed(1)),
    shortRepeatRate: Number((shortRepeatRate * 100).toFixed(1)),
    localShortRepeatRate: Number((localShortRepeatRate * 100).toFixed(1)),
    shortRepeatMomentum: Number(shortRepeatMomentum.toFixed(1)),
    recentHitThreshold,
    b2bRiskScore,
    b2bSupportScore,
    whiteRiskScore,
    downtrendRiskScore,
    b2bBlocked,
    b2bFriendly,
    recentHitTooSoon,
    previewMin: Number(previewMin.toFixed(1)),
  };
}

function buildKMTable(allGapsSorted, maxHorizon = null) {
  if (!allGapsSorted.length) return new Float32Array(1).fill(1);
  const n = allGapsSorted.length;
  const naturalLimit = allGapsSorted[n - 1] + 30;
  const limit = Math.max(
    1,
    Math.min(
      naturalLimit,
      Number.isFinite(maxHorizon) ? Math.max(1, Math.round(maxHorizon)) : naturalLimit
    )
  );
  const table = new Float32Array(limit + 1).fill(1);
  let surv = 1;
  let leftPointer = 0;

  for (let t = 1; t <= limit; t += 1) {
    while (leftPointer < n && allGapsSorted[leftPointer] < t) leftPointer += 1;
    let rightPointer = leftPointer;
    while (rightPointer < n && allGapsSorted[rightPointer] === t) rightPointer += 1;
    const atRisk = n - leftPointer;
    const events = rightPointer - leftPointer;
    if (atRisk === 0) {
      table[t] = table[t - 1];
      continue;
    }
    surv *= (1 - (events / atRisk));
    table[t] = surv;
  }

  return table;
}

function kmProb(kmTable, roundsSince, roundsAhead) {
  const from = Math.min(roundsSince, kmTable.length - 1);
  const to = Math.min(roundsSince + roundsAhead, kmTable.length - 1);
  const sFrom = kmTable[from];
  if (sFrom <= 0) return 100;
  return clampNumber(Math.round((1 - (kmTable[to] / sFrom)) * 1000) / 10, 0, 100);
}

function kmIntervalProb(kmTable, roundsSince, startAhead, endAhead) {
  const from = Math.min(roundsSince, kmTable.length - 1);
  const startExclusive = Math.min(
    roundsSince + Math.max(0, startAhead) - 1,
    kmTable.length - 1
  );
  const end = Math.min(roundsSince + Math.max(0, endAhead), kmTable.length - 1);
  const sFrom = kmTable[from];
  if (sFrom <= 0 || end <= startExclusive) return 0;
  const startSurvival = kmTable[Math.max(from, startExclusive)];
  const endSurvival = kmTable[end];
  return clampNumber(
    Math.round((((startSurvival - endSurvival) / sFrom) * 1000)) / 10,
    0,
    100
  );
}

function summarizeCalibrationRows(rows) {
  const buckets = Array.from({ length: 10 }, (_, index) => ({
    lo: index * 10,
    hi: (index * 10) + 9,
    wins: 0,
    total: 0,
  }));

  let globalWins = 0;
  let globalLosses = 0;

  for (const row of rows) {
    const probPercent = clampNumber(Math.round(Number(row.probW || 0) * 100), 0, 99);
    const bucketIndex = clampNumber(Math.floor(probPercent / 10), 0, 9);
    buckets[bucketIndex].total += 1;
    if (row.outcome === 'win') {
      buckets[bucketIndex].wins += 1;
      globalWins += 1;
    } else if (row.outcome === 'loss') {
      globalLosses += 1;
    }
  }

  const globalTotal = globalWins + globalLosses;
  return {
    buckets,
    globalTotal,
    globalRate: globalTotal >= MIN_GLOBAL_CALIBRATION ? (globalWins / globalTotal) * 100 : null,
  };
}

function buildCalibrationModel(rows, issueMode, regimeMode) {
  const usable = (rows || [])
    .filter((row) => row && row.probW != null && row.outcome !== 'early')
    .slice(0, CALIBRATION_RECENT_LIMIT);
  const sameRegimeRows = regimeMode
    ? usable.filter((row) => String(row.regimeMode || 'full') === String(regimeMode))
    : usable;
  const sameModeRows = issueMode
    ? usable.filter((row) => String(row.issueMode || '') === String(issueMode))
    : [];
  const sameModeSameRegimeRows = (issueMode && regimeMode)
    ? sameModeRows.filter((row) => String(row.regimeMode || 'full') === String(regimeMode))
    : sameModeRows;

  return {
    sameModeSameRegime: summarizeCalibrationRows(sameModeSameRegimeRows),
    sameMode: summarizeCalibrationRows(sameModeRows),
    sameRegime: summarizeCalibrationRows(sameRegimeRows),
    allModes: summarizeCalibrationRows(usable),
  };
}

function calibrateProbability(rawPercent, calibrationRows, issueMode, regimeMode) {
  const raw = clampNumber(Number(rawPercent || 0), 0, 100);
  const bucketIndex = clampNumber(Math.floor(Math.min(raw, 99) / 10), 0, 9);
  const model = buildCalibrationModel(calibrationRows, issueMode, regimeMode);
  const sameModeSameRegimeBucket = model.sameModeSameRegime.buckets[bucketIndex];
  const sameModeBucket = model.sameMode.buckets[bucketIndex];
  const sameRegimeBucket = model.sameRegime.buckets[bucketIndex];
  const allModesBucket = model.allModes.buckets[bucketIndex];

  if (issueMode && regimeMode && sameModeSameRegimeBucket.total >= MIN_BUCKET_CALIBRATION) {
    return {
      calibrated: clampNumber((sameModeSameRegimeBucket.wins / sameModeSameRegimeBucket.total) * 100, 0, 100),
      bucketLabel: `${sameModeSameRegimeBucket.lo}-${sameModeSameRegimeBucket.hi}%`,
      support: sameModeSameRegimeBucket.total,
      mode: `${issueMode}_${regimeMode}_bucket`,
    };
  }

  if (issueMode && regimeMode && model.sameModeSameRegime.globalRate != null) {
    return {
      calibrated: model.sameModeSameRegime.globalRate,
      bucketLabel: 'global',
      support: model.sameModeSameRegime.globalTotal,
      mode: `${issueMode}_${regimeMode}_global`,
    };
  }

  if (issueMode && sameModeBucket.total >= MIN_BUCKET_CALIBRATION) {
    return {
      calibrated: clampNumber((sameModeBucket.wins / sameModeBucket.total) * 100, 0, 100),
      bucketLabel: `${sameModeBucket.lo}-${sameModeBucket.hi}%`,
      support: sameModeBucket.total,
      mode: `${issueMode}_bucket`,
    };
  }

  if (issueMode && model.sameMode.globalRate != null) {
    return {
      calibrated: model.sameMode.globalRate,
      bucketLabel: 'global',
      support: model.sameMode.globalTotal,
      mode: `${issueMode}_global`,
    };
  }

  if (regimeMode && sameRegimeBucket.total >= MIN_BUCKET_CALIBRATION) {
    return {
      calibrated: clampNumber((sameRegimeBucket.wins / sameRegimeBucket.total) * 100, 0, 100),
      bucketLabel: `${sameRegimeBucket.lo}-${sameRegimeBucket.hi}%`,
      support: sameRegimeBucket.total,
      mode: `${regimeMode}_bucket`,
    };
  }

  if (regimeMode && model.sameRegime.globalRate != null) {
    return {
      calibrated: model.sameRegime.globalRate,
      bucketLabel: 'global',
      support: model.sameRegime.globalTotal,
      mode: `${regimeMode}_global`,
    };
  }

  if (allModesBucket.total >= MIN_BUCKET_CALIBRATION) {
    return {
      calibrated: clampNumber((allModesBucket.wins / allModesBucket.total) * 100, 0, 100),
      bucketLabel: `${allModesBucket.lo}-${allModesBucket.hi}%`,
      support: allModesBucket.total,
      mode: 'target_bucket',
    };
  }

  if (model.allModes.globalRate != null) {
    return {
      calibrated: model.allModes.globalRate,
      bucketLabel: 'global',
      support: model.allModes.globalTotal,
      mode: 'target_global',
    };
  }

  return {
    calibrated: raw,
    bucketLabel: `${allModesBucket.lo}-${allModesBucket.hi}%`,
    support: allModesBucket.total,
    mode: 'raw',
  };
}

function computeOracleForecast(rounds, target, options = {}) {
  const normalizedRounds = Array.isArray(rounds)
    && rounds.every((round) => Number.isFinite(round?.id) && Number.isFinite(round?.val))
      ? rounds
      : normalizeRounds(rounds);
  const { minVal, scanN, window: winSize, minHits } = target;
  if (!normalizedRounds.length) return null;

  const nowId = normalizedRounds[normalizedRounds.length - 1].id;
  const hits = normalizedRounds.filter((round) => round.val >= minVal);

  if (hits.length < minHits + 1) {
    if (!hits.length) {
      return { ...target, noData: true, hits: 0, nowId, reason: 'No hits yet' };
    }
    return {
      ...target,
      noData: true,
      hits: hits.length,
      nowId,
      reason: `Need ${minHits + 1} hits to predict`,
      lastHit: hits[hits.length - 1],
    };
  }

  const allGapsRaw = [];
  for (let i = 1; i < hits.length; i += 1) {
    allGapsRaw.push(hits[i].id - hits[i - 1].id);
  }
  if (!allGapsRaw.length) {
    return { ...target, noData: true, hits: hits.length, nowId, reason: 'Not enough gap data' };
  }

  const allGapsSorted = [...allGapsRaw].sort((a, b) => a - b);
  const fullStats = buildRobustStats(allGapsSorted);
  const fullHistogram = buildHistogramClusters(fullStats.trimmed);
  const recentCount = Math.min(scanN, allGapsRaw.length);
  const recentSorted = [...allGapsRaw.slice(-recentCount)].sort((a, b) => a - b);
  const recentStats = buildRobustStats(recentSorted);
  const recentHistogram = buildHistogramClusters(recentStats.trimmed);
  const medDrift = fullStats.med > 0 ? Math.abs(recentStats.med - fullStats.med) / fullStats.med : 0;
  const iqrDrift = fullStats.iqr > 0 ? Math.abs(recentStats.iqr - fullStats.iqr) / fullStats.iqr : 0;
  const upperDrift = fullStats.p75 > 0 ? Math.abs(recentStats.p75 - fullStats.p75) / fullStats.p75 : 0;
  const clusterDrift = (
    fullHistogram.primary?.center > 0 &&
    recentHistogram.primary?.center > 0
  )
    ? Math.abs(recentHistogram.primary.center - fullHistogram.primary.center) / fullHistogram.primary.center
    : medDrift;
  const regimeDrift = (
    (medDrift * 0.45) +
    (iqrDrift * 0.2) +
    (upperDrift * 0.15) +
    (clusterDrift * 0.2)
  );
  const regimeMode = regimeDrift > REGIME_DRIFT_THRESHOLD ? 'recent' : 'full';
  const selectedSorted = regimeMode === 'recent' ? recentSorted : allGapsSorted;
  const selectedStats = regimeMode === 'recent' ? recentStats : fullStats;
  const selectedCount = selectedStats.trimmed.length;
  const sampleRequirements = getSampleRequirements(target);

  const lastHit = hits[hits.length - 1];
  const roundsSince = nowId - lastHit.id;
  const isTooEarly = roundsSince < selectedStats.p10;
  const isOverdue = roundsSince > selectedStats.med;
  const isHardGap = roundsSince > selectedStats.p90;
  const isExtreme = roundsSince > selectedStats.p99;
  const survivingGaps = selectedStats.trimmed.filter((gap) => gap > roundsSince);
  const histogram = buildHistogramClusters(selectedStats.trimmed);
  const candidateClusters = [histogram.primary, histogram.secondary].filter(Boolean);
  const futureCluster = candidateClusters
    .filter((cluster) => cluster.center > roundsSince)
    .sort((a, b) => a.center - b.center)[0];
  const primaryCluster = futureCluster || (!isOverdue ? histogram.primary : null);
  const openWindow = isExtreme && survivingGaps.length === 0;
  const lowData = selectedCount < sampleRequirements.minForecastGaps;
  const kmReliable = selectedCount >= sampleRequirements.minKMGaps;
  const recentPattern = computeRecentPatternDiagnostics(
    normalizedRounds,
    target,
    roundsSince,
    allGapsRaw,
    selectedStats
  );

  let predictedGap;
  let predBasis;
  let predMethod;

  if (openWindow) {
    const tailStep = Math.max(
      1,
      selectedStats.iqr,
      selectedStats.max - selectedStats.p75,
      Math.round(selectedStats.med * 0.25)
    );
    predictedGap = selectedStats.max + tailStep;
    predBasis = `tail extrapolation (${regimeMode}, no survivor gaps)`;
    predMethod = 'tail';
  } else if (!isOverdue && primaryCluster) {
    predictedGap = Math.max(primaryCluster.center, roundsSince + 1);
    predBasis = `histogram cluster (${regimeMode}, ${selectedCount} gaps)`;
    predMethod = 'cluster';
  } else if (survivingGaps.length > 0) {
    predictedGap = computeConditionalExpectedGap(survivingGaps, roundsSince);
    predBasis = `conditional expectation (${survivingGaps.length} survivors)`;
    predMethod = 'survival';
  } else {
    predictedGap = Math.max(selectedStats.med, roundsSince + 1);
    predBasis = `fallback median (${regimeMode})`;
    predMethod = 'fallback';
  }

  let predictedRound = lastHit.id + predictedGap;
  if (predictedRound <= nowId) {
    predictedGap = roundsSince + Math.max(1, Math.round(selectedStats.iqr / 2), 1);
    predictedRound = lastHit.id + predictedGap;
  }

  const halfWin = Math.floor(winSize / 2);
  const windowLo = predictedRound - halfWin;
  const windowHi = windowLo + winSize - 1;
  const selectedGapsForRates = selectedStats.trimmed;
  const droughtPct = selectedCount > 0
    ? Math.round((selectedGapsForRates.filter((gap) => gap <= roundsSince).length / selectedCount) * 100)
    : 0;
  const roundsUntilWindowLo = Math.max(0, windowLo - nowId);
  const roundsUntilWindowHi = Math.max(0, windowHi - nowId);
  const inWindow = nowId >= windowLo && nowId <= windowHi;

  let pHit1 = null;
  let pHit5 = null;
  let pHit10 = null;
  let pHit20 = null;
  let pHitWindow = null;
  let pHitNearWindow = null;
  const nearWindowRounds = Math.max(1, Math.min(75, Math.max(winSize, Math.round(selectedStats.med * 0.35))));

  if (kmReliable) {
    const kmHorizon = Math.max(
      roundsSince + nearWindowRounds + 5,
      predictedGap + winSize + 5,
      selectedStats.p99 + winSize + 5
    );
    const kmTable = buildKMTable(selectedSorted, kmHorizon);
    pHit1 = kmProb(kmTable, roundsSince, 1);
    pHit5 = kmProb(kmTable, roundsSince, 5);
    pHit10 = kmProb(kmTable, roundsSince, 10);
    pHit20 = kmProb(kmTable, roundsSince, 20);
    pHitWindow = roundsUntilWindowHi <= 0
      ? 0
      : kmIntervalProb(kmTable, roundsSince, roundsUntilWindowLo, roundsUntilWindowHi);
    pHitNearWindow = kmProb(kmTable, roundsSince, nearWindowRounds);
  }

  const thresholds = getIssueThresholds(target);
  const winGapLo = predictedGap - halfWin;
  const winGapHi = winGapLo + winSize - 1;
  const hitsInWindow = selectedGapsForRates.filter((gap) => gap >= winGapLo && gap <= winGapHi).length;
  const empiricalWindowHitRate = selectedCount > 0 ? (hitsInWindow / selectedCount) * 100 : 0;
  const chanceWindowRate = computeChanceWindowRate(selectedGapsForRates, winSize);
  const predictiveLift = empiricalWindowHitRate - chanceWindowRate;
  const baselineStd = Math.max(
    1,
    Math.sqrt((chanceWindowRate * Math.max(1, 100 - chanceWindowRate)) / Math.max(1, selectedCount))
  );
  const standardizedLift = predictiveLift / baselineStd;
  const randomLike = standardizedLift <= (target.minVal <= 15 ? 0.85 : target.minVal <= 100 ? 0.65 : 0.45);
  const patternThresholds = getPatternThresholds(target);

  const rawConfidence = kmReliable && pHitWindow != null
    ? pHitWindow
    : empiricalWindowHitRate;

  const reliabilityFlags = [];
  if (regimeMode === 'recent') reliabilityFlags.push('recent_regime');
  if (lowData) reliabilityFlags.push('low_data');
  if (!kmReliable) reliabilityFlags.push('km_low_sample');
  if (openWindow) reliabilityFlags.push('extreme_tail');
  if (randomLike) reliabilityFlags.push('random_like');
  if (!primaryCluster || primaryCluster.supportPct < 0.18) reliabilityFlags.push('weak_cluster');
  if (recentPattern.whiteCluster) reliabilityFlags.push('white_cluster');
  if (recentPattern.lowPressureCluster) reliabilityFlags.push('low_pressure_cluster');
  if (recentPattern.emergingWhiteRisk) reliabilityFlags.push('white_risk');
  if (recentPattern.downtrend) reliabilityFlags.push('downtrend');
  if (recentPattern.downtrendEarly) reliabilityFlags.push('downtrend_risk');
  if (recentPattern.b2bBlocked) reliabilityFlags.push('b2b_risk');

  const probReliable = kmReliable && selectedCount >= sampleRequirements.minKMGaps;
  if (!probReliable) {
    pHit1 = null;
    pHit5 = null;
    pHit10 = null;
    pHit20 = null;
    pHitWindow = null;
    pHitNearWindow = null;
  }

  const clusterSupportPct = primaryCluster ? primaryCluster.supportPct * 100 : 0;
  const windowReadyThreshold = Math.max(1, Math.ceil(winSize * thresholds.readinessFactor));
  const windowReady = inWindow || roundsUntilWindowLo <= windowReadyThreshold;
  const transitionWindowReady = (
    windowReady ||
    (
      recentPattern.transitionReady &&
      roundsUntilWindowLo <= Math.max(
        windowReadyThreshold + Math.ceil(winSize * 0.6),
        target.minVal <= 30 ? 6 : target.minVal <= 100 ? 10 : Math.ceil(winSize * 0.9)
      )
    )
  );
  const signalProb = rawConfidence;
  const strongProbability = signalProb >= thresholds.minProbability;
  const strongEdge = predictiveLift >= thresholds.minLift || standardizedLift >= 1.0;
  const strongCluster = clusterSupportPct >= thresholds.minClusterSupport;
  const severeWhiteCluster = recentPattern.whiteRiskScore >= (target.minVal <= 15 ? 34 : target.minVal <= 100 ? 30 : 24);
  const activeTailWhitePressure = (
    recentPattern.tailSoftWhitePct >= Math.max(48, patternThresholds.softWhitePctThreshold - 22) ||
    recentPattern.tailLowPressurePct >= patternThresholds.tailLowPressurePctThreshold ||
    recentPattern.endingSoftWhiteStreak >= patternThresholds.softStreakThreshold ||
    recentPattern.endingLowPressureStreak >= Math.max(2, Math.ceil(winSize * 0.4)) ||
    (
      recentPattern.weightedSoftWhitePct >= Math.max(52, patternThresholds.softWhitePctThreshold - 16) &&
      recentPattern.tailSoftWhitePct >= Math.max(44, patternThresholds.softWhitePctThreshold - 26)
    ) ||
    (
      recentPattern.weightedLowPressurePct >= Math.max(48, patternThresholds.lowPressurePctThreshold - 10) &&
      recentPattern.tailLowPressurePct >= Math.max(42, patternThresholds.tailLowPressurePctThreshold - 6)
    )
  );
  const activeWhiteRisk = (
    (
      (recentPattern.whiteCluster || recentPattern.lowPressureCluster) &&
      activeTailWhitePressure &&
      recentPattern.whiteRiskScore >= (target.minVal <= 15 ? 20 : 18)
    ) ||
    (
      recentPattern.emergingWhiteRisk &&
      recentPattern.whiteRiskScore >= 18
    )
  ) && !recentPattern.whiteRelease
    && !recentPattern.releaseWatch
    && !recentPattern.compressionSupport
    && recentPattern.transitionSupportScore < (target.minVal <= 30 ? 18 : target.minVal <= 100 ? 15 : 12);
  const severeDowntrend = recentPattern.downtrendRiskScore >= (target.minVal <= 30 ? 26 : target.minVal <= 100 ? 22 : 18);
  const activeDowntrendRisk = (
    recentPattern.downtrend ||
    (recentPattern.downtrendEarly && !recentPattern.reboundSupport && recentPattern.downtrendRiskScore >= 14)
  );
  const positiveB2B = (
    recentPattern.b2bFriendly &&
    recentPattern.b2bSupportScore >= Math.max(12, recentPattern.b2bRiskScore + 4) &&
    !activeWhiteRisk &&
    !activeDowntrendRisk
  );
  const unsafeB2B = recentPattern.b2bBlocked && !positiveB2B;
  const hardRiskBlock = severeWhiteCluster || severeDowntrend || unsafeB2B;
  const supportMinusRisk = recentPattern.supportScore - recentPattern.riskScore;
  const transitionDrivenIssue = (
    target.minVal >= 30 &&
    transitionWindowReady &&
    recentPattern.transitionReady &&
    recentPattern.transitionSupportScore >= (target.minVal <= 30 ? 18 : target.minVal <= 100 ? 16 : 12) &&
    signalProb >= Math.max(8, thresholds.minProbability * 0.62) &&
    clusterSupportPct >= Math.max(12, thresholds.minClusterSupport - 6) &&
    supportMinusRisk >= -1 &&
    (predictiveLift >= Math.max(-0.5, thresholds.minLift * 0.15) || standardizedLift >= 0.15) &&
    !activeWhiteRisk &&
    !activeDowntrendRisk &&
    !unsafeB2B
  );
  const earlyB2BEligible = (
    target.minVal <= 15 ||
    (
      target.minVal <= 30 &&
      recentPattern.b2bSupportScore >= 18 &&
      recentPattern.transitionSupportScore >= 8
    )
  );
  const earlySupportiveEntry = (
    isTooEarly &&
    recentPattern.recentHitTooSoon &&
    earlyB2BEligible &&
    (
      positiveB2B ||
      (
        recentPattern.b2bFriendly &&
        recentPattern.b2bSupportScore >= Math.max(14, recentPattern.b2bRiskScore + 6)
      )
    ) &&
    transitionWindowReady &&
    signalProb >= Math.max(20, thresholds.minProbability * 0.55) &&
    clusterSupportPct >= Math.max(10, thresholds.minClusterSupport - 12) &&
    supportMinusRisk >= -4 &&
    !activeWhiteRisk &&
    !activeDowntrendRisk
  );
  const patternDrivenIssue = (
    transitionWindowReady &&
    recentPattern.patternScore >= (target.minVal <= 15 ? 18 : target.minVal <= 100 ? 14 : 10) &&
    signalProb >= Math.max(14, thresholds.minProbability * 0.55) &&
    clusterSupportPct >= Math.max(14, thresholds.minClusterSupport - 4) &&
    (recentPattern.reboundSupport || recentPattern.transitionSupportScore >= (target.minVal <= 30 ? 14 : 11)) &&
    supportMinusRisk >= Math.max(4, target.minVal <= 15 ? 8 : 5) &&
    !activeWhiteRisk &&
    !activeDowntrendRisk
  );
  const highProbabilitySupport = (
    windowReady &&
    signalProb >= Math.max(55, thresholds.minProbability + 10) &&
    strongCluster &&
    predictiveLift >= Math.max(0, thresholds.minLift * 0.35) &&
    supportMinusRisk >= 0 &&
    !activeWhiteRisk &&
    !activeDowntrendRisk
  );
  const rareCompressionIssue = (
    target.minVal >= 50 &&
    recentPattern.compressionSupport &&
    !activeWhiteRisk &&
    !activeDowntrendRisk &&
    !unsafeB2B &&
    transitionWindowReady &&
    signalProb >= Math.max(8, thresholds.minProbability * 0.72) &&
    clusterSupportPct >= Math.max(12, thresholds.minClusterSupport - 5) &&
    supportMinusRisk >= -2 &&
    recentPattern.transitionSupportScore >= (target.minVal <= 100 ? 14 : 12) &&
    (predictiveLift >= Math.max(-0.5, thresholds.minLift * 0.1) || standardizedLift >= 0.15)
  );
  const earlyCompressionEntry = (
    isTooEarly &&
    target.minVal >= 50 &&
    rareCompressionIssue &&
    signalProb >= Math.max(10, thresholds.minProbability * 0.7) &&
    clusterSupportPct >= Math.max(10, thresholds.minClusterSupport - 6) &&
    supportMinusRisk >= -3
  );
  const supportiveB2BIssue = (
    positiveB2B &&
    transitionWindowReady &&
    signalProb >= Math.max(18, thresholds.minProbability * 0.72) &&
    clusterSupportPct >= Math.max(16, thresholds.minClusterSupport - 4) &&
    predictiveLift >= Math.max(0, thresholds.minLift * 0.2) &&
    supportMinusRisk >= 0
  );
  const randomLikeHardBlock = randomLike && !transitionDrivenIssue && !supportiveB2BIssue;
  const transitionFloor = target.minVal <= 15 ? 10 : target.minVal <= 30 ? 12 : target.minVal <= 100 ? 14 : 16;
  const strongTransition = recentPattern.transitionSupportScore >= transitionFloor;
  const highTargetNeedsPreview = target.minVal >= 100 ? (recentPattern.tailPreviewHits > 0 || strongTransition) : true;

  const lockScore = (
    (signalProb * 0.35) +
    Math.min(20, predictiveLift * 2) +
    Math.min(18, clusterSupportPct * 0.3) +
    Math.min(18, recentPattern.transitionSupportScore * 0.45) +
    Math.min(12, Math.max(0, supportMinusRisk) * 0.4) +
    (positiveB2B ? 8 : 0) +
    (strongEdge ? 4 : 0) -
    Math.min(18, recentPattern.whiteRiskScore * 0.35) -
    Math.min(16, recentPattern.downtrendRiskScore * 0.3) -
    Math.min(14, recentPattern.b2bRiskScore * 0.3)
  );
  const lockThreshold =
    target.minVal <= 10 ? 38 :
    target.minVal <= 30 ? 34 :
    target.minVal <= 100 ? 30 :
    target.minVal <= 200 ? 28 :
    target.minVal <= 500 ? 26 : 24;

  const issuePrediction = (
    !lowData &&
    !openWindow &&
    transitionWindowReady &&
    (!isTooEarly || transitionDrivenIssue || earlySupportiveEntry || earlyCompressionEntry) &&
    strongProbability &&
    highTargetNeedsPreview &&
    !hardRiskBlock &&
    !randomLikeHardBlock &&
    (strongCluster || strongTransition || transitionDrivenIssue) &&
    lockScore >= lockThreshold
  );
  const issueMode = issuePrediction ? 'simple' : 'observe';

  const calibration = calibrateProbability(rawConfidence, options.calibrationRows || [], issueMode, regimeMode);
  let confidence = calibration.calibrated;
  if (lowData) confidence = Math.min(confidence, 25);
  if (!kmReliable) confidence = Math.min(confidence, 20);
  if (randomLike) confidence = Math.min(confidence, Math.max(8, confidence - 12));
  if (openWindow) confidence = Math.min(confidence, 18);
  confidence -= clampNumber(recentPattern.whiteRiskScore * 0.35, 0, 18);
  confidence -= clampNumber(recentPattern.downtrendRiskScore * 0.32, 0, 16);
  confidence -= clampNumber(recentPattern.b2bRiskScore * 0.35, 0, 14);
  confidence += clampNumber(recentPattern.supportScore * 0.12, 0, 10);
  confidence = Math.round(clampNumber(confidence, 4, 95));

  let avoidReason = null;
  if (!issuePrediction) {
    if (lowData) avoidReason = 'low_data';
    else if (openWindow) avoidReason = 'extreme_tail';
    else if (activeWhiteRisk && !strongTransition) avoidReason = 'white_cluster';
    else if (activeDowntrendRisk && !strongTransition) avoidReason = 'downtrend';
    else if (unsafeB2B && !positiveB2B) avoidReason = 'recent_b2b_risk';
    else if (!transitionWindowReady || isTooEarly) avoidReason = 'too_early';
    else if (!strongProbability) avoidReason = 'weak_probability';
    else if (randomLikeHardBlock) avoidReason = 'random_like';
    else if (lockScore < lockThreshold) avoidReason = 'observe_only';
    else avoidReason = 'observe_only';
  }

  let chaseRaw = Math.round(
    (signalProb * 0.68) +
    (clusterSupportPct * 0.18) +
    clampNumber(Math.max(0, predictiveLift) * 4, 0, 20) +
    clampNumber(recentPattern.supportScore * 0.18, 0, 18) -
    clampNumber(recentPattern.riskScore * 0.16, 0, 16) +
    clampNumber(recentPattern.patternScore * 0.08, -8, 10) +
    clampNumber(recentPattern.transitionSupportScore * 0.16, 0, 12)
  );
  if (inWindow) chaseRaw += 10;
  else if (windowReady) chaseRaw += 4;
  else chaseRaw -= 12;
  if (severeWhiteCluster) chaseRaw -= 18;
  else if (activeWhiteRisk) chaseRaw -= 10;
  if (severeDowntrend) chaseRaw -= 14;
  else if (activeDowntrendRisk) chaseRaw -= 8;
  if (unsafeB2B) chaseRaw -= 16;
  if (positiveB2B) chaseRaw += 8;
  if (recentPattern.whiteRelease) chaseRaw += 6;
  if (recentPattern.reboundSupport) chaseRaw += 5;
  if (lowData) chaseRaw -= 20;
  if (!kmReliable) chaseRaw -= 16;
  if (randomLike) chaseRaw -= 18;
  if (openWindow) chaseRaw -= 24;
  chaseRaw = clampNumber(chaseRaw, 0, 100);

  const chaseSignal = issuePrediction ? 'CHASE' : 'SKIP';
  const chaseColor = chaseRaw >= 70
    ? '#39ff8a'
    : chaseRaw >= 50
      ? '#ffd250'
      : chaseRaw >= 30
        ? '#ff9f43'
        : '#ff4040';

  return {
    ...target,
    noData: false,
    nowId,
    hits: hits.length,
    lastHit,
    roundsSince,
    n: selectedCount,
    fullGapCount: allGapsSorted.length,
    regimeMode,
    regimeDrift: Math.round(regimeDrift * 100),
    med: selectedStats.med,
    p10: selectedStats.p10,
    p25: selectedStats.p25,
    p75: selectedStats.p75,
    p90: selectedStats.p90,
    p90all: fullStats.p90,
    p99all: fullStats.p99,
    maxGap: selectedStats.max,
    minGap: selectedStats.min,
    avgGap: selectedStats.avg,
    iqr: selectedStats.iqr,
    clusterCenter: primaryCluster?.center ?? selectedStats.med,
    secondaryClusterCenter: histogram.secondary?.center ?? null,
    clusterSupportPct: Math.round(clusterSupportPct),
    predBasis,
    predMethod,
    predictedGap,
    predictedRound,
    windowLo,
    windowHi,
    windowSize: winSize,
    openWindow,
    isTooEarly,
    isOverdue,
    isHardGap,
    isExtreme,
    droughtPct,
    confidence,
    rawConfidence: Number(rawConfidence.toFixed(1)),
    calibrationBucket: calibration.bucketLabel,
    calibrationSupport: calibration.support,
    calibrationMode: calibration.mode,
    chaseRaw,
    chaseSignal,
    chaseColor,
    issuePrediction,
    issueMode,
    avoidReason,
    windowReady,
    windowReadyThreshold,
    pHit1,
    pHit5,
    pHit10,
    pHit20,
    pHitWindow,
    pHitNearWindow,
    probReliable,
    nearWindowRounds,
    empiricalWindowHitRate: Number(empiricalWindowHitRate.toFixed(1)),
    chanceWindowRate: Number(chanceWindowRate.toFixed(1)),
    predictiveLift: Number(predictiveLift.toFixed(1)),
    standardizedLift: Number(standardizedLift.toFixed(2)),
    roundsUntilWindowLo,
    roundsUntilWindowHi,
    inWindow,
    kmReliable,
    reliabilityFlags,
    recentPattern,
  };
}

function makeOracleLock(forecast, nowId) {
  return {
    label: forecast.label,
    minVal: forecast.minVal,
    color: forecast.color,
    predictedRound: forecast.predictedRound,
    windowLo: forecast.windowLo,
    windowHi: forecast.windowHi,
    windowSize: forecast.windowSize,
    snapAt: nowId,
    lastHitId: forecast.lastHit.id,
    confidence: forecast.confidence,
    predBasis: forecast.predBasis,
    predMethod: forecast.predMethod,
    med: forecast.med,
    iqr: forecast.iqr,
    clusterCenter: forecast.clusterCenter,
    droughtAtSnap: forecast.droughtPct,
    signal: forecast.chaseSignal,
    issueMode: forecast.issueMode || null,
    regimeMode: forecast.regimeMode || null,
    issuePrediction: Boolean(forecast.issuePrediction),
    avoidReason: forecast.avoidReason || null,
  };
}

module.exports = {
  ORACLE_TARGETS,
  normalizeRounds,
  computeOracleForecast,
  makeOracleLock,
};
