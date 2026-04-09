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
const CALIBRATION_RECENT_LIMIT = 180;
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

  const sampleCount = 15;
  const startMax = Math.max(min, max - width + 1);
  const step = sampleCount > 1 ? (startMax - min) / (sampleCount - 1) : 0;
  const rates = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const lo = Math.round(min + (step * i));
    const hi = lo + width - 1;
    const hits = sorted.filter((gap) => gap >= lo && gap <= hi).length;
    rates.push((hits / sorted.length) * 100);
  }
  return mean(rates);
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
    return { minProbability: 52, minLift: 8, minClusterSupport: 24, readinessFactor: 0.55 };
  }
  if (target.minVal <= 30) {
    return { minProbability: 40, minLift: 6, minClusterSupport: 22, readinessFactor: 0.6 };
  }
  if (target.minVal <= 100) {
    return { minProbability: 28, minLift: 4, minClusterSupport: 18, readinessFactor: 0.68 };
  }
  if (target.minVal <= 200) {
    return { minProbability: 18, minLift: 2.5, minClusterSupport: 16, readinessFactor: 0.74 };
  }
  if (target.minVal <= 500) {
    return { minProbability: 11, minLift: 1.5, minClusterSupport: 14, readinessFactor: 0.82 };
  }
  return { minProbability: 7, minLift: 0.8, minClusterSupport: 12, readinessFactor: 0.9 };
}

function computeRecentPatternDiagnostics(rounds, target, roundsSince, allGapsRaw, selectedStats) {
  const lookback = clampNumber(Math.max(target.window * 8, 12), 12, 96);
  const recentRounds = rounds.slice(-lookback);
  const values = recentRounds.map((round) => round.val);
  if (!values.length) {
    return {
      lookback,
      hardWhitePct: 0,
      softWhitePct: 0,
      maxHardWhiteStreak: 0,
      maxSoftWhiteStreak: 0,
      downtrend: false,
      downtrendPct: 0,
      shortRepeatRate: 0,
      b2bBlocked: false,
      b2bFriendly: false,
      recentHitTooSoon: false,
    };
  }

  let hardWhiteCount = 0;
  let softWhiteCount = 0;
  let hardWhiteStreak = 0;
  let softWhiteStreak = 0;
  let maxHardWhiteStreak = 0;
  let maxSoftWhiteStreak = 0;

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

    maxHardWhiteStreak = Math.max(maxHardWhiteStreak, hardWhiteStreak);
    maxSoftWhiteStreak = Math.max(maxSoftWhiteStreak, softWhiteStreak);
  }

  const split = Math.max(1, Math.floor(values.length / 2));
  const firstHalf = values.slice(0, split);
  const secondHalf = values.slice(split);
  const headLogMean = meanLog(firstHalf);
  const tailLogMean = meanLog(secondHalf);
  const trendDelta = headLogMean > 0 ? ((tailLogMean - headLogMean) / headLogMean) : 0;
  const headPeak = firstHalf.length ? Math.max(...firstHalf) : 0;
  const tailPeak = secondHalf.length ? Math.max(...secondHalf) : 0;

  const hardWhitePct = (hardWhiteCount / values.length) * 100;
  const softWhitePct = (softWhiteCount / values.length) * 100;
  const whiteHardPctThreshold = target.minVal <= 10 ? 72 : target.minVal <= 50 ? 60 : 52;
  const whiteSoftPctThreshold = target.minVal <= 10 ? 88 : 80;
  const hardStreakThreshold = target.minVal <= 10 ? 4 : 3;
  const softStreakThreshold = target.minVal <= 10 ? 6 : 5;
  const downtrend = trendDelta <= -0.14 && tailPeak <= (headPeak * 0.82);

  const shortRepeatWindow = Math.max(2, Math.min(target.window, 6));
  const shortRepeatRate = allGapsRaw.length
    ? allGapsRaw.filter((gap) => gap <= shortRepeatWindow).length / allGapsRaw.length
    : 0;
  const recentHitThreshold = Math.max(
    1,
    Math.min(shortRepeatWindow, Math.max(1, Math.round(selectedStats.p10 || shortRepeatWindow)))
  );
  const recentHitTooSoon = roundsSince <= recentHitThreshold;
  const b2bBlocked = recentHitTooSoon && shortRepeatRate < 0.2;
  const b2bFriendly = recentHitTooSoon && shortRepeatRate >= 0.35;
  const whiteCluster = (
    hardWhitePct >= whiteHardPctThreshold ||
    softWhitePct >= whiteSoftPctThreshold ||
    maxHardWhiteStreak >= hardStreakThreshold ||
    maxSoftWhiteStreak >= softStreakThreshold
  );

  return {
    lookback,
    hardWhitePct: Number(hardWhitePct.toFixed(1)),
    softWhitePct: Number(softWhitePct.toFixed(1)),
    maxHardWhiteStreak,
    maxSoftWhiteStreak,
    whiteCluster,
    downtrend,
    downtrendPct: Number((trendDelta * 100).toFixed(1)),
    shortRepeatRate: Number((shortRepeatRate * 100).toFixed(1)),
    b2bBlocked,
    b2bFriendly,
    recentHitTooSoon,
  };
}

function buildKMTable(allGapsSorted) {
  if (!allGapsSorted.length) return new Float32Array(1).fill(1);
  const n = allGapsSorted.length;
  const limit = allGapsSorted[n - 1] + 30;
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

function buildCalibrationModel(rows) {
  const usable = (rows || [])
    .filter((row) => row && row.probW != null && row.outcome !== 'early')
    .slice(0, CALIBRATION_RECENT_LIMIT);
  const buckets = Array.from({ length: 10 }, (_, index) => ({
    lo: index * 10,
    hi: (index * 10) + 9,
    wins: 0,
    total: 0,
  }));

  for (const row of usable) {
    const probPercent = clampNumber(Math.round(Number(row.probW || 0) * 100), 0, 99);
    const bucketIndex = clampNumber(Math.floor(probPercent / 10), 0, 9);
    buckets[bucketIndex].total += 1;
    if (row.outcome === 'win') buckets[bucketIndex].wins += 1;
  }

  const globalWins = usable.filter((row) => row.outcome === 'win').length;
  const globalLosses = usable.filter((row) => row.outcome === 'loss').length;
  const globalTotal = globalWins + globalLosses;

  return {
    buckets,
    globalRate: globalTotal >= MIN_GLOBAL_CALIBRATION ? (globalWins / globalTotal) * 100 : null,
    globalTotal,
  };
}

function calibrateProbability(rawPercent, calibrationRows) {
  const raw = clampNumber(Number(rawPercent || 0), 0, 100);
  const model = buildCalibrationModel(calibrationRows);
  const bucketIndex = clampNumber(Math.floor(Math.min(raw, 99) / 10), 0, 9);
  const bucket = model.buckets[bucketIndex];

  if (bucket.total >= MIN_BUCKET_CALIBRATION) {
    return {
      calibrated: clampNumber((bucket.wins / bucket.total) * 100, 0, 100),
      bucketLabel: `${bucket.lo}-${bucket.hi}%`,
      support: bucket.total,
      mode: 'bucket',
    };
  }

  if (model.globalRate != null) {
    return {
      calibrated: model.globalRate,
      bucketLabel: 'global',
      support: model.globalTotal,
      mode: 'global',
    };
  }

  return {
    calibrated: raw,
    bucketLabel: `${bucket.lo}-${bucket.hi}%`,
    support: bucket.total,
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
  const recentCount = Math.min(scanN, allGapsRaw.length);
  const recentSorted = [...allGapsRaw.slice(-recentCount)].sort((a, b) => a - b);
  const recentStats = buildRobustStats(recentSorted);
  const regimeDrift = fullStats.med > 0 ? Math.abs(recentStats.med - fullStats.med) / fullStats.med : 0;
  const regimeMode = regimeDrift > REGIME_DRIFT_THRESHOLD ? 'recent' : 'full';
  const selectedSorted = regimeMode === 'recent' ? recentSorted : allGapsSorted;
  const selectedStats = regimeMode === 'recent' ? recentStats : fullStats;
  const selectedCount = selectedSorted.length;

  const lastHit = hits[hits.length - 1];
  const roundsSince = nowId - lastHit.id;
  const isTooEarly = roundsSince < selectedStats.p10;
  const isOverdue = roundsSince > selectedStats.med;
  const isHardGap = roundsSince > selectedStats.p90;
  const isExtreme = roundsSince > selectedStats.p99;
  const survivingGaps = selectedSorted.filter((gap) => gap > roundsSince);
  const histogram = buildHistogramClusters(selectedStats.trimmed);
  const candidateClusters = [histogram.primary, histogram.secondary].filter(Boolean);
  const futureCluster = candidateClusters
    .filter((cluster) => cluster.center > roundsSince)
    .sort((a, b) => a.center - b.center)[0];
  const primaryCluster = futureCluster || histogram.primary;
  const openWindow = isExtreme && survivingGaps.length === 0;
  const lowData = selectedCount < MIN_FORECAST_GAPS;
  const kmReliable = selectedCount >= MIN_KM_GAPS;
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
  const droughtPct = selectedCount > 0
    ? Math.round((selectedSorted.filter((gap) => gap <= roundsSince).length / selectedCount) * 100)
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
  let nearWindowRounds = Math.max(1, Math.min(75, Math.max(winSize, Math.round(selectedStats.med * 0.35))));

  if (kmReliable) {
    const kmTable = buildKMTable(selectedSorted);
    pHit1 = kmProb(kmTable, roundsSince, 1);
    pHit5 = kmProb(kmTable, roundsSince, 5);
    pHit10 = kmProb(kmTable, roundsSince, 10);
    pHit20 = kmProb(kmTable, roundsSince, 20);
    pHitWindow = roundsUntilWindowHi <= 0
      ? 0
      : kmIntervalProb(kmTable, roundsSince, roundsUntilWindowLo, roundsUntilWindowHi);
    pHitNearWindow = kmProb(kmTable, roundsSince, nearWindowRounds);
  }

  const winGapLo = predictedGap - halfWin;
  const winGapHi = winGapLo + winSize - 1;
  const hitsInWindow = selectedSorted.filter((gap) => gap >= winGapLo && gap <= winGapHi).length;
  const empiricalWindowHitRate = selectedCount > 0 ? (hitsInWindow / selectedCount) * 100 : 0;
  const chanceWindowRate = computeChanceWindowRate(selectedSorted, winSize);
  const predictiveLift = empiricalWindowHitRate - chanceWindowRate;
  const randomLike = predictiveLift <= 2;

  const rawConfidence = kmReliable && pHitWindow != null
    ? pHitWindow
    : empiricalWindowHitRate;
  const calibration = calibrateProbability(rawConfidence, options.calibrationRows || []);
  let confidence = calibration.calibrated;
  if (lowData) confidence = Math.min(confidence, 25);
  if (!kmReliable) confidence = Math.min(confidence, 45);
  if (randomLike) confidence = Math.min(confidence, Math.max(8, confidence - 12));
  if (openWindow) confidence = Math.min(confidence, 18);
  if (recentPattern.whiteCluster) confidence = Math.min(confidence, 42);
  if (recentPattern.downtrend) confidence = Math.min(confidence, 46);
  if (recentPattern.b2bBlocked) confidence = Math.min(confidence, 40);
  confidence = Math.round(clampNumber(confidence, 4, 95));

  const reliabilityFlags = [];
  if (regimeMode === 'recent') reliabilityFlags.push('recent_regime');
  if (lowData) reliabilityFlags.push('low_data');
  if (!kmReliable) reliabilityFlags.push('km_low_sample');
  if (openWindow) reliabilityFlags.push('extreme_tail');
  if (randomLike) reliabilityFlags.push('random_like');
  if (!primaryCluster || primaryCluster.supportPct < 0.18) reliabilityFlags.push('weak_cluster');
  if (recentPattern.whiteCluster) reliabilityFlags.push('white_cluster');
  if (recentPattern.downtrend) reliabilityFlags.push('downtrend');
  if (recentPattern.b2bBlocked) reliabilityFlags.push('b2b_risk');

  const clusterSupportPct = primaryCluster ? primaryCluster.supportPct * 100 : 0;
  const thresholds = getIssueThresholds(target);
  const windowReadyThreshold = Math.max(1, Math.ceil(winSize * thresholds.readinessFactor));
  const windowReady = inWindow || roundsUntilWindowLo <= windowReadyThreshold;
  const signalProb = kmReliable && pHitWindow != null ? pHitWindow : confidence;
  const strongProbability = signalProb >= thresholds.minProbability;
  const strongEdge = predictiveLift >= thresholds.minLift;
  const strongCluster = clusterSupportPct >= thresholds.minClusterSupport;
  const patternBlocked = recentPattern.whiteCluster || recentPattern.downtrend || recentPattern.b2bBlocked;
  const issuePrediction = (
    !lowData &&
    kmReliable &&
    !openWindow &&
    !randomLike &&
    windowReady &&
    !isTooEarly &&
    strongProbability &&
    strongEdge &&
    strongCluster &&
    !patternBlocked
  );

  let avoidReason = null;
  if (!issuePrediction) {
    if (recentPattern.whiteCluster) avoidReason = 'white_cluster';
    else if (recentPattern.downtrend) avoidReason = 'downtrend';
    else if (recentPattern.b2bBlocked) avoidReason = 'recent_b2b_risk';
    else if (lowData) avoidReason = 'low_data';
    else if (!kmReliable) avoidReason = 'km_low_sample';
    else if (openWindow) avoidReason = 'extreme_tail';
    else if (randomLike) avoidReason = 'random_like';
    else if (!windowReady || isTooEarly) avoidReason = 'too_early';
    else if (!strongCluster) avoidReason = 'weak_cluster';
    else if (!strongEdge) avoidReason = 'no_edge';
    else if (!strongProbability) avoidReason = 'weak_probability';
    else avoidReason = 'observe_only';
  }

  let chaseRaw = Math.round(
    (signalProb * 0.68) +
    (clusterSupportPct * 0.18) +
    clampNumber(Math.max(0, predictiveLift) * 4, 0, 20)
  );
  if (inWindow) chaseRaw += 10;
  else if (windowReady) chaseRaw += 4;
  else chaseRaw -= 12;
  if (recentPattern.whiteCluster) chaseRaw -= 18;
  if (recentPattern.downtrend) chaseRaw -= 14;
  if (recentPattern.b2bBlocked) chaseRaw -= 16;
  if (lowData) chaseRaw -= 20;
  if (!kmReliable) chaseRaw -= 16;
  if (randomLike) chaseRaw -= 18;
  if (openWindow) chaseRaw -= 24;
  chaseRaw = clampNumber(chaseRaw, 0, 100);

  const watchworthy = (
    !issuePrediction &&
    !patternBlocked &&
    !lowData &&
    !openWindow &&
    signalProb >= (thresholds.minProbability * 0.75) &&
    predictiveLift >= Math.max(1, thresholds.minLift * 0.6)
  );
  const waitworthy = (
    !issuePrediction &&
    !patternBlocked &&
    !lowData &&
    !openWindow &&
    signalProb >= (thresholds.minProbability * 0.55)
  );

  const chaseSignal = issuePrediction
    ? 'CHASE'
    : watchworthy
      ? 'WATCH'
      : waitworthy && windowReady
        ? 'WAIT'
        : 'SKIP';
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
    avoidReason,
    windowReady,
    windowReadyThreshold,
    pHit1,
    pHit5,
    pHit10,
    pHit20,
    pHitWindow,
    pHitNearWindow,
    nearWindowRounds,
    empiricalWindowHitRate: Number(empiricalWindowHitRate.toFixed(1)),
    chanceWindowRate: Number(chanceWindowRate.toFixed(1)),
    predictiveLift: Number(predictiveLift.toFixed(1)),
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
