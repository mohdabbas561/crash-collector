'use strict';

const ORACLE_TARGETS = Object.freeze([
  { label: '5x', minVal: 5, color: '#00ff88', window: 4, scanN: 150, minHits: 7 },
  { label: '10x', minVal: 10, color: '#00d4ff', window: 6, scanN: 130, minHits: 6 },
  { label: '15x', minVal: 15, color: '#ff6b9d', window: 7, scanN: 110, minHits: 5 },
  { label: '30x', minVal: 30, color: '#ff9f43', window: 13, scanN: 90, minHits: 4 },
  { label: '50x', minVal: 50, color: '#4db8ff', window: 20, scanN: 75, minHits: 4 },
  { label: '100x', minVal: 100, color: '#39ff8a', window: 30, scanN: 65, minHits: 4 },
  { label: '200x', minVal: 200, color: '#c77dff', window: 50, scanN: 55, minHits: 3 },
  { label: '500x', minVal: 500, color: '#ff4da6', window: 75, scanN: 48, minHits: 3 },
  { label: '1000x', minVal: 1000, color: '#7aa2ff', window: 100, scanN: 42, minHits: 3 },
]);

const REGIME_DRIFT_THRESHOLD = 0.30;
const MIN_FORECAST_GAPS = 10;
const MIN_KM_GAPS = 20;
const MIN_BUCKET_CALIBRATION = 8;
const MIN_GLOBAL_CALIBRATION = 24;
const LOW_HARD_CAP = 1.25;

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

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

function stddev(values, avg) {
  if (values.length < 2) return 0;
  const center = Number.isFinite(avg) ? avg : mean(values);
  const variance = values.reduce((sum, value) => sum + ((value - center) ** 2), 0) / (values.length - 1);
  return Math.sqrt(Math.max(0, variance));
}

function normalizeRounds(rounds) {
  const map = new Map();
  for (const round of rounds || []) {
    const id = Number(round?.roundId ?? round?.id);
    const val = Number.parseFloat(round?.multiplier ?? round?.val ?? round?.gameResult ?? round?.result);
    if (!Number.isFinite(id) || !Number.isFinite(val) || val <= 0) continue;
    map.set(id, { id, val: Number(val.toFixed(4)) });
  }
  return [...map.values()].sort((a, b) => a.id - b.id);
}

function trimSorted(sorted, trimRatio = 0.1) {
  if (!sorted.length) return [];
  if (sorted.length < 10) return sorted.slice();
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
  return filtered.length >= Math.max(6, Math.floor(sorted.length * 0.55))
    ? filtered
    : sorted.slice();
}

function buildRobustStats(sourceSorted) {
  const filtered = tukeyFilterSorted(sourceSorted);
  const trimmed = trimSorted(filtered, filtered.length >= 12 ? 0.1 : 0);
  const working = trimmed.length ? trimmed : filtered;

  const p10 = quantile(working, 10);
  const p25 = quantile(working, 25);
  const med = quantile(working, 50);
  const p75 = quantile(working, 75);
  const p90 = quantile(working, 90);
  const p99 = quantile(working, 99);

  return {
    filtered,
    trimmed: working,
    min: working[0],
    max: working[working.length - 1],
    p10: Math.round(p10),
    p25: Math.round(p25),
    med: Math.round(med),
    p75: Math.round(p75),
    p90: Math.round(p90),
    p99: Math.round(p99),
    iqr: Math.max(1, Math.round(p75 - p25)),
    avg: Math.round(mean(working)),
  };
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
  if (denom <= 0 || Math.abs(meanY) < 0.00001) return 0;
  const slope = numer / denom;
  return ((slope * (n - 1)) / Math.abs(meanY)) * 100;
}

function getLowPressureSoftThreshold(minVal) {
  if (minVal <= 10) return 1.9;
  if (minVal <= 15) return 2.1;
  if (minVal <= 30) return 2.4;
  if (minVal <= 50) return 2.7;
  if (minVal <= 100) return 3.0;
  if (minVal <= 200) return 3.3;
  if (minVal <= 500) return 3.7;
  return 4.0;
}

function computeRegimeMode(allSorted, recentSorted, rounds, target) {
  if (recentSorted.length < 24 || allSorted.length < 40) return 'full';
  const medAll = quantile(allSorted, 50);
  const medRecent = quantile(recentSorted, 50);
  const drift = medAll > 0 ? Math.abs(medRecent - medAll) / medAll : 0;

  const lastValues = rounds.slice(-24).map((round) => round.val);
  const prevValues = rounds.slice(-48, -24).map((round) => round.val);
  const lowSoft = getLowPressureSoftThreshold(target.minVal);
  const recentLowRate = lastValues.length ? (lastValues.filter((value) => value <= lowSoft).length / lastValues.length) : 0;
  const prevLowRate = prevValues.length ? (prevValues.filter((value) => value <= lowSoft).length / prevValues.length) : 0;
  const pressureShift = recentLowRate - prevLowRate;

  if (drift > REGIME_DRIFT_THRESHOLD || pressureShift > 0.24 || pressureShift < -0.24) {
    return 'recent';
  }
  return 'full';
}

function deriveBinWidth(sorted) {
  if (sorted.length < 3) return 1;
  const q25 = quantile(sorted, 25);
  const q75 = quantile(sorted, 75);
  const iqr = Math.max(1, q75 - q25);
  const fd = Math.max(1, Math.round((2 * iqr) / Math.cbrt(sorted.length)));
  const span = Math.max(1, sorted[sorted.length - 1] - sorted[0]);
  const sqrtRule = Math.max(1, Math.round(span / Math.max(2, Math.sqrt(sorted.length))));
  return clampNumber(Math.min(fd, sqrtRule), 1, Math.max(1, Math.round(span / 3)));
}

function buildHistogramClusters(sorted) {
  if (!sorted.length) return { binWidth: 1, primary: null, secondary: null, bins: [] };
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const binWidth = deriveBinWidth(sorted);
  const binCount = Math.max(1, Math.floor((max - min) / binWidth) + 1);

  const bins = Array.from({ length: binCount }, (_, idx) => ({
    index: idx,
    lo: min + (idx * binWidth),
    hi: min + ((idx + 1) * binWidth) - 1,
    count: 0,
    sum: 0,
  }));

  for (const gap of sorted) {
    const idx = clampNumber(Math.floor((gap - min) / binWidth), 0, binCount - 1);
    bins[idx].count += 1;
    bins[idx].sum += gap;
  }

  const smooth = bins.map((bin, idx) => {
    const left = bins[idx - 1]?.count || 0;
    const right = bins[idx + 1]?.count || 0;
    return bin.count + ((left + right) * 0.55);
  });

  const peaks = [];
  for (let i = 0; i < bins.length; i += 1) {
    const left = smooth[i - 1] ?? -Infinity;
    const right = smooth[i + 1] ?? -Infinity;
    if (smooth[i] >= left && smooth[i] >= right && bins[i].count > 0) {
      peaks.push({ index: i, strength: smooth[i] });
    }
  }
  peaks.sort((a, b) => b.strength - a.strength);

  function expand(peakIndex) {
    const peakStrength = smooth[peakIndex];
    const threshold = peakStrength * 0.42;
    let lo = peakIndex;
    let hi = peakIndex;
    while (lo > 0 && smooth[lo - 1] >= threshold) lo -= 1;
    while (hi < bins.length - 1 && smooth[hi + 1] >= threshold) hi += 1;
    const region = bins.slice(lo, hi + 1);
    const supportCount = region.reduce((sum, bin) => sum + bin.count, 0);
    const weightedSum = region.reduce((sum, bin) => sum + bin.sum, 0);
    return {
      lo: region[0].lo,
      hi: region[region.length - 1].hi,
      center: supportCount > 0 ? Math.round(weightedSum / supportCount) : Math.round((region[0].lo + region[region.length - 1].hi) / 2),
      supportCount,
      supportPct: supportCount / sorted.length,
      peakStrength,
    };
  }

  const primary = peaks.length ? expand(peaks[0].index) : null;
  let secondary = null;
  for (const peak of peaks.slice(1)) {
    const candidate = expand(peak.index);
    if (
      primary &&
      Math.abs(candidate.center - primary.center) >= binWidth &&
      candidate.supportPct >= Math.max(0.12, primary.supportPct * 0.4)
    ) {
      secondary = candidate;
      break;
    }
  }

  return { binWidth, bins, primary, secondary };
}

function buildKMTable(allGapsSorted) {
  if (!allGapsSorted.length) return new Float32Array(2).fill(1);
  const n = allGapsSorted.length;
  const maxGap = allGapsSorted[n - 1];
  const limit = maxGap + 120;
  const table = new Float32Array(limit + 1).fill(1);
  let survival = 1;
  let left = 0;
  for (let t = 1; t <= limit; t += 1) {
    while (left < n && allGapsSorted[left] < t) left += 1;
    let right = left;
    while (right < n && allGapsSorted[right] === t) right += 1;
    const atRisk = n - left;
    const events = right - left;
    if (atRisk > 0) {
      survival *= (1 - (events / atRisk));
      survival = clampNumber(survival, 0, 1);
    }
    table[t] = survival;
  }
  return table;
}

function kmProb(table, roundsSince, k) {
  if (!table?.length) return 0;
  const from = clampNumber(Math.round(roundsSince), 0, table.length - 1);
  const to = clampNumber(Math.round(roundsSince + k), 0, table.length - 1);
  const sFrom = table[from];
  if (sFrom <= 0) return 100;
  return clampNumber((1 - (table[to] / sFrom)) * 100, 0, 100);
}

function computeConditionalExpectedGap(survivors, roundsSince) {
  if (!survivors.length) return null;
  const sorted = survivors.slice().sort((a, b) => a - b);
  const trimmed = trimSorted(sorted, sorted.length >= 12 ? 0.1 : 0);
  const working = trimmed.length ? trimmed : sorted;
  const expected = mean(working);
  return Math.max(roundsSince + 1, Math.round(expected));
}

function computeChanceWindowRate(sorted, width, maxSamples = 4000) {
  if (!sorted.length) return { rate: 0, std: 0 };
  const winWidth = Math.max(1, Math.round(width));
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  if (max <= min) return { rate: 100, std: 0 };

  const startMin = min;
  const startMax = Math.max(min, max - winWidth + 1);
  const totalWindows = Math.max(1, startMax - startMin + 1);

  let starts = [];
  if (totalWindows <= maxSamples) {
    starts = Array.from({ length: totalWindows }, (_, idx) => startMin + idx);
  } else {
    const step = (totalWindows - 1) / (maxSamples - 1);
    for (let i = 0; i < maxSamples; i += 1) {
      starts.push(startMin + Math.round(i * step));
    }
    starts = [...new Set(starts)].sort((a, b) => a - b);
  }

  const n = sorted.length;
  let lo = 0;
  let hi = 0;
  const rates = [];
  for (const start of starts) {
    const end = start + winWidth - 1;
    while (lo < n && sorted[lo] < start) lo += 1;
    while (hi < n && sorted[hi] <= end) hi += 1;
    const windowCount = Math.max(0, hi - lo);
    rates.push((windowCount / n) * 100);
  }

  const avg = mean(rates);
  return { rate: avg, std: stddev(rates, avg) };
}

function longestStreak(values, predicate) {
  let best = 0;
  let run = 0;
  for (const value of values) {
    if (predicate(value)) {
      run += 1;
      if (run > best) best = run;
    } else {
      run = 0;
    }
  }
  return best;
}

function trailingStreak(values, predicate) {
  let run = 0;
  for (let i = values.length - 1; i >= 0; i -= 1) {
    if (!predicate(values[i])) break;
    run += 1;
  }
  return run;
}

function computePatternSupport(rounds, target, windowSize) {
  const values = rounds.map((round) => Number(round.val || 0));
  const n = values.length;
  const patternLen = target.minVal <= 30 ? 12 : target.minVal <= 100 ? 10 : 8;
  if (n < (patternLen + windowSize + 30)) {
    return {
      ready: false,
      supportPct: 0,
      randomPct: 0,
      lift: 0,
      bestDistance: null,
      matches: 0,
      sampleSize: 0,
    };
  }

  const transformed = values.map((value) => Math.log(Math.max(1.0001, value)));
  const currentStart = n - patternLen;
  const currentPattern = transformed.slice(currentStart, n);
  const candidateMaxStart = currentStart - windowSize - 1;
  if (candidateMaxStart < 0) {
    return {
      ready: false,
      supportPct: 0,
      randomPct: 0,
      lift: 0,
      bestDistance: null,
      matches: 0,
      sampleSize: 0,
    };
  }

  const maxCandidates = 5500;
  const stride = Math.max(1, Math.ceil((candidateMaxStart + 1) / maxCandidates));
  const topK = 24;
  const top = [];
  let sampleSize = 0;
  let randomHits = 0;

  function nextWindowHit(startIdx) {
    const lo = startIdx + patternLen;
    const hi = lo + windowSize - 1;
    for (let i = lo; i <= hi && i < values.length; i += 1) {
      if (values[i] >= target.minVal) return true;
    }
    return false;
  }

  function pushTop(item) {
    if (top.length < topK) {
      top.push(item);
      top.sort((a, b) => a.distance - b.distance);
      return;
    }
    if (item.distance >= top[top.length - 1].distance) return;
    top[top.length - 1] = item;
    top.sort((a, b) => a.distance - b.distance);
  }

  for (let start = 0; start <= candidateMaxStart; start += stride) {
    sampleSize += 1;
    const hit = nextWindowHit(start);
    if (hit) randomHits += 1;

    let distance = 0;
    for (let i = 0; i < patternLen; i += 1) {
      distance += Math.abs(transformed[start + i] - currentPattern[i]);
    }
    distance /= patternLen;
    pushTop({ start, distance, hit });
  }

  const matchHits = top.filter((item) => item.hit).length;
  const supportPct = top.length ? (matchHits / top.length) * 100 : 0;
  const randomPct = sampleSize ? (randomHits / sampleSize) * 100 : 0;
  return {
    ready: top.length >= 8,
    supportPct,
    randomPct,
    lift: supportPct - randomPct,
    bestDistance: top.length ? top[0].distance : null,
    matches: top.length,
    sampleSize,
  };
}

function computeRecentPatternDiagnostics(rounds, target, roundsSince, allGapsRaw, selectedStats) {
  const recentValues = rounds.slice(-24).map((round) => round.val);
  const shorter = rounds.slice(-12).map((round) => round.val);
  const veryShort = rounds.slice(-6).map((round) => round.val);

  const lowSoft = getLowPressureSoftThreshold(target.minVal);
  const highTarget = target.minVal >= 100;
  const hardWhiteRate = recentValues.length
    ? (recentValues.filter((value) => value <= LOW_HARD_CAP).length / recentValues.length)
    : 0;
  const softWhiteRate = recentValues.length
    ? (recentValues.filter((value) => value <= lowSoft).length / recentValues.length)
    : 0;
  const hardWhiteStreak = longestStreak(recentValues, (value) => value <= LOW_HARD_CAP);
  const softWhiteStreak = longestStreak(recentValues, (value) => value <= lowSoft);
  const hardWhiteTailStreak = trailingStreak(recentValues, (value) => value <= LOW_HARD_CAP);
  const softWhiteTailStreak = trailingStreak(recentValues, (value) => value <= lowSoft);

  const trend24 = computeTrendPercent(recentValues);
  const trend12 = computeTrendPercent(shorter);
  const trend6 = computeTrendPercent(veryShort);

  const nearHitThreshold = target.minVal <= 30
    ? Math.max(2, Math.pow(target.minVal, 0.72))
    : target.minVal <= 100
      ? 6
      : target.minVal <= 200
        ? 8
        : target.minVal <= 500
          ? 12
          : 16;
  const strongRoundThreshold = target.minVal <= 30
    ? Math.max(2.2, Math.pow(target.minVal, 0.82))
    : target.minVal <= 100
      ? 10
      : target.minVal <= 200
        ? 14
        : target.minVal <= 500
          ? 20
          : 30;
  const nearHitRate = shorter.length
    ? (shorter.filter((value) => value >= nearHitThreshold).length / shorter.length)
    : 0;
  const strongRate = shorter.length
    ? (shorter.filter((value) => value >= strongRoundThreshold).length / shorter.length)
    : 0;

  const tailRebound = veryShort.filter((value) => value >= nearHitThreshold).length;
  const tailStrongRate = veryShort.length
    ? (veryShort.filter((value) => value >= strongRoundThreshold).length / veryShort.length)
    : 0;
  const recentGaps = allGapsRaw.slice(-Math.min(14, allGapsRaw.length));
  const shortGapRate = recentGaps.length
    ? (recentGaps.filter((gap) => gap <= Math.max(2, Math.round(selectedStats.p25 * 0.55))).length / recentGaps.length)
    : 0;
  const immediateGapRate = recentGaps.length
    ? (recentGaps.filter((gap) => gap <= 2).length / recentGaps.length)
    : 0;

  const b2bSupportScore = clampNumber(
    (shortGapRate * 45) +
    (immediateGapRate * 30) +
    (tailRebound >= 2 ? 12 : 0) +
    (roundsSince <= 2 ? 8 : 0) +
    (nearHitRate * 18),
    0,
    100
  );

  const preWhiteCluster = highTarget
    ? (
      softWhiteRate >= 0.66 &&
      hardWhiteRate >= 0.28 &&
      trend12 < -5 &&
      trend6 <= -1 &&
      (hardWhiteTailStreak >= 2 || softWhiteTailStreak >= 4 || trend6 <= -2.5)
    )
    : (
      softWhiteRate >= 0.58 &&
      hardWhiteRate >= 0.22 &&
      trend12 < -4 &&
      trend6 <= 0 &&
      (hardWhiteTailStreak >= 2 || softWhiteTailStreak >= 3 || trend6 <= -2.2)
    );

  const activeWhitePressure = highTarget
    ? (hardWhiteTailStreak >= 3 || softWhiteTailStreak >= 5)
    : (hardWhiteTailStreak >= 2 || softWhiteTailStreak >= 4);

  const whiteCluster = highTarget
    ? (
      activeWhitePressure ||
      hardWhiteRate >= 0.52 ||
      (softWhiteRate >= 0.82 && trend24 < -5) ||
      (hardWhiteStreak >= 6 && hardWhiteTailStreak >= 2) ||
      (softWhiteStreak >= 10 && softWhiteTailStreak >= 4)
    )
    : (
      activeWhitePressure ||
      hardWhiteRate >= 0.40 ||
      (softWhiteRate >= 0.72 && trend24 < -3) ||
      (hardWhiteStreak >= 4 && hardWhiteTailStreak >= 2) ||
      (softWhiteStreak >= 7 && softWhiteTailStreak >= 3)
    );

  const hadRecentWhitePressure = highTarget
    ? (
      hardWhiteRate >= 0.40 ||
      softWhiteRate >= 0.72 ||
      hardWhiteStreak >= 5 ||
      softWhiteStreak >= 8
    )
    : (
      hardWhiteRate >= 0.30 ||
      softWhiteRate >= 0.62 ||
      hardWhiteStreak >= 3 ||
      softWhiteStreak >= 6
    );

  const whiteEndingSignal = (
    whiteCluster &&
    hardWhiteTailStreak <= 1 &&
    softWhiteTailStreak <= 2 &&
    tailRebound >= 2 &&
    trend6 > (highTarget ? 2.6 : 3.4) &&
    (
      tailStrongRate >= (highTarget ? 0.16 : 0.24) ||
      strongRate >= (highTarget ? 0.12 : 0.16)
    )
  );

  const downtrend = highTarget
    ? (
      trend24 < -8 &&
      trend12 < -5 &&
      trend6 < 1.6 &&
      nearHitRate < 0.14 &&
      softWhiteRate > 0.58
    )
    : (
      trend24 < -7 &&
      trend12 < -4 &&
      trend6 < 2.2 &&
      nearHitRate < 0.18 &&
      softWhiteRate > 0.5
    );

  const upshift = highTarget
    ? (
      trend12 > 3 &&
      nearHitRate >= 0.14 &&
      strongRate >= 0.10
    )
    : (
      trend12 > 4 &&
      nearHitRate >= 0.24 &&
      strongRate >= 0.14
    );

  const lowRegimeEndingSignal = (
    (whiteCluster || preWhiteCluster || downtrend || hadRecentWhitePressure) &&
    hardWhiteTailStreak <= (highTarget ? 2 : 1) &&
    softWhiteTailStreak <= (highTarget ? 3 : 2) &&
    (tailRebound >= 2 || tailStrongRate >= (highTarget ? 0.20 : 0.30)) &&
    trend6 > (highTarget ? 1.8 : 2.6) &&
    trend12 > (highTarget ? -2.2 : -1.8) &&
    nearHitRate >= (highTarget ? 0.12 : 0.16)
  );

  const randomLike = (
    Math.abs(trend24) < 1.4 &&
    Math.abs(trend12) < 1.2 &&
    Math.abs(nearHitRate - strongRate) < 0.06 &&
    shortGapRate < 0.12
  );

  return {
    lowSoftThreshold: Number(lowSoft.toFixed(2)),
    hardWhiteRate: Number((hardWhiteRate * 100).toFixed(1)),
    softWhiteRate: Number((softWhiteRate * 100).toFixed(1)),
    hardWhiteStreak,
    softWhiteStreak,
    hardWhiteTailStreak,
    softWhiteTailStreak,
    trend24: Number(trend24.toFixed(2)),
    trend12: Number(trend12.toFixed(2)),
    trend6: Number(trend6.toFixed(2)),
    nearHitThreshold: Number(nearHitThreshold.toFixed(2)),
    strongRoundThreshold: Number(strongRoundThreshold.toFixed(2)),
    nearHitRate: Number((nearHitRate * 100).toFixed(1)),
    strongRate: Number((strongRate * 100).toFixed(1)),
    shortGapRate: Number((shortGapRate * 100).toFixed(1)),
    immediateGapRate: Number((immediateGapRate * 100).toFixed(1)),
    b2bSupportScore: Number(b2bSupportScore.toFixed(1)),
    preWhiteCluster,
    whiteCluster,
    whiteEndingSignal,
    lowRegimeEndingSignal,
    downtrend,
    upshift,
    randomLike,
  };
}

function calibrateConfidence(rawConfidence, calibrationRows = []) {
  const rows = (calibrationRows || []).filter((row) => row && (row.outcome === 'win' || row.outcome === 'loss' || row.outcome === 'early'));
  const resolved = rows.filter((row) => row.outcome === 'win' || row.outcome === 'loss');
  if (resolved.length < MIN_GLOBAL_CALIBRATION) {
    const b = Math.floor(rawConfidence / 10) * 10;
    return {
      confidence: Math.round(rawConfidence),
      bucketLabel: `${b}-${b + 9}`,
      support: resolved.length,
      mode: 'raw',
    };
  }

  const globalWinRate = (resolved.filter((row) => row.outcome === 'win').length / resolved.length) * 100;
  const bucketStart = Math.floor(clampNumber(rawConfidence, 0, 99.9) / 10) * 10;
  const bucketRows = resolved.filter((row) => {
    const p = Number(row.probW);
    if (!Number.isFinite(p)) return false;
    const pct = clampNumber(p * 100, 0, 100);
    const b = Math.floor(Math.min(99.9, pct) / 10) * 10;
    return b === bucketStart;
  });

  let empirical = globalWinRate;
  let support = resolved.length;
  let mode = 'global';
  if (bucketRows.length >= MIN_BUCKET_CALIBRATION) {
    empirical = (bucketRows.filter((row) => row.outcome === 'win').length / bucketRows.length) * 100;
    support = bucketRows.length;
    mode = 'bucket';
  }

  const mixed = (rawConfidence * 0.62) + (empirical * 0.38);
  return {
    confidence: Math.round(clampNumber(mixed, 3, 96)),
    bucketLabel: `${bucketStart}-${bucketStart + 9}`,
    support,
    mode,
  };
}

function getIssueThreshold(minVal) {
  if (minVal >= 1000) return 48;
  if (minVal >= 500) return 46;
  if (minVal >= 200) return 44;
  if (minVal >= 100) return 42;
  if (minVal >= 50) return 43;
  if (minVal >= 30) return 44;
  if (minVal >= 15) return 41;
  if (minVal >= 10) return 40;
  return 37;
}

function computeOracleForecast(rounds, target, options = {}) {
  const cleanRounds = Array.isArray(rounds) && rounds.length && rounds[0]?.id
    ? rounds
    : normalizeRounds(rounds);

  const {
    label, minVal, color, scanN, window: winSize, minHits,
  } = target;
  if (!cleanRounds.length) return null;

  const nowId = cleanRounds[cleanRounds.length - 1].id;
  const hits = cleanRounds.filter((round) => round.val >= minVal);
  if (hits.length < (minHits + 1)) {
    return {
      ...target,
      noData: true,
      nowId,
      hits: hits.length,
      reason: hits.length ? `Need ${minHits + 1} hits to predict` : 'No hits yet',
      lastHit: hits[hits.length - 1] || null,
      predictedRound: 0,
      windowLo: 0,
      windowHi: 0,
      roundsUntilWindowLo: 0,
      roundsUntilWindowHi: 0,
      inWindow: false,
      confidence: 0,
      liveConfidence: 0,
      issuePrediction: false,
      activePrediction: false,
      issueMode: 'observe',
      regimeMode: 'full',
      avoidReason: 'weak_probability',
      lockDriftAlert: false,
      lockDriftReason: null,
    };
  }

  const allGapsRaw = [];
  for (let i = 1; i < hits.length; i += 1) {
    allGapsRaw.push(hits[i].id - hits[i - 1].id);
  }
  if (allGapsRaw.length < MIN_FORECAST_GAPS) {
    return {
      ...target,
      noData: true,
      nowId,
      hits: hits.length,
      reason: `Need at least ${MIN_FORECAST_GAPS} gaps`,
      lastHit: hits[hits.length - 1],
      predictedRound: 0,
      windowLo: 0,
      windowHi: 0,
      roundsUntilWindowLo: 0,
      roundsUntilWindowHi: 0,
      inWindow: false,
      confidence: 0,
      liveConfidence: 0,
      issuePrediction: false,
      activePrediction: false,
      issueMode: 'observe',
      regimeMode: 'full',
      avoidReason: 'weak_probability',
      lockDriftAlert: false,
      lockDriftReason: null,
    };
  }

  const allGapsSorted = allGapsRaw.slice().sort((a, b) => a - b);
  const recentN = Math.min(allGapsRaw.length, Math.max(scanN, 24));
  const recentSorted = allGapsRaw.slice(-recentN).sort((a, b) => a - b);
  const regimeMode = computeRegimeMode(allGapsSorted, recentSorted, cleanRounds, target);
  const selectedSorted = regimeMode === 'recent' ? recentSorted : allGapsSorted;
  const selectedStats = buildRobustStats(selectedSorted);
  const lastHit = hits[hits.length - 1];
  const roundsSince = nowId - lastHit.id;
  const droughtPct = Math.round((selectedSorted.filter((gap) => gap <= roundsSince).length / selectedSorted.length) * 100);
  const isTooEarly = roundsSince < selectedStats.p10;
  const isOverdue = roundsSince > selectedStats.med;
  const isHardGap = roundsSince > selectedStats.p90;
  const isExtreme = roundsSince > selectedStats.p99;
  const survivors = selectedSorted.filter((gap) => gap > roundsSince);
  const openWindow = survivors.length === 0;

  const clusters = buildHistogramClusters(selectedStats.trimmed.length ? selectedStats.trimmed : selectedSorted);
  const primaryCluster = clusters.primary;
  const secondaryCluster = clusters.secondary;

  const recentPattern = computeRecentPatternDiagnostics(cleanRounds, target, roundsSince, allGapsRaw, selectedStats);
  const patternSupport = computePatternSupport(cleanRounds, target, winSize);

  let predictedGap = null;
  let predMethod = 'cluster';
  if (!openWindow && primaryCluster && !isOverdue) {
    predictedGap = Math.max(roundsSince + 1, primaryCluster.center);
    predMethod = 'cluster';
  } else if (!openWindow) {
    predictedGap = computeConditionalExpectedGap(survivors, roundsSince);
    predMethod = 'survivor_expectation';
  } else {
    const tail = selectedSorted.slice(Math.max(0, Math.floor(selectedSorted.length * 0.85)));
    const tailDiffs = [];
    for (let i = 1; i < tail.length; i += 1) {
      tailDiffs.push(tail[i] - tail[i - 1]);
    }
    const tailStep = clampNumber(Math.round(mean(tailDiffs) || selectedStats.iqr || 1), 1, Math.max(1, selectedStats.iqr * 2));
    predictedGap = Math.max(roundsSince + 1, selectedStats.max + tailStep);
    predMethod = 'tail_extrapolation';
  }

  if (recentPattern.b2bSupportScore >= 62 && roundsSince <= 2 && minVal <= 50) {
    predictedGap = Math.max(roundsSince + 1, Math.min(predictedGap, roundsSince + Math.max(1, Math.round(selectedStats.p25 * 0.5))));
    predMethod = 'b2b_support';
  }

  if (
    minVal >= 100 &&
    (recentPattern.upshift || recentPattern.whiteEndingSignal || recentPattern.lowRegimeEndingSignal || recentPattern.b2bSupportScore >= 58) &&
    predictedGap > selectedStats.p25
  ) {
    const transitionGap = Math.round((selectedStats.p25 * 0.65) + (selectedStats.med * 0.35));
    predictedGap = Math.max(roundsSince + 1, Math.min(predictedGap, transitionGap));
    predMethod = 'high_target_transition';
  }

  if (
    recentPattern.whiteCluster &&
    !recentPattern.whiteEndingSignal &&
    !recentPattern.lowRegimeEndingSignal &&
    predictedGap < selectedStats.p25
  ) {
    predictedGap = selectedStats.p25;
    predMethod = 'white_safety';
  }

  predictedGap = Math.max(roundsSince + 1, Math.round(predictedGap || selectedStats.med || roundsSince + 1));
  const halfWin = Math.floor(winSize / 2);
  const windowLoGap = Math.max(1, predictedGap - halfWin);
  const windowHiGap = windowLoGap + winSize - 1;

  let predictedRound = lastHit.id + predictedGap;
  let windowLo = lastHit.id + windowLoGap;
  let windowHi = lastHit.id + windowHiGap;

  if (windowHi <= nowId) {
    const drift = (nowId - windowHi) + 1;
    windowLo += drift;
    windowHi += drift;
    predictedRound = clampNumber(predictedRound + drift, windowLo, windowHi);
  }

  const roundsUntilWindowLo = Math.max(0, windowLo - nowId);
  const roundsUntilWindowHi = Math.max(0, windowHi - nowId);
  const inWindow = nowId >= windowLo && nowId <= windowHi;
  const nearWindowRounds = Math.max(0, roundsUntilWindowLo - Math.min(10, Math.round(winSize * 0.6)));

  const kmReliable = selectedSorted.length >= MIN_KM_GAPS;
  const kmTable = kmReliable ? buildKMTable(selectedSorted) : null;
  const pHit1 = kmReliable ? kmProb(kmTable, roundsSince, 1) : 0;
  const pHit5 = kmReliable ? kmProb(kmTable, roundsSince, 5) : 0;
  const pHit10 = kmReliable ? kmProb(kmTable, roundsSince, 10) : 0;
  const pHit20 = kmReliable ? kmProb(kmTable, roundsSince, 20) : 0;
  const pHitWindow = kmReliable
    ? clampNumber(
      kmProb(kmTable, roundsSince, roundsUntilWindowHi) - kmProb(kmTable, roundsSince, roundsUntilWindowLo),
      0,
      100
    )
    : clampNumber(
      (selectedSorted.filter((gap) => gap >= (roundsSince + roundsUntilWindowLo) && gap <= (roundsSince + roundsUntilWindowHi)).length / selectedSorted.length) * 100,
      0,
      100
    );

  const pHitNearWindow = kmReliable
    ? kmProb(kmTable, roundsSince, Math.max(1, roundsUntilWindowHi + Math.round(winSize * 0.35)))
    : clampNumber(
      (selectedSorted.filter((gap) => gap >= (roundsSince + Math.max(1, roundsUntilWindowLo - Math.round(winSize * 0.5))) && gap <= (roundsSince + roundsUntilWindowHi + Math.round(winSize * 0.35))).length / selectedSorted.length) * 100,
      0,
      100
    );

  const scoredWindowLoGap = Math.max(1, windowLo - lastHit.id);
  const scoredWindowHiGap = Math.max(scoredWindowLoGap, windowHi - lastHit.id);
  const hitsInWindow = selectedSorted.filter((gap) => gap >= scoredWindowLoGap && gap <= scoredWindowHiGap).length;
  const empiricalWindowHitRate = (hitsInWindow / selectedSorted.length) * 100;

  const chanceSource = regimeMode === 'recent'
    ? trimSorted(tukeyFilterSorted(recentSorted), recentSorted.length >= 12 ? 0.1 : 0)
    : selectedSorted;
  const chance = computeChanceWindowRate(chanceSource, winSize);
  const chanceWindowRate = chance.rate;
  const predictiveLift = empiricalWindowHitRate - chanceWindowRate;
  const standardizedLift = chance.std > 0.0001 ? predictiveLift / chance.std : 0;

  const randomLiftWeak = patternSupport.ready
    ? (patternSupport.lift < 1.2 && predictiveLift < 0.8)
    : predictiveLift < -0.2;

  let rawConfidence = (
    (pHitWindow * 0.56) +
    (pHit10 * 0.18) +
    (empiricalWindowHitRate * 0.16) +
    (clampNumber(patternSupport.lift, -15, 20) * 0.5) +
    (clampNumber(standardizedLift * 4, -12, 16))
  );

  if (recentPattern.b2bSupportScore >= 58) rawConfidence += 6;
  if (recentPattern.upshift) rawConfidence += 6;
  if (recentPattern.whiteEndingSignal) rawConfidence += 8;
  if (recentPattern.lowRegimeEndingSignal) rawConfidence += 9;
  if (recentPattern.preWhiteCluster && !recentPattern.lowRegimeEndingSignal) rawConfidence -= 12;
  if (
    recentPattern.whiteCluster &&
    !recentPattern.whiteEndingSignal &&
    !recentPattern.lowRegimeEndingSignal
  ) rawConfidence -= 19;
  if (recentPattern.downtrend) rawConfidence -= 11;
  if (recentPattern.randomLike || randomLiftWeak) rawConfidence -= 8;
  if (!kmReliable && minVal >= 100) rawConfidence -= 4;
  if (isTooEarly) rawConfidence -= 6;
  if (openWindow) rawConfidence -= 5;

  rawConfidence = clampNumber(rawConfidence, 0, 100);
  const calibration = calibrateConfidence(rawConfidence, options.calibrationRows || []);
  const confidence = calibration.confidence;

  const threshold = getIssueThreshold(minVal);
  const highTargetMomentum = minVal >= 100 && recentPattern.nearHitRate >= 14 && recentPattern.trend6 > 2;
  const strongTransition =
    recentPattern.whiteEndingSignal ||
    recentPattern.lowRegimeEndingSignal ||
    recentPattern.upshift ||
    highTargetMomentum ||
    (recentPattern.b2bSupportScore >= 65);
  const strongEdge = (pHitWindow >= (threshold - 6)) || (predictiveLift >= 3.5) || (patternSupport.ready && patternSupport.lift >= 4);
  const highXAnticipation = (
    minVal >= 15 &&
    !recentPattern.whiteCluster &&
    !recentPattern.preWhiteCluster &&
    !recentPattern.downtrend &&
    roundsUntilWindowLo <= Math.max(6, Math.round(winSize * 0.9)) &&
    pHitNearWindow >= Math.max(24, threshold - 12) &&
    confidence >= Math.max(18, threshold - 12) &&
    (
      predictiveLift >= -0.5 ||
      (patternSupport.ready && patternSupport.lift >= 0.5)
    )
  );

  const whiteEscapeScore = minVal >= 100 ? 56 : 68;
  const preWhiteEscapeScore = whiteEscapeScore + 6;
  const hardWhiteBlock = (
    (recentPattern.whiteCluster || recentPattern.preWhiteCluster) &&
    !recentPattern.whiteEndingSignal &&
    !recentPattern.lowRegimeEndingSignal &&
    !recentPattern.upshift &&
    (
      (recentPattern.whiteCluster && recentPattern.b2bSupportScore < whiteEscapeScore) ||
      (recentPattern.preWhiteCluster && recentPattern.b2bSupportScore < preWhiteEscapeScore)
    )
  );
  const hardDowntrendBlock = (
    recentPattern.downtrend &&
    !strongTransition &&
    recentPattern.b2bSupportScore < 60
  );
  const randomHardBlock = randomLiftWeak && !strongTransition && pHitWindow < 26;
  const kmHardBlock = (
    !kmReliable &&
    minVal >= 500 &&
    !strongTransition &&
    pHitWindow < 18 &&
    predictiveLift < 0
  );
  const hardBlock = hardWhiteBlock || hardDowntrendBlock || randomHardBlock || kmHardBlock;

  let avoidReason = null;
  if (hardWhiteBlock) avoidReason = recentPattern.preWhiteCluster ? 'pre_white_cluster' : 'white_cluster';
  else if (hardDowntrendBlock) avoidReason = 'downtrend';
  else if (randomHardBlock) avoidReason = 'random_like';
  else if (kmHardBlock) avoidReason = 'weak_probability';
  else if (isTooEarly && confidence < threshold) avoidReason = 'too_early';
  else if (confidence < threshold) avoidReason = 'weak_probability';

  const issuePrediction = !hardBlock && (
    confidence >= threshold ||
    (strongTransition && confidence >= (threshold - 7) && strongEdge) ||
    highXAnticipation
  );

  const issueMode = issuePrediction
    ? (
      recentPattern.b2bSupportScore >= 66 && roundsSince <= 2 ? 'b2b_support'
        : (recentPattern.whiteEndingSignal || recentPattern.lowRegimeEndingSignal) ? 'white_rebound'
          : highXAnticipation ? 'highx_anticipation'
          : patternSupport.ready && patternSupport.lift >= 4 ? 'pattern_support'
            : strongTransition ? 'transition_support'
              : 'strict'
    )
    : 'observe';

  const windowReadyThreshold = Math.max(1, Math.round(winSize * 0.75));
  const windowReady = roundsUntilWindowLo <= windowReadyThreshold;
  const chaseRaw = Math.round(clampNumber(
    confidence +
    (issuePrediction ? 14 : -6) +
    (inWindow ? 10 : 0) +
    (strongTransition ? 6 : 0) -
    (hardWhiteBlock ? 14 : 0) -
    (hardDowntrendBlock ? 10 : 0),
    0,
    100
  ));
  const chaseSignal = issuePrediction ? 'CHASE' : 'SKIP';
  const chaseColor = issuePrediction ? '#39ff8a' : '#ff5555';

  const predBasis = `${predMethod} (${regimeMode}, ${selectedSorted.length} gaps)` +
    (patternSupport.ready ? ` + pattern lift ${patternSupport.lift.toFixed(1)}%` : '');

  const reliabilityFlags = [];
  if (selectedSorted.length < MIN_FORECAST_GAPS) reliabilityFlags.push('low_data');
  if (!kmReliable) reliabilityFlags.push('km_unreliable');
  if (regimeMode === 'recent') reliabilityFlags.push('regime_shift');
  if (recentPattern.preWhiteCluster) reliabilityFlags.push('pre_white_cluster');
  if (recentPattern.whiteCluster) reliabilityFlags.push('white_cluster');
  if (recentPattern.downtrend) reliabilityFlags.push('downtrend');
  if (randomLiftWeak || recentPattern.randomLike) reliabilityFlags.push('random_like');
  if (!patternSupport.ready) reliabilityFlags.push('pattern_low_sample');

  return {
    ...target,
    noData: false,
    nowId,
    hits: hits.length,
    n: selectedSorted.length,
    lastHit,
    roundsSince,
    allGapsRaw,
    allGapsSorted,
    recentSorted,
    med: selectedStats.med,
    p10: selectedStats.p10,
    p25: selectedStats.p25,
    p75: selectedStats.p75,
    p90: selectedStats.p90,
    p99: selectedStats.p99,
    minGap: selectedStats.min,
    maxGap: selectedStats.max,
    avgGap: selectedStats.avg,
    iqr: selectedStats.iqr,
    clusterCenter: primaryCluster?.center ?? selectedStats.med,
    secondaryClusterCenter: secondaryCluster?.center ?? null,
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
    riskOverride: !issuePrediction,
    softDowntrendBlock: hardDowntrendBlock,
    softWhiteBlock: hardWhiteBlock,
    b2bMomentum: Number(recentPattern.b2bSupportScore.toFixed(1)),
    b2bMomentumPct: Number(recentPattern.b2bSupportScore.toFixed(1)),
    b2bMomentumRatio: Number((recentPattern.b2bSupportScore / 100).toFixed(4)),
    transitionSupport: Boolean(strongTransition || highXAnticipation),
    windowReady,
    windowReadyThreshold,
    pHit1: Number(pHit1.toFixed(1)),
    pHit5: Number(pHit5.toFixed(1)),
    pHit10: Number(pHit10.toFixed(1)),
    pHit20: Number(pHit20.toFixed(1)),
    pHitWindow: Number(pHitWindow.toFixed(1)),
    pHitNearWindow: Number(pHitNearWindow.toFixed(1)),
    probReliable: kmReliable,
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
    recentPattern: {
      ...recentPattern,
      patternSupportPct: Number(patternSupport.supportPct.toFixed(1)),
      patternRandomPct: Number(patternSupport.randomPct.toFixed(1)),
      patternLift: Number(patternSupport.lift.toFixed(1)),
      patternMatches: patternSupport.matches,
      patternSampleSize: patternSupport.sampleSize,
      patternReady: patternSupport.ready,
      bestPatternDistance: patternSupport.bestDistance != null ? Number(patternSupport.bestDistance.toFixed(4)) : null,
    },
    regimeMode,
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
