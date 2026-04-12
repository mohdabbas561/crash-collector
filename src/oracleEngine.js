'use strict';

// ═══════════════════════════════════════════════════════════════════════════════
// Oracle Engine V4 — Multi-Layer Crash Prediction System
// ═══════════════════════════════════════════════════════════════════════════════
//
// 7 Prediction Layers:
//   1. Adaptive Markov Chain (bucket state transitions)
//   2. Regime-Aware Kaplan-Meier Survival
//   3. EWMA Cluster Tracker (crossover-based white cluster detection)
//   4. B2B Momentum Detector (gap acceleration + hit rate surge)
//   5. White Cluster 5-Phase Lifecycle (PRE_WHITE → WHITE_ACTIVE → WHITE_ENDING)
//   6. Pattern Similarity Engine (DTW-lite)
//   7. Enhanced Regime Detector (JS divergence + multi-classification)
//
// Ensemble: inverse-error weighted average with hard blocks and floors.
//
// Designed for 40k+ historical round datasets.
// ═══════════════════════════════════════════════════════════════════════════════

const ORACLE_TARGETS = Object.freeze([
  { label: '5x',    minVal: 5,    color: '#00ff88', window: 4,   scanN: 150, minHits: 7 },
  { label: '10x',   minVal: 10,   color: '#00d4ff', window: 6,   scanN: 130, minHits: 6 },
  { label: '15x',   minVal: 15,   color: '#ff6b9d', window: 7,   scanN: 110, minHits: 5 },
  { label: '30x',   minVal: 30,   color: '#ff9f43', window: 13,  scanN: 90,  minHits: 4 },
  { label: '50x',   minVal: 50,   color: '#4db8ff', window: 20,  scanN: 75,  minHits: 4 },
  { label: '100x',  minVal: 100,  color: '#39ff8a', window: 30,  scanN: 65,  minHits: 4 },
  { label: '200x',  minVal: 200,  color: '#c77dff', window: 50,  scanN: 55,  minHits: 3 },
  { label: '500x',  minVal: 500,  color: '#ff4da6', window: 75,  scanN: 48,  minHits: 3 },
  { label: '1000x', minVal: 1000, color: '#7aa2ff', window: 100, scanN: 42,  minHits: 3 },
]);

// ─── Configuration ───────────────────────────────────────────────────────────
const CFG = Object.freeze({
  // Markov chain
  markovRecentWeight: 0.60,
  markovRecentWindow: 500,
  markovBuckets: [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, Infinity],

  // Kaplan-Meier
  kmMinGaps: 20,
  kmRegimeRecentN: 200,
  kmRegimeBlendRecent: 0.55,

  // EWMA
  ewmaAlphaShort: 0.25,
  ewmaAlphaLong: 0.08,
  ewmaCrossoverThreshold: 0.04,

  // B2B
  b2bGapWindow: 20,
  b2bShortHitWindow: 12,
  b2bLongHitWindow: 120,
  b2bImmediateGapMax: 2,
  b2bAccelerationThreshold: -0.15,

  // White cluster lifecycle
  whiteHardCap: 1.25,
  whiteSoftMultiplier: 2.0,
  whitePreEntryTrendRounds: 6,
  whitePreEntryLowConcentration: 0.55,
  whitePreEntryVolCompression: 0.35,
  whiteActiveHardRate: 0.42,
  whiteActiveSoftRate: 0.72,
  whiteActiveStreak: 5,
  whiteEndingReboundMultiple: 2.2,
  whiteEndingEwmaReversal: 1,
  whiteEndingTailRecovery: 1,

  // Pattern similarity
  patternLen: 15,
  patternTopK: 24,
  patternMaxCandidates: 5000,

  // Regime detector
  regimeRecentWindow: 300,
  regimeJsThreshold: 0.015,
  regimeLagThreshold: 0.12,
  regimeDriftThreshold: 0.06,

  // Ensemble
  ensembleMinLayers: 2,
  calibrationWindow: 200,

  // Confidence caps/floors
  whiteActiveConfidenceCap: 24,
  preWhiteConfidenceCap: 42,
  whiteEndingBoost: 14,
  b2bConfidenceFloor: 40,
  b2bScoreThreshold: 60,

  // Issue thresholds per target range
  issueThresholdLow: 26,     // ≤15x
  issueThresholdMid: 30,     // ≤50x
  issueThresholdHigh: 33,    // ≤200x
  issueThresholdMoon: 35,    // >200x
});


// ─── Utilities ───────────────────────────────────────────────────────────────

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

function mean(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function stddev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2;
  return Math.sqrt(s / (arr.length - 1));
}

function quantile(sorted, p) {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function sortedCopy(arr) {
  return arr.slice().sort((a, b) => a - b);
}

function trimSortedRange(sorted, loPct = 5, hiPct = 95) {
  if (!Array.isArray(sorted) || !sorted.length) return [];
  if (sorted.length < 12) return sorted.slice();
  const loIdx = clamp(Math.floor((loPct / 100) * (sorted.length - 1)), 0, sorted.length - 1);
  const hiIdx = clamp(Math.ceil((hiPct / 100) * (sorted.length - 1)), loIdx, sorted.length - 1);
  const trimmed = sorted.slice(loIdx, hiIdx + 1);
  return trimmed.length >= 8 ? trimmed : sorted.slice();
}

function normalizeRounds(rounds) {
  const map = new Map();
  for (const r of rounds || []) {
    const id = Number(r?.roundId ?? r?.id);
    const val = Number.parseFloat(r?.multiplier ?? r?.val ?? r?.gameResult ?? r?.result);
    if (!Number.isFinite(id) || !Number.isFinite(val) || val <= 0) continue;
    map.set(id, { id, val: Number(val.toFixed(4)) });
  }
  return [...map.values()].sort((a, b) => a.id - b.id);
}

function getBucketIndex(val) {
  const buckets = CFG.markovBuckets;
  for (let i = 0; i < buckets.length; i++) {
    if (val < buckets[i]) return i;
  }
  return buckets.length - 1;
}

function getIssueThreshold(minVal) {
  if (minVal <= 15) return CFG.issueThresholdLow;
  if (minVal <= 50) return CFG.issueThresholdMid;
  if (minVal <= 200) return CFG.issueThresholdHigh;
  return CFG.issueThresholdMoon;
}

function getWindowBaselineProbability(minVal, winSize) {
  const oneRoundBase = clamp(1 / Math.max(1, minVal), 0, 1);
  return clamp(1 - ((1 - oneRoundBase) ** Math.max(1, winSize)), 0, 1);
}


// ─── Layer 1: Adaptive Markov Chain ──────────────────────────────────────────

function buildMarkovLayer(rounds, target, winSize) {
  const nBuckets = CFG.markovBuckets.length;
  const n = rounds.length;
  if (n < Math.max(30, winSize + 8)) return { prob: null, reliability: 0 };

  const hitPrefix = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) {
    hitPrefix[i + 1] = hitPrefix[i] + (rounds[i].val >= target.minVal ? 1 : 0);
  }
  const hasHitInWindow = (lo, hi) => {
    const left = clamp(lo, 0, n - 1);
    const right = clamp(hi, left, n - 1);
    return (hitPrefix[right + 1] - hitPrefix[left]) > 0;
  };

  const recentStart = Math.max(1, n - CFG.markovRecentWindow - winSize);
  const pairTotalAll = new Float64Array(nBuckets * nBuckets);
  const pairHitsAll = new Float64Array(nBuckets * nBuckets);
  const pairTotalRecent = new Float64Array(nBuckets * nBuckets);
  const pairHitsRecent = new Float64Array(nBuckets * nBuckets);
  const singleTotalAll = new Float64Array(nBuckets);
  const singleHitsAll = new Float64Array(nBuckets);
  const singleTotalRecent = new Float64Array(nBuckets);
  const singleHitsRecent = new Float64Array(nBuckets);

  let unconditionalTotal = 0;
  let unconditionalHits = 0;
  let unconditionalRecentTotal = 0;
  let unconditionalRecentHits = 0;

  for (let i = 1; i < (n - winSize); i++) {
    const prev2 = getBucketIndex(rounds[i - 1].val);
    const prev1 = getBucketIndex(rounds[i].val);
    const pairIdx = prev2 * nBuckets + prev1;
    const futureHit = hasHitInWindow(i + 1, i + winSize) ? 1 : 0;
    const inRecent = i >= recentStart;

    pairTotalAll[pairIdx] += 1;
    pairHitsAll[pairIdx] += futureHit;
    singleTotalAll[prev1] += 1;
    singleHitsAll[prev1] += futureHit;
    unconditionalTotal += 1;
    unconditionalHits += futureHit;

    if (inRecent) {
      pairTotalRecent[pairIdx] += 1;
      pairHitsRecent[pairIdx] += futureHit;
      singleTotalRecent[prev1] += 1;
      singleHitsRecent[prev1] += futureHit;
      unconditionalRecentTotal += 1;
      unconditionalRecentHits += futureHit;
    }
  }

  if (n < 2) return { prob: null, reliability: 0 };
  const prev2 = getBucketIndex(rounds[n - 2].val);
  const prev1 = getBucketIndex(rounds[n - 1].val);
  const pairIdx = prev2 * nBuckets + prev1;

  let totalAll = pairTotalAll[pairIdx];
  let hitsAll = pairHitsAll[pairIdx];
  let totalRecent = pairTotalRecent[pairIdx];
  let hitsRecent = pairHitsRecent[pairIdx];

  if (totalAll < 8) {
    totalAll = singleTotalAll[prev1];
    hitsAll = singleHitsAll[prev1];
    totalRecent = singleTotalRecent[prev1];
    hitsRecent = singleHitsRecent[prev1];
  }

  if (totalAll <= 0) {
    const baseAll = unconditionalTotal > 0 ? (unconditionalHits / unconditionalTotal) : 0;
    const baseRecent = unconditionalRecentTotal > 0 ? (unconditionalRecentHits / unconditionalRecentTotal) : baseAll;
    const prob = unconditionalRecentTotal >= 12
      ? (CFG.markovRecentWeight * baseRecent + (1 - CFG.markovRecentWeight) * baseAll)
      : baseAll;
    return {
      prob: clamp(prob, 0, 1),
      reliability: clamp(unconditionalTotal / 200, 0, 0.45),
    };
  }

  const pAll = hitsAll / totalAll;
  const pRecent = totalRecent > 0 ? (hitsRecent / totalRecent) : pAll;
  const prob = totalRecent >= 8
    ? (CFG.markovRecentWeight * pRecent + (1 - CFG.markovRecentWeight) * pAll)
    : pAll;
  const reliability = clamp((totalAll + totalRecent) / 80, 0, 1);
  return { prob: clamp(prob, 0, 1), reliability };
}


// ─── Layer 2: Regime-Aware Kaplan-Meier ──────────────────────────────────────

function buildKMTable(gapsSorted) {
  if (!gapsSorted.length) return new Float32Array(2).fill(1);
  const n = gapsSorted.length;
  const maxGap = gapsSorted[n - 1];
  const limit = maxGap + 120;
  const table = new Float32Array(limit + 1).fill(1);
  let survival = 1;
  let left = 0;
  for (let t = 1; t <= limit; t++) {
    while (left < n && gapsSorted[left] < t) left++;
    let right = left;
    while (right < n && gapsSorted[right] === t) right++;
    const atRisk = n - left;
    const events = right - left;
    if (atRisk > 0) {
      survival *= (1 - events / atRisk);
      survival = clamp(survival, 0, 1);
    }
    table[t] = survival;
  }
  return table;
}

function kmProb(table, roundsSince, k) {
  if (!table || !table.length) return 0;
  const from = clamp(Math.round(roundsSince), 0, table.length - 1);
  const to = clamp(Math.round(roundsSince + k), 0, table.length - 1);
  const sFrom = table[from];
  if (sFrom <= 0) return 100;
  return clamp((1 - table[to] / sFrom) * 100, 0, 100);
}

function buildKMLayer(allGaps, recentGaps, roundsSince, winSize, regime) {
  if (allGaps.length < CFG.kmMinGaps) return { pHitWindow: 0, pHit1: 0, pHit5: 0, reliability: 0 };

  const allSorted = sortedCopy(allGaps);
  const recentSorted = sortedCopy(recentGaps);

  const regimeLabel = regime?.label || 'RANDOM';
  const regimeReliable = Number(regime?.reliability || 0) >= 0.35;
  const useRecent =
    regimeReliable &&
    recentSorted.length >= CFG.kmMinGaps &&
    ['TRENDING_UP', 'TRENDING_DOWN', 'CLUSTERED_LOW', 'VOLATILE', 'DISPERSED'].includes(regimeLabel);

  const selectedGaps = useRecent
    ? trimSortedRange(recentSorted, 8, 92)
    : trimSortedRange(allSorted, 4, 96);

  const table = buildKMTable(selectedGaps);

  const pHit1 = kmProb(table, roundsSince, 1);
  const pHit5 = kmProb(table, roundsSince, 5);
  const pHitWindow = kmProb(table, roundsSince, winSize);

  const reliability = clamp(selectedGaps.length / (CFG.kmMinGaps * 2.5), 0, 1);

  return { pHitWindow, pHit1, pHit5, reliability, table };
}


// ─── Layer 3: EWMA Cluster Tracker ───────────────────────────────────────────

function computeEWMA(values, alpha) {
  if (!values.length) return 0;
  let ewma = values[0];
  for (let i = 1; i < values.length; i++) {
    ewma = alpha * values[i] + (1 - alpha) * ewma;
  }
  return ewma;
}

function buildEWMALayer(rounds, target) {
  const n = rounds.length;
  if (n < 30) return { shortEwma: 0, longEwma: 0, crossover: 'none', signalStrength: 0, reliability: 0 };

  // Binary hit series: 1 if >= target, 0 otherwise
  const hitSeries = rounds.map(r => r.val >= target.minVal ? 1.0 : 0.0);

  const shortEwma = computeEWMA(hitSeries, CFG.ewmaAlphaShort);
  const longEwma = computeEWMA(hitSeries, CFG.ewmaAlphaLong);

  // Check recent crossover history for stability
  const recentLen = Math.min(20, n);
  const recentShorts = [];
  const recentLongs = [];
  {
    let s = hitSeries[0], l = hitSeries[0];
    for (let i = 1; i < n; i++) {
      s = CFG.ewmaAlphaShort * hitSeries[i] + (1 - CFG.ewmaAlphaShort) * s;
      l = CFG.ewmaAlphaLong * hitSeries[i] + (1 - CFG.ewmaAlphaLong) * l;
      if (i >= n - recentLen) {
        recentShorts.push(s);
        recentLongs.push(l);
      }
    }
  }

  const gap = shortEwma - longEwma;
  let crossover = 'none';
  if (gap > CFG.ewmaCrossoverThreshold) crossover = 'bullish';   // cluster ending, hits returning
  else if (gap < -CFG.ewmaCrossoverThreshold) crossover = 'bearish'; // cluster entering, hits declining

  const signalStrength = Math.abs(gap) / Math.max(0.001, longEwma || 0.05);
  const reliability = clamp(n / 200, 0, 1);

  return {
    shortEwma: Number(shortEwma.toFixed(6)),
    longEwma: Number(longEwma.toFixed(6)),
    crossover,
    signalStrength: Number(clamp(signalStrength, 0, 5).toFixed(4)),
    reliability,
  };
}


// ─── Layer 4: B2B Momentum Detector ──────────────────────────────────────────

function buildB2BLayer(allGaps, rounds, target, roundsSince) {
  const n = rounds.length;
  if (allGaps.length < 5) return { b2bScore: 0, gapAcceleration: 0, hitRateSurge: 0, immediateB2B: false, reliability: 0 };

  // Gap acceleration: are gaps between hits shrinking?
  const recentGaps = allGaps.slice(-CFG.b2bGapWindow);
  let gapAcceleration = 0;
  if (recentGaps.length >= 4) {
    const firstHalf = recentGaps.slice(0, Math.floor(recentGaps.length / 2));
    const secondHalf = recentGaps.slice(Math.floor(recentGaps.length / 2));
    const firstMean = mean(firstHalf);
    const secondMean = mean(secondHalf);
    if (firstMean > 0) {
      gapAcceleration = (secondMean - firstMean) / firstMean;
    }
  }

  // Hit rate surge: short-term hit rate vs long-term
  const shortHits = rounds.slice(-CFG.b2bShortHitWindow).filter(r => r.val >= target.minVal).length;
  const longHits = rounds.slice(-CFG.b2bLongHitWindow).filter(r => r.val >= target.minVal).length;
  const shortRate = shortHits / Math.max(1, Math.min(n, CFG.b2bShortHitWindow));
  const longRate = longHits / Math.max(1, Math.min(n, CFG.b2bLongHitWindow));
  const hitRateSurge = longRate > 0 ? (shortRate - longRate) / longRate : 0;

  // Immediate b2b: last gaps were very short
  const lastFewGaps = allGaps.slice(-3);
  const immediateB2B = (
    roundsSince <= CFG.b2bImmediateGapMax &&
    lastFewGaps.filter(g => g <= CFG.b2bImmediateGapMax).length >= 2
  );

  // Composite score
  const accelerationScore = clamp(-gapAcceleration * 100, 0, 40); // negative accel = shrinking gaps = bullish
  const surgeScore = clamp(hitRateSurge * 30, 0, 30);
  const immediateScore = immediateB2B ? 20 : 0;
  const recencyScore = roundsSince <= 2 ? 10 : roundsSince <= 5 ? 5 : 0;
  const b2bScore = clamp(accelerationScore + surgeScore + immediateScore + recencyScore, 0, 100);

  const reliability = clamp(allGaps.length / 15, 0, 1);

  return {
    b2bScore: Number(b2bScore.toFixed(1)),
    gapAcceleration: Number(gapAcceleration.toFixed(4)),
    hitRateSurge: Number(hitRateSurge.toFixed(4)),
    immediateB2B,
    reliability,
  };
}


// ─── Layer 5: White Cluster 5-Phase Lifecycle ────────────────────────────────

function computeWhitePhase(rounds, target) {
  const n = rounds.length;
  if (n < 20) return { phase: 'NORMAL', signals: {}, reliability: 0 };




  // White pressure should reflect true low-multiplier clusters, not scale linearly with high targets.
  // Capped target-relative thresholds reduce false "always-white" behavior on high X.
  const softThreshold = target.minVal <= 10 ? 1.9
    : target.minVal <= 30 ? 2.05
    : target.minVal <= 100 ? 2.15
    : target.minVal <= 500 ? 2.22
    : 2.28;

  const recent24 = rounds.slice(-24).map(r => r.val);
  const recent12 = rounds.slice(-12).map(r => r.val);
  const recent8 = rounds.slice(-8).map(r => r.val);
  const recent6 = rounds.slice(-6).map(r => r.val);
  const recent4 = rounds.slice(-4).map(r => r.val);

  // Hard white rate (≤1.25x)
  const hardWhiteRate24 = recent24.filter(v => v <= CFG.whiteHardCap).length / recent24.length;
  const hardWhiteRate12 = recent12.filter(v => v <= CFG.whiteHardCap).length / recent12.length;

  // Soft white rate (below target-dependent threshold)
  const softWhiteRate24 = recent24.filter(v => v <= softThreshold).length / recent24.length;
  const softWhiteRate12 = recent12.filter(v => v <= softThreshold).length / recent12.length;
  const softWhiteRate8 = recent8.filter(v => v <= softThreshold).length / Math.max(1, recent8.length);
  const hardWhiteRate8 = recent8.filter(v => v <= CFG.whiteHardCap).length / Math.max(1, recent8.length);

  // Trend (log-multiplier slope)
  const logRecent = recent12.map(v => Math.log(Math.max(1.0001, v)));
  let trend12 = 0;
  if (logRecent.length >= 4) {
    const mX = (logRecent.length - 1) / 2;
    const mY = mean(logRecent);
    let num = 0, den = 0;
    for (let i = 0; i < logRecent.length; i++) {
      const dx = i - mX;
      num += dx * (logRecent[i] - mY);
      den += dx * dx;
    }
    trend12 = den > 0 ? (num / den) * (logRecent.length - 1) / Math.max(0.0001, Math.abs(mY)) * 100 : 0;
  }
  const logRecent6 = recent6.map(v => Math.log(Math.max(1.0001, v)));
  let trend6 = 0;
  if (logRecent6.length >= 3) {
    const mX = (logRecent6.length - 1) / 2;
    const mY = mean(logRecent6);
    let num = 0, den = 0;
    for (let i = 0; i < logRecent6.length; i++) {
      const dx = i - mX;
      num += dx * (logRecent6[i] - mY);
      den += dx * dx;
    }
    trend6 = den > 0 ? (num / den) * (logRecent6.length - 1) / Math.max(0.0001, Math.abs(mY)) * 100 : 0;
  }

  // Volatility (stddev of log-multipliers)
  const vol12 = stddev(logRecent);
  const longLog = rounds.slice(-60).map(r => Math.log(Math.max(1.0001, r.val)));
  const vol60 = stddev(longLog);
  const volCompression = vol60 > 0.01 ? 1 - (vol12 / vol60) : 0;

  // White streak
  let whiteStreak = 0;
  for (let i = n - 1; i >= 0; i--) {
    if (rounds[i].val <= softThreshold) whiteStreak++;
    else break;
  }

  // Strongest recent multiplier (for rebound detection)
  const maxRecent4 = Math.max(...recent4);
  const maxRecent6 = Math.max(...recent6);
  const medianAll = quantile(sortedCopy(rounds.slice(-200).map(r => r.val)), 50);

  // Recovery signals
  const reboundSpike = maxRecent4 >= softThreshold * CFG.whiteEndingReboundMultiple;
  const ewmaReversalCount = (() => {
    let count = 0;
    for (let i = Math.max(0, recent6.length - CFG.whiteEndingEwmaReversal); i < recent6.length; i++) {
      if (recent6[i] > softThreshold) count++;
    }
    return count;
  })();
  const tailRecovery = recent4.filter(v => v >= medianAll).length >= CFG.whiteEndingTailRecovery;
  const recoveryReady =
    reboundSpike ||
    tailRecovery ||
    ewmaReversalCount >= CFG.whiteEndingEwmaReversal ||
    trend6 > 6 ||
    recent4.filter(v => v > softThreshold).length >= 2;

  // ── Phase classification ──
  let phase = 'NORMAL';

  // WHITE_ACTIVE: currently in a white cluster


  const whiteActive = (
    hardWhiteRate24 >= CFG.whiteActiveHardRate ||
    softWhiteRate24 >= CFG.whiteActiveSoftRate ||
    hardWhiteRate12 >= 0.34 ||
    softWhiteRate12 >= 0.78 ||
    whiteStreak >= CFG.whiteActiveStreak
  );

  // WHITE_ENDING: in cluster but recovery signals firing
  const whiteEnding = whiteActive && recoveryReady;

  // PRE_WHITE: not yet in cluster but signals approaching
  const preWhite = !whiteActive && !recoveryReady && (
    (trend12 < -6 && softWhiteRate12 >= 0.62) ||
    (trend6 < -4 && softWhiteRate8 >= 0.625) ||
    (hardWhiteRate8 >= 0.25 && volCompression >= 0.12 && trend6 <= 0) ||
    (whiteStreak >= 2 && softWhiteRate12 >= 0.55 && trend6 < -1.5) ||
    (
      hardWhiteRate12 >= 0.18 &&
      softWhiteRate24 >= Math.max(CFG.whitePreEntryLowConcentration, 0.62) &&
      volCompression >= (CFG.whitePreEntryVolCompression * 0.6)
    )
  );

  if (whiteEnding) phase = 'WHITE_ENDING';
  else if (whiteActive) phase = 'WHITE_ACTIVE';
  else if (preWhite) phase = 'PRE_WHITE';
  else phase = 'NORMAL';

  const signals = {
    hardWhiteRate24: Number((hardWhiteRate24 * 100).toFixed(1)),
    softWhiteRate24: Number((softWhiteRate24 * 100).toFixed(1)),
    hardWhiteRate12: Number((hardWhiteRate12 * 100).toFixed(1)),
    softWhiteRate12: Number((softWhiteRate12 * 100).toFixed(1)),
    softWhiteRate8: Number((softWhiteRate8 * 100).toFixed(1)),
    hardWhiteRate8: Number((hardWhiteRate8 * 100).toFixed(1)),
    trend12: Number(trend12.toFixed(2)),
    trend6: Number(trend6.toFixed(2)),
    volCompression: Number(volCompression.toFixed(3)),
    whiteStreak,
    softThreshold: Number(softThreshold.toFixed(2)),
    reboundSpike,
    ewmaReversalCount,
    tailRecovery,
    recoveryReady,
    maxRecent4: Number(maxRecent4.toFixed(2)),
  };

  return { phase, signals, reliability: clamp(n / 50, 0, 1) };
}


// ─── Layer 6: Pattern Similarity (DTW-lite) ──────────────────────────────────

function buildPatternLayer(rounds, target, winSize) {
  const n = rounds.length;
  const pLen = CFG.patternLen;
  if (n < pLen + winSize + 30) return { supportPct: 0, lift: 0, bestDistance: null, ready: false, reliability: 0 };

  const transformed = rounds.map(r => Math.log(Math.max(1.0001, r.val)));
  const currentPattern = transformed.slice(n - pLen, n);
  const candidateMaxStart = n - pLen - winSize - 1;
  if (candidateMaxStart < 0) return { supportPct: 0, lift: 0, bestDistance: null, ready: false, reliability: 0 };

  const stride = Math.max(1, Math.ceil((candidateMaxStart + 1) / CFG.patternMaxCandidates));
  const topK = CFG.patternTopK;
  const top = [];
  let sampleSize = 0;
  let randomHits = 0;

  for (let start = 0; start <= candidateMaxStart; start += stride) {
    sampleSize++;

    // Check if hit occurred in window after this pattern
    const lo = start + pLen;
    const hi = lo + winSize - 1;
    let hit = false;
    for (let i = lo; i <= hi && i < n; i++) {
      if (rounds[i].val >= target.minVal) { hit = true; break; }
    }
    if (hit) randomHits++;

    // DTW-lite distance (simplified: allow ±1 warp)
    let distance = 0;
    for (let i = 0; i < pLen; i++) {
      const ci = currentPattern[i];
      const pi = transformed[start + i];
      // Check neighbors for better alignment
      const piPrev = i > 0 ? transformed[start + i - 1] : pi;
      const piNext = i < pLen - 1 ? transformed[start + i + 1] : pi;
      const bestMatch = Math.min(Math.abs(ci - pi), Math.abs(ci - piPrev), Math.abs(ci - piNext));
      distance += bestMatch;
    }
    distance /= pLen;

    // Maintain top-K
    if (top.length < topK) {
      top.push({ distance, hit });
      top.sort((a, b) => a.distance - b.distance);
    } else if (distance < top[top.length - 1].distance) {
      top[top.length - 1] = { distance, hit };
      top.sort((a, b) => a.distance - b.distance);
    }
  }

  const matchHits = top.filter(item => item.hit).length;
  const supportPct = top.length ? (matchHits / top.length) * 100 : 0;
  const randomPct = sampleSize ? (randomHits / sampleSize) * 100 : 0;
  const lift = supportPct - randomPct;

  return {
    supportPct: Number(supportPct.toFixed(1)),
    randomPct: Number(randomPct.toFixed(1)),
    lift: Number(lift.toFixed(1)),
    bestDistance: top.length ? Number(top[0].distance.toFixed(4)) : null,
    ready: top.length >= 8,
    reliability: clamp(sampleSize / 500, 0, 1),
  };
}


// ─── Layer 7: Enhanced Regime Detector ───────────────────────────────────────

function buildRegimeLayer(rounds, target) {
  const n = rounds.length;
  const w = CFG.regimeRecentWindow;
  if (n < w * 2) return { label: 'RANDOM', score: 0, reliability: 0 };

  const recentVals = rounds.slice(-w).map(r => r.val);
  const prevVals = rounds.slice(-w * 2, -w).map(r => r.val);
  if (!recentVals.length || !prevVals.length) return { label: 'RANDOM', score: 0, reliability: 0 };

  // Jensen-Shannon divergence between recent and previous distributions
  const bins = [1, 1.5, 2, 3, 5, 10, 25, 50, 100, 500, Infinity];
  function histogram(vals) {
    const counts = new Array(bins.length - 1).fill(0);
    for (const v of vals) {
      for (let j = 0; j < bins.length - 1; j++) {
        if (v >= bins[j] && v < bins[j + 1]) { counts[j]++; break; }
      }
    }
    const total = counts.reduce((s, x) => s + x, 0);
    return total > 0 ? counts.map(c => c / total) : counts.map(() => 1 / counts.length);
  }

  const pRecent = histogram(recentVals);
  const pPrev = histogram(prevVals);
  const pMid = pRecent.map((x, i) => 0.5 * (x + pPrev[i]));

  function klDiv(p, q) {
    let s = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] <= 0) continue;
      const qSafe = q[i] <= 0 ? 1e-12 : q[i];
      s += p[i] * Math.log(p[i] / qSafe);
    }
    return s;
  }
  const js = 0.5 * klDiv(pRecent, pMid) + 0.5 * klDiv(pPrev, pMid);

  // Hit rate drift
  const recentHitRate = recentVals.filter(v => v >= target.minVal).length / recentVals.length;
  const prevHitRate = prevVals.filter(v => v >= target.minVal).length / prevVals.length;
  const baseline = 1 / target.minVal;
  const drift = Math.abs(recentHitRate - baseline);
  const absoluteHitDrift = recentHitRate - prevHitRate;
  const relativeHitDrift = baseline > 0 ? absoluteHitDrift / baseline : 0;

  // Lag-1 correlation
  const recentHits = recentVals.map(v => v >= target.minVal ? 1 : 0);
  let lagCorr = 0;
  if (recentHits.length > 3) {
    const x = recentHits.slice(0, -1);
    const y = recentHits.slice(1);
    const mx = mean(x), my = mean(y);
    let num = 0, dx2 = 0, dy2 = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      num += dx * dy;
      dx2 += dx * dx;
      dy2 += dy * dy;
    }
    lagCorr = (dx2 > 0 && dy2 > 0) ? num / Math.sqrt(dx2 * dy2) : 0;
  }

  // Low-rate dominance
  const lowRateRecent = recentVals.filter(v => v < 2).length / recentVals.length;
  const lowRatePrev = prevVals.filter(v => v < 2).length / prevVals.length;

  // Classify
  let label = 'RANDOM';
  const strongerLowPressure = lowRateRecent >= 0.54 && lowRateRecent > lowRatePrev + 0.08;
  const trendUpSignal =
    absoluteHitDrift >= Math.max(0.01, baseline * 0.45) ||
    relativeHitDrift >= 0.7;
  const trendDownSignal =
    absoluteHitDrift <= -Math.max(0.01, baseline * 0.45) ||
    relativeHitDrift <= -0.7;

  if (strongerLowPressure) {
    label = 'CLUSTERED_LOW';
  } else if ((js > CFG.regimeJsThreshold * 1.35 || Math.abs(relativeHitDrift) >= 0.7) && trendUpSignal) {
    label = 'TRENDING_UP';
  } else if ((js > CFG.regimeJsThreshold * 1.35 || Math.abs(relativeHitDrift) >= 0.7) && trendDownSignal) {
    label = 'TRENDING_DOWN';
  } else if (js > CFG.regimeJsThreshold && stddev(recentVals) > stddev(prevVals) * 1.3) {
    label = 'VOLATILE';
  } else if (Math.abs(lagCorr) > CFG.regimeLagThreshold) {
    label = 'CLUSTERED';
  } else if (js <= CFG.regimeJsThreshold && drift <= CFG.regimeDriftThreshold) {
    label = 'RANDOM';
  } else {
    label = 'DISPERSED';
  }

  return {
    label,
    js: Number(js.toFixed(6)),
    drift: Number(drift.toFixed(6)),
    lagCorr: Number(lagCorr.toFixed(4)),
    lowRateRecent: Number((lowRateRecent * 100).toFixed(1)),
    hitRateDrift: Number(((recentHitRate - prevHitRate) * 100).toFixed(2)),
    relativeHitDrift: Number((relativeHitDrift * 100).toFixed(2)),
    reliability: clamp(n / (w * 2), 0, 1),
  };
}


// ─── Ensemble Confidence Scorer ──────────────────────────────────────────────

function computeEnsembleConfidence({
  markov,
  km,
  ewma,
  b2b,
  whitePhase,
  pattern,
  regime,
  target,
  roundsSince,
  droughtPct,
  calibrationRows,
}) {
  // Collect layer probabilities and weights
  const layers = [];
  const baselineP = getWindowBaselineProbability(target.minVal, target.window);

  if (markov.prob !== null && markov.reliability > 0.1) {
    layers.push({ name: 'markov', prob: markov.prob, weight: 1.5, reliability: markov.reliability });
  }
  if (km.pHitWindow > 0 && km.reliability > 0.1) {
    layers.push({ name: 'km', prob: km.pHitWindow / 100, weight: 2.0, reliability: km.reliability });






  }
  if (pattern.ready && pattern.reliability > 0.1) {
    const patternProb = clamp((pattern.supportPct + Math.max(0, pattern.lift)) / 200, 0, 1);
    layers.push({ name: 'pattern', prob: patternProb, weight: 1.2, reliability: pattern.reliability });







  }

  // EWMA modifies base probability
  let ewmaModifier = 0;
  if (ewma.reliability > 0.1) {
    if (ewma.crossover === 'bullish') ewmaModifier = 0.05 * ewma.signalStrength;
    else if (ewma.crossover === 'bearish') ewmaModifier = -0.05 * ewma.signalStrength;
  }

  // B2B layer
  if (b2b.b2bScore > 30 && b2b.reliability > 0.3) {
    const b2bProb = clamp(baselineP + (baselineP * (b2b.b2bScore / 100) * 0.45), 0, 1);
    layers.push({ name: 'b2b', prob: b2bProb, weight: 1.3 * (b2b.b2bScore / 100), reliability: b2b.reliability });
  }

  if (layers.length < 1) {
    const kmP = clamp((km?.pHitWindow || 0) / 100, 0, 1);
    const b2bBoost = clamp((b2b?.b2bScore || 0) / 100, 0, 1) * baselineP * 0.6;
    const patternBoost = clamp((pattern?.lift || 0) / 140, -0.04, 0.08);
    const fallbackP = clamp(
      baselineP + (kmP - baselineP) * 0.45 + b2bBoost + patternBoost,
      0,
      1
    );

    let fallbackRaw = clamp(
      (fallbackP * 100 * 0.72) +
      ((km?.pHitWindow || 0) * 0.16) +
      ((b2b?.b2bScore || 0) * 0.14) +
      ((regime?.label === 'TRENDING_UP') ? 5 : 0) -
      ((regime?.label === 'TRENDING_DOWN') ? 6 : 0),
      10,
      68
    );

    if (whitePhase?.phase === 'WHITE_ACTIVE') fallbackRaw = Math.min(fallbackRaw, CFG.whiteActiveConfidenceCap);
    else if (whitePhase?.phase === 'PRE_WHITE') fallbackRaw = Math.min(fallbackRaw, CFG.preWhiteConfidenceCap);
    else if (whitePhase?.phase === 'WHITE_ENDING') fallbackRaw = Math.min(78, fallbackRaw + CFG.whiteEndingBoost * 0.5);

    const fallbackConfidence = Math.round(clamp(fallbackRaw, 0, 100));
    return {
      confidence: fallbackConfidence,
      rawConfidence: Number(fallbackRaw.toFixed(1)),
      ensembleP: Number(fallbackP.toFixed(6)),
      baselineP: Number(baselineP.toFixed(6)),
      edge: Number((fallbackP - baselineP).toFixed(6)),
      ev: Number((fallbackP * target.minVal - 1).toFixed(4)),
      predMethod: 'fallback_baseline',
      layerBreakdown: [],
    };
  }

  // Inverse-error weighted blend


  let wSum = 0, pSum = 0;
  const layerBreakdown = [];
  for (const layer of layers) {
    const error = Math.abs(layer.prob - baselineP);
    const invErr = 1 / Math.max(0.001, error + 0.01);
    const w = layer.weight * layer.reliability * invErr;
    wSum += w;
    pSum += w * layer.prob;
    layerBreakdown.push({
      name: layer.name,
      prob: Number((layer.prob * 100).toFixed(1)),
      weight: Number(w.toFixed(3)),
      reliability: Number(layer.reliability.toFixed(2)),
    });
  }

  let ensembleP = wSum > 0 ? pSum / wSum : baselineP;
  ensembleP = clamp(ensembleP + ewmaModifier, 0, 1);

  // Drought pressure bonus (overdue targets get gentle lift)
  if (droughtPct > 75) {
    const droughtBoost = clamp((droughtPct - 75) / 100 * 0.08, 0, 0.08);
    ensembleP = clamp(ensembleP + droughtBoost, 0, 1);
  }

  // Edge over baseline
  const edge = ensembleP - baselineP;
  const edgeScore = clamp(edge / Math.max(0.03, baselineP), -1.1, 2.2);


  // EV
  const ev = ensembleP * target.minVal - 1;

  // Raw confidence: 0-100 scale







  let rawConfidence = clamp(
    (clamp(ensembleP * 100, 0, 100) * 0.62) +
    (edgeScore * 14) +
    (km.pHitWindow * 0.14) +
    (clamp(pattern.lift, -8, 18) * 0.22) +
    (b2b.b2bScore * 0.12) +
    (ev > 0 ? 5 : 0),
    0,
    100
  );

  // Regime adjustments
  if (regime.label === 'TRENDING_UP') rawConfidence += 6;
  if (regime.label === 'TRENDING_DOWN') rawConfidence -= 7;
  if (regime.label === 'CLUSTERED_LOW') rawConfidence -= 10;
  if (regime.label === 'VOLATILE') rawConfidence -= 4;
  if (regime.label === 'RANDOM') rawConfidence -= 1;

  // White phase hard blocks
  const wp = whitePhase.phase;
  if (wp === 'WHITE_ACTIVE') rawConfidence = Math.min(rawConfidence, CFG.whiteActiveConfidenceCap);
  else if (wp === 'PRE_WHITE') rawConfidence = Math.min(rawConfidence, CFG.preWhiteConfidenceCap);
  else if (wp === 'WHITE_ENDING') rawConfidence += CFG.whiteEndingBoost;

  // B2B floor
  if (b2b.b2bScore >= CFG.b2bScoreThreshold && wp === 'NORMAL') {
    rawConfidence = Math.max(rawConfidence, CFG.b2bConfidenceFloor);
  }

  // Too-early penalty
  if (roundsSince < 2 && target.minVal >= 50) rawConfidence -= 5;


  rawConfidence = clamp(rawConfidence, 0, 100);

  // Calibration against historical accuracy


  let confidence = rawConfidence;
  const resolved = (calibrationRows || []).filter(r => r.outcome === 'win' || r.outcome === 'loss');
  if (resolved.length >= 24) {

    const globalWinRate = resolved.filter(r => r.outcome === 'win').length / resolved.length * 100;
    confidence = Math.round(rawConfidence * 0.86 + globalWinRate * 0.14);
  }
  confidence = clamp(Math.round(confidence), 0, 100);

  // Determine best prediction method
  let predMethod = 'ensemble';
  if (layers.length === 1) predMethod = layers[0].name;
  else {
    const topLayer = layerBreakdown.reduce((best, l) => l.weight > best.weight ? l : best, layerBreakdown[0]);
    predMethod = `ensemble_${topLayer.name}_lead`;
  }

  return {
    confidence,
    rawConfidence: Number(rawConfidence.toFixed(1)),
    ensembleP: Number(ensembleP.toFixed(6)),
    baselineP: Number(baselineP.toFixed(6)),
    edge: Number(edge.toFixed(6)),
    ev: Number(ev.toFixed(4)),
    predMethod,
    layerBreakdown,
  };
}


// ─── Main Forecast Function ──────────────────────────────────────────────────

function computeOracleForecast(rounds, target, options = {}) {
  const cleanRounds = Array.isArray(rounds) && rounds.length && rounds[0]?.id
    ? rounds
    : normalizeRounds(rounds);

  const { label, minVal, color, window: winSize, minHits } = target;
  if (!cleanRounds.length) return null;

  const nowId = cleanRounds[cleanRounds.length - 1].id;
  const hits = cleanRounds.filter(r => r.val >= minVal);

  if (hits.length < minHits + 1) {
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
      avoidReason: 'insufficient_data',
      whitePhase: 'NORMAL',
      b2bScore: 0,
      regimeLabel: 'RANDOM',
      layerBreakdown: [],
      engineVersion: 'oracle_v4',
    };
  }

  // Build gap series
  const allGapsRaw = [];
  for (let i = 1; i < hits.length; i++) {
    allGapsRaw.push(hits[i].id - hits[i - 1].id);
  }

  if (allGapsRaw.length < 10) {
    return {
      ...target,
      noData: true,
      nowId,
      hits: hits.length,
      reason: `Need at least 10 gaps (have ${allGapsRaw.length})`,
      lastHit: hits[hits.length - 1],
      predictedRound: 0, windowLo: 0, windowHi: 0,
      roundsUntilWindowLo: 0, roundsUntilWindowHi: 0,
      inWindow: false, confidence: 0, liveConfidence: 0,
      issuePrediction: false, activePrediction: false,
      issueMode: 'observe', avoidReason: 'insufficient_gaps',
      whitePhase: 'NORMAL', b2bScore: 0, regimeLabel: 'RANDOM',
      layerBreakdown: [], engineVersion: 'oracle_v4',
    };
  }

  const lastHit = hits[hits.length - 1];
  const roundsSince = nowId - lastHit.id;
  const allGapsSorted = sortedCopy(allGapsRaw);
  const recentN = Math.min(allGapsRaw.length, Math.max(target.scanN, 24));
  const recentGaps = allGapsRaw.slice(-recentN);

  // Gap statistics
  const gapStats = {
    min: allGapsSorted[0],
    max: allGapsSorted[allGapsSorted.length - 1],
    med: Math.round(quantile(allGapsSorted, 50)),
    p10: Math.round(quantile(allGapsSorted, 10)),
    p25: Math.round(quantile(allGapsSorted, 25)),
    p75: Math.round(quantile(allGapsSorted, 75)),
    p90: Math.round(quantile(allGapsSorted, 90)),
    p99: Math.round(quantile(allGapsSorted, 99)),
    avg: Math.round(mean(allGapsSorted)),
    iqr: Math.max(1, Math.round(quantile(allGapsSorted, 75) - quantile(allGapsSorted, 25))),
  };

  const droughtPct = Math.round(allGapsSorted.filter(g => g <= roundsSince).length / allGapsSorted.length * 100);
  const isTooEarly = roundsSince < gapStats.p10;
  const isOverdue = roundsSince > gapStats.med;
  const isHardGap = roundsSince > gapStats.p90;

  // ── Run all 7 layers ──

  // Layer 7 first (regime affects other layers)
  const regime = buildRegimeLayer(cleanRounds, target);

  // Layer 1: Markov
  const markov = buildMarkovLayer(cleanRounds, target, winSize);

  // Layer 2: KM
  const km = buildKMLayer(allGapsRaw, recentGaps, roundsSince, winSize, regime);

  // Layer 3: EWMA
  const ewma = buildEWMALayer(cleanRounds, target);

  // Layer 4: B2B
  const b2b = buildB2BLayer(allGapsRaw, cleanRounds, target, roundsSince);

  // Layer 5: White Phase
  const whitePhase = computeWhitePhase(cleanRounds, target);

  // Layer 6: Pattern
  const pattern = buildPatternLayer(cleanRounds, target, winSize);

  // ── Ensemble ──
  const ensemble = computeEnsembleConfidence({
    markov, km, ewma, b2b,
    whitePhase,
    pattern,
    regime,
    target,
    roundsSince,
    droughtPct,
    calibrationRows: options.calibrationRows || [],
  });

  // ── Predict gap and window ──
  let predictedGap;
  let gapMethod = ensemble.predMethod;

  // Use KM survival median as base prediction
  if (km.table && km.table.length > roundsSince + 1) {
    // Find the gap at which survival drops to 50% conditional on current drought
    const sNow = km.table[Math.min(roundsSince, km.table.length - 1)];
    let medianGap = gapStats.med;
    if (sNow > 0) {
      for (let g = roundsSince + 1; g < km.table.length; g++) {
        if (km.table[g] / sNow <= 0.5) { medianGap = g; break; }
      }
    }
    predictedGap = Math.max(roundsSince + 1, medianGap);
  } else {
    predictedGap = Math.max(roundsSince + 1, gapStats.med);
  }

  // B2B override: if strong b2b signal, pull prediction closer
  if (b2b.b2bScore >= 60 && roundsSince <= 3 && minVal <= 50) {
    predictedGap = Math.max(roundsSince + 1, Math.min(predictedGap, roundsSince + Math.max(1, Math.round(gapStats.p25 * 0.5))));
    gapMethod = 'b2b_pull';
  }

  // White cluster override: push prediction further out
  if (whitePhase.phase === 'WHITE_ACTIVE' && predictedGap < gapStats.p75) {
    predictedGap = Math.max(predictedGap, gapStats.p75);
    gapMethod = 'white_delay';
  }

  // White ending override: pull prediction closer
  if (whitePhase.phase === 'WHITE_ENDING' && predictedGap > gapStats.p25) {
    predictedGap = Math.max(roundsSince + 1, Math.round((gapStats.p25 * 0.6 + gapStats.med * 0.4)));
    gapMethod = 'white_recovery';
  }

  predictedGap = Math.max(roundsSince + 1, Math.round(predictedGap));

  // Window positioning
  const halfWin = Math.floor(winSize / 2);
  const windowLoGap = Math.max(1, predictedGap - halfWin);
  const windowHiGap = windowLoGap + winSize - 1;

  let predictedRound = lastHit.id + predictedGap;
  let windowLo = lastHit.id + windowLoGap;
  let windowHi = lastHit.id + windowHiGap;

  // FIXED: Do NOT shift window forward. If window has passed, it stays as-is.
  // The replay loop in api.js will detect the miss and create a fresh lock.
  // Shifting causes the +1 slide bug the user reported.

  const roundsUntilWindowLo = Math.max(0, windowLo - nowId);
  const roundsUntilWindowHi = Math.max(0, windowHi - nowId);
  const inWindow = nowId >= windowLo && nowId <= windowHi;

  // ── Issue decision ──
  const confidence = ensemble.confidence;
  const threshold = getIssueThreshold(minVal);
  const thresholdFloor = minVal <= 15 ? 22 : minVal <= 50 ? 24 : minVal <= 200 ? 26 : 28;
  const patternStrong = pattern.ready && pattern.lift >= 4;
  const kmStrong = km.pHitWindow >= Math.max(10, Math.round(threshold * 0.9));
  const b2bStrong = b2b.b2bScore >= 50;
  let thresholdEase = 0;
  if (b2bStrong) thresholdEase += 6;
  if (patternStrong) thresholdEase += 4;
  if (regime.label === 'TRENDING_UP') thresholdEase += 4;
  if (regime.label === 'DISPERSED') thresholdEase += 2;
  if (kmStrong) thresholdEase += 3;
  if (roundsSince >= gapStats.p25) thresholdEase += 2;
  if (roundsSince >= gapStats.med) thresholdEase += 3;
  if (minVal >= 100 && whitePhase.phase !== 'WHITE_ACTIVE') thresholdEase += 2;
  const effectiveThreshold = clamp(threshold - thresholdEase, thresholdFloor, threshold);

  // Hard blocks — WHITE_ACTIVE blocks only, but WHITE_ENDING ALWAYS passes through
  const whiteBlock = whitePhase.phase === 'WHITE_ACTIVE';
  const preWhiteBlock =
    whitePhase.phase === 'PRE_WHITE' &&
    confidence < effectiveThreshold - 4 &&
    !b2bStrong &&
    !patternStrong;
  const downtrendBlock =
    regime.label === 'TRENDING_DOWN' &&
    confidence < effectiveThreshold - 4 &&
    !b2bStrong;
  const randomWeakStructure =
    !b2b.immediateB2B &&
    !b2bStrong &&
    !patternStrong &&
    km.pHitWindow < Math.max(6, Math.round(effectiveThreshold * 0.55));
  const randomBlock =
    regime.label === 'RANDOM' &&
    confidence < effectiveThreshold - 8 &&
    randomWeakStructure &&
    !inWindow;
  const hardBlock = whiteBlock || preWhiteBlock || downtrendBlock || randomBlock;

  // Strong transition signals that override blocks
  const strongB2BThreshold = minVal <= 15 ? 50 : minVal <= 50 ? 54 : 58;
  const strongB2B = b2b.b2bScore >= strongB2BThreshold;
  const strongWhiteRecovery = whitePhase.phase === 'WHITE_ENDING';
  const strongOverdue = droughtPct >= 85;
  const strongPattern = pattern.ready && pattern.lift >= 6;
  const strongTrendUp = regime.label === 'TRENDING_UP' && confidence >= effectiveThreshold - 8;
  const strongTransition = strongB2B || strongWhiteRecovery || strongOverdue || strongPattern || strongTrendUp;

  const issuePrediction =
    (!hardBlock && confidence >= effectiveThreshold) ||
    (strongTransition && confidence >= effectiveThreshold - 10 && !whiteBlock);








  let avoidReason = null;
  if (!issuePrediction) {
    if (whiteBlock) avoidReason = 'white_cluster';
    else if (preWhiteBlock) avoidReason = 'pre_white_cluster';
    else if (downtrendBlock) avoidReason = 'downtrend';
    else if (randomBlock) avoidReason = 'random_like';
    else if (confidence >= effectiveThreshold - 4) avoidReason = 'near_threshold';
    else if (isTooEarly) avoidReason = 'too_early';
    else avoidReason = 'weak_probability';
  }

  let issueMode = 'observe';
  if (issuePrediction) {
    if (strongB2B && roundsSince <= 3) issueMode = 'b2b_support';
    else if (strongWhiteRecovery) issueMode = 'white_rebound';
    else if (pattern.ready && pattern.lift >= 4) issueMode = 'pattern_support';
    else if (regime.label === 'TRENDING_UP') issueMode = 'trend_support';
    else issueMode = 'strict';
  } else {
    if (whiteBlock || preWhiteBlock) issueMode = 'cluster_guard';
    else if (downtrendBlock) issueMode = 'trend_guard';
    else if (confidence >= effectiveThreshold - 4 || kmStrong || patternStrong || b2bStrong) issueMode = 'watch';
    else if (isTooEarly) issueMode = 'prep';
  }

  // Chase signal
  const chaseSignal = issuePrediction ? 'CHASE' : 'SKIP';
  const chaseColor = issuePrediction ? '#39ff8a' : '#ff5555';

  return {
    ...target,
    noData: false,
    nowId,
    hits: hits.length,
    n: allGapsSorted.length,
    lastHit,
    roundsSince,
    allGapsRaw,
    allGapsSorted,

    // Gap stats
    med: gapStats.med,
    p10: gapStats.p10,
    p25: gapStats.p25,
    p75: gapStats.p75,
    p90: gapStats.p90,
    p99: gapStats.p99,
    minGap: gapStats.min,
    maxGap: gapStats.max,
    avgGap: gapStats.avg,
    iqr: gapStats.iqr,

    // Prediction
    predMethod: gapMethod,
    predBasis: `${gapMethod} (${regime.label}, ${allGapsSorted.length} gaps)`,
    predictedGap,
    predictedRound,
    windowLo,
    windowHi,
    windowSize: winSize,

    // Status flags
    isTooEarly,
    isOverdue,
    isHardGap,
    droughtPct,

    // Confluence
    confidence,
    threshold,
    effectiveThreshold,
    rawConfidence: ensemble.rawConfidence,
    ensembleP: ensemble.ensembleP,
    baselineP: ensemble.baselineP,
    ensembleEdge: ensemble.edge,
    ensembleEV: ensemble.ev,

    // Window
    roundsUntilWindowLo,
    roundsUntilWindowHi,
    inWindow,

    // Issue
    issuePrediction,
    activePrediction: false,
    issueMode,
    avoidReason,
    chaseSignal,
    chaseColor,

    // KM probabilities
    pHit1: Number((km.pHit1 || 0).toFixed(1)),
    pHit5: Number((km.pHit5 || 0).toFixed(1)),
    pHitWindow: Number((km.pHitWindow || 0).toFixed(1)),
    kmReliable: km.reliability > 0.3,

    // Layer signals (new — exposed to frontend)
    whitePhase: whitePhase.phase,
    whiteSignals: whitePhase.signals,
    b2bScore: b2b.b2bScore,
    b2bDetails: {
      gapAcceleration: b2b.gapAcceleration,
      hitRateSurge: b2b.hitRateSurge,
      immediateB2B: b2b.immediateB2B,
    },
    regimeLabel: regime.label,
    regimeDetails: {
      js: regime.js,
      drift: regime.drift,
      lagCorr: regime.lagCorr,
      lowRateRecent: regime.lowRateRecent,
      hitRateDrift: regime.hitRateDrift,
    },
    ewmaSignal: {
      crossover: ewma.crossover,
      signalStrength: ewma.signalStrength,
      shortEwma: ewma.shortEwma,
      longEwma: ewma.longEwma,
    },
    markovProb: markov.prob !== null ? Number((markov.prob * 100).toFixed(1)) : null,
    patternSupport: {
      supportPct: pattern.supportPct,
      lift: pattern.lift,
      bestDistance: pattern.bestDistance,
      ready: pattern.ready,
    },
    layerBreakdown: ensemble.layerBreakdown,

    // Engine version
    engineVersion: 'oracle_v4',
  };
}


// ─── Lock Builder ────────────────────────────────────────────────────────────

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
    clusterCenter: forecast.med,
    droughtAtSnap: forecast.droughtPct,
    signal: forecast.chaseSignal,
    issueMode: forecast.issueMode || null,
    regimeMode: forecast.regimeLabel || null,
    issuePrediction: Boolean(forecast.issuePrediction),
    avoidReason: forecast.avoidReason || null,
    generation: 1,
  };
}


// ─── Exports ─────────────────────────────────────────────────────────────────

module.exports = {
  ORACLE_TARGETS,
  normalizeRounds,
  computeOracleForecast,
  makeOracleLock,
};
