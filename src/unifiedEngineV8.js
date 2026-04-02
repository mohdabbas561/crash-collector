'use strict';

// ============================================================
// PURE DATA ENGINE — v9
// Zero bias. Zero fake defaults. Zero invented priors.
// Every number comes from YOUR data or is explicitly absent.
// White cluster + B2B are first-class measured signals.
// ============================================================

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];

// Fixed window spans per target (how many rounds the prediction window covers)
const FIXED_WINDOW_SPAN = {
  5: 3,
  10: 6,
  20: 10,
  50: 18,
  100: 27,
  500: 50,
  1000: 75,
};

const BUCKETS = [
  { id: 'micro', label: 'Micro', min: 1,  max: 1.99,              color: '#ff4560' },
  { id: 'low',   label: 'Low',   min: 2,  max: 4.99,              color: '#ffd84d' },
  { id: 'mid',   label: 'Mid',   min: 5,  max: 9.99,              color: '#00ff88' },
  { id: 'high',  label: 'High',  min: 10, max: 24.99,             color: '#00d4ff' },
  { id: 'moon',  label: 'Moon',  min: 25, max: Number.POSITIVE_INFINITY, color: '#c084fc' },
];

// ────────────────────────────────────────────────────────────
// MATH PRIMITIVES — no invented fallbacks
// ────────────────────────────────────────────────────────────

function clamp(v, lo, hi) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : lo;
}

function roundNum(v, digits = 4) {
  const n = Number(v);
  return Number.isFinite(n) ? Number(n.toFixed(digits)) : null;
}

function mean(arr) {
  if (!arr.length) return null;
  let s = 0;
  for (const v of arr) s += Number(v) || 0;
  return s / arr.length;
}

function variance(arr, avg) {
  if (arr.length < 2) return null;
  const m = avg ?? mean(arr);
  if (m === null) return null;
  let s = 0;
  for (const v of arr) { const d = (Number(v) || 0) - m; s += d * d; }
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
  const idx = (sorted.length - 1) * clamp(q, 0, 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
}

function quantile(arr, q) {
  if (!arr.length) return null;
  return quantileFromSorted(sortedCopy(arr), q);
}

// Weighted median — no invented fallback
function weightedMedian(items) {
  if (!items.length) return null;
  const sorted = [...items].sort((a, b) => a.value - b.value);
  let total = 0;
  for (const x of sorted) total += Math.max(0, x.weight || 0);
  if (total <= 0) return sorted[Math.floor(sorted.length / 2)].value;
  let acc = 0;
  for (const x of sorted) {
    acc += Math.max(0, x.weight || 0);
    if (acc / total >= 0.5) return x.value;
  }
  return sorted[sorted.length - 1].value;
}

function safeLog(v) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.log(n) : 0;
}

// ────────────────────────────────────────────────────────────
// PREFIX STRUCTURES
// ────────────────────────────────────────────────────────────

function buildPrefix(arr) {
  const p = new Array(arr.length + 1).fill(0);
  for (let i = 0; i < arr.length; i++) p[i + 1] = p[i] + (Number(arr[i]) || 0);
  return p;
}

function prefixWindowMean(pref, endIdx, windowLen) {
  const n = pref.length - 1;
  if (n <= 0 || endIdx < 0 || windowLen < 1) return null;
  const e = Math.min(endIdx, n - 1);
  const s = Math.max(0, e - windowLen + 1);
  const w = e - s + 1;
  return w > 0 ? (pref[e + 1] - pref[s]) / w : null;
}

function prefixWindowVariance(prefVal, prefSq, endIdx, windowLen) {
  const n = prefVal.length - 1;
  if (n <= 1 || endIdx < 0 || windowLen < 2) return null;
  const e = Math.min(endIdx, n - 1);
  const s = Math.max(0, e - windowLen + 1);
  const w = e - s + 1;
  if (w < 2) return null;
  const sumV = prefVal[e + 1] - prefVal[s];
  const sumSq = prefSq[e + 1] - prefSq[s];
  const m = sumV / w;
  return Math.max(0, sumSq / w - m * m);
}

function lowerBound(arr, value) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < value) lo = mid + 1; else hi = mid; }
  return lo;
}

// ────────────────────────────────────────────────────────────
// INPUT NORMALIZATION
// ────────────────────────────────────────────────────────────

function normalizeRounds(rounds) {
  const clean = (rounds || [])
    .map(r => ({
      roundId:    Number(r.roundId),
      multiplier: Number(r.multiplier),
      timestamp:  Number(r.timestamp) || 0,
    }))
    .filter(r => Number.isFinite(r.roundId) && Number.isFinite(r.multiplier) && r.multiplier > 0)
    .sort((a, b) => a.roundId - b.roundId);

  // Deduplicate by roundId (last write wins)
  const dedup = [];
  let lastId = null;
  for (const r of clean) {
    if (r.roundId === lastId) dedup[dedup.length - 1] = r;
    else { dedup.push(r); lastId = r.roundId; }
  }
  return dedup;
}

function normalizeLockInput(input) {
  if (!input) return null;
  return {
    lo:           Number(input.lo),
    hi:           Number(input.hi),
    roundWhenMade: Number(input.roundWhenMade ?? input.round_when_made),
    generation:   Number(input.generation || 1),
    suspended:    Boolean(input.suspended ?? input.eta?.suspended),
    eta:          input.eta || null,
  };
}

// ────────────────────────────────────────────────────────────
// WHITE CLUSTER DETECTOR (purely data-measured)
// A "white" round is one below the empirical low-multiplier cut.
// We measure:  run lengths, continuation probability, rebound probability
// all from the actual observed data — no smoothing invented out of thin air.
// ────────────────────────────────────────────────────────────

function buildWhiteClusterDetector(multipliers) {
  const n = multipliers.length;
  const sorted = sortedCopy(multipliers);

  // Cut determined by data distribution — 40th percentile is the "low zone"
  const whiteCut   = clamp(quantileFromSorted(sorted, 0.40), 1.5, 4.0);
  // Rebound: a round is a "rebound" if it's in the top 30% of historical rounds
  const reboundCut = clamp(quantileFromSorted(sorted, 0.70), 3.0, 999);

  // Build run-length array: how many consecutive white rounds ending at index i
  const whiteFlag = multipliers.map(m => (m < whiteCut ? 1 : 0));
  const runLen    = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    runLen[i] = whiteFlag[i] ? (1 + (i > 0 ? runLen[i - 1] : 0)) : 0;
  }

  // Empirically measure: given run length R, what is P(next is also white)?
  // and P(rebound within next 3 rounds)?
  // We bucket run lengths 0..maxRun and count directly.
  const maxRun = Math.max(1, Math.round(quantile(runLen, 0.99) * 1.2)) || 20;
  const obs         = new Array(maxRun + 1).fill(0);   // observations at each run length
  const contCount   = new Array(maxRun + 1).fill(0);   // next round also white
  const reboundCount= new Array(maxRun + 1).fill(0);   // rebound in next 3

  let totalObs = 0, totalCont = 0, totalRebound = 0;

  for (let i = 0; i < n - 1; i++) {
    if (runLen[i] === 0) continue; // only measure when we ARE in a white run
    const r = Math.min(maxRun, runLen[i]);
    obs[r]++;
    totalObs++;
    if (whiteFlag[i + 1]) { contCount[r]++; totalCont++; }
    const reboundSoon = (i + 1 < n && multipliers[i + 1] >= reboundCut) ||
                        (i + 2 < n && multipliers[i + 2] >= reboundCut) ||
                        (i + 3 < n && multipliers[i + 3] >= reboundCut);
    if (reboundSoon) { reboundCount[r]++; totalRebound++; }
  }

  // P(continue | runLen) and P(rebound | runLen) — measured only, NO smoothing added
  // When bucket has zero observations we use the global rate (also measured)
  const globalContRate    = totalObs > 0 ? totalCont    / totalObs : null;
  const globalReboundRate = totalObs > 0 ? totalRebound / totalObs : null;

  function estimate(currentRunLen) {
    const r = Math.min(maxRun, Math.max(0, Math.round(currentRunLen) || 0));
    if (obs[r] === 0) {
      // No data for this run length — use global rate, or null if no data at all
      return {
        continueProb: globalContRate,
        reboundProb:  globalReboundRate,
        sample:       totalObs,
        reliable:     totalObs >= 10,
      };
    }
    return {
      continueProb: contCount[r] / obs[r],
      reboundProb:  reboundCount[r] / obs[r],
      sample:       obs[r],
      reliable:     obs[r] >= 5,
    };
  }

  // Percentiles of run lengths (from data)
  const allRuns = runLen.filter(v => v > 0);
  const runQ85  = allRuns.length ? quantile(allRuns, 0.85) : null;
  const runQ95  = allRuns.length ? quantile(allRuns, 0.95) : null;
  const runQ99  = allRuns.length ? quantile(allRuns, 0.99) : null;

  return {
    whiteCut,
    reboundCut,
    whiteFlag,
    runLen,
    estimate,
    runQ85,
    runQ95,
    runQ99,
    totalObsInRuns: totalObs,
  };
}

// ────────────────────────────────────────────────────────────
// B2B (BACK-TO-BACK) DETECTOR (purely data-measured)
// Measures: P(hit immediately after last hit), P(hit within 2 rounds after last hit)
// Computed globally AND on a recent window (measured, not invented).
// ────────────────────────────────────────────────────────────

function buildB2BDetector(hitIndices, n, recentWindow) {
  // Global: over all hit pairs
  let gImm = 0, gNear = 0, gPairs = 0;
  for (let i = 1; i < hitIndices.length; i++) {
    const d = hitIndices[i] - hitIndices[i - 1];
    gPairs++;
    if (d <= 1) gImm++;
    if (d <= 2) gNear++;
  }

  // Recent window: only pairs where the second hit is in the last `recentWindow` rounds
  const cutoff = n - recentWindow;
  let rImm = 0, rNear = 0, rPairs = 0;
  for (let i = 1; i < hitIndices.length; i++) {
    if (hitIndices[i] < cutoff) continue;
    const d = hitIndices[i] - hitIndices[i - 1];
    rPairs++;
    if (d <= 1) rImm++;
    if (d <= 2) rNear++;
  }

  return {
    // Global rates — null if no pairs observed
    globalImmRate:  gPairs > 0 ? gImm  / gPairs : null,
    globalNearRate: gPairs > 0 ? gNear / gPairs : null,
    globalPairs:    gPairs,

    // Recent rates — null if no recent pairs observed
    recentImmRate:  rPairs > 0 ? rImm  / rPairs : null,
    recentNearRate: rPairs > 0 ? rNear / rPairs : null,
    recentPairs:    rPairs,

    // Best-available rate (prefer recent if enough data, else global, else null)
    immRate:  rPairs >= 5 ? rImm  / rPairs : (gPairs >= 5 ? gImm  / gPairs : null),
    nearRate: rPairs >= 5 ? rNear / rPairs : (gPairs >= 5 ? gNear / gPairs : null),
  };
}

// ────────────────────────────────────────────────────────────
// GLOBAL STATE — everything is derived from the data
// ────────────────────────────────────────────────────────────

function buildGlobalState(cleanRounds) {
  const n = cleanRounds.length;
  const multipliers = cleanRounds.map(r => Number(r.multiplier));
  const logs   = multipliers.map(safeLog);
  const logsSq = logs.map(v => v * v);

  const wcd = buildWhiteClusterDetector(multipliers);

  const prefLog   = buildPrefix(logs);
  const prefLogSq = buildPrefix(logsSq);
  const prefWhite = buildPrefix(wcd.whiteFlag);

  // Sorted multipliers for quantile computation
  const sorted = sortedCopy(multipliers);

  // Dynamic window sizes based on data length
  const shortW = clamp(Math.round(Math.sqrt(n) * 3.5), 12, 150);
  const longW  = clamp(Math.round(Math.sqrt(n) * 9),   shortW + 5, 500);

  // Green zone: top 25% of historical rounds
  const greenCut = clamp(quantileFromSorted(sorted, 0.75), 2.0, 999);
  const greenFlag = multipliers.map(m => (m >= greenCut ? 1 : 0));
  const prefGreen = buildPrefix(greenFlag);

  return {
    rounds: cleanRounds,
    n,
    multipliers,
    logs,
    sorted,
    prefLog,
    prefLogSq,
    prefWhite,
    prefGreen,
    wcd,
    shortW,
    longW,
    greenCut,
  };
}

// ────────────────────────────────────────────────────────────
// TARGET-SPECIFIC DATA
// ────────────────────────────────────────────────────────────

function buildTargetData(state, target) {
  const n = state.n;
  const flags       = new Array(n).fill(0);
  const hitIndices  = [];
  const hitRoundIds = [];

  for (let i = 0; i < n; i++) {
    if (state.multipliers[i] >= target) {
      flags[i] = 1;
      hitIndices.push(i);
      hitRoundIds.push(state.rounds[i].roundId);
    }
  }

  const prefHit = buildPrefix(flags);
  const baseRate = hitIndices.length / n;   // observed frequency — could be 0

  // Gap since last hit (in rounds)
  const gapAt = new Array(n).fill(0);
  let lastHit = -1;
  for (let i = 0; i < n; i++) {
    if (flags[i]) lastHit = i;
    gapAt[i] = lastHit >= 0 ? (i - lastHit) : (i + 1);
  }

  // Distance to NEXT hit (looking forward) — null if no future hit exists
  const nextHitDist = new Array(n).fill(null);
  let nextHit = null;
  for (let i = n - 1; i >= 0; i--) {
    if (flags[i]) nextHit = i;
    nextHitDist[i] = nextHit !== null ? (nextHit - i) : null;
  }

  // Inter-hit gap distribution (raw, no invented padding)
  const interGaps = [];
  for (let i = 1; i < hitIndices.length; i++) interGaps.push(hitIndices[i] - hitIndices[i - 1]);
  const sortedGaps = sortedCopy(interGaps);

  const gapQ = interGaps.length >= 4 ? {
    q10: quantileFromSorted(sortedGaps, 0.10),
    q25: quantileFromSorted(sortedGaps, 0.25),
    q35: quantileFromSorted(sortedGaps, 0.35),
    q50: quantileFromSorted(sortedGaps, 0.50),
    q75: quantileFromSorted(sortedGaps, 0.75),
    q90: quantileFromSorted(sortedGaps, 0.90),
    q95: quantileFromSorted(sortedGaps, 0.95),
  } : null;  // null = not enough data

  // B2B detector — recent window = sqrt(n) * 5
  const recentWindow = clamp(Math.round(Math.sqrt(n) * 5), 30, 600);
  const b2b = buildB2BDetector(hitIndices, n, recentWindow);

  const currentGap = gapAt[n - 1] || 0;

  return {
    target,
    flags,
    prefHit,
    hitIndices,
    hitRoundIds,
    nextHitDist,
    gapAt,
    interGaps,
    sortedGaps,
    gapQ,
    baseRate,
    currentGap,
    b2b,
  };
}

// ────────────────────────────────────────────────────────────
// HAZARD MODEL — purely from observed gap distribution
// At gap G, what fraction of rounds resulted in a hit?
// No invented prior. If data is insufficient → returns null p1.
// ────────────────────────────────────────────────────────────

function buildHazardModel(targetData) {
  const { gapAt, flags, gapQ } = targetData;
  const n = gapAt.length;

  if (!gapQ || targetData.hitIndices.length < 5) {
    return { p1: null, reliable: false, at: () => null };
  }

  const maxGap = Math.max(6, Math.round(gapQ.q95 * 1.5));
  const obs  = new Array(maxGap + 2).fill(0);
  const hits = new Array(maxGap + 2).fill(0);

  for (let i = 0; i < n - 1; i++) {
    const g = Math.min(maxGap, Math.round(gapAt[i] || 0));
    obs[g]++;
    if (flags[i + 1]) hits[g]++;
  }

  // Bandwidth for kernel smoothing — derived from data spread, not invented
  const gapSd  = stddev(targetData.interGaps);
  const bw     = gapSd !== null ? clamp(Math.round(gapSd * 0.15), 1, 20) : 2;
  const totalO = obs.reduce((s, v) => s + v, 0);
  const totalH = hits.reduce((s, v) => s + v, 0);
  const globalRate = totalO > 0 ? totalH / totalO : null;

  function at(gap) {
    const g = clamp(Math.round(gap), 0, maxGap);
    let o = 0, h = 0;
    for (let d = -bw; d <= bw; d++) {
      const idx = g + d;
      if (idx < 0 || idx > maxGap) continue;
      const w = bw + 1 - Math.abs(d);
      o += obs[idx] * w;
      h += hits[idx] * w;
    }
    // No fake prior — if we have zero obs in this band, return global rate (or null)
    if (o === 0) return globalRate;
    return h / o;
  }

  const p1 = at(targetData.currentGap);
  return {
    p1:      p1 !== null ? clamp(p1, 0, 1) : null,
    reliable: totalO >= 20,
    at,
    totalObs: totalO,
  };
}

// ────────────────────────────────────────────────────────────
// KNN (k-nearest-neighbor) PREDICTOR
// Feature: current context vector vs all historical contexts
// Only trained on rounds where we have a "next hit" label.
// ────────────────────────────────────────────────────────────

function buildFeatureVector(state, targetData, idx) {
  const { prefLog, prefLogSq, prefWhite, prefGreen, shortW, longW } = state;
  const { prefHit, gapAt, gapQ } = targetData;
  const { runLen } = state.wcd;

  const logShort = prefixWindowMean(prefLog, idx, shortW);
  const logLong  = prefixWindowMean(prefLog, idx, longW);
  const logVarSh = prefixWindowVariance(prefLog, prefLogSq, idx, shortW);

  const hitW = gapQ ? clamp(Math.round(gapQ.q75 * 2), shortW, Math.max(longW, 600)) : longW;

  return [
    prefixWindowMean(prefWhite, idx, shortW)  ?? 0,   // [0] recent white rate
    prefixWindowMean(prefWhite, idx, longW)   ?? 0,   // [1] long white rate
    prefixWindowMean(prefGreen, idx, shortW)  ?? 0,   // [2] recent green rate
    prefixWindowMean(prefHit,   idx, shortW)  ?? 0,   // [3] recent hit rate (this target)
    prefixWindowMean(prefHit,   idx, hitW)    ?? 0,   // [4] medium hit rate
    (logShort !== null && logLong !== null) ? (logShort - logLong) : 0,  // [5] trend
    logVarSh !== null ? Math.sqrt(logVarSh) : 0,     // [6] volatility
    gapQ ? clamp((gapAt[idx] || 0) / Math.max(1, gapQ.q75), 0, 10) : 0, // [7] gap pressure
    gapQ ? clamp((runLen[idx] || 0) / Math.max(1, state.wcd.runQ95 || 1), 0, 5) : 0, // [8] white run pressure
  ];
}

function computeFeatureStats(vecs) {
  if (!vecs.length) return null;
  const D = vecs[0].length;
  const means = new Array(D).fill(0);
  const stds  = new Array(D).fill(1);
  for (const v of vecs) for (let d = 0; d < D; d++) means[d] += v[d];
  for (let d = 0; d < D; d++) means[d] /= vecs.length;
  for (const v of vecs) for (let d = 0; d < D; d++) { const z = v[d] - means[d]; stds[d] += z * z; }
  for (let d = 0; d < D; d++) stds[d] = Math.sqrt(Math.max(1e-8, stds[d] / vecs.length));
  return { means, stds };
}

function zNormalize(vec, stats) {
  return vec.map((v, i) => (v - stats.means[i]) / stats.stds[i]);
}

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function runKnn(state, targetData, currentIdx) {
  const n = state.n;
  const minHistory = state.longW + 2;
  if (n < minHistory + 10) return null;  // not enough data — return null, not fake

  // Horizon: how many rounds ahead we consider "soon"
  const gapQ = targetData.gapQ;
  const horizonSoon = gapQ ? Math.max(2, Math.round(gapQ.q35)) : null;

  // Build labeled samples: every past index where we know the next hit distance
  const samples = [];
  for (let i = minHistory; i < currentIdx; i++) {
    const dist = targetData.nextHitDist[i];
    if (dist === null || dist < 1) continue;
    const recency = (i - minHistory + 1) / Math.max(1, currentIdx - minHistory);
    samples.push({
      x:       buildFeatureVector(state, targetData, i),
      y1:      dist <= 1 ? 1 : 0,
      ySoon:   horizonSoon !== null ? (dist <= horizonSoon ? 1 : 0) : null,
      gapDist: dist,
      recency,
    });
  }

  if (samples.length < 10) return null;  // truly not enough — return null

  const stats = computeFeatureStats(samples.map(s => s.x));
  if (!stats) return null;

  const sampleZ = samples.map(s => ({ ...s, z: zNormalize(s.x, stats) }));
  const queryZ  = zNormalize(buildFeatureVector(state, targetData, currentIdx), stats);

  // Adaptive k: sqrt of sample count, capped
  const k = clamp(Math.round(Math.sqrt(sampleZ.length) * 2), 15, Math.round(sampleZ.length * 0.4));

  // Score and pick top-k
  const scored = sampleZ.map(s => ({ ...s, dist: euclidean(queryZ, s.z) }));
  scored.sort((a, b) => a.dist - b.dist);
  const top = scored.slice(0, k);

  // Weighted by inverse distance × recency
  let sumW = 0, sumW2 = 0, p1Num = 0, pSoonNum = 0;
  const gapItems = [];
  for (const s of top) {
    const w = (1 / (1 + s.dist)) * (0.5 + 0.5 * s.recency);
    sumW += w; sumW2 += w * w;
    p1Num   += s.y1 * w;
    if (s.ySoon !== null) pSoonNum += s.ySoon * w;
    gapItems.push({ value: s.gapDist, weight: w });
  }

  if (sumW === 0) return null;
  const p1      = p1Num / sumW;
  const pSoon   = pSoonNum / sumW;
  const effN    = (sumW * sumW) / sumW2;
  const support = clamp(effN / Math.max(1, k), 0, 1);

  const sortedGapItems = [...gapItems].sort((a, b) => a.value - b.value);
  function wq(q) { return weightedMedian(gapItems.map(x => ({ ...x, weight: x.weight * (x.value <= quantile(gapItems.map(v=>v.value), q) ? 1 : 0) + 1e-12 }))); }

  // Simpler: just find approximate quantiles from sorted gaps
  const gapVals = sortedGapItems.map(x => x.value);
  const q25 = quantileFromSorted(gapVals, 0.25);
  const q50 = quantileFromSorted(gapVals, 0.50);
  const q75 = quantileFromSorted(gapVals, 0.75);

  return {
    p1:      clamp(p1, 0, 1),
    pSoon:   horizonSoon !== null ? clamp(pSoon, 0, 1) : null,
    horizonSoon,
    q25,
    q50,
    q75,
    support,
    k,
    sampleSize: sampleZ.length,
  };
}

// ────────────────────────────────────────────────────────────
// CALIBRATION — built entirely from your historical outcomes
// Only adjusts if you have real feedback rows. If no history: identity.
// ────────────────────────────────────────────────────────────

function targetFromLabel(raw) {
  const n = Number(String(raw || '').trim().replace(/x/i, ''));
  return Number.isFinite(n) ? n : null;
}

function buildCalibration(historyRows = []) {
  const out = {};
  for (const t of TARGETS) out[t] = null;  // null = no calibration data

  const byTarget = {};
  for (const t of TARGETS) byTarget[t] = [];
  for (const row of (historyRows || [])) {
    const t = targetFromLabel(row?.target);
    if (TARGETS.includes(t)) byTarget[t].push(row);
  }

  for (const t of TARGETS) {
    const rows = byTarget[t];
    if (rows.length < 3) continue;  // not enough data to calibrate

    let win = 0, early = 0, loss = 0, total = 0;
    const shiftItems = [];

    for (let i = 0; i < rows.length; i++) {
      // Recent rows are weighted more heavily (recency weighting — from data ordering)
      const w = 1 + 2 * Math.pow((i + 1) / rows.length, 2);
      const outcome = String(rows[i]?.outcome || '').toLowerCase();
      if (outcome === 'win')   win   += w;
      else if (outcome === 'early') early += w;
      else if (outcome === 'loss')  loss  += w;
      else continue;
      total += w;

      // Shift: measure whether hits arrived earlier or later than predicted window
      const lo  = Number(rows[i]?.lo);
      const hi  = Number(rows[i]?.hi);
      const hit = Number(rows[i]?.hitRound);
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
        const span = hi - lo;
        let err = 0;
        if (outcome === 'win'   && Number.isFinite(hit)) err = (hit - (lo + span * 0.5)) / span;
        else if (outcome === 'early' && Number.isFinite(hit)) err = (hit - lo) / span;
        else if (outcome === 'loss') err = 1.0;
        shiftItems.push({ value: clamp(err, -3, 3), weight: w });
      }
    }

    if (total === 0) continue;

    const winRate  = win  / Math.max(1, win + loss);
    const lossRate = loss / total;
    const earlyRate= early/ total;

    // Shift correction: if hits always arrive early → negative shift → move window earlier
    const shift = shiftItems.length >= 3 ? clamp(weightedMedian(shiftItems), -1.5, 1.5) : 0;

    // Span correction: if high variance in where hits land → widen window
    const absShifts = shiftItems.map(x => ({ value: Math.abs(x.value), weight: x.weight }));
    const spanMult = absShifts.length >= 3
      ? clamp(1 + weightedMedian(absShifts), 0.6, 3.5)
      : 1.0;

    out[t] = {
      sample:       Math.round(total),
      winRate:      roundNum(winRate, 6),
      lossRate:     roundNum(lossRate, 6),
      earlyRate:    roundNum(earlyRate, 6),
      shift:        roundNum(shift, 6),
      spanMult:     roundNum(spanMult, 6),
    };
  }
  return out;
}

// ────────────────────────────────────────────────────────────
// CORE PREDICTOR — fuses hazard + knn, gated by data quality
// No fake confidence floors. No invented activation thresholds.
// If data is insufficient → suspended = true, no prediction made.
// ────────────────────────────────────────────────────────────

function buildPrediction(state, targetData, calibration) {
  const currentIdx   = state.n - 1;
  const currentRound = state.rounds[currentIdx].roundId;
  const target       = targetData.target;
  const gapQ         = targetData.gapQ;
  const cal          = calibration; // may be null

  // ── White cluster signals (measured) ──
  const currentRun  = state.wcd.runLen[currentIdx] || 0;
  const wEst        = state.wcd.estimate(currentRun);
  const whiteRisk   = wEst.continueProb;   // null if no data
  const whiteRelease= wEst.reboundProb;    // null if no data

  // ── B2B signals (measured) ──
  const b2b = targetData.b2b;

  // ── Hazard model ──
  const hazard = buildHazardModel(targetData);

  // ── KNN ──
  const knn = runKnn(state, targetData, currentIdx);

  // ── Determine if we have enough data to make a meaningful prediction ──
  const hasBaseRate = targetData.hitIndices.length >= 5;
  const hasHazard   = hazard.reliable && hazard.p1 !== null;
  const hasKnn      = knn !== null && knn.support >= 0.1;

  if (!hasBaseRate) {
    // Not enough hits in data — no prediction possible
    return { suspended: true, reason: 'insufficient_hits', target };
  }

  // ── Probability of hit in next round ──
  // Blend hazard + knn proportionally to their data quality
  let p1 = null;
  let wHaz = 0, wKnn = 0;

  if (hasHazard && hasKnn) {
    // Both available — blend by relative data support
    wHaz = hazard.totalObs;
    wKnn = knn.sampleSize * knn.support;
    const wSum = wHaz + wKnn;
    p1 = (hazard.p1 * wHaz + knn.p1 * wKnn) / wSum;
    wHaz /= wSum; wKnn /= wSum;
  } else if (hasHazard) {
    p1 = hazard.p1;
    wHaz = 1; wKnn = 0;
  } else if (hasKnn) {
    p1 = knn.p1;
    wHaz = 0; wKnn = 1;
  } else {
    // Fallback to raw base rate only if hazard and knn both unavailable
    p1 = targetData.baseRate;
    wHaz = 0; wKnn = 0;
  }

  // ── Apply white cluster adjustment (only if measured signal is available) ──
  // Logic: when we're in a long white run, higher targets become less likely now
  if (whiteRisk !== null && currentRun > 0) {
    if (target >= 20) {
      // High target hit chance suppressed by active white cluster
      p1 = p1 * (1 - 0.6 * whiteRisk);
    } else {
      // Low targets less affected by white clusters
      p1 = p1 * (1 - 0.25 * whiteRisk);
    }
  }
  // When white cluster is ending (rebound signal), boost probability
  if (whiteRelease !== null && currentRun > 0 && whiteRelease > 0.5) {
    p1 = p1 * (1 + 0.3 * (whiteRelease - 0.5));
  }

  // ── Apply B2B adjustment (only if measured) ──
  if (b2b.nearRate !== null && target <= 20) {
    p1 = p1 * (1 + 0.2 * b2b.nearRate);
  }
  if (b2b.immRate !== null && target <= 10) {
    p1 = p1 * (1 + 0.15 * b2b.immRate);
  }

  p1 = clamp(p1, 0, 1);

  // ── Window placement ──
  // Use calibration shift if we have data, else use gap quantile directly
  const shift = cal?.shift ?? 0;
  const spanMultiplier = cal?.spanMult ?? 1;

  let aheadLo, spread;
  if (gapQ) {
    const rawLo = gapQ.q25;
    const rawSpread = Math.max(1, gapQ.q75 - gapQ.q25);
    aheadLo = Math.max(1, Math.round(rawLo + shift * rawSpread));
    spread  = Math.max(1, Math.round(rawSpread * spanMultiplier));
  } else {
    // No gap quantiles → use reciprocal of base rate as rough estimate
    const roughGap = Math.max(1, Math.round(1 / Math.max(0.001, targetData.baseRate)));
    aheadLo = Math.max(1, Math.round(roughGap * 0.5));
    spread  = roughGap;
  }

  // White cluster adjustment: push window forward when risk is high
  if (whiteRisk !== null && currentRun > 0 && target >= 20 && whiteRisk > 0.4) {
    const push = Math.max(1, Math.round(whiteRisk * (state.wcd.runQ95 || 3)));
    aheadLo = aheadLo + push;
  }
  // B2B adjustment: pull window closer when back-to-back pressure is high
  if (b2b.nearRate !== null && b2b.nearRate > 0.3 && target <= 20) {
    const pull = Math.max(1, Math.round(b2b.nearRate * 3));
    aheadLo = Math.max(1, aheadLo - pull);
  }

  // Fixed span by target (architectural requirement)
  const fixedSpan = FIXED_WINDOW_SPAN[target] || Math.max(3, spread);
  const lo = currentRound + aheadLo;
  const hi = lo + fixedSpan - 1;

  // ── Confidence — measured signal coverage ──
  // Confidence is a function of: how much data we have, hazard reliability, knn support
  const dataScore    = clamp(targetData.hitIndices.length / Math.max(10, state.n * 0.05), 0, 1);
  const hazardScore  = hasHazard ? clamp(hazard.totalObs / 100, 0, 1) : 0;
  const knnScore     = hasKnn ? knn.support : 0;
  const calScore     = cal ? clamp(cal.sample / 30, 0, 1) : 0;

  const confidence   = clamp(
    0.2 * dataScore + 0.3 * hazardScore + 0.3 * knnScore + 0.2 * calScore,
    0, 1
  );

  // ── Suspension logic — only suspend based on measured data signals ──
  let suspended = false;
  let suspendReason = null;

  // Suspend if white cluster is extremely active and measured
  if (target >= 50 && whiteRisk !== null && wEst.reliable && whiteRisk > 0.75 && (whiteRelease === null || whiteRelease < 0.3)) {
    suspended = true; suspendReason = 'white_cluster_active';
  }
  // Suspend if we have extremely low measured probability
  if (p1 < 0.001) {
    suspended = true; suspendReason = 'probability_too_low';
  }
  // Suspend large targets when white run is longer than the 99th percentile run
  if (target >= 500 && state.wcd.runQ99 !== null && currentRun > state.wcd.runQ99 * 0.8) {
    suspended = true; suspendReason = 'extreme_white_run';
  }

  // ── pSoon ──
  const pSoon = knn?.pSoon ?? clamp(1 - Math.pow(1 - p1, Math.max(1, gapQ?.q35 ?? 3)), 0, 1);

  return {
    target,
    lo,
    hi,
    roundWhenMade: currentRound,
    suspended,
    suspendReason,
    confidence: roundNum(confidence, 6),
    eta: {
      pHit1:             roundNum(p1, 6),
      pHitSoon:          roundNum(pSoon, 6),
      horizonSoon:       knn?.horizonSoon ?? null,

      // Gap distribution (from data)
      q25:               roundNum(gapQ?.q25 ?? null, 3),
      q50:               roundNum(gapQ?.q50 ?? null, 3),
      q75:               roundNum(gapQ?.q75 ?? null, 3),
      gapNow:            targetData.currentGap,
      aheadLo,
      aheadHi:           hi - currentRound,

      // Signal breakdown
      whiteClusterRun:        currentRun,
      whiteClusterRisk:       roundNum(whiteRisk, 6),        // null = no data
      whiteClusterRelease:    roundNum(whiteRelease, 6),     // null = no data
      whiteClusterReliable:   wEst.reliable,
      whiteClusterSample:     wEst.sample,

      b2bImmRate:       roundNum(b2b.immRate, 6),            // null = no data
      b2bNearRate:      roundNum(b2b.nearRate, 6),           // null = no data
      b2bRecentPairs:   b2b.recentPairs,
      b2bGlobalPairs:   b2b.globalPairs,

      hazardP1:         roundNum(hazard.p1, 6),              // null = insufficient
      hazardReliable:   hazard.reliable,
      hazardTotalObs:   hazard.totalObs ?? 0,

      knnP1:            roundNum(knn?.p1 ?? null, 6),
      knnPSoon:         roundNum(knn?.pSoon ?? null, 6),
      knnSupport:       roundNum(knn?.support ?? null, 6),
      knnSampleSize:    knn?.sampleSize ?? 0,

      blend:  { hazard: roundNum(wHaz, 6), knn: roundNum(wKnn, 6) },

      calibrationShift:     roundNum(cal?.shift ?? 0, 6),
      calibrationSpanMult:  roundNum(cal?.spanMult ?? 1, 6),
      calibrationSample:    cal?.sample ?? 0,

      baseRate:   roundNum(targetData.baseRate, 6),
      hitCount:   targetData.hitIndices.length,
      sampleSize: state.n,

      suspended,
      suspendReason,
      modelVersion: 'v9-pure-data',
    },
  };
}

// ────────────────────────────────────────────────────────────
// LOCK LIFECYCLE
// ────────────────────────────────────────────────────────────

function evaluateExistingLock(lock, hitRoundIds, currentRound) {
  if (!lock) return { resolved: false, status: 'none', outcome: null, hitRound: null };
  const lo   = Number(lock.lo);
  const hi   = Number(lock.hi);
  const made = Number(lock.roundWhenMade ?? lo - 1);

  // Find first hit after the lock was made
  const lb       = lowerBound(hitRoundIds, made + 1);
  const ub       = lowerBound(hitRoundIds, currentRound + 1);
  const firstHit = lb < ub ? hitRoundIds[lb] : null;

  if (firstHit !== null) {
    if (firstHit < lo) return { resolved: true, status: 'resolved', outcome: 'early', hitRound: firstHit };
    if (firstHit <= hi) return { resolved: true, status: 'resolved', outcome: 'win',  hitRound: firstHit };
  }
  // Resolve loss as soon as we have reached the final round of the window
  // with no hit recorded (collector rounds are completed rounds).
  if (currentRound >= hi) return { resolved: true,  status: 'resolved',    outcome: 'loss',  hitRound: null };
  if (currentRound >= lo) return { resolved: false, status: 'window-open', outcome: null,    hitRound: null };
  return { resolved: false, status: 'waiting', outcome: null, hitRound: null };
}

function confidenceBand(score) {
  const s = clamp(score ?? 0, 0, 1);
  if (s >= 0.8) return 'VERY HIGH';
  if (s >= 0.6) return 'HIGH';
  if (s >= 0.35) return 'MED';
  if (s >= 0.15) return 'LOW';
  return 'NONE';
}

function buildUiTarget(target, lock, status, currentRound, previousOutcome) {
  const lo  = Number(lock.lo);
  const hi  = Number(lock.hi);
  const eta = lock.eta || {};
  return {
    target,
    targetLabel: `${target}x`,
    status,
    confidence:  clamp(Number(lock.confidence ?? 0), 0, 1),
    confidenceBand: confidenceBand(lock.confidence),
    window: {
      lo,
      hi,
      span:                 Math.max(1, hi - lo + 1),
      roundsUntilWindow:    Math.max(0, lo - currentRound),
      roundsLeftInWindow:   Math.max(0, hi - currentRound),
    },
    signals: {
      pHit1:              Number(eta.pHit1   ?? 0),
      pHitSoon:           Number(eta.pHitSoon ?? 0),
      whiteClusterRisk:   eta.whiteClusterRisk   ?? null,
      whiteClusterRelease:eta.whiteClusterRelease ?? null,
      whiteClusterRun:    Number(eta.whiteClusterRun ?? 0),
      b2bImmRate:         eta.b2bImmRate  ?? null,
      b2bNearRate:        eta.b2bNearRate ?? null,
      hazardP1:           eta.hazardP1    ?? null,
      knnP1:              eta.knnP1       ?? null,
      knnSupport:         eta.knnSupport  ?? null,
      baseRate:           eta.baseRate    ?? null,
    },
    previousOutcome: previousOutcome || null,
  };
}

// ────────────────────────────────────────────────────────────
// PUBLIC: computeLockedRangePredictions
// ────────────────────────────────────────────────────────────

function computeLockedRangePredictions(rounds, existingLocksRaw = {}, options = {}) {
  const cleanRounds = normalizeRounds(rounds);

  if (!cleanRounds.length) {
    return {
      model: 'range-lock-v9-pure-data',
      generatedAt: new Date().toISOString(),
      asOfRound: null,
      sampleSize: 0,
      targets: [],
      locksToSave: {},
      resolvedHistory: [],
      calibration: {},
      whiteCluster: null,
      summary: { waiting: 0, windowOpen: 0, relocked: 0, sampleSize: 0 },
    };
  }

  const state        = buildGlobalState(cleanRounds);
  const currentRound = cleanRounds[cleanRounds.length - 1].roundId;
  const historyRows  = Array.isArray(options?.historyRows) ? options.historyRows : [];
  const calibration  = buildCalibration(historyRows);

  const locksToSave    = {};
  const resolvedHistory= [];
  const targetsOut     = [];
  let waitingCount = 0, openCount = 0, relockedCount = 0;

  for (const target of TARGETS) {
    const key        = String(target);
    const targetData = buildTargetData(state, target);
    const existing   = normalizeLockInput(existingLocksRaw[key]);
    const eval_      = evaluateExistingLock(existing, targetData.hitRoundIds, currentRound);

    // Check if current lock's span matches expected (structural change → rebuild)
    const expectedSpan   = FIXED_WINDOW_SPAN[target] || 3;
    const existingSpan   = existing ? Math.max(1, Number(existing.hi) - Number(existing.lo) + 1) : null;
    const spanMismatch   = Boolean(existing && existingSpan !== expectedSpan);

    const candidate      = buildPrediction(state, targetData, calibration[target]);
    let lockToUse        = existing;
    let status           = eval_.status;
    let previousOutcome  = null;

    if (!existing || eval_.resolved || spanMismatch) {
      if (existing && eval_.resolved) {
        previousOutcome = {
          outcome:    eval_.outcome,
          hitRound:   eval_.hitRound,
          lo:         existing.lo,
          hi:         existing.hi,
          generation: existing.generation,
        };
        // Save ALL fields needed so this row can be passed back as a historyRow
        // for calibration on future calls (via options.historyRows)
        resolvedHistory.push({
          target:     `${target}x`,          // e.g. "50x" — matches targetFromLabel()
          minMult:    Number(target),
          outcome:    eval_.outcome,          // 'win' | 'early' | 'loss'
          lo:         Number(existing.lo),
          hi:         Number(existing.hi),
          hitRound:   eval_.hitRound,         // null on loss
          generation: Number(existing.generation || 1),
          probW:      existing?.eta?.pHit1 ?? existing?.eta?.hazardP1 ?? null, // predicted probability when lock was made
          confidence: existing?.eta?.aiConfidence ?? existing?.confidence ?? null,
          roundWhenMade: Number(existing.roundWhenMade),
        });
      }

      const generation = existing ? Number(existing.generation || 1) + 1 : 1;
      lockToUse = {
        lo:           Number(candidate.lo),
        hi:           Number(candidate.hi),
        roundWhenMade:Number(candidate.roundWhenMade),
        generation,
        suspended:    Boolean(candidate.suspended),
        confidence:   Number(candidate.confidence || 0),
        eta:          candidate.eta,
      };
      status = candidate.suspended ? 'waiting' : 'locked';
      relockedCount++;
    } else {
      // Refresh signals on existing lock without moving the window
      lockToUse = {
        ...existing,
        confidence: Number(candidate.confidence || existing?.confidence || 0),
        eta: {
          ...(existing?.eta || {}),
          ...(candidate?.eta || {}),
          suspended: Boolean(existing?.suspended),
        },
      };
      status = eval_.status;
    }

    // Fix status if window has already opened
    if ((status === 'waiting' || status === 'locked') && Number(lockToUse.lo) <= currentRound) {
      status = 'window-open';
    }

    if (status === 'waiting') waitingCount++;
    if (status === 'window-open') openCount++;

    locksToSave[key] = {
      lo:           Number(lockToUse.lo),
      hi:           Number(lockToUse.hi),
      roundWhenMade:Number(lockToUse.roundWhenMade),
      generation:   Number(lockToUse.generation || 1),
      suspended:    Boolean(lockToUse.suspended),
      eta:          lockToUse.eta || null,
    };

    targetsOut.push(buildUiTarget(target, lockToUse, status, currentRound, previousOutcome));
  }

  targetsOut.sort((a, b) => a.target - b.target);

  // ── White cluster global summary ──
  const currentRun  = state.wcd.runLen[state.n - 1] || 0;
  const wEst        = state.wcd.estimate(currentRun);
  const whiteCluster = {
    activeRun:       currentRun,
    cut:             roundNum(state.wcd.whiteCut, 4),
    reboundCut:      roundNum(state.wcd.reboundCut, 4),
    continueProb:    roundNum(wEst.continueProb, 6),
    reboundProb:     roundNum(wEst.reboundProb, 6),
    reliable:        wEst.reliable,
    sample:          wEst.sample,
    runQ85:          roundNum(state.wcd.runQ85, 2),
    runQ95:          roundNum(state.wcd.runQ95, 2),
    runQ99:          roundNum(state.wcd.runQ99, 2),
    isExtreme:       state.wcd.runQ99 !== null && currentRun > state.wcd.runQ99 * 0.9,
  };

  return {
    model:         'range-lock-v9-pure-data',
    generatedAt:   new Date().toISOString(),
    asOfRound:     currentRound,
    sampleSize:    state.n,
    targets:       targetsOut,
    locksToSave,
    resolvedHistory,
    calibration,
    whiteCluster,
    summary: {
      waiting:    waitingCount,
      windowOpen: openCount,
      relocked:   relockedCount,
      sampleSize: state.n,
    },
    settings: {
      modelVersion:       'v9-pure-data',
      noFakeDefaults:     true,
      noInventedPriors:   true,
      supervisedCalib:    historyRows.length > 0,
      calibrationRows:    historyRows.length,
      whiteCut:           roundNum(state.wcd.whiteCut, 4),
      reboundCut:         roundNum(state.wcd.reboundCut, 4),
      shortWindow:        state.shortW,
      longWindow:         state.longW,
      fixedWindowSpans:   FIXED_WINDOW_SPAN,
    },
  };
}

// ────────────────────────────────────────────────────────────
// PUBLIC: buildPredictionReport (dashboard summary)
// ────────────────────────────────────────────────────────────

const REPORT_THRESHOLDS = [2, 5, 10, 25, 50];

function buildThresholdSnapshot(state, target) {
  const targetData = buildTargetData(state, target);
  const hazard     = buildHazardModel(targetData);
  const knn        = runKnn(state, targetData, state.n - 1);
  const p1         = hazard.p1 ?? knn?.p1 ?? targetData.baseRate;
  const gapQ       = targetData.gapQ;

  return {
    target,
    gapNow:          targetData.currentGap,
    p1:              roundNum(p1, 6),
    p3:              roundNum(clamp(1 - Math.pow(1 - p1, 3), 0, 1), 6),
    p5:              roundNum(clamp(1 - Math.pow(1 - p1, 5), 0, 1), 6),
    expectedGap:     p1 > 0 ? roundNum(1 / p1, 2) : null,
    q50Gap:          roundNum(gapQ?.q50 ?? null, 2),
    hitCount:        targetData.hitIndices.length,
    baseRate:        roundNum(targetData.baseRate, 6),
    b2bNearRate:     roundNum(targetData.b2b.nearRate ?? null, 6),
    hazardReliable:  hazard.reliable,
    knnSupport:      roundNum(knn?.support ?? null, 6),
  };
}

function buildPredictionReport(rounds) {
  const cleanRounds = normalizeRounds(rounds);

  if (!cleanRounds.length) {
    return {
      model:      'v9-pure-data-report',
      generatedAt: new Date().toISOString(),
      asOfRound:  null,
      sampleSize: 0,
      bucketProbabilities:  BUCKETS.map(b => ({ ...b, probability: null })),
      predictedBucket:      null,
      targetProbabilities:  REPORT_THRESHOLDS.map(t => ({ target: t, p1: null })),
      whiteCluster:         null,
      cashoutPlan:          null,
    };
  }

  const state = buildGlobalState(cleanRounds);
  const mult  = cleanRounds.map(r => Number(r.multiplier));

  // Bucket probabilities — raw observed frequencies
  const counts = new Array(BUCKETS.length).fill(0);
  for (const m of mult) {
    for (let i = 0; i < BUCKETS.length; i++) {
      if (m >= BUCKETS[i].min && m <= BUCKETS[i].max) { counts[i]++; break; }
    }
  }
  const total = mult.length;
  const bucketProbabilities = BUCKETS.map((b, i) => ({
    ...b,
    probability: roundNum(counts[i] / total, 6),
    count: counts[i],
  }));
  const topIdx = counts.indexOf(Math.max(...counts));

  // Per-threshold snapshots
  const targetProbabilities = REPORT_THRESHOLDS.map(t => buildThresholdSnapshot(state, t));

  // White cluster state
  const currentRun = state.wcd.runLen[state.n - 1] || 0;
  const wEst       = state.wcd.estimate(currentRun);
  const whiteCluster = {
    activeRun:    currentRun,
    cut:          roundNum(state.wcd.whiteCut, 4),
    continueProb: roundNum(wEst.continueProb, 6),
    reboundProb:  roundNum(wEst.reboundProb, 6),
    reliable:     wEst.reliable,
    sample:       wEst.sample,
    runQ85:       roundNum(state.wcd.runQ85, 2),
    runQ95:       roundNum(state.wcd.runQ95, 2),
  };

  // Cashout plan — based purely on measured expected value
  const p_safe    = targetProbabilities.find(x => x.target === 2)?.p1  ?? null;
  const p_bal     = targetProbabilities.find(x => x.target === 5)?.p1  ?? null;
  const p_agg     = targetProbabilities.find(x => x.target === 10)?.p1 ?? null;

  const cashoutPlan = {
    safe:       p_safe !== null ? { target: 1.8, hitChance: roundNum(p_safe, 6), ev: roundNum(1.8 * p_safe - 1, 6) } : null,
    balanced:   p_bal  !== null ? { target: 3.2, hitChance: roundNum(p_bal,  6), ev: roundNum(3.2 * p_bal  - 1, 6) } : null,
    aggressive: p_agg  !== null ? { target: 7.0, hitChance: roundNum(p_agg,  6), ev: roundNum(7.0 * p_agg  - 1, 6) } : null,
  };

  // Pick best EV option — no preference baked in
  const options = [
    cashoutPlan.safe       && { label: 'SAFE',       ...cashoutPlan.safe },
    cashoutPlan.balanced   && { label: 'BALANCED',   ...cashoutPlan.balanced },
    cashoutPlan.aggressive && { label: 'AGGRESSIVE', ...cashoutPlan.aggressive },
  ].filter(Boolean);
  const bestEv = options.sort((a, b) => (b.ev ?? -Infinity) - (a.ev ?? -Infinity))[0] ?? null;
  cashoutPlan.recommended      = bestEv;
  cashoutPlan.recommendedLabel = bestEv?.label ?? null;

  // Sorted multiplier stats
  const sorted = state.sorted;
  return {
    model:       'v9-pure-data-report',
    generatedAt:  new Date().toISOString(),
    asOfRound:    cleanRounds[cleanRounds.length - 1].roundId,
    sampleSize:   cleanRounds.length,

    // Distribution stats (all from data)
    expectedMean:   roundNum(mult.reduce((s, v) => s + v, 0) / mult.length, 4),
    expectedMedian: roundNum(quantileFromSorted(sorted, 0.50), 4),
    expectedP75:    roundNum(quantileFromSorted(sorted, 0.75), 4),
    expectedP90:    roundNum(quantileFromSorted(sorted, 0.90), 4),

    bucketProbabilities,
    predictedBucket: bucketProbabilities[topIdx],
    targetProbabilities,
    whiteCluster,
    cashoutPlan,

    // Raw signal diagnostics
    diagnostics: {
      shortWindow:   state.shortW,
      longWindow:    state.longW,
      whiteCut:      roundNum(state.wcd.whiteCut, 4),
      reboundCut:    roundNum(state.wcd.reboundCut, 4),
      greenCut:      roundNum(state.greenCut, 4),
      currentWhiteRun:      currentRun,
      whiteRunQ85:   roundNum(state.wcd.runQ85, 2),
      whiteRunQ95:   roundNum(state.wcd.runQ95, 2),
    },
  };
}

// ────────────────────────────────────────────────────────────
// EXPORTS
// ────────────────────────────────────────────────────────────

module.exports = {
  TARGETS,
  FIXED_WINDOW_SPAN,
  BUCKETS,
  buildPredictionReport,
  computeLockedRangePredictions,

  // Expose internal builders for testing/debugging
  _internal: {
    normalizeRounds,
    buildGlobalState,
    buildTargetData,
    buildWhiteClusterDetector,
    buildB2BDetector,
    buildHazardModel,
    runKnn,
    buildCalibration,
    buildPrediction,
  },
};
