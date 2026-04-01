'use strict';

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const REPORT_THRESHOLDS = [2, 5, 10, 25, 50];

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
  return Math.max(lo, Math.min(hi, n));
}

function roundNum(v, digits = 4) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Number(n.toFixed(digits));
}

function mean(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += Number(v) || 0;
  return s / arr.length;
}

function stddev(arr, avg = null) {
  if (arr.length <= 1) return 0;
  const m = avg == null ? mean(arr) : avg;
  let s = 0;
  for (const v of arr) {
    const d = (Number(v) || 0) - m;
    s += d * d;
  }
  return Math.sqrt(s / arr.length);
}

function quantileFromSorted(sorted, q) {
  if (!sorted.length) return 0;
  const qq = clamp(q, 0, 1);
  const idx = (sorted.length - 1) * qq;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return (sorted[lo] * (1 - w)) + (sorted[hi] * w);
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return quantileFromSorted(sorted, q);
}

function weightedQuantile(items, q) {
  if (!items.length) return 1;
  const sorted = [...items].sort((a, b) => a.value - b.value);
  const qq = clamp(q, 0, 1);
  let total = 0;
  for (const item of sorted) total += Math.max(0, Number(item.weight) || 0);
  if (total <= 0) return sorted[Math.floor((sorted.length - 1) * qq)].value;
  let acc = 0;
  for (const item of sorted) {
    acc += Math.max(0, Number(item.weight) || 0);
    if ((acc / total) >= qq) return item.value;
  }
  return sorted[sorted.length - 1].value;
}

function sigmoid(v) {
  const x = clamp(v, -20, 20);
  return 1 / (1 + Math.exp(-x));
}

function safeLog(v) {
  return Math.log(Math.max(1, Number(v) || 1));
}

function normalizeDistribution(dist) {
  let sum = 0;
  for (const v of dist) sum += Number(v) || 0;
  if (sum <= 0) {
    const uniform = 1 / Math.max(1, dist.length);
    return dist.map(() => uniform);
  }
  return dist.map(v => (Number(v) || 0) / sum);
}

function bucketIndex(multiplier) {
  const m = Number(multiplier) || 1;
  for (let i = 0; i < BUCKETS.length; i++) {
    const b = BUCKETS[i];
    if (m >= b.min && m <= b.max) return i;
  }
  return BUCKETS.length - 1;
}

function bucketMidpoint(bucket) {
  if (!Number.isFinite(bucket.max)) return bucket.min * 1.4;
  return (bucket.min + bucket.max) / 2;
}

function buildPrefix(arr) {
  const pref = new Array(arr.length + 1).fill(0);
  for (let i = 0; i < arr.length; i++) pref[i + 1] = pref[i] + (Number(arr[i]) || 0);
  return pref;
}

function prefixSum(pref, lo, hi) {
  if (hi < lo) return 0;
  const l = clamp(lo, 0, pref.length - 1);
  const r = clamp(hi + 1, 0, pref.length - 1);
  if (r <= l) return 0;
  return pref[r] - pref[l];
}

function prefixRate(pref, endIdx, len) {
  const n = pref.length - 1;
  if (n <= 0 || endIdx < 0) return 0;
  const e = clamp(endIdx, 0, n - 1);
  const l = clamp(e - Math.max(1, len) + 1, 0, n - 1);
  const total = prefixSum(pref, l, e);
  const width = (e - l + 1);
  return width > 0 ? (total / width) : 0;
}

function prefixStd(pref, prefSq, endIdx, len) {
  const n = pref.length - 1;
  if (n <= 1 || endIdx < 0) return 0;
  const e = clamp(endIdx, 0, n - 1);
  const l = clamp(e - Math.max(1, len) + 1, 0, n - 1);
  const width = Math.max(1, e - l + 1);
  const sum = prefixSum(pref, l, e);
  const sumSq = prefixSum(prefSq, l, e);
  const m = sum / width;
  return Math.sqrt(Math.max(0, (sumSq / width) - (m * m)));
}

function lowerBound(arr, value) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function normalizeRounds(rounds) {
  const clean = (rounds || [])
    .map(r => ({
      roundId: Number(r.roundId),
      multiplier: Number(r.multiplier),
      timestamp: Number(r.timestamp) || Date.now(),
    }))
    .filter(r => Number.isFinite(r.roundId) && Number.isFinite(r.multiplier) && r.multiplier > 0)
    .sort((a, b) => a.roundId - b.roundId);

  if (!clean.length) return [];
  const dedup = [];
  let lastId = null;
  for (const r of clean) {
    if (r.roundId === lastId) dedup[dedup.length - 1] = r;
    else {
      dedup.push(r);
      lastId = r.roundId;
    }
  }
  return dedup;
}

function targetFromLabel(raw) {
  const s = String(raw || '').trim().toLowerCase();
  if (!s) return null;
  const n = Number(s.replace('x', ''));
  return Number.isFinite(n) ? n : null;
}

function buildHistoryCalibration(historyRows = []) {
  // === v8 TRUE SUPERVISED LEARNING & SIMPLIFIED UPGRADE START ===
  // Justification: turn every historical win/early/loss into persistent calibration for thresholds, shift and span.
  const out = {};
  for (const t of TARGETS) {
    out[t] = {
      sample: 0,
      winRate: 0.5,
      earlyRate: 0,
      lossRate: 0,
      reliability: 0.5,
      shift: 0,
      spanMultiplier: 1,
      minP1: 0.08,
      minConfidence: 0.42,
    };
  }

  const rowsByTarget = {};
  for (const t of TARGETS) rowsByTarget[t] = [];
  for (const row of (historyRows || [])) {
    const target = targetFromLabel(row?.target);
    if (!TARGETS.includes(target)) continue;
    rowsByTarget[target].push(row);
  }

  for (const t of TARGETS) {
    const rows = rowsByTarget[t];
    if (!rows.length) continue;
    let win = 0;
    let early = 0;
    let loss = 0;
    const shiftItems = [];
    const absItems = [];
    const winProb = [];
    const lossProb = [];

    const n = rows.length;
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      const w = 1 + (2 * (((n - i) / Math.max(1, n)) ** 2));
      const outcome = String(row?.outcome || '').toLowerCase();
      if (outcome === 'win') win += w;
      else if (outcome === 'early') early += w;
      else if (outcome === 'loss') loss += w;

      const lo = Number(row?.lo);
      const hi = Number(row?.hi);
      const hit = row?.hitRound == null ? null : Number(row.hitRound);
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo) {
        const span = Math.max(1, hi - lo + 1);
        let err = 0;
        if (outcome === 'early' && Number.isFinite(hit)) err = (hit - lo) / span;
        else if (outcome === 'win' && Number.isFinite(hit)) err = (hit - (lo + ((span - 1) * 0.5))) / span;
        else if (outcome === 'loss') err = 1;
        shiftItems.push({ value: clamp(err, -3, 3), weight: w });
        absItems.push({ value: Math.abs(clamp(err, -3, 3)), weight: w });
      }

      const p = Number(row?.probW);
      if (Number.isFinite(p)) {
        if (outcome === 'win') winProb.push(p);
        if (outcome === 'loss') lossProb.push(p);
      }
    }

    const total = win + early + loss;
    const wl = Math.max(1, win + loss);
    const winRate = win / wl;
    const earlyRate = total > 0 ? (early / total) : 0;
    const lossRate = total > 0 ? (loss / total) : 0;
    const reliability = clamp((win + (0.35 * early) + 2) / Math.max(4, total + 4), 0.02, 0.98);
    const shift = shiftItems.length ? clamp(weightedQuantile(shiftItems, 0.5), -0.9, 0.9) : 0;
    const spanMultiplier = absItems.length ? clamp(1 + weightedQuantile(absItems, 0.6), 0.7, 3) : 1;

    const meanWinProb = winProb.length ? mean(winProb) : null;
    const meanLossProb = lossProb.length ? mean(lossProb) : null;
    const minP1 = (meanWinProb != null && meanLossProb != null)
      ? clamp((0.52 * meanLossProb) + (0.48 * meanWinProb), 0.01, 0.9)
      : clamp((0.04 + (0.3 * lossRate) + (0.1 * earlyRate) + (0.2 * (1 - reliability))), 0.01, 0.9);
    const minConfidence = clamp(0.18 + (0.5 * lossRate) + (0.16 * earlyRate) - (0.14 * reliability), 0.1, 0.95);

    out[t] = {
      sample: Math.round(total),
      winRate: roundNum(winRate, 6),
      earlyRate: roundNum(earlyRate, 6),
      lossRate: roundNum(lossRate, 6),
      reliability: roundNum(reliability, 6),
      shift: roundNum(shift, 6),
      spanMultiplier: roundNum(spanMultiplier, 6),
      minP1: roundNum(minP1, 6),
      minConfidence: roundNum(minConfidence, 6),
    };
  }
  return out;
  // === UPGRADE END ===
}

function buildGlobalState(cleanRounds) {
  const n = cleanRounds.length;
  const multipliers = cleanRounds.map(r => Number(r.multiplier));
  const logs = multipliers.map(m => safeLog(m));
  const sortedMult = [...multipliers].sort((a, b) => a - b);

  const whiteCut = clamp(quantileFromSorted(sortedMult, 0.43), 2.2, 3.8);
  const reboundCut = clamp(quantileFromSorted(sortedMult, 0.68), 3.6, 12);
  const greenCut = clamp(quantileFromSorted(sortedMult, 0.58), 4.5, 8.5);

  const whiteFlags = multipliers.map(m => (m < whiteCut ? 1 : 0));
  const greenFlags = multipliers.map(m => (m >= greenCut ? 1 : 0));
  const prefWhite = buildPrefix(whiteFlags);
  const prefGreen = buildPrefix(greenFlags);
  const prefLog = buildPrefix(logs);
  const prefLogSq = buildPrefix(logs.map(v => v * v));

  const whiteRunAt = new Array(n).fill(0);
  for (let i = 0; i < n; i++) whiteRunAt[i] = whiteFlags[i] ? (1 + (i > 0 ? whiteRunAt[i - 1] : 0)) : 0;
  const whiteRunQ85 = quantile(whiteRunAt, 0.85);
  const whiteRunQ95 = quantile(whiteRunAt, 0.95);

  const byRun = new Map();
  let globalObs = 0;
  let globalContinue = 0;
  let globalRebound = 0;
  for (let i = 1; i < n - 3; i++) {
    const run = whiteRunAt[i];
    if (run <= 0) continue;
    const bucket = Math.min(10, run);
    if (!byRun.has(bucket)) byRun.set(bucket, { obs: 0, cont: 0, rebound: 0 });
    const row = byRun.get(bucket);
    row.obs += 1;
    globalObs += 1;
    if (multipliers[i + 1] < whiteCut) {
      row.cont += 1;
      globalContinue += 1;
    }
    const rebound = multipliers[i + 1] >= reboundCut || multipliers[i + 2] >= reboundCut || multipliers[i + 3] >= reboundCut;
    if (rebound) {
      row.rebound += 1;
      globalRebound += 1;
    }
  }

  const whiteModel = {
    estimate(runLength) {
      const bucket = Math.min(10, Math.max(0, Number(runLength) || 0));
      const row = byRun.get(bucket);
      if (!row || row.obs < 4) {
        return {
          continueProb: (globalContinue + 1) / Math.max(2, globalObs + 2),
          reboundProb: (globalRebound + 1) / Math.max(2, globalObs + 2),
          sample: globalObs,
        };
      }
      return {
        continueProb: (row.cont + 1) / (row.obs + 2),
        reboundProb: (row.rebound + 1) / (row.obs + 2),
        sample: row.obs,
      };
    },
  };

  return {
    rounds: cleanRounds,
    n,
    multipliers,
    logs,
    whiteCut,
    reboundCut,
    prefWhite,
    prefGreen,
    prefLog,
    prefLogSq,
    whiteRunAt,
    whiteRunQ85: Math.max(1, whiteRunQ85),
    whiteRunQ95: Math.max(1, whiteRunQ95),
    whiteModel,
  };
}

function buildTargetData(state, target) {
  const n = state.n;
  const flags = new Array(n).fill(0);
  const hitIndices = [];
  const hitRoundIds = [];
  for (let i = 0; i < n; i++) {
    if (state.multipliers[i] >= target) {
      flags[i] = 1;
      hitIndices.push(i);
      hitRoundIds.push(state.rounds[i].roundId);
    }
  }

  const prefHit = buildPrefix(flags);
  const gapAt = new Array(n).fill(0);
  let lastHit = -1;
  for (let i = 0; i < n; i++) {
    if (flags[i]) lastHit = i;
    gapAt[i] = lastHit >= 0 ? (i - lastHit) : (i + 1);
  }

  const nextHitDist = new Array(n).fill(null);
  let nextHit = null;
  for (let i = n - 1; i >= 0; i--) {
    if (flags[i]) nextHit = i;
    nextHitDist[i] = nextHit == null ? null : (nextHit - i);
  }

  const interGaps = [];
  for (let i = 1; i < hitIndices.length; i++) interGaps.push(hitIndices[i] - hitIndices[i - 1]);
  if (!interGaps.length) interGaps.push(Math.max(1, n));

  const sortedGaps = [...interGaps].sort((a, b) => a - b);
  const gapQ = {
    q20: quantileFromSorted(sortedGaps, 0.2),
    q35: quantileFromSorted(sortedGaps, 0.35),
    q50: quantileFromSorted(sortedGaps, 0.5),
    q75: quantileFromSorted(sortedGaps, 0.75),
    q90: quantileFromSorted(sortedGaps, 0.9),
    q95: quantileFromSorted(sortedGaps, 0.95),
  };

  let b2bImm = 0;
  let b2bNear = 0;
  if (hitIndices.length > 1) {
    for (let i = 1; i < hitIndices.length; i++) {
      const d = hitIndices[i] - hitIndices[i - 1];
      if (d <= 1) b2bImm += 1;
      if (d <= 2) b2bNear += 1;
    }
  }
  const pairDen = Math.max(1, hitIndices.length - 1);
  const b2bBase = b2bImm / pairDen;
  const b2bNearBase = b2bNear / pairDen;

  const recentLookback = clamp(Math.round(Math.sqrt(Math.max(1, n)) * 10), 80, 1500);
  let recentPairs = 0;
  let recentNear = 0;
  let recentImm = 0;
  let prevHitIdx = null;
  const start = Math.max(0, n - recentLookback);
  for (let i = start; i < n; i++) {
    if (!flags[i]) continue;
    if (prevHitIdx != null) {
      const d = i - prevHitIdx;
      recentPairs += 1;
      if (d <= 1) recentImm += 1;
      if (d <= 2) recentNear += 1;
    }
    prevHitIdx = i;
  }
  const b2bRecent = recentPairs > 0 ? (recentImm / recentPairs) : b2bBase;
  const b2bNearRecent = recentPairs > 0 ? (recentNear / recentPairs) : b2bNearBase;

  return {
    target,
    flags,
    prefHit,
    hitIndices,
    hitRoundIds,
    nextHitDist,
    gapAt,
    interGaps,
    gapQ,
    baseRate: hitIndices.length / Math.max(1, n),
    currentGap: gapAt[n - 1] || 0,
    b2bBase,
    b2bNearBase,
    b2bRecent,
    b2bNearRecent,
  };
}

function buildFeatureFns(state, targetData) {
  const density = clamp(Math.round(Math.sqrt(Math.max(1, state.n))), 24, 220);
  const shortW = clamp(Math.round(density * 0.5), 10, 120);
  const longW = clamp(Math.round(density * 1.4), shortW + 5, 420);
  const hitW = clamp(Math.round(Math.max(shortW, targetData.gapQ.q75 * 1.8)), shortW, Math.max(longW, 1200));
  const gapScale = Math.max(1, targetData.gapQ.q75 || targetData.gapQ.q50 || 1);
  const whiteScale = Math.max(1, state.whiteRunQ95);

  const getFeatures = (idx) => {
    const whiteShort = prefixRate(state.prefWhite, idx, shortW);
    const whiteLong = prefixRate(state.prefWhite, idx, longW);
    const greenShort = prefixRate(state.prefGreen, idx, shortW);
    const hitShort = prefixRate(targetData.prefHit, idx, shortW);
    const hitLong = prefixRate(targetData.prefHit, idx, hitW);
    const trend = prefixRate(state.prefLog, idx, shortW) - prefixRate(state.prefLog, idx, longW);
    const vol = prefixStd(state.prefLog, state.prefLogSq, idx, shortW);
    const gapNorm = clamp((targetData.gapAt[idx] || 0) / gapScale, 0, 8);
    const whiteRunNorm = clamp((state.whiteRunAt[idx] || 0) / whiteScale, 0, 4);
    return [whiteShort, whiteLong, greenShort, hitShort, hitLong, trend, vol, gapNorm, whiteRunNorm];
  };
  return { getFeatures, shortW, longW, hitW };
}

function buildSamples(state, targetData, featureFns) {
  const n = state.n;
  const start = Math.max(featureFns.longW + 2, 24);
  const horizonSoon = clamp(Math.round(Math.max(2, targetData.gapQ.q35 || 2)), 2, 30);
  const samples = [];
  for (let i = start; i < n - 1; i++) {
    const dist = targetData.nextHitDist[i];
    if (!Number.isFinite(dist) || dist < 1) continue;
    const recency = 0.55 + (0.45 * ((i - start + 1) / Math.max(1, n - start)));
    samples.push({
      idx: i,
      x: featureFns.getFeatures(i),
      y1: dist <= 1 ? 1 : 0,
      ySoon: dist <= horizonSoon ? 1 : 0,
      gapToHit: dist,
      recency,
    });
  }
  return { samples, horizonSoon };
}

function computeFeatureStats(samples) {
  if (!samples.length) return { means: [], stds: [] };
  const dims = samples[0].x.length;
  const means = new Array(dims).fill(0);
  const stds = new Array(dims).fill(0);
  for (const s of samples) {
    for (let d = 0; d < dims; d++) means[d] += s.x[d];
  }
  for (let d = 0; d < dims; d++) means[d] /= samples.length;
  for (const s of samples) {
    for (let d = 0; d < dims; d++) {
      const z = s.x[d] - means[d];
      stds[d] += z * z;
    }
  }
  for (let d = 0; d < dims; d++) stds[d] = Math.max(1e-6, Math.sqrt(stds[d] / samples.length));
  return { means, stds };
}

function zScore(vec, stats) {
  return vec.map((v, i) => (v - stats.means[i]) / stats.stds[i]);
}

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function runKnn(currentZ, sampleZ, k) {
  if (!sampleZ.length) return { p1: 0, pSoon: 0, q25: 1, q50: 2, q75: 3, effectiveSample: 0, support: 0 };
  const scored = [];
  for (const row of sampleZ) {
    const dist = euclidean(currentZ, row.z);
    const w = (1 / (1 + dist)) * row.recency;
    scored.push({ ...row, dist, w });
  }
  scored.sort((a, b) => a.dist - b.dist);
  const top = scored.slice(0, Math.max(1, Math.min(k, scored.length)));

  let sumW = 0;
  let sumW2 = 0;
  let numP1 = 0;
  let numSoon = 0;
  const gapItems = [];
  for (const row of top) {
    const w = Math.max(0, row.w);
    sumW += w;
    sumW2 += w * w;
    numP1 += row.y1 * w;
    numSoon += row.ySoon * w;
    gapItems.push({ value: row.gapToHit, weight: w });
  }
  const p1 = sumW > 0 ? (numP1 / sumW) : 0;
  const pSoon = sumW > 0 ? (numSoon / sumW) : 0;
  const effectiveSample = sumW2 > 0 ? ((sumW * sumW) / sumW2) : 0;
  return {
    p1: clamp(p1, 0, 1),
    pSoon: clamp(pSoon, 0, 1),
    q25: weightedQuantile(gapItems, 0.25),
    q50: weightedQuantile(gapItems, 0.5),
    q75: weightedQuantile(gapItems, 0.75),
    effectiveSample,
    support: clamp(effectiveSample / Math.max(1, top.length), 0, 1),
  };
}

function buildHazard(targetData) {
  const gaps = targetData.gapAt;
  const flags = targetData.flags;
  if (gaps.length <= 2) return { p1: targetData.baseRate, q50: 1 / Math.max(1e-6, targetData.baseRate), support: 0 };
  const maxGap = clamp(Math.round(Math.max(6, targetData.gapQ.q95 * 1.4)), 6, 5000);
  const obs = new Array(maxGap + 1).fill(0);
  const hits = new Array(maxGap + 1).fill(0);
  let totalObs = 0;
  let totalHits = 0;
  for (let i = 0; i < gaps.length - 1; i++) {
    const g = clamp(Math.round(gaps[i] || 0), 0, maxGap);
    obs[g] += 1;
    totalObs += 1;
    if (flags[i + 1]) {
      hits[g] += 1;
      totalHits += 1;
    }
  }
  const globalP = clamp(totalHits / Math.max(1, totalObs), 0.000001, 0.999);
  const bw = clamp(Math.round(stddev(gaps) * 0.18), 1, 25);
  const prior = Math.max(4, Math.sqrt(Math.max(1, totalObs)));
  const at = (gap) => {
    const g = clamp(Math.round(gap), 0, maxGap);
    let o = 0;
    let h = 0;
    for (let d = -bw; d <= bw; d++) {
      const idx = g + d;
      if (idx < 0 || idx > maxGap) continue;
      const w = (bw + 1) - Math.abs(d);
      o += obs[idx] * w;
      h += hits[idx] * w;
    }
    return clamp((h + (prior * globalP)) / Math.max(1e-6, o + prior), 0.000001, 0.999);
  };
  const p1 = at(targetData.currentGap);
  const q50 = clamp(1 / Math.max(0.000001, p1), 1, Math.max(2, targetData.gapQ.q95 * 1.5));
  return { p1, q50, support: clamp(totalObs / Math.max(20, gaps.length), 0, 1), at };
}

function evaluateExistingLock(lock, hitRoundIds, currentRound) {
  if (!lock) return { resolved: false, status: 'pending', outcome: null, hitRound: null };
  const lo = Number(lock.lo);
  const hi = Number(lock.hi);
  const made = Number(lock.roundWhenMade ?? lock.round_when_made ?? lo - 1);
  const lb = lowerBound(hitRoundIds, made + 1);
  const ub = lowerBound(hitRoundIds, currentRound + 1);
  const firstHit = lb < ub ? hitRoundIds[lb] : null;

  if (firstHit != null) {
    if (firstHit < lo) return { resolved: true, status: 'resolved', outcome: 'early', hitRound: firstHit };
    if (firstHit <= hi) return { resolved: true, status: 'resolved', outcome: 'win', hitRound: firstHit };
  }
  if (currentRound > hi) return { resolved: true, status: 'resolved', outcome: 'loss', hitRound: null };
  if (currentRound >= lo) return { resolved: false, status: 'window-open', outcome: null, hitRound: null };
  return { resolved: false, status: 'waiting', outcome: null, hitRound: null };
}

function confidenceBandWord(score) {
  const s = clamp(score, 0, 1);
  if (s >= 0.8) return 'VERY HIGH';
  if (s >= 0.6) return 'HIGH';
  if (s >= 0.35) return 'MED';
  if (s >= 0.15) return 'LOW';
  return 'NONE';
}

// === v8 TRUE SUPERVISED LEARNING & SIMPLIFIED UPGRADE START ===
// Justification: unify pattern + gap + calibration + adaptive online learning into one deterministic predictor.
function buildTargetCandidate(state, targetData, calibration) {
  const currentIdx = state.n - 1;
  const currentRound = state.rounds[currentIdx].roundId;
  const target = Number(targetData?.target || 0);
  const featureFns = buildFeatureFns(state, targetData);
  const { samples, horizonSoon } = buildSamples(state, targetData, featureFns);
  const stats = computeFeatureStats(samples);
  const sampleZ = samples.map(s => ({ ...s, z: zScore(s.x, stats) }));
  const currentZ = zScore(featureFns.getFeatures(currentIdx), stats);

  const kAdaptive = clamp(Math.round(Math.sqrt(Math.max(1, sampleZ.length)) * 1.9), 20, Math.max(20, Math.round(sampleZ.length * 0.45)));
  const knn = runKnn(currentZ, sampleZ, kAdaptive);
  const hazard = buildHazard(targetData);

  const whiteRun = state.whiteRunAt[currentIdx] || 0;
  const whiteShort = prefixRate(state.prefWhite, currentIdx, featureFns.shortW);
  const whiteLong = prefixRate(state.prefWhite, currentIdx, featureFns.longW);
  const greenShort = prefixRate(state.prefGreen, currentIdx, featureFns.shortW);
  const trend = prefixRate(state.prefLog, currentIdx, featureFns.shortW) - prefixRate(state.prefLog, currentIdx, featureFns.longW);
  const vol = prefixStd(state.prefLog, state.prefLogSq, currentIdx, featureFns.shortW);

  const whiteEstimate = state.whiteModel.estimate(whiteRun);
  const whiteRisk = clamp((0.5 * whiteEstimate.continueProb) + (0.3 * sigmoid((whiteRun - state.whiteRunQ85) / Math.max(1, state.whiteRunQ95 - state.whiteRunQ85))) + (0.2 * clamp((whiteShort - whiteLong) * 2 + (0.45 - greenShort), 0, 1)), 0, 1);
  const whiteRelease = clamp((0.65 * whiteEstimate.reboundProb) + (0.35 * clamp(greenShort + Math.max(0, trend), 0, 1)), 0, 1);

  const b2bPressure = clamp((0.55 * targetData.b2bNearRecent) + (0.45 * targetData.b2bNearBase), 0, 1);
  const b2bImmediate = clamp((0.6 * targetData.b2bRecent) + (0.4 * targetData.b2bBase), 0, 1);
  const trendDownRisk = clamp(sigmoid((-trend * 4.2) + ((whiteShort - whiteLong) * 4.8) + ((vol - 0.5) * 1.2)), 0, 1);

  const baseRate = clamp(targetData.baseRate, 0.000001, 0.999);
  const p1Base = baseRate;
  const p1Hazard = hazard.p1;
  const pSoonHazard = clamp(1 - ((1 - p1Hazard) ** Math.max(1, horizonSoon)), 0, 1);

  const reliability = clamp(calibration?.reliability ?? 0.5, 0.02, 0.98);
  const wKnnRaw = knn.support * (0.42 + (0.38 * reliability));
  const wHazRaw = hazard.support * (0.36 + (0.24 * (1 - reliability)));
  const wBaseRaw = Math.max(0.1, 1 - (wKnnRaw + wHazRaw));
  const wSum = wKnnRaw + wHazRaw + wBaseRaw;
  const wKnn = wKnnRaw / wSum;
  const wHaz = wHazRaw / wSum;
  const wBase = wBaseRaw / wSum;

  let p1 = (wKnn * knn.p1) + (wHaz * p1Hazard) + (wBase * p1Base);
  let pSoon = (wKnn * knn.pSoon) + (wHaz * pSoonHazard) + (wBase * clamp(1 - ((1 - p1Base) ** Math.max(1, horizonSoon)), 0, 1));

  if (target >= 20) {
    p1 *= (1 - (0.58 * whiteRisk)) * (1 + (0.24 * whiteRelease));
    pSoon *= (1 - (0.5 * whiteRisk)) * (1 + (0.2 * whiteRelease));
  } else {
    p1 *= (1 - (0.28 * whiteRisk)) * (1 + (0.3 * b2bPressure));
    pSoon *= (1 - (0.24 * whiteRisk)) * (1 + (0.24 * b2bPressure));
  }
  if (target >= 100) {
    p1 *= (1 + (0.18 * b2bImmediate));
    pSoon *= (1 + (0.14 * b2bImmediate));
  }
  p1 = clamp(p1, 0.000001, 0.999);
  pSoon = clamp(Math.max(pSoon, p1), 0.000001, 0.999);

  const calShift = clamp(calibration?.shift ?? 0, -1, 1);
  const calSpan = clamp(calibration?.spanMultiplier ?? 1, 0.65, 3.2);
  const q25Raw = Math.max(1, knn.q25 || hazard.q50 || targetData.gapQ.q35 || 1);
  const q50Raw = Math.max(q25Raw, knn.q50 || hazard.q50 || targetData.gapQ.q50 || q25Raw + 1);
  const q75Raw = Math.max(q50Raw, knn.q75 || targetData.gapQ.q75 || q50Raw + 1);

  let aheadLo = Math.max(1, Math.round(q25Raw + (calShift * 0.85)));
  let aheadMid = Math.max(aheadLo, Math.round(q50Raw + Math.max(0, calShift)));
  let spread = Math.max(1, Math.round((q75Raw - q25Raw) * calSpan));

  if (target >= 20 && whiteRisk >= 0.45) {
    const push = Math.max(1, Math.round(whiteRisk * spread * 0.9));
    aheadLo += push;
    aheadMid += Math.max(1, Math.round(push * 1.25));
  }
  if (target <= 20 && b2bPressure >= 0.35) {
    const pull = Math.max(1, Math.round((b2bPressure + b2bImmediate) * 1.8));
    aheadLo = Math.max(1, aheadLo - pull);
    aheadMid = Math.max(aheadLo, aheadMid - Math.round(pull * 0.8));
  }

  const adaptiveCap = Math.max(Math.round(targetData.gapQ.q95 * 1.6), Math.round((1 / Math.max(0.000001, p1)) * 1.2), 8);
  aheadLo = clamp(aheadLo, 1, adaptiveCap);
  aheadMid = clamp(aheadMid, aheadLo, adaptiveCap + Math.max(2, spread));

  const span = Math.max(2, spread + 1);
  const lo = currentRound + aheadLo;
  const hi = Math.max(currentRound + aheadMid, lo + span - 1);

  const minP1 = clamp(calibration?.minP1 ?? (0.04 + (targetData.baseRate * 0.6)), 0.01, 0.95);
  const minConfidence = clamp(calibration?.minConfidence ?? 0.4, 0.1, 0.95);
  const confidence = clamp((0.2 + (0.46 * pSoon) + (0.18 * knn.support) + (0.12 * reliability) + (0.04 * (1 - trendDownRisk))), 0.05, 0.99);
  const actionScore = clamp((0.55 * p1) + (0.45 * confidence), 0, 1);
  let active = actionScore >= ((minP1 + minConfidence) * 0.5);
  if (target >= 50 && whiteRisk >= 0.8 && whiteRelease <= 0.25) active = false;
  if (target <= 20 && b2bPressure >= 0.55 && p1 >= (minP1 * 0.8)) active = true;

  return {
    target,
    lo,
    hi,
    roundWhenMade: currentRound,
    suspended: !active,
    confidence: roundNum(confidence, 6),
    eta: {
      pHit1: roundNum(p1, 6),
      pHitSoon: roundNum(pSoon, 6),
      horizonSoon,
      q25: roundNum(q25Raw, 3),
      q50: roundNum(q50Raw, 3),
      q75: roundNum(q75Raw, 3),
      aheadLo,
      aheadHi: hi - currentRound,
      spread: roundNum(spread, 3),
      minP1: roundNum(minP1, 6),
      minConfidence: roundNum(minConfidence, 6),
      actionScore: roundNum(actionScore, 6),
      confidenceBand: confidenceBandWord(confidence),
      calibrationShift: roundNum(calShift, 6),
      calibrationSpanMultiplier: roundNum(calSpan, 6),
      calibrationSample: Number(calibration?.sample || 0),
      reliability: roundNum(reliability, 6),
      whiteClusterSeverity: roundNum(whiteRisk, 6),
      whiteReleaseSignal: roundNum(whiteRelease, 6),
      whiteRun: Number(whiteRun || 0),
      trendDownRisk: roundNum(trendDownRisk, 6),
      b2bPressure: roundNum(b2bPressure, 6),
      b2bImmediate: roundNum(b2bImmediate, 6),
      knnSupport: roundNum(knn.support, 6),
      knnEffectiveSample: roundNum(knn.effectiveSample, 3),
      hazardP1: roundNum(p1Hazard, 6),
      baseRate: roundNum(baseRate, 6),
      blend: { knn: roundNum(wKnn, 6), hazard: roundNum(wHaz, 6), base: roundNum(wBase, 6) },
      aiProbability: roundNum(actionScore, 6),
      aiConfidence: roundNum(confidence, 6),
      aiScore: roundNum(actionScore, 6),
      aiSource: 'v8-unified-supervised',
      waitingModelVersion: 'v8-unified-supervised',
      suspended: !active,
    },
  };
}
// === UPGRADE END ===

function normalizeLockInput(input) {
  if (!input) return null;
  const eta = input.eta || input.eta_json || null;
  return {
    lo: Number(input.lo),
    hi: Number(input.hi),
    roundWhenMade: Number(input.roundWhenMade ?? input.round_when_made),
    generation: Number(input.generation || 1),
    suspended: Boolean(input.suspended ?? eta?.suspended),
    eta,
  };
}

function buildUiTarget(target, lock, status, currentRound, previousOutcome = null) {
  const lo = Number(lock.lo);
  const hi = Number(lock.hi);
  const eta = lock.eta || {};
  return {
    target,
    targetLabel: `${target}x`,
    status,
    confidence: clamp(Number(lock.confidence ?? eta.aiConfidence ?? eta.confidence ?? 0.5), 0, 1),
    window: {
      lo,
      hi,
      span: Math.max(1, hi - lo + 1),
      aheadLo: Math.max(0, lo - currentRound),
      aheadHi: Math.max(0, hi - currentRound),
      roundsUntilWindow: Math.max(0, lo - currentRound),
      roundsLeftInWindow: Math.max(0, hi - currentRound),
    },
    signals: {
      pHit1: Number(eta.pHit1 || 0),
      pHitSoon: Number(eta.pHitSoon || 0),
      quickHit: Number(eta.pHit1 || 0),
      hardGapImpulse: Number(eta.trendDownRisk || 0),
      whiteClusterSeverity: Number(eta.whiteClusterSeverity || 0),
      whiteReleaseSignal: Number(eta.whiteReleaseSignal || 0),
      b2bPressure: Number(eta.b2bPressure || 0),
      aiProbability: Number(eta.aiProbability || eta.actionScore || 0),
      aiConfidence: Number(eta.aiConfidence || 0),
      aiScore: Number(eta.aiScore || eta.aiProbability || 0),
    },
    previousOutcome,
  };
}

function buildAlertSummary(targetsOut, state) {
  const rows = Array.isArray(targetsOut) ? targetsOut : [];
  const ranked = rows
    .map((row) => ({
      target: row.target,
      targetLabel: row.targetLabel,
      score: clamp(Number(row?.signals?.aiScore || 0), 0, 1),
      aheadLo: Number(row?.window?.aheadLo || 0),
      aheadHi: Number(row?.window?.aheadHi || 0),
      status: row?.status || 'unknown',
    }))
    .sort((a, b) => (b.score - a.score) || (a.aheadLo - b.aheadLo));
  const top = ranked[0] || null;
  const whiteRun = state.whiteRunAt[state.n - 1] || 0;
  const white = state.whiteModel.estimate(whiteRun);
  const whitePressure = clamp((0.62 * white.continueProb) + (0.38 * sigmoid(whiteRun - state.whiteRunQ85)), 0, 1);
  return {
    dominantSignal: top && top.score >= 0.45 ? (top.target <= 20 ? 'b2b' : 'high-target') : (whitePressure >= 0.55 ? 'white_cluster' : 'balanced'),
    topCandidate: top,
    whiteCluster: {
      run: Number(whiteRun),
      risk: roundNum(white.continueProb, 6),
      release: roundNum(white.reboundProb, 6),
      pressure: roundNum(whitePressure, 6),
      sample: Number(white.sample || 0),
    },
  };
}

function computeLockedRangePredictions(rounds, existingLocksRaw = {}, options = {}) {
  // === v8 TRUE SUPERVISED LEARNING & SIMPLIFIED UPGRADE START ===
  // Justification: closed-loop lock lifecycle; resolve prior lock, save outcome, rebuild next lock in same tick.
  const cleanRounds = normalizeRounds(rounds);
  if (!cleanRounds.length) {
    return {
      model: 'range-lock-v8-unified',
      generatedAt: new Date().toISOString(),
      asOfRound: null,
      sampleSize: 0,
      targets: [],
      locksToSave: {},
      resolvedHistory: [],
      calibration: {},
      aiPredictor: { dominantSignal: 'none' },
      summary: { pending: 0, waiting: 0, windowOpen: 0, relocked: 0, sampleSize: 0 },
      settings: { adaptive: true, pureData: true, supervisedLearning: true },
    };
  }

  const state = buildGlobalState(cleanRounds);
  const currentRound = cleanRounds[cleanRounds.length - 1].roundId;
  const historyRows = Array.isArray(options?.historyRows) ? options.historyRows : [];
  const calibration = buildHistoryCalibration(historyRows);

  const locksToSave = {};
  const resolvedHistory = [];
  const targetsOut = [];
  let waitingCount = 0;
  let openCount = 0;
  let relockedCount = 0;
  let pendingCount = 0;

  for (const target of TARGETS) {
    const key = String(target);
    const targetData = buildTargetData(state, target);
    const existing = normalizeLockInput(existingLocksRaw[key]);
    const evalResult = evaluateExistingLock(existing, targetData.hitRoundIds, currentRound);
    const candidate = buildTargetCandidate(state, targetData, calibration[target]);

    let lockToUse = existing;
    let status = evalResult.status;
    let previousOutcome = null;

    if (!existing || evalResult.resolved) {
      if (existing && evalResult.resolved) {
        previousOutcome = {
          outcome: evalResult.outcome,
          hitRound: evalResult.hitRound,
          lo: existing.lo,
          hi: existing.hi,
          generation: existing.generation,
        };
        resolvedHistory.push({
          target: `${target}x`,
          minMult: target,
          outcome: evalResult.outcome,
          lo: Number(existing.lo),
          hi: Number(existing.hi),
          hitRound: evalResult.hitRound,
          generation: Number(existing.generation || 1),
          confidence: Number(existing?.eta?.aiProbability ?? existing?.eta?.confidence ?? candidate.confidence ?? null),
        });
      }

      const generation = existing ? Number(existing.generation || 1) + 1 : 1;
      lockToUse = {
        lo: Number(candidate.lo),
        hi: Number(candidate.hi),
        roundWhenMade: Number(candidate.roundWhenMade),
        generation,
        suspended: Boolean(candidate.suspended),
        confidence: Number(candidate.confidence || 0.5),
        eta: candidate.eta,
      };
      status = candidate.suspended ? 'waiting' : 'locked';
      relockedCount += 1;
    } else {
      lockToUse = {
        ...existing,
        confidence: Number(candidate.confidence || existing?.confidence || existing?.eta?.confidence || 0.5),
        eta: {
          ...(existing?.eta || {}),
          ...(candidate?.eta || {}),
          suspended: Boolean(existing?.suspended),
        },
      };
      status = evalResult.status;
    }

    if ((status === 'waiting' || status === 'pending') && Number(lockToUse.lo) <= currentRound) status = 'window-open';

    if (status === 'pending') pendingCount += 1;
    if (status === 'waiting') waitingCount += 1;
    if (status === 'window-open') openCount += 1;

    locksToSave[key] = {
      lo: Number(lockToUse.lo),
      hi: Number(lockToUse.hi),
      roundWhenMade: Number(lockToUse.roundWhenMade),
      generation: Number(lockToUse.generation || 1),
      suspended: Boolean(lockToUse.suspended),
      eta: lockToUse.eta || null,
    };

    targetsOut.push(buildUiTarget(target, lockToUse, status, currentRound, previousOutcome));
  }

  targetsOut.sort((a, b) => a.target - b.target);
  const alertSummary = buildAlertSummary(targetsOut, state);

  return {
    model: 'range-lock-v8-unified',
    generatedAt: new Date().toISOString(),
    asOfRound: currentRound,
    sampleSize: state.n,
    targets: targetsOut,
    locksToSave,
    resolvedHistory,
    calibration,
    aiPredictor: { mode: alertSummary.dominantSignal, top: alertSummary.topCandidate, whiteCluster: alertSummary.whiteCluster },
    alertSummary,
    settings: {
      adaptive: true,
      pureData: true,
      supervisedLearning: true,
      immediateRecalcOnResolve: true,
      noSimulation: true,
      modelVersion: 'v8-unified-supervised',
    },
    summary: {
      pending: pendingCount,
      waiting: waitingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: state.n,
    },
  };
  // === UPGRADE END ===
}

function computeThresholdSnapshot(cleanRounds, threshold) {
  const state = buildGlobalState(cleanRounds);
  const data = buildTargetData(state, threshold);
  const calibration = { reliability: 0.5, shift: 0, spanMultiplier: 1, minP1: clamp(data.baseRate * 0.7, 0.01, 0.8), minConfidence: 0.35 };
  const candidate = buildTargetCandidate(state, data, calibration);
  const p1 = clamp(Number(candidate?.eta?.pHit1 || data.baseRate), 0, 1);
  const p3 = clamp(1 - ((1 - p1) ** 3), 0, 1);
  const p5 = clamp(1 - ((1 - p1) ** 5), 0, 1);
  return {
    target: threshold,
    gapNow: Number(data.currentGap || 0),
    p1: roundNum(p1, 6),
    p3: roundNum(Math.max(p3, p1), 6),
    p5: roundNum(Math.max(p5, p3, p1), 6),
    expectedGap: roundNum(Math.max(1, 1 / Math.max(0.000001, p1)), 2),
    softGapPressure: roundNum(clamp((data.currentGap - data.gapQ.q50) / Math.max(1, data.gapQ.q75 - data.gapQ.q50 + 1), 0, 1), 6),
    hardGapPressure: roundNum(clamp((data.currentGap - data.gapQ.q75) / Math.max(1, data.gapQ.q95 - data.gapQ.q75 + 1), 0, 1), 6),
  };
}

function buildPredictionReport(rounds) {
  // === v8 TRUE SUPERVISED LEARNING & SIMPLIFIED UPGRADE START ===
  // Justification: dashboard probabilities use the same unified v8 learner (no separate stale model path).
  const cleanRounds = normalizeRounds(rounds);
  if (!cleanRounds.length) {
    return {
      model: 'unified-v8-report',
      generatedAt: new Date().toISOString(),
      asOfRound: null,
      sampleSize: 0,
      expectedMultiplier: 0,
      expectedMedian: 0,
      expectedP75: 0,
      expectedP90: 0,
      predictedBucket: { ...BUCKETS[0], probability: 0, confidence: 0, confidenceBand: 'NONE' },
      bucketProbabilities: BUCKETS.map((b) => ({ ...b, probability: 0 })),
      targetProbabilities: REPORT_THRESHOLDS.map((t) => ({ target: t, gapNow: 0, p1: 0, p3: 0, p5: 0, expectedGap: null })),
      diagnostics: {},
      cashoutPlan: null,
      similarPatterns: [],
      signals: [],
    };
  }

  const mult = cleanRounds.map(r => Number(r.multiplier));
  const sorted = [...mult].sort((a, b) => a - b);
  const counts = new Array(BUCKETS.length).fill(0);
  for (const m of mult) counts[bucketIndex(m)] += 1;
  const bucketProb = normalizeDistribution(counts);
  const topIdx = bucketProb.reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0);
  const expectedMean = bucketProb.reduce((s, p, i) => s + (p * bucketMidpoint(BUCKETS[i])), 0);

  const state = buildGlobalState(cleanRounds);
  const whiteRun = state.whiteRunAt[state.n - 1] || 0;
  const white = state.whiteModel.estimate(whiteRun);
  const trend = prefixRate(state.prefLog, state.n - 1, 32) - prefixRate(state.prefLog, state.n - 1, 120);

  const targetProbabilities = REPORT_THRESHOLDS.map((t) => computeThresholdSnapshot(cleanRounds, t));
  const pSafe = targetProbabilities.find(x => x.target === 2)?.p1 ?? 0.5;
  const pBalanced = targetProbabilities.find(x => x.target === 5)?.p1 ?? 0.3;
  const pAgg = targetProbabilities.find(x => x.target === 10)?.p1 ?? 0.12;

  const cashoutPlan = {
    safe: { target: 1.8, hitChance: roundNum(pSafe, 6), edge: roundNum((1.8 * pSafe) - 1, 6) },
    balanced: { target: 3.2, hitChance: roundNum(pBalanced, 6), edge: roundNum((3.2 * pBalanced) - 1, 6) },
    aggressive: { target: 7, hitChance: roundNum(pAgg, 6), edge: roundNum((7 * pAgg) - 1, 6) },
  };
  const rec = [cashoutPlan.safe, cashoutPlan.balanced, cashoutPlan.aggressive].map((x, idx) => ({ ...x, idx })).sort((a, b) => b.edge - a.edge)[0];
  const labels = ['SAFE', 'BALANCED', 'AGGRESSIVE'];
  cashoutPlan.recommended = rec;
  cashoutPlan.recommendedLabel = labels[rec.idx] || 'BALANCED';
  cashoutPlan.zoneLow = roundNum(rec.target * 0.92, 3);
  cashoutPlan.zoneHigh = roundNum(rec.target * 1.08, 3);
  cashoutPlan.reason = 'Unified v8 data-driven expected value ranking.';

  const confidence = clamp((0.32 * bucketProb[topIdx]) + (0.24 * (1 - clamp(stddev(mult.slice(-120).map(safeLog)) / 0.8, 0, 1))) + (0.18 * (1 - white.continueProb)) + (0.16 * clamp(pBalanced, 0, 1)) + (0.1 * clamp(1 + trend, 0, 1)), 0.05, 0.99);

  return {
    model: 'unified-v8-report',
    generatedAt: new Date().toISOString(),
    asOfRound: cleanRounds[cleanRounds.length - 1].roundId,
    sampleSize: cleanRounds.length,
    expectedMultiplier: roundNum(expectedMean, 4),
    expectedMedian: roundNum(quantileFromSorted(sorted, 0.5), 4),
    expectedP75: roundNum(quantileFromSorted(sorted, 0.75), 4),
    expectedP90: roundNum(quantileFromSorted(sorted, 0.9), 4),
    predictedBucket: {
      ...BUCKETS[topIdx],
      probability: roundNum(bucketProb[topIdx], 6),
      confidence: roundNum(confidence, 6),
      confidenceBand: confidenceBandWord(confidence),
    },
    bucketProbabilities: BUCKETS.map((b, i) => ({ ...b, probability: roundNum(bucketProb[i], 6) })),
    targetProbabilities,
    diagnostics: {
      whiteCluster: { run: Number(whiteRun), risk: roundNum(white.continueProb, 6), release: roundNum(white.reboundProb, 6) },
      trend: { score: roundNum(trend, 6) },
    },
    cashoutPlan,
    similarPatterns: [],
    signals: [
      `White cluster risk ${roundNum(white.continueProb * 100, 2)}%`,
      `White release signal ${roundNum(white.reboundProb * 100, 2)}%`,
      `Trend score ${roundNum(trend, 4)}`,
    ],
  };
  // === UPGRADE END ===
}

module.exports = {
  TARGETS,
  buildPredictionReport,
  computeLockedRangePredictions,
};
