'use strict';

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const WINDOW_SPAN_PRIOR = {
  5: 3,
  10: 6,
  20: 10,
  50: 17,
  100: 25,
  500: 50,
  1000: 60,
};
const WINDOW_AHEAD_CAP = {
  5: 9,
  10: 12,
  20: 17,
  50: 30,
  100: 42,
  500: 85,
  1000: 110,
};
const ACTIVATE_MIN_CONF = {
  5: 0.36,
  10: 0.4,
  20: 0.45,
  50: 0.52,
  100: 0.58,
  500: 0.66,
  1000: 0.72,
};
const ACTIVATE_MIN_P1 = {
  5: 0.12,
  10: 0.1,
  20: 0.08,
  50: 0.06,
  100: 0.05,
  500: 0.03,
  1000: 0.02,
};
const WAITING_MODEL_VERSION = 'v11-adaptive-live';

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function roundNum(v, digits = 4) {
  if (!Number.isFinite(Number(v))) return 0;
  return Number(Number(v).toFixed(digits));
}

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function stddev(arr, avg = null) {
  if (arr.length <= 1) return 0;
  const m = avg == null ? mean(arr) : avg;
  const variance = arr.reduce((s, v) => s + ((v - m) ** 2), 0) / arr.length;
  return Math.sqrt(variance);
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const qq = clamp(q, 0, 1);
  const idx = (sorted.length - 1) * qq;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + (sorted[hi] * w);
}

function weightedQuantile(items, q) {
  if (!items.length) return 1;
  const qq = clamp(q, 0, 1);
  const sorted = [...items].sort((a, b) => a.value - b.value);
  const total = sorted.reduce((s, x) => s + x.weight, 0);
  if (!total) return sorted[Math.floor(sorted.length * qq)].value;
  let acc = 0;
  for (const item of sorted) {
    acc += item.weight;
    if ((acc / total) >= qq) return item.value;
  }
  return sorted[sorted.length - 1].value;
}

function weightedMean(items) {
  if (!items.length) return 0;
  let num = 0;
  let den = 0;
  for (const item of items) {
    num += item.value * item.weight;
    den += item.weight;
  }
  return den > 0 ? (num / den) : 0;
}

function wilsonBounds(wins, losses, z = 1.96) {
  const n = wins + losses;
  if (!n) return { low: 0, mid: 0.5, high: 1 };
  const p = wins / n;
  const z2 = z * z;
  const denom = 1 + (z2 / n);
  const center = (p + (z2 / (2 * n))) / denom;
  const margin = (z / denom) * Math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)));
  return {
    low: clamp(center - margin, 0, 1),
    mid: clamp(center, 0, 1),
    high: clamp(center + margin, 0, 1),
  };
}

function safeLog(v) {
  return Math.log(Math.max(1, Number(v) || 1));
}

function findFirstInRange(sortedRoundIds, lo, hi) {
  if (!sortedRoundIds || !sortedRoundIds.length || lo > hi) return null;
  let left = 0;
  let right = sortedRoundIds.length - 1;
  let pos = sortedRoundIds.length;
  while (left <= right) {
    const mid = (left + right) >> 1;
    if (sortedRoundIds[mid] >= lo) {
      pos = mid;
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  if (pos >= sortedRoundIds.length) return null;
  const v = sortedRoundIds[pos];
  return v <= hi ? v : null;
}

function buildPrefix(arr) {
  const pref = new Array(arr.length + 1).fill(0);
  for (let i = 0; i < arr.length; i++) pref[i + 1] = pref[i] + arr[i];
  return pref;
}

function rangeMean(pref, lo, hi) {
  if (hi < lo) return 0;
  const l = clamp(lo, 0, pref.length - 1);
  const r = clamp(hi + 1, 0, pref.length - 1);
  const len = Math.max(1, r - l);
  return (pref[r] - pref[l]) / len;
}

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: learn empirical hazard P(hit next | current gap) from real history to catch b2b + regime transitions.
function buildHazardCurve(gaps, rounds, target, maxGapBucket) {
  const maxGap = clamp(Math.round(maxGapBucket || 180), 20, 600);
  const obs = new Array(maxGap + 1).fill(0);
  const hits = new Array(maxGap + 1).fill(0);

  let totalObs = 0;
  let totalHits = 0;
  for (let i = 0; i < gaps.length - 1; i++) {
    const g = clamp(Math.round(gaps[i] || 0), 0, maxGap);
    obs[g] += 1;
    totalObs += 1;
    if (rounds[i + 1].multiplier >= target) {
      hits[g] += 1;
      totalHits += 1;
    }
  }

  const globalP = totalObs > 0 ? (totalHits / totalObs) : 0;
  const hazard = new Array(maxGap + 1).fill(clamp(globalP, 0.000001, 0.95));

  for (let g = 0; g <= maxGap; g++) {
    let o = 0;
    let h = 0;
    for (let d = -2; d <= 2; d++) {
      const k = g + d;
      if (k < 0 || k > maxGap) continue;
      const w = d === 0 ? 1.8 : (Math.abs(d) === 1 ? 1.1 : 0.6);
      o += obs[k] * w;
      h += hits[k] * w;
    }
    const alpha = 8;
    hazard[g] = clamp((h + (alpha * globalP)) / Math.max(1e-6, o + alpha), 0.000001, 0.95);
  }

  return {
    hazard,
    maxGap,
    globalP: clamp(globalP, 0.000001, 0.95),
    totalObs,
  };
}
// === UPGRADE END ===

function preprocess(rounds) {
  const cleanRounds = rounds
    .map(r => ({
      roundId: Number(r.roundId),
      multiplier: Number(r.multiplier),
      timestamp: Number(r.timestamp) || Date.now(),
    }))
    .filter(r => Number.isFinite(r.roundId) && Number.isFinite(r.multiplier) && r.multiplier > 0)
    .sort((a, b) => a.roundId - b.roundId);

  const n = cleanRounds.length;
  const logs = cleanRounds.map(r => safeLog(r.multiplier));
  const tokens = cleanRounds.map((r) => {
    const m = r.multiplier;
    if (m < 1.2) return 0;
    if (m < 1.5) return 1;
    if (m < 2) return 2;
    if (m < 3) return 3;
    if (m < 5) return 4;
    if (m < 10) return 5;
    if (m < 20) return 6;
    if (m < 50) return 7;
    return 8;
  });

  const prefLog = buildPrefix(logs);
  const sqLogs = logs.map(v => v * v);
  const prefSq = buildPrefix(sqLogs);
  const under2Flags = cleanRounds.map(r => (r.multiplier < 2 ? 1 : 0));
  const prefUnder2 = buildPrefix(under2Flags);

  const lowStreak = new Array(n).fill(0);
  const whiteStreak = new Array(n).fill(0);
  const highStreak = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    lowStreak[i] = cleanRounds[i].multiplier < 2 ? 1 + (i > 0 ? lowStreak[i - 1] : 0) : 0;
    whiteStreak[i] = cleanRounds[i].multiplier < 3 ? 1 + (i > 0 ? whiteStreak[i - 1] : 0) : 0;
    highStreak[i] = cleanRounds[i].multiplier >= 10 ? 1 + (i > 0 ? highStreak[i - 1] : 0) : 0;
  }

  const streakSorted = [...lowStreak].sort((a, b) => a - b);
  const under2Window = clamp(Math.round(Math.sqrt(Math.max(1, n)) + 8), 16, 120);
  const under2Rates = [];
  for (let i = under2Window - 1; i < n; i++) {
    under2Rates.push(rangeMean(prefUnder2, i - under2Window + 1, i));
  }
  const under2Sorted = [...under2Rates].sort((a, b) => a - b);
  const whiteProfile = {
    lowQ85: quantile(streakSorted, 0.85),
    lowQ95: quantile(streakSorted, 0.95),
    under2Q85: quantile(under2Sorted, 0.85),
    under2Q95: quantile(under2Sorted, 0.95),
    under2Window,
  };

  const gapMaps = {};
  const nextHitMaps = {};
  const hitRoundIds = {};
  const gapStats = {};

  for (const target of TARGETS) {
    const gap = new Array(n).fill(0);
    const nextHit = new Array(n).fill(null);
    const hits = [];
    let last = -1;
    for (let i = 0; i < n; i++) {
      if (cleanRounds[i].multiplier >= target) {
        last = i;
        hits.push(cleanRounds[i].roundId);
      }
      gap[i] = last === -1 ? i + 1 : i - last;
    }
    let next = null;
    for (let i = n - 1; i >= 0; i--) {
      nextHit[i] = next;
      if (cleanRounds[i].multiplier >= target) next = i;
    }

    const interGaps = [];
    for (let i = 1; i < hits.length; i++) interGaps.push(hits[i] - hits[i - 1]);
    const sorted = [...interGaps].sort((a, b) => a - b);
    const avg = sorted.length ? mean(sorted) : 0;
    const sd = sorted.length ? stddev(sorted, avg) : 0;
    const hazardMaxGap = Math.max(
      24,
      Math.round((sorted.length ? quantile(sorted, 0.99) : Math.max(12, avg || 12)) * 2.2 + 8)
    );
    const hazardCurve = buildHazardCurve(gap, cleanRounds, target, hazardMaxGap);

    gapMaps[target] = gap;
    nextHitMaps[target] = nextHit;
    hitRoundIds[target] = hits;
    gapStats[target] = {
      count: sorted.length,
      mean: roundNum(avg, 3),
      sd: roundNum(sd, 3),
      q10: roundNum(sorted.length ? quantile(sorted, 0.1) : 0, 3),
      q25: roundNum(sorted.length ? quantile(sorted, 0.25) : 0, 3),
      q50: roundNum(sorted.length ? quantile(sorted, 0.5) : 0, 3),
      q75: roundNum(sorted.length ? quantile(sorted, 0.75) : 0, 3),
      q90: roundNum(sorted.length ? quantile(sorted, 0.9) : 0, 3),
      q95: roundNum(sorted.length ? quantile(sorted, 0.95) : 0, 3),
      q99: roundNum(sorted.length ? quantile(sorted, 0.99) : 0, 3),
      interGaps,
      hazardByGap: hazardCurve.hazard,
      hazardGlobal: roundNum(hazardCurve.globalP, 6),
      hazardMaxGap: hazardCurve.maxGap,
      hazardSample: Number(hazardCurve.totalObs || 0),
    };
  }

  function rangeStd(lo, hi) {
    if (hi < lo) return 0;
    const l = clamp(lo, 0, n - 1);
    const h = clamp(hi, 0, n - 1);
    const m = rangeMean(prefLog, l, h);
    const len = (h - l + 1);
    const sq = (prefSq[h + 1] - prefSq[l]) / Math.max(1, len);
    const variance = Math.max(0, sq - (m * m));
    return Math.sqrt(variance);
  }

  function trendRegimeAt(idx) {
    const s1 = rangeMean(prefLog, idx - 11, idx);
    const s0 = rangeMean(prefLog, idx - 23, idx - 12);
    const l1 = rangeMean(prefLog, idx - 39, idx);
    const l0 = rangeMean(prefLog, idx - 79, idx - 40);
    const trend = ((s1 - s0) * 0.65) + ((l1 - l0) * 0.35);
    const volNow = rangeStd(idx - 29, idx);
    const volBase = rangeStd(idx - 159, idx - 30) || 1;
    const volRatio = volNow / volBase;
    const under2Rate = rangeMean(prefUnder2, idx - whiteProfile.under2Window + 1, idx);
    const lowNow = lowStreak[clamp(idx, 0, n - 1)];
    let regime = 'balanced';
    if (lowNow >= whiteProfile.lowQ85 && under2Rate >= whiteProfile.under2Q85 && trend <= 0.01) regime = 'white';
    else if (trend <= -0.04 && volRatio <= 1.02) regime = 'compression';
    else if (trend >= 0.035 && volRatio >= 1.02) regime = 'expansion';
    else if (volRatio >= 1.18) regime = 'chaotic';
    else if (trend <= -0.02) regime = 'soft-down';
    else if (trend >= 0.02) regime = 'soft-up';
    return { trend, volRatio, regime, under2Rate };
  }

  const trendByIdx = new Array(n).fill(0);
  const volRatioByIdx = new Array(n).fill(1);
  const regimeByIdx = new Array(n).fill('balanced');
  const under2RateByIdx = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const tr = trendRegimeAt(i);
    trendByIdx[i] = tr.trend;
    volRatioByIdx[i] = tr.volRatio;
    regimeByIdx[i] = tr.regime;
    under2RateByIdx[i] = tr.under2Rate;
  }

  for (const target of TARGETS) {
    const stat = gapStats[target] || {};
    const inter = Array.isArray(stat.interGaps) ? stat.interGaps : [];
    const totalInter = inter.length;
    const recentSpan = clamp(Math.round(Math.sqrt(Math.max(1, n)) * 9), 120, 3000);
    const recentInter = inter.slice(Math.max(0, totalInter - recentSpan));

    const calcQuick = (arr) => {
      if (!arr.length) return { le1: 0, le2: 0, le3: 0 };
      let c1 = 0;
      let c2 = 0;
      let c3 = 0;
      for (const g of arr) {
        if (g <= 1) c1++;
        if (g <= 2) c2++;
        if (g <= 3) c3++;
      }
      return {
        le1: c1 / arr.length,
        le2: c2 / arr.length,
        le3: c3 / arr.length,
      };
    };

    const byRegime = {};
    for (let i = 0; i < n - 1; i++) {
      const reg = regimeByIdx[i] || 'balanced';
      const bucket = byRegime[reg] || { obs: 0, h1: 0, h2: 0, h3: 0 };
      bucket.obs += 1;
      if (cleanRounds[i + 1].multiplier >= target) bucket.h1 += 1;
      if (i + 2 < n && (cleanRounds[i + 1].multiplier >= target || cleanRounds[i + 2].multiplier >= target)) bucket.h2 += 1;
      if (i + 3 < n && (cleanRounds[i + 1].multiplier >= target || cleanRounds[i + 2].multiplier >= target || cleanRounds[i + 3].multiplier >= target)) bucket.h3 += 1;
      byRegime[reg] = bucket;
    }

    const byRegimeOut = {};
    for (const [reg, b] of Object.entries(byRegime)) {
      const obs = Math.max(1, b.obs);
      byRegimeOut[reg] = {
        obs: b.obs,
        p1: b.h1 / obs,
        p2: b.h2 / obs,
        p3: b.h3 / obs,
      };
    }

    stat.quick = {
      global: calcQuick(inter),
      recent: calcQuick(recentInter),
      byRegime: byRegimeOut,
    };
    gapStats[target] = stat;
  }

  function stateAt(idx, target) {
    const seq = [];
    for (let i = idx - 7; i <= idx; i++) seq.push(tokens[clamp(i, 0, n - 1)]);
    const ii = clamp(idx, 0, n - 1);
    return {
      gapT: gapMaps[target][ii],
      gap5: gapMaps[5][ii],
      gap10: gapMaps[10][ii],
      gap20: gapMaps[20][ii],
      gap50: gapMaps[50][ii],
      gap100: gapMaps[100][ii],
      trend: trendByIdx[ii],
      volRatio: volRatioByIdx[ii],
      regime: regimeByIdx[ii],
      under2Rate: under2RateByIdx[ii],
      lowStreak: lowStreak[ii],
      whiteStreak: whiteStreak[ii],
      highStreak: highStreak[ii],
      seq,
    };
  }

  return {
    rounds: cleanRounds,
    n,
    prefUnder2,
    gapMaps,
    nextHitMaps,
    hitRoundIds,
    gapStats,
    whiteProfile,
    trendByIdx,
    volRatioByIdx,
    regimeByIdx,
    under2RateByIdx,
    whiteStreak,
    stateAt,
  };
}

function stateDistance(a, b, target) {
  let d = 0;
  d += 1.35 * Math.abs(Math.log1p(a.gapT) - Math.log1p(b.gapT));
  d += 0.22 * Math.abs(Math.log1p(a.gap5) - Math.log1p(b.gap5));
  d += 0.26 * Math.abs(Math.log1p(a.gap10) - Math.log1p(b.gap10));
  d += 0.24 * Math.abs(Math.log1p(a.gap20) - Math.log1p(b.gap20));
  d += 0.2 * Math.abs(Math.log1p(a.gap50) - Math.log1p(b.gap50));
  if (target >= 100) d += 0.35 * Math.abs(Math.log1p(a.gap100) - Math.log1p(b.gap100));
  d += 2.9 * Math.abs(a.trend - b.trend);
  d += 1.5 * Math.abs(a.volRatio - b.volRatio);
  d += 0.9 * Math.abs(a.under2Rate - b.under2Rate);
  d += 0.06 * Math.abs(a.lowStreak - b.lowStreak);
  d += 0.05 * Math.abs(a.whiteStreak - b.whiteStreak);
  d += 0.05 * Math.abs(a.highStreak - b.highStreak);

  let seqMismatch = 0;
  for (let i = 0; i < a.seq.length; i++) {
    const w = 0.6 + ((i + 1) / a.seq.length);
    seqMismatch += w * Math.abs(a.seq[i] - b.seq[i]);
  }
  d += 0.12 * (seqMismatch / a.seq.length);
  return d;
}

function collectNeighbors(pre, target, currentIdx, currentState, options = {}) {
  const nextMap = pre.nextHitMaps[target];
  const items = [];
  const stats = pre.gapStats[target] || {};
  const maxAhead = Math.max(12, Math.round((stats.q99 || stats.q95 || stats.q90 || stats.mean || 12) * 6));
  const baseLookback = target >= 500 ? 300000 : (target >= 100 ? 220000 : 170000);
  const maxLookback = Math.max(1200, Number(options.maxLookback || baseLookback));
  const start = Math.max(120, currentIdx - maxLookback);
  const rawSpan = Math.max(1, currentIdx - start);
  const maxScan = Math.max(3000, Number(options.maxScan || 70000));
  const stride = Math.max(1, Math.floor(rawSpan / maxScan));

  for (let idx = start; idx < currentIdx; idx += stride) {
    const nextIdx = nextMap[idx];
    if (nextIdx == null) continue;
    const ttn = nextIdx - idx;
    if (ttn <= 0 || ttn > maxAhead) continue;

    const st = pre.stateAt(idx, target);
    const dist = stateDistance(currentState, st, target);
    const recency = 0.35 + (0.65 * ((idx + 1) / Math.max(1, currentIdx)) ** 1.4);
    const regimeBoost = st.regime === currentState.regime ? 1.12 : 1;
    const weight = Math.exp(-dist * 0.9) * recency * regimeBoost;
    if (weight < 0.002) continue;

    items.push({ value: ttn, weight, dist });
  }

  items.sort((a, b) => b.weight - a.weight);
  const keepMin = Math.max(80, Number(options.keepMin || 140));
  const keepMax = Math.max(keepMin, Number(options.keepMax || 1700));
  const keep = clamp(Math.round(Math.sqrt(Math.max(1, currentIdx)) * 8.5), keepMin, keepMax);
  return items.slice(0, keep);
}

function gapPressure(currentGap, stats) {
  const q75 = stats.q75 || stats.q50 || Math.max(2, stats.mean || 2);
  const q90 = stats.q90 || (q75 + 2);
  const softDen = Math.max(2, (q90 - q75) + 1);
  const hardDen = Math.max(4, (q90 * 0.35) + 2);
  const soft = clamp((currentGap - q75) / softDen, 0, 1);
  const hard = clamp((currentGap - q90) / hardDen, 0, 1);
  return { soft, hard };
}

function hazardEta(target, currentState, stats, pressure) {
  const meanGap = Math.max(2, Number(stats.mean || stats.q50 || 2));
  const baseFromMean = clamp(1 / meanGap, 0.00008, 0.45);

  const hazardByGap = Array.isArray(stats.hazardByGap) ? stats.hazardByGap : null;
  const hazardMaxGap = Number(stats.hazardMaxGap || 0);
  const gapIdx = clamp(Math.round(currentState.gapT || 0), 0, Math.max(0, hazardMaxGap));
  const pGap = (hazardByGap && hazardByGap.length) ? Number(hazardByGap[gapIdx]) : NaN;
  const pPrev = (hazardByGap && hazardByGap.length) ? Number(hazardByGap[Math.max(0, gapIdx - 1)]) : NaN;
  const pNext = (hazardByGap && hazardByGap.length) ? Number(hazardByGap[Math.min(hazardByGap.length - 1, gapIdx + 1)]) : NaN;
  const localHazard = Number.isFinite(pGap)
    ? clamp((0.2 * (Number.isFinite(pPrev) ? pPrev : pGap)) + (0.6 * pGap) + (0.2 * (Number.isFinite(pNext) ? pNext : pGap)), 0.00005, 0.95)
    : NaN;

  const baseP = Number.isFinite(localHazard)
    ? clamp((0.72 * localHazard) + (0.28 * baseFromMean), 0.00005, 0.62)
    : baseFromMean;

  let factor = 1;
  factor *= 1 + (0.38 * pressure.soft) + (0.75 * pressure.hard);

  if (currentState.regime === 'expansion') factor *= target >= 20 ? 1.16 : 1.1;
  if (currentState.regime === 'compression') factor *= target >= 20 ? 0.84 : 0.9;
  if (currentState.regime === 'chaotic') factor *= target >= 50 ? 1.07 : 1.03;
  if (currentState.regime === 'soft-up') factor *= 1.04;
  if (currentState.regime === 'soft-down') factor *= 0.96;

  if (Number.isFinite(localHazard) && Number.isFinite(pPrev)) {
    const hazardSlope = clamp((localHazard - pPrev) * 5.5, -0.45, 0.55);
    factor *= 1 + hazardSlope;
  }

  factor *= 1 + (0.18 * clamp(currentState.trend, -0.6, 0.6));
  factor = clamp(factor, 0.35, 2.2);

  const p = clamp(baseP * factor, 0.00005, 0.62);
  const q = 1 - p;

  const stepFor = (quantileTarget) => {
    if (p >= 0.999) return 1;
    const raw = Math.log(1 - quantileTarget) / Math.log(Math.max(0.000001, q));
    return Math.max(1, raw);
  };

  return {
    pHit1: roundNum(p, 6),
    q20: roundNum(stepFor(0.2), 3),
    q50: roundNum(stepFor(0.5), 3),
    q80: roundNum(stepFor(0.8), 3),
  };
}

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: detect persistent white clusters with gap gating so post-hit b2b windows are not delayed too much.
function whiteClusterSeverity(pre, currentState, target, currentIdx = null) {
  const idx = Number.isFinite(currentIdx) ? currentIdx : (pre.n - 1);
  const p = pre.whiteProfile || {};
  const targetStats = pre.gapStats?.[target] || {};
  const lowQ85 = Number(p.lowQ85 || 0);
  const lowQ95 = Number(p.lowQ95 || lowQ85 + 1);
  const under2Q85 = Number(p.under2Q85 || 0);
  const under2Q95 = Number(p.under2Q95 || Math.max(under2Q85 + 0.01, 0.01));
  const gapQ50 = Number(targetStats.q50 || targetStats.mean || 1);
  const gapQ90 = Number(targetStats.q90 || Math.max(gapQ50 + 1, 2));

  const sLow = clamp((currentState.lowStreak - lowQ85) / Math.max(1, lowQ95 - lowQ85), 0, 1);
  const sRate = clamp((currentState.under2Rate - under2Q85) / Math.max(0.001, under2Q95 - under2Q85), 0, 1);
  const lookback = clamp(Math.round((p.under2Window || 32) * 1.2), 18, 180);
  const recentUnder2 = rangeMean(pre.prefUnder2, idx - lookback + 1, idx);
  const sPersist = clamp((recentUnder2 - under2Q85) / Math.max(0.001, under2Q95 - under2Q85), 0, 1);
  const sTrend = currentState.trend < 0 ? clamp((-currentState.trend) / 0.08, 0, 1) : 0;
  const gapGate = clamp((currentState.gapT - gapQ50) / Math.max(1, gapQ90 - gapQ50), 0, 1);
  const regimeGate = currentState.regime === 'white' ? 1 : (currentState.regime === 'compression' ? 0.6 : 0.4);
  const targetScale = clamp(Math.log(Math.max(2, target)) / Math.log(1000), 0.25, 1);
  const base = (0.33 * sLow) + (0.25 * sRate) + (0.2 * sPersist) + (0.22 * sTrend);
  return clamp(base * regimeGate * targetScale * (0.05 + (0.95 * gapGate)), 0, 1);
}
// === UPGRADE END ===

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: learn model blend weights from real historical prediction errors (no fixed blend bias).
function learnBlendWeights(pre, target, currentIdx, stats) {
  const evalCount = clamp(Math.round(Math.sqrt(Math.max(1, currentIdx)) * 2.5), 45, 180);
  const fromIdx = Math.max(140, currentIdx - (evalCount * 2));
  const step = Math.max(1, Math.floor((currentIdx - fromIdx) / Math.max(1, evalCount)));
  let errNeighbor = 0;
  let errHazard = 0;
  let errPrior = 0;
  let samples = 0;

  for (let idx = fromIdx; idx < currentIdx; idx += step) {
    const nextIdx = pre.nextHitMaps[target][idx];
    if (nextIdx == null || nextIdx <= idx) continue;
    const actualAhead = nextIdx - idx;
    const state = pre.stateAt(idx, target);
    const neighbors = collectNeighbors(pre, target, idx, state, {
      maxLookback: target >= 100 ? 140000 : 100000,
      maxScan: 24000,
      keepMin: 100,
      keepMax: 900,
    });
    const pressure = gapPressure(pre.gapMaps[target][idx], stats);
    const hazard = hazardEta(target, state, stats, pressure);

    const nPred = neighbors.length ? weightedQuantile(neighbors, 0.5) : Math.max(1, stats.q50 || stats.mean || 2);
    const hPred = Math.max(1, hazard.q50 || stats.q50 || stats.mean || 2);
    const pPred = Math.max(1, stats.q50 || stats.mean || 2);

    const a = Math.log1p(actualAhead);
    errNeighbor += Math.abs(a - Math.log1p(nPred));
    errHazard += Math.abs(a - Math.log1p(hPred));
    errPrior += Math.abs(a - Math.log1p(pPred));
    samples++;
  }

  if (!samples) {
    return {
      neighbor: 0.45,
      hazard: 0.35,
      prior: 0.2,
      samples: 0,
      errors: { neighbor: 0, hazard: 0, prior: 0 },
    };
  }

  const eN = errNeighbor / samples;
  const eH = errHazard / samples;
  const eP = errPrior / samples;
  const invN = 1 / Math.max(1e-6, eN);
  const invH = 1 / Math.max(1e-6, eH);
  const invP = 1 / Math.max(1e-6, eP);
  const sumInv = invN + invH + invP;

  return {
    neighbor: invN / sumInv,
    hazard: invH / sumInv,
    prior: invP / sumInv,
    samples,
    errors: {
      neighbor: roundNum(eN, 6),
      hazard: roundNum(eH, 6),
      prior: roundNum(eP, 6),
    },
  };
}
// === UPGRADE END ===

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: explicit b2b/quick-hit detection to avoid late windows after fresh hits.
function quickHitSignal(neighbors, hazard, currentGap, stats) {
  if (!neighbors.length) {
    const h2 = clamp(1 - ((1 - clamp(hazard.pHit1 || 0, 0, 1)) ** 2), 0, 1);
    return clamp(h2, 0, 1);
  }

  let totalW = 0;
  let wLe1 = 0;
  let wLe2 = 0;
  let wLe3 = 0;
  for (const n of neighbors) {
    totalW += n.weight;
    if (n.value <= 1.5) wLe1 += n.weight;
    if (n.value <= 2.5) wLe2 += n.weight;
    if (n.value <= 3.5) wLe3 += n.weight;
  }

  const pLe1 = totalW > 0 ? (wLe1 / totalW) : 0;
  const pLe2 = totalW > 0 ? (wLe2 / totalW) : 0;
  const pLe3 = totalW > 0 ? (wLe3 / totalW) : 0;
  const h2 = clamp(1 - ((1 - clamp(hazard.pHit1 || 0, 0, 1)) ** 2), 0, 1);
  const baseQuick = clamp((0.42 * pLe1) + (0.28 * pLe2) + (0.12 * pLe3) + (0.18 * h2), 0, 1);

  const q25 = Number(stats.q25 || stats.q10 || 1);
  const q50 = Number(stats.q50 || stats.mean || q25 + 1);
  const freshnessBoost = currentGap <= q25 ? 1 : (currentGap <= q50 ? 0.72 : 0.45);
  return clamp(baseQuick * freshnessBoost, 0, 1);
}
// === UPGRADE END ===

function betaMean(success, total, priorMean = 0.12, strength = 14) {
  const a = Math.max(0.0001, priorMean * strength);
  const b = Math.max(0.0001, (1 - priorMean) * strength);
  return (success + a) / Math.max(0.0001, total + a + b);
}

function empiricalQuickHitSignal(stats, currentState, currentGap, hazard) {
  const quick = stats?.quick || {};
  const g = quick.global || { le1: 0, le2: 0, le3: 0 };
  const r = quick.recent || g;
  const byRegime = quick.byRegime || {};
  const reg = byRegime[currentState?.regime || 'balanced'] || null;
  const regObs = Number(reg?.obs || 0);
  const regWeight = clamp(regObs / 220, 0, 1);

  const baseLe1 = (0.4 * (g.le1 || 0)) + (0.6 * (r.le1 || 0));
  const baseLe2 = (0.38 * (g.le2 || 0)) + (0.62 * (r.le2 || 0));
  const baseLe3 = (0.36 * (g.le3 || 0)) + (0.64 * (r.le3 || 0));

  const regP1 = reg ? betaMean((reg.p1 || 0) * regObs, regObs, baseLe1, 18) : baseLe1;
  const regP2 = reg ? betaMean((reg.p2 || 0) * regObs, regObs, baseLe2, 18) : baseLe2;
  const regP3 = reg ? betaMean((reg.p3 || 0) * regObs, regObs, baseLe3, 18) : baseLe3;

  const le1 = ((1 - regWeight) * baseLe1) + (regWeight * regP1);
  const le2 = ((1 - regWeight) * baseLe2) + (regWeight * regP2);
  const le3 = ((1 - regWeight) * baseLe3) + (regWeight * regP3);

  const q25 = Number(stats?.q25 || stats?.q10 || 1);
  const q50 = Number(stats?.q50 || stats?.mean || q25 + 1);
  const freshness = currentGap <= q25 ? 1 : (currentGap <= q50 ? 0.75 : 0.42);
  const h2 = clamp(1 - ((1 - clamp(hazard.pHit1 || 0, 0, 1)) ** 2), 0, 1);
  const score = clamp(((0.36 * le1) + (0.31 * le2) + (0.15 * le3) + (0.18 * h2)) * freshness, 0, 1);

  return {
    score,
    le1: clamp(le1, 0, 1),
    le2: clamp(le2, 0, 1),
    le3: clamp(le3, 0, 1),
    regimeObs: regObs,
  };
}

function whiteReleaseSignal(pre, currentState, target, currentIdx) {
  const idx = clamp(currentIdx, 0, Math.max(0, pre.n - 1));
  const white = whiteClusterSeverity(pre, currentState, target, idx);
  if (white <= 0) return 0;

  const start = Math.max(0, idx - 11);
  const recent = pre.rounds.slice(start, idx + 1).map(r => safeLog(r.multiplier));
  if (recent.length < 6) return clamp(white * 0.35, 0, 1);

  const last3 = mean(recent.slice(-3));
  const prev3 = mean(recent.slice(-6, -3));
  const rebound = clamp((last3 - prev3) / 0.42, 0, 1);
  const trendTail = clamp((currentState.trend + 0.03) / 0.12, 0, 1);
  const volSupport = clamp((currentState.volRatio - 0.92) / 0.55, 0, 1);
  const targetScale = target <= 10 ? 1 : (target <= 50 ? 0.78 : 0.58);

  return clamp(white * ((0.34 * rebound) + (0.38 * trendTail) + (0.28 * volSupport)) * targetScale, 0, 1);
}

function hardGapImpulse(stats, pressure, target, currentGap) {
  const q95 = Number(stats?.q95 || stats?.q90 || stats?.q75 || 1);
  const q99 = Number(stats?.q99 || Math.max(q95 + 2, 2));
  const tailAge = clamp((currentGap - q95) / Math.max(1, q99 - q95 + 1), 0, 1);
  const base = clamp((0.58 * (pressure.hard || 0)) + (0.27 * (pressure.soft || 0)) + (0.3 * tailAge), 0, 1);
  const targetScale = target >= 100 ? 1 : (target >= 20 ? 0.9 : 0.78);
  return clamp(base * targetScale, 0, 1);
}

function buildWindow(pre, target, currentIdx, calibration = null) {
  // === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
  // Justification: adaptive window center/span from blended predictors + white-cluster + calibration feedback.
  const currentRound = pre.rounds[currentIdx].roundId;
  const currentState = pre.stateAt(currentIdx, target);
  const stats = pre.gapStats[target] || { mean: 0, q50: 0, q75: 0, q90: 0, interGaps: [] };
  const gapNow = pre.gapMaps[target][currentIdx];
  const pressure = gapPressure(gapNow, stats);
  const hazard = hazardEta(target, currentState, stats, pressure);
  const neighbors = collectNeighbors(pre, target, currentIdx, currentState, {
    maxLookback: target >= 500 ? 320000 : (target >= 100 ? 240000 : 190000),
    maxScan: 80000,
    keepMin: 150,
    keepMax: 1900,
  });
  const blend = learnBlendWeights(pre, target, currentIdx, stats);

  const priorQ20 = Math.max(1, stats.q25 || stats.q10 || stats.q50 || 2);
  const priorQ50 = Math.max(1, stats.q50 || stats.mean || hazard.q50 || 2);
  const priorQ80 = Math.max(priorQ50 + 1, stats.q75 || stats.q90 || hazard.q80 || (priorQ50 + 2));

  const neighQ20 = neighbors.length ? weightedQuantile(neighbors, 0.2) : priorQ20;
  const neighQ50 = neighbors.length ? weightedQuantile(neighbors, 0.5) : priorQ50;
  const neighQ80 = neighbors.length ? weightedQuantile(neighbors, 0.8) : priorQ80;

  const q20 = (blend.neighbor * neighQ20) + (blend.hazard * hazard.q20) + (blend.prior * priorQ20);
  const q50 = (blend.neighbor * neighQ50) + (blend.hazard * hazard.q50) + (blend.prior * priorQ50);
  const q80 = (blend.neighbor * neighQ80) + (blend.hazard * hazard.q80) + (blend.prior * priorQ80);

  const whiteSeverity = whiteClusterSeverity(pre, currentState, target, currentIdx);
  const quickRaw = quickHitSignal(neighbors, hazard, gapNow, stats);
  const quickEmpirical = empiricalQuickHitSignal(stats, currentState, gapNow, hazard);
  const quickSignal = clamp((0.68 * quickRaw) + (0.32 * quickEmpirical.score), 0, 1);
  const whiteRelease = whiteReleaseSignal(pre, currentState, target, currentIdx);
  const hardImpulse = hardGapImpulse(stats, pressure, target, gapNow);
  const earlyDominance = clamp(
    Number(calibration?.earlyRate || 0) - Number(calibration?.lossRate || 0),
    -0.8,
    0.8
  );
  const interGaps = Array.isArray(stats.interGaps) ? stats.interGaps : [];
  let pGap1 = 0;
  let pGap2 = 0;
  let pGap3 = 0;
  if (interGaps.length) {
    let c1 = 0;
    let c2 = 0;
    let c3 = 0;
    for (const g of interGaps) {
      if (g <= 1) c1++;
      if (g <= 2) c2++;
      if (g <= 3) c3++;
    }
    pGap1 = c1 / interGaps.length;
    pGap2 = c2 / interGaps.length;
    pGap3 = c3 / interGaps.length;
  }
  const spread = Math.max(1, q80 - q20);
  const windowSpan = Math.max(1, Number(WINDOW_SPAN_PRIOR[target] || 3));
  const earlyBias = Math.max(0, earlyDominance);
  const lateBias = Math.max(0, -earlyDominance);

  let centerAhead = q50;
  centerAhead *= 1 + Number(calibration?.shift || 0);
  centerAhead *= 1 - (0.86 * earlyBias);
  centerAhead *= 1 + (0.58 * lateBias);
  const whiteDrift = target <= 10
    ? (-0.14 * whiteSeverity)
    : (target <= 50 ? (0.08 * whiteSeverity) : (0.22 * whiteSeverity));
  centerAhead *= 1 + whiteDrift;
  centerAhead *= 1 - (whiteRelease * (target <= 20 ? 0.36 : 0.22));
  centerAhead *= 1 - (pressure.hard * 0.24);

  if (quickSignal > 0) {
    const quickAnchor = Math.max(1, (0.68 * neighQ20) + (0.32 * hazard.q20));
    const quickPull = clamp((quickSignal - 0.08) / 0.62, 0, 1);
    const mix = 0.72 * quickPull;
    centerAhead = ((1 - mix) * centerAhead) + (mix * quickAnchor);
  }

  if (hardImpulse > 0) {
    const hardAnchor = Math.max(1, (0.62 * q20) + (0.38 * hazard.q20));
    const mix = 0.58 * hardImpulse;
    centerAhead = ((1 - mix) * centerAhead) + (mix * hardAnchor);
  }

  if (gapNow <= 1 && (pGap2 > 0.03 || pGap3 > 0.08)) {
    const b2bStrength = clamp((0.62 * pGap2) + (0.38 * pGap3), 0, 0.82);
    const b2bAnchor = Math.max(1, (0.74 * neighQ20) + (0.26 * 1.2));
    centerAhead = ((1 - b2bStrength) * centerAhead) + (b2bStrength * b2bAnchor);
  }

  if (earlyBias > 0) {
    centerAhead -= Math.max(0, Math.round(windowSpan * Math.min(0.38, earlyBias * 0.75)));
  }
  if (lateBias > 0) {
    centerAhead += Math.max(0, Math.round(windowSpan * Math.min(0.32, lateBias * 0.7)));
  }
  centerAhead = clamp(
    centerAhead,
    Math.max(1, q20 - Math.max(1, (windowSpan * 0.35))),
    Math.max(1, q80 + Math.max(1, (windowSpan * 0.35)))
  );

  const skewDen = Math.max(0.000001, q80 - q20);
  const leftSkew = clamp((q50 - q20) / skewDen, 0.1, 0.9);
  const halfLeft = Math.round((windowSpan - 1) * leftSkew);

  const dynamicMaxAhead = Math.max(
    windowSpan + 1,
    Math.round((stats.q99 || stats.q95 || stats.q90 || stats.mean || 20) * 6)
  );
  const cappedMaxAhead = Math.max(
    windowSpan + 1,
    Math.min(dynamicMaxAhead, Number(WINDOW_AHEAD_CAP[target] || dynamicMaxAhead))
  );
  let loAhead = Math.max(1, Math.round(centerAhead) - halfLeft);
  loAhead = Math.min(loAhead, Math.max(1, cappedMaxAhead - windowSpan + 1));
  const hiAhead = loAhead + windowSpan - 1;

  const componentCenters = [neighQ50, hazard.q50, priorQ50].map(v => Math.log1p(Math.max(1, v)));
  const engineAgreement = clamp(1 / (1 + stddev(componentCenters)), 0, 1);
  const support = clamp(neighbors.length / Math.max(20, Math.sqrt(Math.max(1, pre.n)) * 3), 0, 1);
  const uncertainty = clamp(1 - (spread / Math.max(2, q80 + q20)), 0, 1);
  const calibScale = clamp(Number(calibration?.confidenceScale || 0.5), 0.1, 1);
  const calibrationPenalty = clamp(
    (Number(calibration?.earlyRate || 0) + Number(calibration?.lossRate || 0)) * 0.35,
    0,
    0.35
  );
  const regimeEvidence = clamp((quickEmpirical.regimeObs || 0) / 220, 0, 1);
  const consensus = clamp(1 - Math.abs(quickRaw - quickEmpirical.score), 0, 1);
  const blendBalance = clamp(1 - Math.abs(blend.neighbor - blend.hazard), 0, 1);
  const confidence = clamp(
    (0.38 * engineAgreement) +
    (0.2 * support) +
    (0.16 * uncertainty) +
    (0.14 * calibScale) +
    (0.06 * regimeEvidence) +
    (0.03 * consensus) +
    (0.03 * blendBalance) -
    (0.08 * calibrationPenalty),
    0.04,
    0.98
  );

  return {
    target,
    lo: currentRound + loAhead,
    hi: currentRound + hiAhead,
    roundWhenMade: currentRound,
    eta: {
      q20: roundNum(q20, 2),
      q50: roundNum(q50, 2),
      q80: roundNum(q80, 2),
      neighQ20: roundNum(neighQ20, 2),
      neighQ50: roundNum(neighQ50, 2),
      neighQ80: roundNum(neighQ80, 2),
      hazardQ20: roundNum(hazard.q20, 2),
      hazardQ50: roundNum(hazard.q50, 2),
      hazardQ80: roundNum(hazard.q80, 2),
      pHit1: roundNum(hazard.pHit1, 6),
      priorQ20: roundNum(priorQ20, 2),
      priorQ50: roundNum(priorQ50, 2),
      priorQ80: roundNum(priorQ80, 2),
      blendCenter: roundNum(centerAhead, 2),
      centerAhead: roundNum(centerAhead, 2),
      windowSpan,
      neighbors: neighbors.length,
      blendSamples: Number(blend.samples || 0),
      blendErrors: blend.errors || null,
      blend: {
        neighbor: roundNum(blend.neighbor, 4),
        hazard: roundNum(blend.hazard, 4),
        prior: roundNum(blend.prior, 4),
      },
      uncertainty: roundNum(1 - uncertainty, 4),
      engineAgreement: roundNum(engineAgreement, 4),
      regime: currentState.regime,
      currentGap: gapNow,
      under2Rate: roundNum(currentState.under2Rate || 0, 4),
      whiteClusterSeverity: roundNum(whiteSeverity, 4),
      historicalGapMean: stats.mean || 0,
      historicalQ75: stats.q75 || 0,
      historicalQ90: stats.q90 || 0,
      softGapPressure: roundNum(pressure.soft, 4),
      hardGapPressure: roundNum(pressure.hard, 4),
      quickHitSignal: roundNum(quickSignal, 4),
      quickHitRaw: roundNum(quickRaw, 4),
      quickHitEmpirical: roundNum(quickEmpirical.score, 4),
      quickEmpiricalP1: roundNum(quickEmpirical.le1, 4),
      quickEmpiricalP2: roundNum(quickEmpirical.le2, 4),
      quickEmpiricalP3: roundNum(quickEmpirical.le3, 4),
      quickEmpiricalRegimeObs: Number(quickEmpirical.regimeObs || 0),
      pGapLe1: roundNum(pGap1, 4),
      pGapLe2: roundNum(pGap2, 4),
      pGapLe3: roundNum(pGap3, 4),
      whiteReleaseSignal: roundNum(whiteRelease, 4),
      hardGapImpulse: roundNum(hardImpulse, 4),
      confidence: roundNum(confidence, 4),
      reason: currentState.regime === 'white'
        ? 'White-cluster regime: release + empirical quick-hit gating + adaptive blend.'
        : 'Adaptive blend (neighbor + hazard + prior) with empirical quick-hit/regime gating.',
      calibrationShift: roundNum(calibration?.shift || 0, 4),
      calibrationSample: Number(calibration?.sample || 0),
      calibrationDirectional: Number(calibration?.directionalSamples || 0),
      calibrationError: roundNum(calibration?.meanNormError || 0, 4),
      calibrationWinRate: roundNum(calibration?.winRate || 0, 4),
      calibrationEarlyRate: roundNum(calibration?.earlyRate || 0, 4),
      calibrationLossRate: roundNum(calibration?.lossRate || 0, 4),
      calibrationWilsonLow: roundNum(calibration?.wilsonLow || 0.5, 4),
      calibrationSpanMultiplier: 1,
    },
    confidence: roundNum(confidence, 4),
  };
  // === UPGRADE END ===
}

function enforceNextWindowStart(nextLock, fixedSpan, minLo) {
  const lock = { ...nextLock };
  const span = Math.max(1, Number(fixedSpan) || 1);
  const minStart = Math.max(1, Number(minLo) || 1);
  if (!Number.isFinite(lock.lo) || !Number.isFinite(lock.hi)) return lock;
  if (lock.lo >= minStart) return lock;
  lock.lo = minStart;
  lock.hi = minStart + span - 1;
  lock.eta = {
    ...(lock.eta || {}),
    nonOverlapAdjusted: true,
    minAllowedLo: minStart,
  };
  return lock;
}

function buildAdaptiveWaitingWindow(nextLock, target, fixedSpan, currentRound, minLo, existing = null) {
  const base = enforceNextWindowStart(nextLock, fixedSpan, minLo);
  const eta = base.eta || {};
  const span = Math.max(1, Number(fixedSpan) || 1);
  const historicalQ90 = Math.max(1, Number(eta.historicalQ90 || eta.hazardQ80 || eta.q80 || eta.q50 || 1));
  const historicalGapMean = Math.max(1, Number(eta.historicalGapMean || eta.q50 || 1));
  const dynamicCapRaw = Math.max(
    span + 1,
    Math.round((target >= 100 ? 2.6 : 2.2) * historicalQ90),
    Math.round((target >= 100 ? 2.1 : 1.8) * historicalGapMean),
    Math.round(Math.max(0, Number(eta.currentGap || 0)) + Math.max(6, span * 1.4))
  );
  const cap = clamp(dynamicCapRaw, span + 1, 220);
  const maxStartAhead = Math.max(1, cap - span + 1);

  const q20 = Math.max(1, Number(eta.q20 || 1));
  const q50 = Math.max(q20, Number(eta.q50 || q20));
  const hazardQ20 = Math.max(1, Number(eta.hazardQ20 || q20));
  const currentGap = Math.max(0, Number(eta.currentGap || 0));
  const pHit1 = clamp(Number(eta.pHit1 || 0), 0, 1);
  const pGap2 = clamp(Number(eta.pGapLe2 || 0), 0, 1);
  const pGap3 = clamp(Number(eta.pGapLe3 || 0), 0, 1);
  const quick = clamp(Number(eta.quickHitEmpirical ?? eta.quickHitSignal ?? 0), 0, 1);
  const hard = clamp(Number(eta.hardGapImpulse ?? eta.hardGapPressure ?? 0), 0, 1);
  const white = clamp(Number(eta.whiteClusterSeverity || 0), 0, 1);
  const release = clamp(Number(eta.whiteReleaseSignal || 0), 0, 1);
  const confidence = clamp(Number(eta.confidence || 0), 0, 1);
  const centerAhead = Math.max(1, Number(eta.centerAhead || q50));
  const expectedFromP1 = pHit1 > 0 ? clamp(1 / pHit1, 1, maxStartAhead) : maxStartAhead;
  const b2bPull = clamp((target <= 20 ? ((0.65 * pGap2) + (0.35 * pGap3)) : ((0.45 * pGap2) + (0.55 * pGap3))), 0, 1);

  // Data-driven pre-window start: blend hazard/gap/cluster signals (not fixed offsets).
  let startAhead = (
    (0.34 * q20) +
    (0.24 * hazardQ20) +
    (0.18 * q50) +
    (0.24 * expectedFromP1)
  );
  startAhead = (0.66 * startAhead) + (0.34 * centerAhead);
  startAhead -= (target <= 20 ? 4.2 : 2.2) * quick;
  startAhead -= (target <= 20 ? 2.8 : 1.5) * b2bPull;
  startAhead -= (target >= 50 ? 1.8 : 1.0) * hard;

  if (target <= 10) {
    startAhead -= 1.05 * white;
    startAhead += 1.45 * release;
  } else {
    startAhead += 0.58 * white;
    startAhead += 0.2 * release;
  }

  if (confidence < 0.45) {
    startAhead += (0.45 - confidence) * 3.5;
  }

  if (existing?.suspended && Number.isFinite(Number(existing.lo))) {
    const prevAhead = clamp(Number(existing.lo) - Number(currentRound || 0), 1, maxStartAhead);
    const keepPrev = clamp(0.32 + (0.42 * (1 - confidence)), 0.3, 0.76);
    startAhead = (keepPrev * prevAhead) + ((1 - keepPrev) * startAhead);
  }

  const roundedAhead = clamp(Math.round(startAhead), 1, maxStartAhead);
  const absoluteLo = Math.max(Number(minLo || (currentRound + 1)), Number(currentRound || 0) + roundedAhead);
  const absoluteHi = absoluteLo + span - 1;

  return {
    ...base,
    lo: absoluteLo,
    hi: absoluteHi,
    eta: {
      ...eta,
      adaptiveWaiting: true,
      waitingModelVersion: WAITING_MODEL_VERSION,
      waitingStartAhead: roundedAhead,
      waitingBlendConfidence: roundNum(confidence, 4),
      waitingBlendExpectedFromP1: roundNum(expectedFromP1, 2),
      waitingBlendB2BPull: roundNum(b2bPull, 4),
      waitingDynamicCap: cap,
      waitingSource: 'hazard+gap+quick+hard+cluster',
    },
  };
}

function evaluateLock(lock, target, pre, currentRound) {
  if (!lock) return { resolved: false, status: 'missing' };
  const suspended = Boolean(lock.suspended);
  const roundWhenMade = Number(lock.roundWhenMade || lock.round_when_made || 0);
  const lo = Number(lock.lo);
  const hi = Number(lock.hi);
  if (!Number.isFinite(roundWhenMade) || !Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo || lo <= roundWhenMade) {
    return { resolved: true, outcome: 'loss', hitRound: null };
  }

  const hits = pre.hitRoundIds[target] || [];
  const firstHitAfterMade = findFirstInRange(hits, roundWhenMade + 1, currentRound);

  if (firstHitAfterMade != null) {
    if (firstHitAfterMade < lo) return { resolved: true, outcome: 'early', hitRound: firstHitAfterMade };
    if (firstHitAfterMade <= hi) return { resolved: true, outcome: 'win', hitRound: firstHitAfterMade };
  }

  // Window is inclusive. If current round reached/ended hi without a hit, it is a miss now.
  if (currentRound >= hi) return { resolved: true, outcome: 'loss', hitRound: null };

  if (currentRound < lo) return { resolved: false, status: suspended ? 'waiting' : 'pending' };
  return { resolved: false, status: suspended ? 'waiting' : 'window-open' };
}

function shouldActivateWindow(target, nextLock, calibration = null) {
  const eta = nextLock?.eta || {};
  const confidence = clamp(Number(eta.confidence || 0), 0, 1);
  const p1 = clamp(Number(eta.pHit1 || 0), 0, 1);
  const quick = clamp(Number(eta.quickHitEmpirical ?? eta.quickHitSignal ?? 0), 0, 1);
  const hard = clamp(Number(eta.hardGapImpulse || 0), 0, 1);
  const white = clamp(Number(eta.whiteClusterSeverity || 0), 0, 1);
  const release = clamp(Number(eta.whiteReleaseSignal || 0), 0, 1);
  const calWilson = clamp(Number(calibration?.wilsonLow || 0.5), 0, 1);
  const calPenalty = clamp(
    (Number(calibration?.lossRate || 0) + Number(calibration?.earlyRate || 0)) * 0.35,
    0,
    0.3
  );

  const minConf = Number(ACTIVATE_MIN_CONF[target] || 0.5);
  const minP1 = Number(ACTIVATE_MIN_P1[target] || 0.05);

  let score = (
    (0.44 * confidence) +
    (0.22 * quick) +
    (0.18 * hard) +
    (0.16 * p1)
  );
  if (target <= 10) score += (0.1 * white);
  if (target <= 10) score -= (0.08 * release);
  score = clamp((0.82 * score) + (0.18 * calWilson) - calPenalty, 0, 1);

  let active = false;
  if (target <= 20) {
    active = (score >= (minConf - 0.08)) && (quick >= 0.18 || p1 >= minP1 || hard >= 0.18);
  } else if (target <= 100) {
    active = (score >= minConf) && (p1 >= minP1 || hard >= 0.22);
  } else if (target <= 500) {
    active = (score >= minConf) && (hard >= 0.28 || p1 >= minP1);
  } else {
    active = (score >= minConf) && (hard >= 0.34 || p1 >= minP1);
  }

  return {
    active,
    score: roundNum(score, 6),
    minConf: roundNum(minConf, 6),
    minP1: roundNum(minP1, 6),
  };
}

function normalizeLockInput(input) {
  if (!input) return null;
  const eta = input.eta || input.eta_json || null;
  const suspended = Boolean(input.suspended ?? eta?.suspended);
  return {
    lo: Number(input.lo),
    hi: Number(input.hi),
    roundWhenMade: Number(input.roundWhenMade ?? input.round_when_made),
    generation: Number(input.generation || 1),
    suspended,
    eta,
  };
}

function buildCalibrationMap(historyRows, pre) {
  // === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
  // Justification: derive timing-shift, span multiplier, and confidence scale from real win/loss/early outcomes.
  const out = {};
  for (const target of TARGETS) {
    const label = `${target}x`;
    const rows = (historyRows || [])
      .filter(r => String(r.target || '').toLowerCase() === label)
      .slice(0, 320);

    if (!rows.length) {
      out[target] = {
        shift: 0,
        spanMultiplier: 1,
        sample: 0,
        directionalSamples: 0,
        meanNormError: 0,
        absNormError: 0,
        winRate: 0.5,
        earlyRate: 0,
        lossRate: 0,
        wilsonLow: 0.5,
        confidenceScale: 0.5,
      };
      continue;
    }

    const hits = pre?.hitRoundIds?.[target] || [];
    let winCount = 0;
    let lossCount = 0;
    let earlyCount = 0;
    const errItems = [];
    const absErrItems = [];
    let directionalSamples = 0;

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const lo = Number(row.lo);
      const hi = Number(row.hi);
      const hitRound = Number(row.hitRound);
      const outcome = String(row.outcome || '').toLowerCase();
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) continue;

      const span = Math.max(1, (hi - lo + 1));
      const center = lo + ((span - 1) * 0.5);
      const recencyRatio = (rows.length - i) / Math.max(1, rows.length);
      const recency = 1 + (2 * (recencyRatio ** 2));
      let err = 0;
      let directional = false;

      if (outcome === 'early') {
        earlyCount++;
        if (Number.isFinite(hitRound)) {
          err = (hitRound - lo) / span;
          directional = true;
        }
      } else if (outcome === 'loss') {
        lossCount++;
        const searchCap = hi + Math.max(span * 12, 100);
        const nextHitAfterHi = findFirstInRange(hits, hi + 1, searchCap);
        if (nextHitAfterHi != null) {
          err = (nextHitAfterHi - hi) / span;
        } else {
          err = 1;
        }
        directional = true;
      } else if (outcome === 'win') {
        winCount++;
        if (Number.isFinite(hitRound)) {
          err = (hitRound - center) / span;
          directional = true;
        }
      } else {
        continue;
      }

      if (directional) {
        const normErr = clamp(err, -3, 3);
        errItems.push({ value: normErr, weight: recency });
        absErrItems.push({ value: Math.abs(normErr), weight: recency });
        directionalSamples++;
      }
    }

    const total = winCount + lossCount + earlyCount;
    const denom = winCount + lossCount;
    const winRate = denom > 0 ? (winCount / denom) : 0.5;
    const earlyRate = total > 0 ? (earlyCount / total) : 0;
    const lossRate = total > 0 ? (lossCount / total) : 0;
    const meanNormErr = weightedMean(errItems);
    const absNormErr = weightedMean(absErrItems);
    const wb = wilsonBounds(winCount, lossCount);
    const sampleFactor = clamp(rows.length / 26, 0, 1);
    const directionalBias = clamp((lossRate - earlyRate), -1, 1);
    const blendedShift = (0.62 * meanNormErr) + (0.38 * directionalBias);
    const shift = clamp(blendedShift * sampleFactor, -0.72, 0.72);
    const spanMultiplier = clamp((1 + absNormErr) * (1 + (earlyRate * 0.45)), 0.75, 2.9);
    const confidenceScale = clamp((wb.low + wb.mid) * 0.5, 0.1, 1);

    out[target] = {
      shift: roundNum(shift, 4),
      spanMultiplier: roundNum(spanMultiplier, 4),
      sample: rows.length,
      directionalSamples,
      winRate: roundNum(winRate, 4),
      earlyRate: roundNum(earlyRate, 4),
      lossRate: roundNum(lossRate, 4),
      meanNormError: roundNum(meanNormErr, 4),
      absNormError: roundNum(absNormErr, 4),
      wilsonLow: roundNum(wb.low, 4),
      confidenceScale: roundNum(confidenceScale, 4),
    };
  }
  return out;
  // === UPGRADE END ===
}

function buildUiTarget(target, lock, status, currentRound, previousOutcome = null) {
  const loAhead = Math.max(1, Number(lock.lo) - currentRound);
  const hiAhead = Math.max(loAhead, Number(lock.hi) - currentRound);
  const roundsUntilWindow = Math.max(0, Number(lock.lo) - currentRound);
  const roundsLeftInWindow = Math.max(0, Number(lock.hi) - currentRound);

  return {
    target,
    targetLabel: `${target}x`,
    generation: Number(lock.generation || 1),
    roundWhenMade: Number(lock.roundWhenMade),
    window: {
      lo: Number(lock.lo),
      hi: Number(lock.hi),
      aheadLo: loAhead,
      aheadHi: hiAhead,
      span: Math.max(1, (Number(lock.hi) - Number(lock.lo) + 1)),
      roundsUntilWindow,
      roundsLeftInWindow,
    },
    status,
    confidence: roundNum(lock.eta?.confidence ?? 0.25, 4),
    regime: lock.eta?.regime || 'unknown',
    currentGap: Number(lock.eta?.currentGap || 0),
    softGapPressure: roundNum(lock.eta?.softGapPressure || 0, 4),
    hardGapPressure: roundNum(lock.eta?.hardGapPressure || 0, 4),
    signals: {
      quickHit: roundNum(
        lock.eta?.quickHitEmpirical ??
        lock.eta?.quickHitSignal ??
        lock.eta?.pHit1 ??
        0,
        6
      ),
      pHit1: roundNum(lock.eta?.pHit1 || 0, 6),
      quickHitRaw: roundNum(lock.eta?.quickHitRaw || 0, 6),
      quickHitEmpirical: roundNum(lock.eta?.quickHitEmpirical || 0, 6),
      quickEmpiricalP1: roundNum(lock.eta?.quickEmpiricalP1 || 0, 6),
      quickEmpiricalP2: roundNum(lock.eta?.quickEmpiricalP2 || 0, 6),
      quickEmpiricalP3: roundNum(lock.eta?.quickEmpiricalP3 || 0, 6),
      quickEmpiricalRegimeObs: Number(lock.eta?.quickEmpiricalRegimeObs || 0),
      whiteClusterSeverity: roundNum(lock.eta?.whiteClusterSeverity || 0, 6),
      whiteReleaseSignal: roundNum(lock.eta?.whiteReleaseSignal || 0, 6),
      hardGapImpulse: roundNum(lock.eta?.hardGapImpulse || 0, 6),
      pGapLe1: roundNum(lock.eta?.pGapLe1 || 0, 6),
      pGapLe2: roundNum(lock.eta?.pGapLe2 || 0, 6),
      pGapLe3: roundNum(lock.eta?.pGapLe3 || 0, 6),
    },
    reason: lock.eta?.reason || 'Range locked from historical cluster-pattern analogs.',
    previousOutcome,
  };
}

function sequenceLogDistance(rounds, idxA, idxB, len = 8) {
  if (!rounds?.length) return 0;
  let dist = 0;
  let weightTotal = 0;
  for (let i = 0; i < len; i++) {
    const a = rounds[clamp(idxA - i, 0, rounds.length - 1)];
    const b = rounds[clamp(idxB - i, 0, rounds.length - 1)];
    const la = safeLog(a?.multiplier || 1);
    const lb = safeLog(b?.multiplier || 1);
    const w = 1 + (((len - i) / len) * 1.9);
    dist += w * Math.abs(la - lb);
    weightTotal += w;
  }
  return weightTotal > 0 ? (dist / weightTotal) : 0;
}

function evalWhiteFuture(rounds, startIdx, endIdx) {
  const s = clamp(startIdx, 0, Math.max(0, rounds.length - 1));
  const e = clamp(endIdx, s, Math.max(0, rounds.length - 1));
  let whiteCount = 0;
  let run = 0;
  let maxRun = 0;
  let firstClusterAhead = null;
  let reboundSoon = false;

  for (let i = s; i <= e; i++) {
    const m = Number(rounds[i]?.multiplier || 0);
    if (i <= (s + 2) && m >= 5) reboundSoon = true;
    if (m < 3) {
      whiteCount++;
      run++;
      if (run > maxRun) maxRun = run;
      if (run >= 3 && firstClusterAhead == null) firstClusterAhead = (i - s + 1);
    } else {
      run = 0;
    }
  }

  const clusterSoon = maxRun >= 3 || whiteCount >= 3;
  return {
    clusterSoon,
    reboundSoon,
    firstClusterAhead,
    whiteCount,
    maxRun,
  };
}

function buildHighTargetChainSignal(pre, target, currentIdx) {
  if (target < 50) {
    return { active: false, p1: 0, p3: 0, score: 0, samples: 0 };
  }
  const rounds = pre.rounds || [];
  const n = Number(pre.n || rounds.length || 0);
  if (!n || currentIdx < 20) {
    return { active: false, p1: 0, p3: 0, score: 0, samples: 0 };
  }

  const nextMap = pre.nextHitMaps?.[target] || [];
  const cur = Number(rounds[currentIdx]?.multiplier || 0);
  const prev1 = Number(rounds[Math.max(0, currentIdx - 1)]?.multiplier || 0);
  const prev2 = Number(rounds[Math.max(0, currentIdx - 2)]?.multiplier || 0);
  const curPeak2 = Math.max(cur, prev1);
  const curPeak3 = Math.max(curPeak2, prev2);
  const anchor = target >= 500
    ? Math.max(120, target * 0.35)
    : (target >= 100 ? Math.max(50, target * 0.42) : Math.max(18, target * 0.48));
  const active = curPeak2 >= anchor;
  const curSuper = curPeak2 >= target ? 1 : 0;
  const curTail = (
    safeLog(cur) +
    safeLog(prev1 || cur) +
    safeLog(prev2 || prev1 || cur)
  ) / 3;

  const start = Math.max(80, currentIdx - (target >= 500 ? 250000 : 210000));
  const rawSpan = Math.max(1, currentIdx - start);
  const maxScan = target >= 500 ? 70000 : 65000;
  const stride = Math.max(1, Math.floor(rawSpan / maxScan));

  let totalW = 0;
  let hit1 = 0;
  let hit3 = 0;
  for (let idx = start; idx < currentIdx - 3; idx += stride) {
    const h0 = Number(rounds[idx]?.multiplier || 0);
    const h1 = Number(rounds[Math.max(0, idx - 1)]?.multiplier || 0);
    const h2 = Number(rounds[Math.max(0, idx - 2)]?.multiplier || 0);
    const histPeak2 = Math.max(h0, h1);
    const histPeak3 = Math.max(histPeak2, h2);
    const histSuper = histPeak2 >= target ? 1 : 0;
    const histHot = histPeak2 >= anchor ? 1 : 0;
    const curHot = active ? 1 : 0;
    const histTail = (safeLog(h0) + safeLog(h1 || h0) + safeLog(h2 || h1 || h0)) / 3;

    let dist = 0;
    dist += 0.9 * Math.abs(curHot - histHot);
    dist += 0.7 * Math.abs(curSuper - histSuper);
    dist += 0.42 * Math.abs(Math.log1p(curPeak3) - Math.log1p(histPeak3));
    dist += 1.25 * Math.abs(curTail - histTail);
    dist += 0.14 * sequenceLogDistance(rounds, currentIdx, idx, 6);

    const recency = 0.45 + (0.55 * ((idx + 1) / Math.max(1, currentIdx)) ** 1.16);
    const hotBoost = histHot ? 1.12 : 0.95;
    const weight = Math.exp(-dist * 0.92) * recency * hotBoost;
    if (weight < 0.001) continue;

    totalW += weight;
    const nextIdx = nextMap[idx];
    const ahead = nextIdx == null ? Number.POSITIVE_INFINITY : (nextIdx - idx);
    if (ahead > 0 && ahead <= 1) hit1 += weight;
    if (ahead > 0 && ahead <= 3) hit3 += weight;
  }

  const quick = pre.gapStats?.[target]?.quick?.global || {};
  const baseP1 = clamp(Number(quick.le1 || 0), 0.0005, 0.95);
  const baseP3 = clamp(Number(quick.le3 || 0), baseP1, 0.98);
  const alpha = totalW >= 22 ? 6 : 10;
  const p1 = clamp((hit1 + (alpha * baseP1)) / Math.max(0.0001, totalW + alpha), 0, 1);
  const p3 = clamp((hit3 + (alpha * baseP3)) / Math.max(0.0001, totalW + alpha), p1, 1);
  const score = clamp((0.42 * p1) + (0.58 * p3), 0, 1);

  return {
    active,
    p1: roundNum(p1, 6),
    p3: roundNum(p3, 6),
    score: roundNum(score, 6),
    samples: Math.round(totalW),
  };
}

function buildIndependentB2BTargetAlert(pre, target, currentIdx) {
  const rounds = pre.rounds || [];
  const n = Number(pre.n || rounds.length || 0);
  if (!n || currentIdx < 40) {
    return {
      target,
      targetLabel: `${target}x`,
      probability: 0,
      p1: 0,
      p2: 0,
      p3: 0,
      p5: 0,
      aheadLo: 1,
      aheadHi: 5,
      evidence: 0,
      samples: 0,
    };
  }

  const stats = pre.gapStats?.[target] || {};
  const nextMap = pre.nextHitMaps?.[target] || [];
  const currentState = pre.stateAt(currentIdx, target);
  const horizonMax = target >= 500 ? 16 : (target >= 100 ? 14 : 12);
  const start = Math.max(120, currentIdx - (target >= 500 ? 260000 : 210000));
  const rawSpan = Math.max(1, currentIdx - start);
  const maxScan = target >= 500 ? 80000 : 70000;
  const stride = Math.max(1, Math.floor(rawSpan / maxScan));

  let totalW = 0;
  let w1 = 0;
  let w2 = 0;
  let w3 = 0;
  let w5 = 0;
  const hitAheads = [];

  for (let idx = start; idx < currentIdx - 1; idx += stride) {
    const st = pre.stateAt(idx, target);
    let dist = stateDistance(currentState, st, target);
    dist += 0.18 * sequenceLogDistance(rounds, currentIdx, idx, 8);

    const megaCur = rounds[currentIdx]?.multiplier >= 100 ? 1 : 0;
    const megaPrevCur = rounds[Math.max(0, currentIdx - 1)]?.multiplier >= 100 ? 1 : 0;
    const megaHist = rounds[idx]?.multiplier >= 100 ? 1 : 0;
    const megaPrevHist = rounds[Math.max(0, idx - 1)]?.multiplier >= 100 ? 1 : 0;
    if (megaCur !== megaHist) dist += 0.28;
    if (megaPrevCur !== megaPrevHist) dist += 0.2;

    const recency = 0.42 + (0.58 * ((idx + 1) / Math.max(1, currentIdx)) ** 1.25);
    const regimeBoost = st.regime === currentState.regime ? 1.12 : 1;
    const weight = Math.exp(-dist * 0.82) * recency * regimeBoost;
    if (weight < 0.001) continue;

    totalW += weight;
    const nextIdx = nextMap[idx];
    const ahead = nextIdx == null ? Number.POSITIVE_INFINITY : (nextIdx - idx);
    if (ahead <= 1) w1 += weight;
    if (ahead <= 2) w2 += weight;
    if (ahead <= 3) w3 += weight;
    if (ahead <= 5) w5 += weight;
    if (ahead > 0 && ahead <= horizonMax) hitAheads.push({ value: ahead, weight });
  }

  const inter = Array.isArray(stats.interGaps) ? stats.interGaps : [];
  let c1 = 0;
  let c2 = 0;
  let c3 = 0;
  let c5 = 0;
  for (const g of inter) {
    if (g <= 1) c1++;
    if (g <= 2) c2++;
    if (g <= 3) c3++;
    if (g <= 5) c5++;
  }
  const interDen = Math.max(1, inter.length);
  const prior1 = clamp(c1 / interDen, 0.0005, 0.95);
  const prior2 = clamp(c2 / interDen, prior1, 0.97);
  const prior3 = clamp(c3 / interDen, prior2, 0.98);
  const prior5 = clamp(c5 / interDen, prior3, 0.995);

  const alpha = totalW >= 26 ? 6 : 10;
  let p1 = (w1 + (alpha * prior1)) / Math.max(0.0001, totalW + alpha);
  let p2 = (w2 + (alpha * prior2)) / Math.max(0.0001, totalW + alpha);
  let p3 = (w3 + (alpha * prior3)) / Math.max(0.0001, totalW + alpha);
  let p5 = (w5 + (alpha * prior5)) / Math.max(0.0001, totalW + alpha);
  p1 = clamp(p1, 0, 1);
  p2 = clamp(Math.max(p2, p1), 0, 1);
  p3 = clamp(Math.max(p3, p2), 0, 1);
  p5 = clamp(Math.max(p5, p3), 0, 1);

  const highChain = buildHighTargetChainSignal(pre, target, currentIdx);
  if (highChain.active && target >= 50) {
    const chainBlend = clamp((Number(highChain.samples || 0) / (target >= 500 ? 38 : 52)), 0, 1) * 0.62;
    const chainP1 = clamp(Number(highChain.p1 || 0), 0, 1);
    const chainP3 = clamp(Number(highChain.p3 || 0), chainP1, 1);
    p1 = clamp(((1 - chainBlend) * p1) + (chainBlend * chainP1), 0, 1);
    p2 = clamp(Math.max(p1, ((1 - chainBlend) * p2) + (chainBlend * Math.max(chainP1, chainP3 * 0.82))), 0, 1);
    p3 = clamp(Math.max(p2, ((1 - chainBlend) * p3) + (chainBlend * chainP3)), 0, 1);
    p5 = clamp(Math.max(p3, ((1 - (chainBlend * 0.5)) * p5) + ((chainBlend * 0.5) * Math.max(chainP3, p3))), 0, 1);
  }

  let aheadLo = 1;
  let aheadHi = 5;
  if (hitAheads.length >= 5) {
    aheadLo = clamp(Math.round(weightedQuantile(hitAheads, 0.2)), 1, horizonMax);
    aheadHi = clamp(Math.round(weightedQuantile(hitAheads, 0.65)), aheadLo, horizonMax);
  } else {
    const center = p1 >= 0.18 ? 1 : (p3 >= 0.3 ? 2 : 3);
    aheadLo = center;
    aheadHi = clamp(center + 2, aheadLo, horizonMax);
  }
  if (highChain.active && Number(highChain.p1 || 0) >= 0.2 && target >= 50) {
    aheadLo = 1;
    aheadHi = Math.min(aheadHi, 3);
  }

  const evidence = clamp(totalW / Math.max(22, Math.sqrt(Math.max(1, n)) * 2.4), 0, 1);
  return {
    target,
    targetLabel: `${target}x`,
    probability: roundNum(p3, 6),
    p1: roundNum(p1, 6),
    p2: roundNum(p2, 6),
    p3: roundNum(p3, 6),
    p5: roundNum(p5, 6),
    aheadLo,
    aheadHi,
    evidence: roundNum(evidence, 4),
    samples: Math.round(totalW),
    highChainActive: Boolean(highChain.active),
    highChainScore: roundNum(highChain.score, 6),
    highChainP3: roundNum(highChain.p3, 6),
    highChainSamples: Number(highChain.samples || 0),
  };
}

function buildIndependentWhiteClusterAlert(pre, currentIdx) {
  const rounds = pre.rounds || [];
  const n = Number(pre.n || rounds.length || 0);
  if (!n || currentIdx < 30) {
    return {
      risk: 0,
      release: 0,
      currentRun: 0,
      aheadLo: 1,
      aheadHi: 5,
      samples: 0,
      whiteRate: 0,
    };
  }

  const horizon = 5;
  const start = Math.max(80, currentIdx - 220000);
  const rawSpan = Math.max(1, currentIdx - start);
  const maxScan = 70000;
  const stride = Math.max(1, Math.floor(rawSpan / maxScan));
  const currentRun = Number(pre.whiteStreak?.[currentIdx] || 0);

  let totalW = 0;
  let eventW = 0;
  let releaseW = 0;
  let contTotalW = 0;
  let contEventW = 0;
  const eventAheads = [];

  const whiteTotal = rounds.reduce((acc, r) => acc + (Number(r.multiplier) < 3 ? 1 : 0), 0);
  const whiteRate = whiteTotal / Math.max(1, n);
  const curTrend = Number(pre.trendByIdx?.[currentIdx] || 0);
  const curVol = Number(pre.volRatioByIdx?.[currentIdx] || 1);
  const curUnder2 = Number(pre.under2RateByIdx?.[currentIdx] || 0);
  const curRegime = String(pre.regimeByIdx?.[currentIdx] || 'balanced');
  const megaCur = rounds[currentIdx]?.multiplier >= 100 ? 1 : 0;
  const megaPrevCur = rounds[Math.max(0, currentIdx - 1)]?.multiplier >= 100 ? 1 : 0;

  for (let idx = start; idx < currentIdx - horizon - 1; idx += stride) {
    let dist = 0;
    dist += 2.4 * Math.abs(curTrend - Number(pre.trendByIdx?.[idx] || 0));
    dist += 1.35 * Math.abs(curVol - Number(pre.volRatioByIdx?.[idx] || 1));
    dist += 0.95 * Math.abs(curUnder2 - Number(pre.under2RateByIdx?.[idx] || 0));
    dist += 0.08 * Math.abs(currentRun - Number(pre.whiteStreak?.[idx] || 0));
    dist += 0.2 * sequenceLogDistance(rounds, currentIdx, idx, 8);

    const megaHist = rounds[idx]?.multiplier >= 100 ? 1 : 0;
    const megaPrevHist = rounds[Math.max(0, idx - 1)]?.multiplier >= 100 ? 1 : 0;
    if (megaCur !== megaHist) dist += 0.35;
    if (megaPrevCur !== megaPrevHist) dist += 0.25;

    const recency = 0.4 + (0.6 * ((idx + 1) / Math.max(1, currentIdx)) ** 1.2);
    const regimeBoost = (String(pre.regimeByIdx?.[idx] || 'balanced') === curRegime) ? 1.1 : 1;
    const weight = Math.exp(-dist * 0.9) * recency * regimeBoost;
    if (weight < 0.001) continue;

    totalW += weight;
    const future = evalWhiteFuture(rounds, idx + 1, idx + horizon);
    if (future.clusterSoon) {
      eventW += weight;
      if (future.firstClusterAhead != null) {
        eventAheads.push({ value: future.firstClusterAhead, weight });
      }
    }
    if (future.reboundSoon) releaseW += weight;
    if (Number(pre.whiteStreak?.[idx] || 0) >= 3) {
      contTotalW += weight;
      if (future.clusterSoon) contEventW += weight;
    }
  }

  let priorEvent = clamp(whiteRate * 1.9, 0.02, 0.9);
  if (currentRun >= 3) {
    priorEvent = Math.max(priorEvent, clamp(0.56 + (0.09 * Math.min(4, currentRun - 3)), 0.56, 0.92));
  }
  const priorRelease = clamp((1 - priorEvent) * 0.75, 0.03, 0.95);
  const alpha = totalW >= 26 ? 6 : 10;

  let risk = (eventW + (alpha * priorEvent)) / Math.max(0.0001, totalW + alpha);
  if (currentRun >= 3 && contTotalW > 0) {
    const contRisk = (contEventW + (6 * priorEvent)) / Math.max(0.0001, contTotalW + 6);
    risk = Math.max(risk, contRisk);
  }
  risk = clamp(risk, 0, 1);
  const release = clamp((releaseW + (alpha * priorRelease)) / Math.max(0.0001, totalW + alpha), 0, 1);

  let aheadLo = 1;
  let aheadHi = 5;
  if (eventAheads.length >= 5) {
    aheadLo = clamp(Math.round(weightedQuantile(eventAheads, 0.2)), 1, 5);
    aheadHi = clamp(Math.round(weightedQuantile(eventAheads, 0.75)), aheadLo, 5);
  } else if (currentRun >= 3) {
    aheadLo = 1;
    aheadHi = 3;
  }

  return {
    risk: roundNum(risk, 6),
    release: roundNum(release, 6),
    currentRun,
    aheadLo,
    aheadHi,
    samples: Math.round(totalW),
    whiteRate: roundNum(whiteRate, 6),
  };
}

function buildWhiteContextB2BAdjusters(pre, currentIdx) {
  const rounds = pre.rounds || [];
  const n = Number(pre.n || rounds.length || 0);
  const out = {};
  for (const target of TARGETS) {
    out[target] = { ratio: 1, condP3: 0, baseP3: 0, sample: 0 };
  }
  if (!n || currentIdx < 50) return out;

  const curRun = Number(pre.whiteStreak?.[currentIdx] || 0);
  const curUnder2 = Number(pre.under2RateByIdx?.[currentIdx] || 0);
  const curTrend = Number(pre.trendByIdx?.[currentIdx] || 0);
  const curRegime = String(pre.regimeByIdx?.[currentIdx] || 'balanced');
  const start = Math.max(80, currentIdx - 220000);
  const rawSpan = Math.max(1, currentIdx - start);
  const maxScan = 70000;
  const stride = Math.max(1, Math.floor(rawSpan / maxScan));

  const totals = {};
  const hits = {};
  for (const target of TARGETS) {
    totals[target] = 0;
    hits[target] = 0;
  }

  for (let idx = start; idx < currentIdx - 3; idx += stride) {
    let dist = 0;
    dist += 0.42 * Math.abs(curRun - Number(pre.whiteStreak?.[idx] || 0));
    dist += 1.5 * Math.abs(curUnder2 - Number(pre.under2RateByIdx?.[idx] || 0));
    dist += 1.15 * Math.abs(curTrend - Number(pre.trendByIdx?.[idx] || 0));
    if (String(pre.regimeByIdx?.[idx] || 'balanced') !== curRegime) dist += 0.55;
    dist += 0.16 * sequenceLogDistance(rounds, currentIdx, idx, 6);

    const recency = 0.45 + (0.55 * ((idx + 1) / Math.max(1, currentIdx)) ** 1.18);
    const weight = Math.exp(-dist * 0.92) * recency;
    if (weight < 0.001) continue;

    for (const target of TARGETS) {
      totals[target] += weight;
      const nextIdx = pre.nextHitMaps?.[target]?.[idx];
      const ahead = nextIdx == null ? Number.POSITIVE_INFINITY : (nextIdx - idx);
      if (ahead > 0 && ahead <= 3) hits[target] += weight;
    }
  }

  for (const target of TARGETS) {
    const totalW = Math.max(0, Number(totals[target] || 0));
    const condP3 = totalW > 0 ? (hits[target] / totalW) : 0;
    let baseP3 = Number(pre.gapStats?.[target]?.quick?.global?.le3 || 0);
    if (!(baseP3 > 0)) {
      const inter = Array.isArray(pre.gapStats?.[target]?.interGaps)
        ? pre.gapStats[target].interGaps
        : [];
      if (inter.length) {
        let c3 = 0;
        for (const g of inter) if (g <= 3) c3++;
        baseP3 = c3 / inter.length;
      }
    }
    baseP3 = clamp(Number(baseP3 || 0), 0, 1);
    const ratio = clamp(condP3 / Math.max(0.02, baseP3), 0.2, 1.45);
    out[target] = {
      ratio: roundNum(ratio, 6),
      condP3: roundNum(condP3, 6),
      baseP3: roundNum(baseP3, 6),
      sample: Math.round(totalW),
    };
  }

  return out;
}

function buildAlertSummary(pre, currentIdx, targetsOut) {
  const byTargetUi = {};
  for (const row of (targetsOut || [])) byTargetUi[Number(row.target)] = row;

  const white = buildIndependentWhiteClusterAlert(pre, currentIdx);
  const contextAdjusters = buildWhiteContextB2BAdjusters(pre, currentIdx);
  const whitePressure = clamp(
    (white.risk * 1.02) - (white.release * 0.86) + (white.currentRun >= 3 ? 0.14 : 0),
    0,
    1
  );
  const whiteBlend = clamp(0.18 + (0.82 * whitePressure), 0, 1);

  const b2bByTarget = TARGETS.map((target) => {
    const calc = buildIndependentB2BTargetAlert(pre, target, currentIdx);
    const ui = byTargetUi[target];
    const adj = contextAdjusters[target] || { ratio: 1, condP3: 0, baseP3: 0, sample: 0 };
    const rawProbability = clamp(Number(calc.probability || 0), 0, 1);
    const isHighTarget = target >= 50;
    const targetWhiteBlend = isHighTarget ? (whiteBlend * 0.42) : whiteBlend;
    let ratioAdj = clamp(Number(adj.ratio || 1), isHighTarget ? 0.58 : 0.2, isHighTarget ? 1.6 : 1.45);
    if (calc.highChainActive) {
      ratioAdj = Math.max(ratioAdj, clamp(0.82 + (0.55 * Number(calc.highChainScore || 0)), 0.82, 1.4));
    }
    const blendFactor = (1 - targetWhiteBlend) + (targetWhiteBlend * ratioAdj);
    const effectiveProbability = clamp(rawProbability * blendFactor, 0, 1);
    const baseRef = clamp(Number(adj.baseP3 || 0), 0, 1);
    const lift = clamp(effectiveProbability / Math.max(0.02, baseRef || 0.02), 0, 4);
    const gain = clamp(effectiveProbability - baseRef, -1, 1);
    const liftNorm = clamp((lift - 1) / 1.3, 0, 1);
    const gainNorm = clamp(gain / 0.2, 0, 1);
    const sampleNorm = clamp(Number(calc.samples || 0) / (target >= 100 ? 20 : 28), 0, 1);
    const evidenceNorm = clamp(Number(calc.evidence || 0), 0, 1);
    const highChainNorm = clamp(Number(calc.highChainScore || 0), 0, 1);
    let actionScore = (
      (0.36 * effectiveProbability) +
      (0.27 * liftNorm) +
      (0.18 * gainNorm) +
      (0.09 * sampleNorm) +
      (0.1 * evidenceNorm)
    );
    if (target === 5) {
      actionScore -= 0.05 * (1 - clamp((lift - 1) / 0.7, 0, 1));
    }
    if (target >= 50) {
      actionScore += 0.04 * highChainNorm;
    }
    actionScore = clamp(actionScore, 0, 1);
    // Use a target-normalized display score so B2B does not get permanently pinned to 5x.
    // This highlights relative pressure/lift for higher targets when true signal exists.
    let relativePressure = clamp(
      (0.48 * liftNorm) +
      (0.24 * gainNorm) +
      (0.18 * evidenceNorm) +
      (0.10 * highChainNorm),
      0,
      1
    );
    if (target === 5 && lift < 1.2) {
      relativePressure = clamp(relativePressure * 0.86, 0, 1);
    }
    const displayProbability = clamp(
      target >= 50
        ? ((0.34 * effectiveProbability) + (0.66 * relativePressure))
        : ((0.76 * effectiveProbability) + (0.24 * relativePressure)),
      0,
      1
    );
    return {
      target,
      targetLabel: `${target}x`,
      probability: roundNum(effectiveProbability, 6),
      effectiveProbability: roundNum(effectiveProbability, 6),
      displayProbability: roundNum(displayProbability, 6),
      relativePressure: roundNum(relativePressure, 6),
      rawProbability: roundNum(rawProbability, 6),
      actionScore: roundNum(actionScore, 6),
      lift: roundNum(lift, 6),
      gain: roundNum(gain, 6),
      p1: roundNum(calc.p1, 6),
      p2: roundNum(calc.p2, 6),
      p3: roundNum(calc.p3, 6),
      p5: roundNum(calc.p5, 6),
      aheadLo: calc.aheadLo,
      aheadHi: calc.aheadHi,
      evidence: roundNum(calc.evidence, 4),
      samples: Number(calc.samples || 0),
      whiteContextRatio: roundNum(adj.ratio, 6),
      whiteContextP3: roundNum(adj.condP3, 6),
      baseP3: roundNum(adj.baseP3, 6),
      whiteContextSample: Number(adj.sample || 0),
      highChainActive: Boolean(calc.highChainActive),
      highChainScore: roundNum(calc.highChainScore, 6),
      highChainP3: roundNum(calc.highChainP3, 6),
      highChainSamples: Number(calc.highChainSamples || 0),
      confidence: roundNum(Number(ui?.confidence || 0), 4),
      source: 'historical_pattern_context',
    };
  });

  const byAction = b2bByTarget
    .slice()
    .sort((a, b) => b.actionScore - a.actionScore);
  let topB2B = byAction[0] || null;
  const topNon5Action = byAction.find((row) => Number(row.target) !== 5) || null;
  if (topB2B?.target === 5 && topNon5Action) {
    const closeScore = topNon5Action.actionScore >= (topB2B.actionScore - 0.035);
    const non5Actionable = (
      topNon5Action.effectiveProbability >= 0.1 ||
      topNon5Action.highChainActive ||
      topNon5Action.lift >= 1.28
    );
    if (closeScore && non5Actionable) topB2B = topNon5Action;
  }
  const topB2BByProbability = b2bByTarget
    .slice()
    .sort((a, b) => b.effectiveProbability - a.effectiveProbability)[0] || null;
  const topB2BRaw = b2bByTarget
    .slice()
    .sort((a, b) => b.rawProbability - a.rawProbability)[0] || null;
  const topB2BHigh = b2bByTarget
    .filter((row) => Number(row.target) >= 50)
    .slice()
    .sort((a, b) => b.effectiveProbability - a.effectiveProbability)[0] || null;
  const topB2BHighRaw = b2bByTarget
    .filter((row) => Number(row.target) >= 50)
    .slice()
    .sort((a, b) => b.rawProbability - a.rawProbability)[0] || null;
  const suppressionTarget = Number(topB2BRaw?.target || topB2BByProbability?.target || 0);
  const suppressionEligible = suppressionTarget > 0 && suppressionTarget <= 20;
  const b2bSuppressedByWhite = (
    suppressionEligible &&
    whitePressure >= 0.55 &&
    (topB2BRaw?.rawProbability || 0) > (topB2B?.effectiveProbability || 0) + 0.08
  );
  let dominantSignal = 'neutral';
  if ((topB2BHigh?.effectiveProbability || 0) >= 0.2 && (topB2BHighRaw?.rawProbability || 0) >= 0.18) {
    dominantSignal = 'b2b_spike';
  } else if (b2bSuppressedByWhite || whitePressure >= 0.58 || (white.currentRun >= 3 && white.risk >= 0.52)) {
    dominantSignal = 'white_cluster';
  } else if ((topB2B?.effectiveProbability || 0) >= 0.32) {
    dominantSignal = 'b2b';
  } else if (white.risk >= 0.34) {
    dominantSignal = 'white_watch';
  }
  const hardGapRisk = (targetsOut || []).length
    ? Math.max(...targetsOut.map(t => Number(t?.signals?.hardGapImpulse || 0)))
    : 0;

  return {
    source: 'range-lock-v10-selective',
    generatedAt: new Date().toISOString(),
    b2bByTarget,
    topB2B: topB2B
      ? {
        target: topB2B.target,
        targetLabel: topB2B.targetLabel,
        probability: roundNum(topB2B.effectiveProbability, 6),
        effectiveProbability: roundNum(topB2B.effectiveProbability, 6),
        rawProbability: roundNum(topB2B.rawProbability, 6),
        actionScore: roundNum(topB2B.actionScore, 6),
        lift: roundNum(topB2B.lift, 6),
        gain: roundNum(topB2B.gain, 6),
        aheadLo: topB2B.aheadLo,
        aheadHi: topB2B.aheadHi,
        whiteContextRatio: roundNum(topB2B.whiteContextRatio, 6),
        source: topB2B.source,
      }
      : null,
    topB2BRaw: topB2BRaw
      ? {
        target: topB2BRaw.target,
        targetLabel: topB2BRaw.targetLabel,
        probability: roundNum(topB2BRaw.rawProbability, 6),
        aheadLo: topB2BRaw.aheadLo,
        aheadHi: topB2BRaw.aheadHi,
      }
      : null,
    topB2BHigh: topB2BHigh
      ? {
        target: topB2BHigh.target,
        targetLabel: topB2BHigh.targetLabel,
        probability: roundNum(topB2BHigh.effectiveProbability, 6),
        effectiveProbability: roundNum(topB2BHigh.effectiveProbability, 6),
        rawProbability: roundNum(topB2BHigh.rawProbability, 6),
        aheadLo: topB2BHigh.aheadLo,
        aheadHi: topB2BHigh.aheadHi,
      }
      : null,
    dominantSignal,
    whitePressure: roundNum(whitePressure, 6),
    b2bSuppressedByWhite,
    highB2BPressure: roundNum(topB2BHigh?.effectiveProbability || 0, 6),
    whiteClusterRisk: roundNum(white.risk, 6),
    whiteReleaseSignal: roundNum(white.release, 6),
    whiteCurrentRun: Number(white.currentRun || 0),
    whiteAheadLo: Number(white.aheadLo || 1),
    whiteAheadHi: Number(white.aheadHi || 5),
    whiteSamples: Number(white.samples || 0),
    whiteBaseRate: roundNum(white.whiteRate, 6),
    hardGapRisk: roundNum(hardGapRisk, 6),
  };
}

function computeLockedRangePredictions(rounds, existingLocksRaw = {}, options = {}) {
  const pre = preprocess(rounds || []);
  const calibration = buildCalibrationMap(options.historyRows || [], pre);
  if (pre.n < 800) {
    return {
      model: 'range-lock-v10-selective',
      generatedAt: new Date().toISOString(),
      asOfRound: pre.rounds[pre.n - 1]?.roundId || null,
      targets: [],
      locksToSave: {},
      resolvedHistory: [],
      summary: { pending: 0, windowOpen: 0, relocked: 0, sampleSize: pre.n },
      calibration,
      settings: { windowSpan: WINDOW_SPAN_PRIOR, adaptive: true },
      warning: 'Need at least 800 rounds before reliable range locks.',
    };
  }

  const currentIdx = pre.n - 1;
  const currentRound = pre.rounds[currentIdx].roundId;
  const locksToSave = {};
  const resolvedHistory = [];
  const targetsOut = [];
  const evaluatedByTarget = {};
  let anyResolvedThisTick = false;

  for (const target of TARGETS) {
    const key = String(target);
    const existing = normalizeLockInput(existingLocksRaw[key]);
    const evalResult = evaluateLock(existing, target, pre, currentRound);
    evaluatedByTarget[key] = { existing, evalResult };
    if (evalResult.resolved) anyResolvedThisTick = true;
  }

  let pendingCount = 0;
  let waitingCount = 0;
  let openCount = 0;
  let relockedCount = 0;

  for (const target of TARGETS) {
    const key = String(target);
    const evaluated = evaluatedByTarget[key] || {};
    const existing = evaluated.existing || normalizeLockInput(existingLocksRaw[key]);
    const evalResult = evaluated.evalResult || evaluateLock(existing, target, pre, currentRound);
    const fixedSpan = Math.max(1, Number(WINDOW_SPAN_PRIOR[target] || 3));
    const existingSpan = existing ? Math.max(1, (Number(existing.hi) - Number(existing.lo) + 1)) : null;
    const spanMismatch = Boolean(existing && Number.isFinite(existingSpan) && existingSpan !== fixedSpan);

    let lockToUse = existing;
    let status = 'pending';
    let previousOutcome = null;

    // Keep WAIT windows stable while alive, but refresh if:
    // 1) suspended window already expired, or
    // 2) suspended lock was produced by older waiting model version, or
    // 3) suspended wait has reached its own start, or
    // 4) another target resolved on this tick (global state shift).
    const suspendedExpired = Boolean(
      existing?.suspended &&
      Number.isFinite(Number(existing.hi)) &&
      currentRound > Number(existing.hi)
    );
    const suspendedVersionMismatch = Boolean(
      existing?.suspended &&
      String(existing?.eta?.waitingModelVersion || '') !== WAITING_MODEL_VERSION
    );
    const suspendedReachedStart = Boolean(
      existing?.suspended &&
      Number.isFinite(Number(existing.lo)) &&
      currentRound >= Number(existing.lo)
    );
    const suspendedRecomputeOnAnyResolve = Boolean(
      anyResolvedThisTick &&
      existing?.suspended &&
      !evalResult.resolved
    );
    const suspendedNeedsRefresh = (
      suspendedExpired ||
      suspendedVersionMismatch ||
      suspendedReachedStart ||
      suspendedRecomputeOnAnyResolve
    );

    if (!existing || evalResult.resolved || spanMismatch || suspendedNeedsRefresh) {
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
          lo: existing.lo,
          hi: existing.hi,
          hitRound: evalResult.hitRound,
          generation: existing.generation,
          confidence: Number(existing?.eta?.confidence ?? null),
        });
      }

      const nextLock = buildWindow(pre, target, currentIdx, calibration[target]);
      // Only force non-overlap after a true miss.
      // For win/early we allow next lock to start from next round so engine can adapt.
      const minNextLo = (
        existing &&
        evalResult.resolved &&
        evalResult.outcome === 'loss'
      )
        ? Math.max(currentRound + 1, Number(existing.hi || 0) + 1)
        : (currentRound + 1);
      const activation = shouldActivateWindow(target, nextLock, calibration[target]);
      if (!activation.active) {
        const generation = existing ? Number(existing.generation || 1) : 1;
        // Engine-derived WAIT window with anti-drift guard:
        // do not keep pushing WAIT farther every tick unless the engine brings it earlier.
        const waitingCandidate = buildAdaptiveWaitingWindow(
          nextLock,
          target,
          fixedSpan,
          currentRound,
          minNextLo,
          existing
        );
        let waitingNext = waitingCandidate;
        if (
          existing?.suspended &&
          Number.isFinite(Number(existing.lo)) &&
          Number.isFinite(Number(existing.hi))
        ) {
          const existingLo = Number(existing.lo);
          const existingHi = Number(existing.hi);
          const candidateLo = Number(waitingCandidate.lo);
          // Keep sticky WAIT only while the previous window is still fully ahead.
          // If current round already reached prior lo, allow recompute forward now.
          const canCountdown = existingLo > currentRound;
          const candidatePushesOut = candidateLo > existingLo;
          const existingBlendConfidence = clamp(Number(existing?.eta?.waitingBlendConfidence ?? existing?.eta?.confidence ?? 0), 0, 1);
          const candidateBlendConfidence = clamp(
            Number(waitingCandidate?.eta?.waitingBlendConfidence ?? waitingCandidate?.eta?.confidence ?? 0),
            0,
            1
          );
          const allowPushOut = candidatePushesOut && (candidateBlendConfidence > (existingBlendConfidence + 0.22));
          if (canCountdown && candidatePushesOut && !allowPushOut) {
            waitingNext = {
              ...waitingCandidate,
              lo: existingLo,
              hi: existingHi,
              eta: {
                ...(waitingCandidate.eta || {}),
                waitingSticky: true,
                waitingStickyLo: existingLo,
              },
            };
          }
        }
        const suspendedLo = Number(waitingNext.lo);
        const suspendedHi = Number(waitingNext.hi);
        lockToUse = {
          lo: suspendedLo,
          hi: suspendedHi,
          roundWhenMade: currentRound,
          generation,
          suspended: true,
          eta: {
            ...(waitingNext.eta || {}),
            suspended: true,
            suspendedReason: 'low-signal',
            activationScore: activation.score,
            activationMinConfidence: activation.minConf,
            activationMinP1: activation.minP1,
          },
        };
        status = 'waiting';
      } else {
        const adjustedNext = enforceNextWindowStart(nextLock, fixedSpan, minNextLo);
        const generation = existing ? Number(existing.generation || 1) + 1 : 1;
        lockToUse = {
          lo: adjustedNext.lo,
          hi: adjustedNext.hi,
          roundWhenMade: adjustedNext.roundWhenMade,
          generation,
          suspended: false,
          eta: {
            ...(adjustedNext.eta || {}),
            suspended: false,
            activationScore: activation.score,
            activationMinConfidence: activation.minConf,
            activationMinP1: activation.minP1,
          },
        };
        status = 'locked';
        relockedCount++;
      }
    } else {
      status = evalResult.status || 'pending';
    }

    if (status === 'pending') pendingCount++;
    if (status === 'waiting') waitingCount++;
    if (status === 'window-open') openCount++;

    locksToSave[key] = {
      lo: Number(lockToUse.lo),
      hi: Number(lockToUse.hi),
      roundWhenMade: Number(lockToUse.roundWhenMade),
      generation: Number(lockToUse.generation || 1),
      suspended: Boolean(lockToUse.suspended),
      eta: lockToUse.eta || null,
    };

    targetsOut.push(buildUiTarget(target, locksToSave[key], status, currentRound, previousOutcome));
  }

  targetsOut.sort((a, b) => a.target - b.target);

  return {
    model: 'range-lock-v10-selective',
    generatedAt: new Date().toISOString(),
    asOfRound: currentRound,
    sampleSize: pre.n,
    targets: targetsOut,
    locksToSave,
    resolvedHistory,
    calibration,
    settings: {
      windowSpan: WINDOW_SPAN_PRIOR,
      windowAheadCap: WINDOW_AHEAD_CAP,
      adaptive: true,
      fixedWindowSpan: true,
      regimeAwareQuickHit: true,
      whiteReleaseModel: true,
      hardGapImpulse: true,
      calibrationSpanAdaptive: false,
      selectiveLocking: true,
    },
    alertSummary: buildAlertSummary(pre, currentIdx, targetsOut),
    summary: {
      pending: pendingCount,
      waiting: waitingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: pre.n,
    },
  };
}

module.exports = { TARGETS, computeLockedRangePredictions };
