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
  const highStreak = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    lowStreak[i] = cleanRounds[i].multiplier < 2 ? 1 + (i > 0 ? lowStreak[i - 1] : 0) : 0;
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

  function stateAt(idx, target) {
    const seq = [];
    for (let i = idx - 7; i <= idx; i++) seq.push(tokens[clamp(i, 0, n - 1)]);
    const tr = trendRegimeAt(idx);
    return {
      gapT: gapMaps[target][idx],
      gap5: gapMaps[5][idx],
      gap10: gapMaps[10][idx],
      gap20: gapMaps[20][idx],
      gap50: gapMaps[50][idx],
      gap100: gapMaps[100][idx],
      trend: tr.trend,
      volRatio: tr.volRatio,
      regime: tr.regime,
      under2Rate: tr.under2Rate,
      lowStreak: lowStreak[idx],
      highStreak: highStreak[idx],
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
  d += 0.05 * Math.abs(a.highStreak - b.highStreak);

  let seqMismatch = 0;
  for (let i = 0; i < a.seq.length; i++) {
    const w = 0.6 + ((i + 1) / a.seq.length);
    seqMismatch += w * Math.abs(a.seq[i] - b.seq[i]);
  }
  d += 0.12 * (seqMismatch / a.seq.length);
  return d;
}

function collectNeighbors(pre, target, currentIdx, currentState) {
  const nextMap = pre.nextHitMaps[target];
  const items = [];
  const stats = pre.gapStats[target] || {};
  const maxAhead = Math.max(12, Math.round((stats.q99 || stats.q95 || stats.q90 || stats.mean || 12) * 6));
  const start = 120;
  for (let idx = start; idx < currentIdx; idx++) {
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
  const keep = clamp(Math.round(Math.sqrt(Math.max(1, currentIdx)) * 6), 80, 850);
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
  const baseP = clamp(1 / meanGap, 0.00008, 0.45);

  let factor = 1;
  factor *= 1 + (0.38 * pressure.soft) + (0.75 * pressure.hard);

  if (currentState.regime === 'expansion') factor *= target >= 20 ? 1.16 : 1.1;
  if (currentState.regime === 'compression') factor *= target >= 20 ? 0.84 : 0.9;
  if (currentState.regime === 'chaotic') factor *= target >= 50 ? 1.07 : 1.03;
  if (currentState.regime === 'soft-up') factor *= 1.04;
  if (currentState.regime === 'soft-down') factor *= 0.96;

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
    const neighbors = collectNeighbors(pre, target, idx, state);
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

function buildWindow(pre, target, currentIdx, calibration = null) {
  // === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
  // Justification: adaptive window center/span from blended predictors + white-cluster + calibration feedback.
  const currentRound = pre.rounds[currentIdx].roundId;
  const currentState = pre.stateAt(currentIdx, target);
  const stats = pre.gapStats[target] || { mean: 0, q50: 0, q75: 0, q90: 0, interGaps: [] };
  const gapNow = pre.gapMaps[target][currentIdx];
  const pressure = gapPressure(gapNow, stats);
  const hazard = hazardEta(target, currentState, stats, pressure);
  const neighbors = collectNeighbors(pre, target, currentIdx, currentState);
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
  const quickSignal = quickHitSignal(neighbors, hazard, gapNow, stats);
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

  let centerAhead = q50;
  centerAhead *= 1 + Number(calibration?.shift || 0);
  centerAhead *= 1 - (0.58 * Math.max(0, earlyDominance));
  centerAhead *= 1 + (0.42 * Math.max(0, -earlyDominance));
  centerAhead *= 1 + (whiteSeverity * 0.34);
  centerAhead *= 1 - (pressure.hard * 0.24);

  if (quickSignal > 0) {
    const quickAnchor = Math.max(1, (0.68 * neighQ20) + (0.32 * hazard.q20));
    const quickPull = clamp((quickSignal - 0.08) / 0.62, 0, 1);
    const mix = 0.72 * quickPull;
    centerAhead = ((1 - mix) * centerAhead) + (mix * quickAnchor);
  }

  if (gapNow <= 1 && (pGap2 > 0.03 || pGap3 > 0.08)) {
    const b2bStrength = clamp((0.62 * pGap2) + (0.38 * pGap3), 0, 0.82);
    const b2bAnchor = Math.max(1, (0.74 * neighQ20) + (0.26 * 1.2));
    centerAhead = ((1 - b2bStrength) * centerAhead) + (b2bStrength * b2bAnchor);
  }

  centerAhead = Math.max(1, centerAhead);

  const skewDen = Math.max(0.000001, q80 - q20);
  const leftSkew = clamp((q50 - q20) / skewDen, 0.1, 0.9);
  const halfLeft = Math.round((windowSpan - 1) * leftSkew);

  const dynamicMaxAhead = Math.max(
    windowSpan + 1,
    Math.round((stats.q99 || stats.q95 || stats.q90 || stats.mean || 20) * 6)
  );
  let loAhead = Math.max(1, Math.round(centerAhead) - halfLeft);
  loAhead = Math.min(loAhead, Math.max(1, dynamicMaxAhead - windowSpan + 1));
  const hiAhead = loAhead + windowSpan - 1;

  const componentCenters = [neighQ50, hazard.q50, priorQ50].map(v => Math.log1p(Math.max(1, v)));
  const engineAgreement = clamp(1 / (1 + stddev(componentCenters)), 0, 1);
  const support = clamp(neighbors.length / Math.max(20, Math.sqrt(Math.max(1, pre.n)) * 3), 0, 1);
  const uncertainty = clamp(1 - (spread / Math.max(2, q80 + q20)), 0, 1);
  const calibScale = clamp(Number(calibration?.confidenceScale || 0.5), 0.1, 1);
  const confidence = clamp(
    (0.38 * engineAgreement) +
    (0.24 * support) +
    (0.22 * uncertainty) +
    (0.16 * calibScale),
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
      pGapLe1: roundNum(pGap1, 4),
      pGapLe2: roundNum(pGap2, 4),
      pGapLe3: roundNum(pGap3, 4),
      confidence: roundNum(confidence, 4),
      reason: currentState.regime === 'white'
        ? 'White cluster detected; center shifted using real outcome calibration (fixed target span).'
        : 'Adaptive blend (neighbor + hazard + prior) weighted from backtested real errors (fixed target span).',
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

function evaluateLock(lock, target, pre, currentRound) {
  if (!lock) return { resolved: false, status: 'missing' };
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

  if (currentRound < lo) return { resolved: false, status: 'pending' };
  return { resolved: false, status: 'window-open' };
}

function normalizeLockInput(input) {
  if (!input) return null;
  return {
    lo: Number(input.lo),
    hi: Number(input.hi),
    roundWhenMade: Number(input.roundWhenMade ?? input.round_when_made),
    generation: Number(input.generation || 1),
    eta: input.eta || input.eta_json || null,
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
    reason: lock.eta?.reason || 'Range locked from historical cluster-pattern analogs.',
    previousOutcome,
  };
}

function computeLockedRangePredictions(rounds, existingLocksRaw = {}, options = {}) {
  const pre = preprocess(rounds || []);
  const calibration = buildCalibrationMap(options.historyRows || [], pre);
  if (pre.n < 800) {
    return {
      model: 'range-lock-v7-adaptive',
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

  let pendingCount = 0;
  let openCount = 0;
  let relockedCount = 0;

  for (const target of TARGETS) {
    const key = String(target);
    const existing = normalizeLockInput(existingLocksRaw[key]);
    const evalResult = evaluateLock(existing, target, pre, currentRound);
    const fixedSpan = Math.max(1, Number(WINDOW_SPAN_PRIOR[target] || 3));
    const existingSpan = existing ? Math.max(1, (Number(existing.hi) - Number(existing.lo) + 1)) : null;
    const spanMismatch = Boolean(existing && Number.isFinite(existingSpan) && existingSpan !== fixedSpan);

    let lockToUse = existing;
    let status = 'pending';
    let previousOutcome = null;

    if (!existing || evalResult.resolved || spanMismatch) {
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
      const generation = existing ? Number(existing.generation || 1) + 1 : 1;
      lockToUse = {
        lo: nextLock.lo,
        hi: nextLock.hi,
        roundWhenMade: nextLock.roundWhenMade,
        generation,
        eta: nextLock.eta,
      };
      status = 'locked';
      relockedCount++;
    } else {
      status = evalResult.status || 'pending';
    }

    if (status === 'pending') pendingCount++;
    if (status === 'window-open') openCount++;

    locksToSave[key] = {
      lo: Number(lockToUse.lo),
      hi: Number(lockToUse.hi),
      roundWhenMade: Number(lockToUse.roundWhenMade),
      generation: Number(lockToUse.generation || 1),
      eta: lockToUse.eta || null,
    };

    targetsOut.push(buildUiTarget(target, locksToSave[key], status, currentRound, previousOutcome));
  }

  targetsOut.sort((a, b) => a.target - b.target);

  return {
    model: 'range-lock-v7-adaptive',
    generatedAt: new Date().toISOString(),
    asOfRound: currentRound,
    sampleSize: pre.n,
    targets: targetsOut,
    locksToSave,
    resolvedHistory,
    calibration,
    settings: {
      windowSpan: WINDOW_SPAN_PRIOR,
      adaptive: true,
      fixedWindowSpan: true,
    },
    summary: {
      pending: pendingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: pre.n,
    },
  };
}

module.exports = { TARGETS, computeLockedRangePredictions };
