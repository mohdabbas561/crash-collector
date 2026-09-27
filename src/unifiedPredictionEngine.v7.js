'use strict';

const LockedEngine = (() => {

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
// Justification: detect persistent white clusters and adjust timing/width to reduce late windows.
function whiteClusterSeverity(pre, currentState, target) {
  const p = pre.whiteProfile || {};
  const lowQ85 = Number(p.lowQ85 || 0);
  const lowQ95 = Number(p.lowQ95 || lowQ85 + 1);
  const under2Q85 = Number(p.under2Q85 || 0);
  const under2Q95 = Number(p.under2Q95 || Math.max(under2Q85 + 0.01, 0.01));

  const sLow = clamp((currentState.lowStreak - lowQ85) / Math.max(1, lowQ95 - lowQ85), 0, 1);
  const sRate = clamp((currentState.under2Rate - under2Q85) / Math.max(0.001, under2Q95 - under2Q85), 0, 1);
  const sTrend = currentState.trend < 0 ? clamp((-currentState.trend) / 0.08, 0, 1) : 0;
  const targetScale = clamp(Math.log(Math.max(2, target)) / Math.log(1000), 0.25, 1);
  return clamp(((sLow + sRate + sTrend) / 3) * targetScale, 0, 1);
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

  const whiteSeverity = whiteClusterSeverity(pre, currentState, target);
  const interSorted = [...(stats.interGaps || [])].sort((a, b) => a - b);
  const spread = Math.max(1, q80 - q20);
  const baseSpan = Math.max(2, spread + 1);
  const dataSpan = Math.max(2, (quantile(interSorted, 0.65) - quantile(interSorted, 0.3) + 1));
  const calSpan = Number(calibration?.spanMultiplier || 1);
  const minSpan = Math.max(2, Math.round(quantile(interSorted, 0.1) || WINDOW_SPAN_PRIOR[target] || 3));
  const maxSpan = Math.max(minSpan + 1, Math.round(quantile(interSorted, 0.85) || ((WINDOW_SPAN_PRIOR[target] || 8) * 2.4)));
  let windowSpan = ((baseSpan + dataSpan) * 0.5) * calSpan * (1 + (whiteSeverity * 0.65));
  windowSpan = clamp(Math.round(windowSpan), minSpan, maxSpan);

  let centerAhead = q50;
  centerAhead *= 1 + Number(calibration?.shift || 0);
  centerAhead *= 1 + (whiteSeverity * 0.55);
  centerAhead *= 1 - (pressure.hard * 0.18);
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
      confidence: roundNum(confidence, 4),
      reason: currentState.regime === 'white'
        ? 'White cluster detected; window widened/shifted using real outcome calibration.'
        : 'Adaptive blend (neighbor + hazard + prior) weighted from backtested real errors.',
      calibrationShift: roundNum(calibration?.shift || 0, 4),
      calibrationSample: Number(calibration?.sample || 0),
      calibrationDirectional: Number(calibration?.directionalSamples || 0),
      calibrationError: roundNum(calibration?.meanNormError || 0, 4),
      calibrationWinRate: roundNum(calibration?.winRate || 0, 4),
      calibrationWilsonLow: roundNum(calibration?.wilsonLow || 0.5, 4),
      calibrationSpanMultiplier: roundNum(calibration?.spanMultiplier || 1, 4),
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
      const recency = 1 + ((rows.length - i) / rows.length);
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
    const sampleFactor = clamp(rows.length / 28, 0, 1);
    const shift = clamp(meanNormErr * sampleFactor, -0.42, 0.42);
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

    let lockToUse = existing;
    let status = 'pending';
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
      windowSpan: Object.fromEntries(targetsOut.map(t => [t.target, t.window.span])),
      adaptive: true,
    },
    summary: {
      pending: pendingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: pre.n,
    },
  };
}


  return { TARGETS, computeLockedRangePredictions };
})();

const NextRoundEngine = (() => {

const BUCKETS = [
  { id: 'micro', label: 'Micro', min: 1, max: 1.99, color: '#ff4560' },
  { id: 'low', label: 'Low', min: 2, max: 4.99, color: '#ffd84d' },
  { id: 'mid', label: 'Mid', min: 5, max: 9.99, color: '#00ff88' },
  { id: 'high', label: 'High', min: 10, max: 24.99, color: '#00d4ff' },
  { id: 'moon', label: 'Moon', min: 25, max: Number.POSITIVE_INFINITY, color: '#c084fc' },
];

const THRESHOLDS = [2, 5, 10, 25, 50];
const HORIZONS = [1, 3, 5];
const CACHE_TTL_MS = 0;

const cache = {
  key: null,
  createdAt: 0,
  report: null,
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function roundNum(v, digits = 4) {
  if (!isFiniteNumber(v)) return 0;
  return Number(v.toFixed(digits));
}

function toLog(multiplier) {
  return Math.log(Math.max(1, Number(multiplier) || 1));
}

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

function stddev(arr, avg = null) {
  if (arr.length <= 1) return 0;
  const m = avg == null ? mean(arr) : avg;
  const variance = arr.reduce((sum, v) => sum + ((v - m) ** 2), 0) / arr.length;
  return Math.sqrt(variance);
}

function distributionFromCounts(counts) {
  const total = counts.reduce((sum, c) => sum + c, 0);
  if (!total) return counts.map(() => 0);
  return counts.map(c => c / total);
}

function normalizeDistribution(dist) {
  const sum = dist.reduce((s, v) => s + v, 0);
  if (!sum) {
    const uniform = 1 / Math.max(1, dist.length);
    return dist.map(() => uniform);
  }
  return dist.map(v => v / sum);
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
  if (!Number.isFinite(bucket.max)) return bucket.min * 1.45;
  return (bucket.min + bucket.max) / 2;
}

function encodeToken(multiplier) {
  const m = Number(multiplier) || 1;
  if (m < 1.2) return 0;
  if (m < 1.5) return 1;
  if (m < 2) return 2;
  if (m < 3) return 3;
  if (m < 5) return 4;
  if (m < 10) return 5;
  if (m < 20) return 6;
  if (m < 50) return 7;
  return 8;
}

function roundsSinceHit(rounds, endIdx, threshold) {
  for (let i = endIdx; i >= 0; i--) {
    if (rounds[i].multiplier >= threshold) return endIdx - i;
  }
  return endIdx + 1;
}

function streakLength(rounds, endIdx, predicate, maxDepth = 500) {
  let streak = 0;
  for (let i = endIdx; i >= 0 && streak < maxDepth; i--) {
    if (!predicate(rounds[i].multiplier)) break;
    streak++;
  }
  return streak;
}

function buildFeatureVector(rounds, endIdx, windowSize) {
  const start = endIdx - windowSize + 1;
  const slice = rounds.slice(start, endIdx + 1);
  const logs = slice.map(r => toLog(r.multiplier));
  const avgLog = mean(logs);
  const sdLog = stddev(logs, avgLog);
  const sorted = [...logs].sort((a, b) => a - b);
  const p25 = sorted[Math.floor(sorted.length * 0.25)] ?? sorted[0] ?? 0;
  const p75 = sorted[Math.floor(sorted.length * 0.75)] ?? sorted[sorted.length - 1] ?? 0;
  const recent5 = logs.slice(-5);
  const prev5 = logs.slice(-10, -5);
  const momentum5 = mean(recent5) - mean(prev5.length ? prev5 : logs.slice(0, Math.max(1, logs.length - 5)));
  const lastLog = logs[logs.length - 1] ?? 0;
  const maxLog = sorted[sorted.length - 1] ?? 0;
  const minLog = sorted[0] ?? 0;
  const under2Rate = slice.filter(r => r.multiplier < 2).length / slice.length;
  const over10Rate = slice.filter(r => r.multiplier >= 10).length / slice.length;
  const over25Rate = slice.filter(r => r.multiplier >= 25).length / slice.length;
  const lowStreak = streakLength(rounds, endIdx, m => m < 2);
  const highStreak = streakLength(rounds, endIdx, m => m >= 10);
  const gap2 = roundsSinceHit(rounds, endIdx, 2);
  const gap5 = roundsSinceHit(rounds, endIdx, 5);
  const gap10 = roundsSinceHit(rounds, endIdx, 10);
  const gap25 = roundsSinceHit(rounds, endIdx, 25);

  return [
    avgLog,
    sdLog,
    lastLog,
    momentum5,
    maxLog - minLog,
    p75 - p25,
    under2Rate,
    over10Rate,
    over25Rate,
    lowStreak,
    highStreak,
    gap2,
    gap5,
    gap10,
    gap25,
    lastLog - avgLog,
  ];
}

function computeFeatureStats(rows) {
  if (!rows.length) return { means: [], stds: [] };
  const dims = rows[0].length;
  const means = new Array(dims).fill(0);
  const stds = new Array(dims).fill(0);

  for (const row of rows) {
    for (let d = 0; d < dims; d++) means[d] += row[d];
  }
  for (let d = 0; d < dims; d++) means[d] /= rows.length;

  for (const row of rows) {
    for (let d = 0; d < dims; d++) stds[d] += ((row[d] - means[d]) ** 2);
  }
  for (let d = 0; d < dims; d++) stds[d] = Math.sqrt(stds[d] / rows.length) || 1;
  return { means, stds };
}

function normalizeFeature(row, stats) {
  return row.map((v, i) => (v - stats.means[i]) / (stats.stds[i] || 1));
}

function squaredDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function initCentroids(vectors, k) {
  const n = vectors.length;
  const centroids = [];
  if (!n || k <= 0) return centroids;

  let seedIndex = Math.floor(n * 0.31);
  seedIndex = clamp(seedIndex, 0, n - 1);
  centroids.push([...vectors[seedIndex]]);

  while (centroids.length < k) {
    let bestIdx = 0;
    let bestDist = -1;
    for (let i = 0; i < n; i++) {
      let minDist = Number.POSITIVE_INFINITY;
      for (const c of centroids) {
        minDist = Math.min(minDist, squaredDistance(vectors[i], c));
      }
      if (minDist > bestDist) {
        bestDist = minDist;
        bestIdx = i;
      }
    }
    centroids.push([...vectors[bestIdx]]);
  }
  return centroids;
}

function runKMeans(vectors, k, maxIter = 12) {
  if (!vectors.length || k <= 0) {
    return { centroids: [], assignments: [], counts: [] };
  }
  const dim = vectors[0].length;
  const centroids = initCentroids(vectors, k);
  const assignments = new Array(vectors.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    const sums = Array.from({ length: k }, () => new Array(dim).fill(0));
    const counts = new Array(k).fill(0);

    for (let i = 0; i < vectors.length; i++) {
      let bestCluster = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        const d = squaredDistance(vectors[i], centroids[c]);
        if (d < bestDist) {
          bestDist = d;
          bestCluster = c;
        }
      }
      if (assignments[i] !== bestCluster) changed = true;
      assignments[i] = bestCluster;
      counts[bestCluster]++;
      for (let d = 0; d < dim; d++) sums[bestCluster][d] += vectors[i][d];
    }

    for (let c = 0; c < k; c++) {
      if (!counts[c]) continue;
      for (let d = 0; d < dim; d++) centroids[c][d] = sums[c][d] / counts[c];
    }

    if (!changed) {
      return { centroids, assignments, counts };
    }
  }

  const counts = new Array(k).fill(0);
  for (const a of assignments) counts[a]++;
  return { centroids, assignments, counts };
}

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: choose cluster count from data complexity (BIC-like criterion), not fixed formula.
function chooseAdaptiveK(vectors) {
  if (!vectors.length) return 0;
  if (vectors.length < 160) return clamp(Math.round(Math.sqrt(vectors.length / 10)), 3, 5);
  const dim = vectors[0].length || 1;
  const maxK = clamp(Math.round(Math.sqrt(vectors.length / 10)), 4, 12);
  let best = { k: 4, bic: Number.POSITIVE_INFINITY };

  for (let k = 4; k <= maxK; k++) {
    const km = runKMeans(vectors, k, 8);
    let sse = 0;
    for (let i = 0; i < vectors.length; i++) {
      const c = km.assignments[i];
      sse += squaredDistance(vectors[i], km.centroids[c]);
    }
    const mse = Math.max(1e-9, sse / Math.max(1, vectors.length));
    const bic = (vectors.length * Math.log(mse)) + (k * dim * Math.log(vectors.length));
    if (bic < best.bic) best = { k, bic };
  }
  return best.k;
}
// === UPGRADE END ===

function nearestCentroidIndex(feature, centroids) {
  let best = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let c = 0; c < centroids.length; c++) {
    const d = squaredDistance(feature, centroids[c]);
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}

function createAccumulator() {
  const thresholdHits = {};
  for (const t of THRESHOLDS) {
    thresholdHits[t] = {};
    for (const h of HORIZONS) thresholdHits[t][h] = 0;
  }
  return {
    count: 0,
    bucketCounts: new Array(BUCKETS.length).fill(0),
    nextMultSum: 0,
    thresholdHits,
  };
}

function accumulateSample(acc, sample, weight = 1) {
  acc.count += weight;
  acc.bucketCounts[sample.nextBucket] += weight;
  acc.nextMultSum += sample.nextMult * weight;

  for (const t of THRESHOLDS) {
    for (const h of HORIZONS) {
      if (sample.futureMax[h] >= t) acc.thresholdHits[t][h] += weight;
    }
  }
}

function finalizeAccumulator(acc) {
  const count = acc.count || 0;
  const dist = normalizeDistribution(distributionFromCounts(acc.bucketCounts));
  const thresholdProbabilities = {};
  for (const t of THRESHOLDS) {
    thresholdProbabilities[t] = {};
    for (const h of HORIZONS) {
      thresholdProbabilities[t][h] = count ? acc.thresholdHits[t][h] / count : 0;
    }
  }
  return {
    count,
    bucketDistribution: dist,
    meanNextMultiplier: count ? acc.nextMultSum / count : 0,
    thresholdProbabilities,
  };
}

function patternSimilarity(currentTokens, currentLogs, candidateTokens, candidateLogs) {
  let weightedSum = 0;
  let weightTotal = 0;
  for (let i = 0; i < currentTokens.length; i++) {
    const w = 1 + (i / currentTokens.length) * 2.3;
    const tokenDiff = Math.abs(currentTokens[i] - candidateTokens[i]);
    const tokenScore = clamp(1 - tokenDiff / 8, 0, 1);
    const logDiff = Math.abs(currentLogs[i] - candidateLogs[i]);
    const valueScore = clamp(1 - logDiff / 2.6, 0, 1);
    weightedSum += w * ((tokenScore * 0.62) + (valueScore * 0.38));
    weightTotal += w;
  }
  return weightTotal ? weightedSum / weightTotal : 0;
}

function findPatternMatches(rounds, patternWindow) {
  const maxH = Math.max(...HORIZONS);
  const n = rounds.length;
  const tokens = rounds.map(r => encodeToken(r.multiplier));
  const logs = rounds.map(r => toLog(r.multiplier));
  const currentTokens = tokens.slice(n - patternWindow);
  const currentLogs = logs.slice(n - patternWindow);
  const rawMatches = [];

  const maxEndIdx = Math.min(n - maxH - 1, n - patternWindow - 1);
  for (let endIdx = patternWindow - 1; endIdx <= maxEndIdx; endIdx++) {
    const start = endIdx - patternWindow + 1;
    const candidateTokens = tokens.slice(start, endIdx + 1);
    const candidateLogs = logs.slice(start, endIdx + 1);
    const similarity = patternSimilarity(currentTokens, currentLogs, candidateTokens, candidateLogs);

    const futureMax = {};
    for (const h of HORIZONS) {
      let mx = 0;
      for (let j = endIdx + 1; j <= endIdx + h; j++) {
        mx = Math.max(mx, rounds[j].multiplier);
      }
      futureMax[h] = mx;
    }

    const nextMult = rounds[endIdx + 1].multiplier;
    rawMatches.push({
      startRoundId: rounds[start].roundId,
      endRoundId: rounds[endIdx].roundId,
      nextRoundId: rounds[endIdx + 1].roundId,
      nextMult,
      nextBucket: bucketIndex(nextMult),
      similarity,
      futureMax,
    });
  }

  if (!rawMatches.length) return [];
  const simSorted = rawMatches.map(m => m.similarity).sort((a, b) => a - b);
  const dynCutoff = quantile(simSorted, 0.82);
  return rawMatches
    .filter(m => m.similarity >= dynCutoff)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 260);
}

function aggregatePatternStats(matches) {
  if (!matches.length) {
    return {
      count: 0,
      avgSimilarity: 0,
      bucketDistribution: normalizeDistribution(new Array(BUCKETS.length).fill(1)),
      meanNextMultiplier: 0,
      thresholdProbabilities: Object.fromEntries(
        THRESHOLDS.map(t => [t, Object.fromEntries(HORIZONS.map(h => [h, 0]))])
      ),
    };
  }

  const acc = createAccumulator();
  let similaritySum = 0;
  for (const m of matches) {
    const weight = Math.max(0.0001, m.similarity ** 2.8);
    similaritySum += m.similarity;
    accumulateSample(acc, m, weight);
  }
  const out = finalizeAccumulator(acc);
  out.avgSimilarity = similaritySum / matches.length;
  return out;
}

function buildMarkovModel(rounds) {
  if (rounds.length < 4) {
    return {
      mode: 'none',
      support: 0,
      distribution: normalizeDistribution(new Array(BUCKETS.length).fill(1)),
    };
  }

  const tokens = rounds.map(r => encodeToken(r.multiplier));
  const pairState = `${tokens[tokens.length - 2]}|${tokens[tokens.length - 1]}`;
  const pairCounts = new Array(BUCKETS.length).fill(0);
  let pairSupport = 0;

  for (let i = 1; i < rounds.length - 1; i++) {
    const state = `${tokens[i - 1]}|${tokens[i]}`;
    if (state !== pairState) continue;
    pairCounts[bucketIndex(rounds[i + 1].multiplier)]++;
    pairSupport++;
  }

  if (pairSupport >= 20) {
    return {
      mode: 'pair',
      support: pairSupport,
      distribution: normalizeDistribution(distributionFromCounts(pairCounts)),
    };
  }

  const tokenState = tokens[tokens.length - 1];
  const tokenCounts = new Array(BUCKETS.length).fill(0);
  let tokenSupport = 0;

  for (let i = 0; i < rounds.length - 1; i++) {
    if (tokens[i] !== tokenState) continue;
    tokenCounts[bucketIndex(rounds[i + 1].multiplier)]++;
    tokenSupport++;
  }

  if (tokenSupport >= 20) {
    return {
      mode: 'single',
      support: tokenSupport,
      distribution: normalizeDistribution(distributionFromCounts(tokenCounts)),
    };
  }

  return {
    mode: 'global',
    support: tokenSupport,
    distribution: normalizeDistribution(new Array(BUCKETS.length).fill(1)),
  };
}

function entropy(dist) {
  const safe = dist.filter(p => p > 0);
  if (!safe.length) return 1;
  const h = -safe.reduce((sum, p) => sum + (p * Math.log(p)), 0);
  return h / Math.log(dist.length || 1);
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  if (!aa || !bb) return 0;
  return dot / (Math.sqrt(aa) * Math.sqrt(bb));
}

// === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
// Justification: blend weights become evidence-driven from support + entropy, avoiding static bias.
function blendWeights({ clusterStats, patternStats, markovStats, baselineStats }) {
  const qCluster = Math.max(0.000001, (1 - entropy(clusterStats.bucketDistribution)) * Math.log1p(clusterStats.count || 0));
  const qPattern = Math.max(0.000001, (1 - entropy(patternStats.bucketDistribution)) * Math.log1p(patternStats.count || 0) * Math.max(0.05, patternStats.avgSimilarity || 0.2));
  const qMarkov = Math.max(0.000001, (1 - entropy(markovStats.distribution)) * Math.log1p(markovStats.support || 0));
  const qBaseline = Math.max(0.000001, 1 - entropy(baselineStats.bucketDistribution));
  const total = qCluster + qPattern + qMarkov + qBaseline;
  return {
    cluster: qCluster / total,
    pattern: qPattern / total,
    markov: qMarkov / total,
    baseline: qBaseline / total,
  };
}
// === UPGRADE END ===

function blendDistribution({ weights, clusterDist, patternDist, markovDist, baselineDist }) {
  const out = new Array(BUCKETS.length).fill(0);
  for (let i = 0; i < BUCKETS.length; i++) {
    out[i] =
      (weights.cluster * clusterDist[i]) +
      (weights.pattern * patternDist[i]) +
      (weights.markov * markovDist[i]) +
      (weights.baseline * baselineDist[i]);
  }
  return normalizeDistribution(out);
}

function probabilityFromDistribution(dist, threshold) {
  let p = 0;
  for (let i = 0; i < BUCKETS.length; i++) {
    const b = BUCKETS[i];
    if (threshold <= b.min) {
      p += dist[i];
      continue;
    }
    if (threshold > b.max) continue;
    if (!Number.isFinite(b.max)) {
      p += dist[i];
      continue;
    }
    const span = Math.max(0.0001, b.max - b.min);
    const hitPart = clamp((b.max - threshold) / span, 0, 1);
    p += dist[i] * hitPart;
  }
  return clamp(p, 0, 1);
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const qq = clamp(q, 0, 1);
  const idx = (sorted.length - 1) * qq;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function estimateQuantileFromDistribution(dist, q) {
  const qq = clamp(q, 0, 0.9999);
  let running = 0;
  for (let i = 0; i < BUCKETS.length; i++) {
    const p = dist[i];
    if (qq <= running + p || i === BUCKETS.length - 1) {
      const b = BUCKETS[i];
      const local = p > 0 ? clamp((qq - running) / p, 0, 1) : 0.5;
      if (!Number.isFinite(b.max)) {
        return b.min * (1 + (local * 1.5));
      }
      return b.min + ((b.max - b.min) * local);
    }
    running += p;
  }
  return BUCKETS[BUCKETS.length - 1].min;
}

function computeTrendContext(rounds) {
  const logs = rounds.map(r => toLog(r.multiplier));
  const last24 = logs.slice(-24);
  const prev24 = logs.slice(-48, -24);
  const last80 = logs.slice(-80);
  const prev80 = logs.slice(-160, -80);
  const shortTrend = mean(last24) - mean(prev24.length ? prev24 : logs.slice(0, Math.max(1, logs.length - 24)));
  const longTrend = mean(last80) - mean(prev80.length ? prev80 : logs.slice(0, Math.max(1, logs.length - 80)));
  const trendScore = clamp(((shortTrend * 0.65) + (longTrend * 0.35)) / 0.28, -1, 1);

  const volRecent = stddev(last24);
  const volBase = stddev(logs.slice(-200));
  const volRatio = volBase > 0 ? volRecent / volBase : 1;
  const volatilityScore = clamp((volRatio - 0.85) / 0.75, 0, 1.5);

  let regime = 'balanced';
  if (trendScore <= -0.4 && volRatio <= 1) regime = 'compression';
  else if (trendScore >= 0.35 && volRatio >= 1.05) regime = 'expansion';
  else if (volRatio >= 1.2) regime = 'chaotic';
  else if (trendScore <= -0.2) regime = 'soft-down';
  else if (trendScore >= 0.2) regime = 'soft-up';

  return { trendScore, volRatio, volatilityScore, regime };
}

function buildGapProfile(rounds, threshold) {
  const hitIndexes = [];
  for (let i = 0; i < rounds.length; i++) {
    if (rounds[i].multiplier >= threshold) hitIndexes.push(i);
  }

  const gaps = [];
  for (let i = 1; i < hitIndexes.length; i++) gaps.push(hitIndexes[i] - hitIndexes[i - 1]);
  const sorted = [...gaps].sort((a, b) => a - b);
  const q75 = sorted.length ? quantile(sorted, 0.75) : 0;
  const q90 = sorted.length ? quantile(sorted, 0.9) : 0;
  const avg = sorted.length ? mean(sorted) : 0;
  const sd = sorted.length ? stddev(sorted, avg) : 0;
  const currentGap = roundsSinceHit(rounds, rounds.length - 1, threshold);

  const softDen = Math.max(2, (q90 - q75) + 1);
  const hardDen = Math.max(3, (q90 * 0.35) + 2);
  const soft = clamp((currentGap - q75) / softDen, 0, 1);
  const hard = clamp((currentGap - q90) / hardDen, 0, 1);
  const z = sd > 0 ? (currentGap - avg) / sd : 0;

  return {
    threshold,
    currentGap,
    q75: roundNum(q75, 2),
    q90: roundNum(q90, 2),
    avg: roundNum(avg, 2),
    z: roundNum(z, 3),
    soft: roundNum(soft, 4),
    hard: roundNum(hard, 4),
  };
}

function buildGapPressure(rounds) {
  const out = {};
  for (const t of THRESHOLDS) out[t] = buildGapProfile(rounds, t);
  return out;
}

function applyGapAndRegimeAdjustments(baseDist, trend, gaps) {
  const g2 = gaps[2] || {};
  const g5 = gaps[5] || {};
  const g10 = gaps[10] || {};
  const g25 = gaps[25] || {};
  const g50 = gaps[50] || {};

  const boosts = new Array(BUCKETS.length).fill(0);
  boosts[1] += (0.06 * (g2.soft || 0)) + (0.11 * (g2.hard || 0));
  boosts[2] += (0.08 * (g5.soft || 0)) + (0.17 * (g5.hard || 0)) + (0.03 * (g2.soft || 0));
  boosts[3] += (0.11 * (g10.soft || 0)) + (0.23 * (g10.hard || 0)) + (0.05 * (g5.soft || 0));
  boosts[4] += (0.14 * (g25.soft || 0)) + (0.28 * (g25.hard || 0)) + (0.08 * (g50.soft || 0)) + (0.2 * (g50.hard || 0));

  const totalSoft = (g2.soft || 0) + (g5.soft || 0) + (g10.soft || 0) + (g25.soft || 0) + (g50.soft || 0);
  const totalHard = (g2.hard || 0) + (g5.hard || 0) + (g10.hard || 0) + (g25.hard || 0) + (g50.hard || 0);
  boosts[0] -= (0.03 * totalSoft) + (0.06 * totalHard);

  if (trend.trendScore > 0) {
    boosts[2] += 0.05 * trend.trendScore;
    boosts[3] += 0.08 * trend.trendScore;
    boosts[4] += 0.1 * trend.trendScore;
    boosts[0] -= 0.06 * trend.trendScore;
  } else if (trend.trendScore < 0) {
    const down = -trend.trendScore;
    boosts[0] += 0.08 * down;
    boosts[1] += 0.05 * down;
    boosts[3] -= 0.04 * down;
    boosts[4] -= 0.05 * down;
  }

  const volBoost = clamp((trend.volatilityScore - 0.35) / 0.9, 0, 1);
  boosts[3] += 0.05 * volBoost;
  boosts[4] += 0.08 * volBoost;

  const adjusted = baseDist.map((p, i) => p * clamp(1 + boosts[i], 0.08, 2.8));
  return {
    adjustedDistribution: normalizeDistribution(adjusted),
    bucketBoosts: boosts.map(v => roundNum(v, 4)),
  };
}

function pickTargetFromCandidates(dist, candidates, mode) {
  let best = {
    target: candidates[0],
    hitChance: 0,
    edge: -1,
    score: -1,
  };

  for (const target of candidates) {
    const hitChance = probabilityFromDistribution(dist, target);
    const edge = (hitChance * target) - 1;
    let score = 0;
    if (mode === 'safe') {
      score = (hitChance ** 1.7) * (target ** 0.35) + (edge * 0.2);
    } else if (mode === 'aggressive') {
      score = (hitChance * target) * (1 + ((1 - hitChance) * 0.85)) + (edge * 0.6) + (target >= 5 ? 0.08 : 0);
    } else {
      score = (hitChance * target) * (0.75 + (0.25 * hitChance)) + (edge * 0.4);
    }
    if (score > best.score) best = { target, hitChance, edge, score };
  }
  return best;
}

function buildCashoutPlan(dist, predictedBucket, confidence, gaps) {
  const p2 = probabilityFromDistribution(dist, 2);
  const hard10 = gaps[10]?.hard || 0;
  const hard25 = gaps[25]?.hard || 0;
  const bullish = predictedBucket.id === 'mid' || predictedBucket.id === 'high' || predictedBucket.id === 'moon' || hard10 > 0.45 || hard25 > 0.3;
  const bearish = predictedBucket.id === 'micro' && confidence >= 0.62 && p2 < 0.52;

  let safeCandidates = [1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6];
  let balancedCandidates = [1.5, 1.6, 1.8, 2, 2.2, 2.5, 3];
  let aggressiveCandidates = [2, 2.5, 3, 4, 5, 7, 10];

  if (bullish) {
    safeCandidates = [1.4, 1.5, 1.6, 1.8, 2, 2.2];
    balancedCandidates = [2, 2.2, 2.5, 3, 4, 5];
    aggressiveCandidates = [3, 4, 5, 7, 10, 15];
  }

  const safe = pickTargetFromCandidates(dist, safeCandidates, 'safe');
  const balanced = pickTargetFromCandidates(dist, balancedCandidates, 'balanced');
  const aggressive = pickTargetFromCandidates(dist, aggressiveCandidates, 'aggressive');

  let recommended = balanced;
  let recommendedLabel = 'BALANCED';
  let reason = 'Best risk/reward for current regime.';

  if (bearish && safe.hitChance >= 0.62) {
    recommended = safe;
    recommendedLabel = 'SAFE';
    reason = 'Micro-pressure is high; protect capital with a tighter exit.';
  } else if (bullish && aggressive.hitChance >= 0.2 && aggressive.score > (balanced.score * 1.08)) {
    recommended = aggressive;
    recommendedLabel = 'AGGRESSIVE';
    reason = 'Expansion pressure detected from hard gaps and trend.';
  } else if (safe.score > (balanced.score * 1.14) && confidence > 0.7) {
    recommended = safe;
    recommendedLabel = 'SAFE';
    reason = 'High confidence with compressed regime favors safer extraction.';
  }

  const pad = recommended.target <= 1.5
    ? 0.08
    : recommended.target <= 3
      ? 0.18
      : recommended.target <= 6
        ? 0.36
        : recommended.target * 0.12;

  const toItem = (x) => ({
    target: roundNum(x.target, 2),
    hitChance: roundNum(x.hitChance, 4),
    edge: roundNum(x.edge, 4),
  });

  return {
    safe: toItem(safe),
    balanced: toItem(balanced),
    aggressive: toItem(aggressive),
    recommended: toItem(recommended),
    recommendedLabel,
    zoneLow: roundNum(Math.max(1.05, recommended.target - pad), 2),
    zoneHigh: roundNum(recommended.target + pad, 2),
    reason,
  };
}

function buildSignals(context) {
  const out = [];
  const {
    lowStreak,
    highStreak,
    patternCount,
    clusterRegime,
    topBucket,
    trendRegime,
    gapPressure,
    recommendedCashout,
  } = context;

  const g10 = gapPressure[10];
  const g25 = gapPressure[25];
  const g50 = gapPressure[50];

  if (lowStreak >= 5) {
    out.push(`Low streak is ${lowStreak} rounds; rebound pressure usually increases after long micro runs.`);
  }
  if (highStreak >= 2) {
    out.push(`Back-to-back high multipliers (${highStreak}) detected; volatility regime is elevated.`);
  }
  if (g10 && (g10.soft > 0 || g10.hard > 0)) {
    out.push(`10x gap ${g10.currentGap} rounds | soft ${roundNum(g10.soft * 100, 1)}% | hard ${roundNum(g10.hard * 100, 1)}% pressure.`);
  }
  if (g25 && (g25.soft > 0 || g25.hard > 0)) {
    out.push(`25x gap ${g25.currentGap} rounds | soft ${roundNum(g25.soft * 100, 1)}% | hard ${roundNum(g25.hard * 100, 1)}% pressure.`);
  }
  if (g50 && g50.hard > 0.2) {
    out.push(`50x hard-gap pressure active (${roundNum(g50.hard * 100, 1)}%), tail spikes can appear abruptly.`);
  }
  if (patternCount >= 60) {
    out.push(`Pattern engine found ${patternCount} close historical analogs, improving signal stability.`);
  } else if (patternCount > 0) {
    out.push(`Pattern analog count is ${patternCount}; confidence depends more on cluster and baseline structure.`);
  }

  out.push(`Regime: cluster=${clusterRegime}, trend=${trendRegime}; model leans ${topBucket.label} (${topBucket.min}x+).`);
  if (recommendedCashout) {
    out.push(`Recommended ${recommendedCashout.recommendedLabel} cashout near ${recommendedCashout.recommended.target.toFixed(2)}x (${recommendedCashout.reason})`);
  }
  return out.slice(0, 6);
}

function computeReport(rounds) {
  const cleanRounds = rounds
    .map(r => ({
      roundId: Number(r.roundId),
      multiplier: Number(r.multiplier),
      timestamp: Number(r.timestamp) || Date.now(),
    }))
    .filter(r => Number.isFinite(r.roundId) && Number.isFinite(r.multiplier) && r.multiplier > 0)
    .sort((a, b) => a.roundId - b.roundId);

  if (cleanRounds.length < 200) {
    const uniform = normalizeDistribution(new Array(BUCKETS.length).fill(1));
    const asOfRound = cleanRounds[cleanRounds.length - 1]?.roundId || null;
    const fallbackCashout = {
      safe: { target: 1.3, hitChance: 0.7, edge: -0.09 },
      balanced: { target: 1.8, hitChance: 0.45, edge: -0.19 },
      aggressive: { target: 3, hitChance: 0.25, edge: -0.25 },
      recommended: { target: 1.8, hitChance: 0.45, edge: -0.19 },
      recommendedLabel: 'BALANCED',
      zoneLow: 1.62,
      zoneHigh: 1.98,
      reason: 'Insufficient training depth, using neutral fallback.',
    };
    return {
      model: 'cluster-pattern-hybrid-v7',
      generatedAt: new Date().toISOString(),
      asOfRound,
      sampleSize: cleanRounds.length,
      expectedMultiplier: roundNum(mean(cleanRounds.map(r => r.multiplier)), 4),
      expectedMedian: roundNum(mean(cleanRounds.map(r => r.multiplier)), 4),
      predictedBucket: {
        ...BUCKETS[0],
        probability: roundNum(1 / BUCKETS.length, 4),
        confidence: 0.15,
        confidenceBand: 'low',
      },
      bucketProbabilities: BUCKETS.map((b, i) => ({ ...b, probability: roundNum(uniform[i], 4) })),
      targetProbabilities: THRESHOLDS.map(t => ({
        target: t,
        gapNow: asOfRound == null ? 0 : roundsSinceHit(cleanRounds, cleanRounds.length - 1, t),
        p1: 0,
        p3: 0,
        p5: 0,
        expectedGap: null,
      })),
      diagnostics: {
        message: 'Not enough data for cluster-pattern modeling yet (need at least 200 rounds).',
      },
      cashoutPlan: fallbackCashout,
      similarPatterns: [],
      signals: ['Insufficient history. Continue collecting rounds to activate full engine.'],
    };
  }

  const windowSize = 24;
  const patternWindow = 16;
  const maxH = Math.max(...HORIZONS);

  const samples = [];
  for (let endIdx = windowSize - 1; endIdx < cleanRounds.length - maxH; endIdx++) {
    const featureRaw = buildFeatureVector(cleanRounds, endIdx, windowSize);
    if (!featureRaw.every(isFiniteNumber)) continue;

    const nextMult = cleanRounds[endIdx + 1].multiplier;
    const futureMax = {};
    for (const h of HORIZONS) {
      let mx = 0;
      for (let j = endIdx + 1; j <= endIdx + h; j++) {
        mx = Math.max(mx, cleanRounds[j].multiplier);
      }
      futureMax[h] = mx;
    }

    samples.push({
      endIdx,
      featureRaw,
      nextMult,
      nextBucket: bucketIndex(nextMult),
      futureMax,
    });
  }

  const featureRows = samples.map(s => s.featureRaw);
  const featureStats = computeFeatureStats(featureRows);
  for (const s of samples) s.featureNorm = normalizeFeature(s.featureRaw, featureStats);

  const k = chooseAdaptiveK(samples.map(s => s.featureNorm));
  const { centroids, assignments } = runKMeans(samples.map(s => s.featureNorm), k, 12);

  const clusterAccumulators = Array.from({ length: k }, () => createAccumulator());
  const baselineAcc = createAccumulator();
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const clusterId = assignments[i];
    accumulateSample(clusterAccumulators[clusterId], sample, 1);
    accumulateSample(baselineAcc, sample, 1);
  }
  const clusterStats = clusterAccumulators.map(a => finalizeAccumulator(a));
  const baselineStats = finalizeAccumulator(baselineAcc);

  const currentEndIdx = cleanRounds.length - 1;
  const currentFeatureRaw = buildFeatureVector(cleanRounds, currentEndIdx, windowSize);
  const currentFeatureNorm = normalizeFeature(currentFeatureRaw, featureStats);
  const currentClusterId = nearestCentroidIndex(currentFeatureNorm, centroids);
  const currentClusterStats = clusterStats[currentClusterId] || baselineStats;

  const patternMatches = findPatternMatches(cleanRounds, patternWindow);
  const patternStats = aggregatePatternStats(patternMatches);

  const markov = buildMarkovModel(cleanRounds);
  const weights = blendWeights({
    clusterStats: currentClusterStats,
    patternStats,
    markovStats: markov,
    baselineStats,
  });

  const blendedBucketDist = blendDistribution({
    weights,
    clusterDist: currentClusterStats.bucketDistribution,
    patternDist: patternStats.bucketDistribution,
    markovDist: markov.distribution,
    baselineDist: baselineStats.bucketDistribution,
  });

  const trendContext = computeTrendContext(cleanRounds);
  const gapPressure = buildGapPressure(cleanRounds);
  const gapAdjusted = applyGapAndRegimeAdjustments(
    blendedBucketDist,
    trendContext,
    gapPressure
  );
  let finalBucketDist = gapAdjusted.adjustedDistribution;
  const bucketBoosts = gapAdjusted.bucketBoosts;

  // === v7 SUPERVISED LEARNING & ADAPTIVE UPGRADE START ===
  // Justification: reduce false bullish calls in prolonged white clusters.
  const lowStreakNow = streakLength(cleanRounds, cleanRounds.length - 1, m => m < 2);
  const lowStreakHistory = [];
  for (let i = 0; i < cleanRounds.length; i++) {
    lowStreakHistory.push(streakLength(cleanRounds, i, m => m < 2, 300));
  }
  const lowSorted = [...lowStreakHistory].sort((a, b) => a - b);
  const lowQ85 = quantile(lowSorted, 0.85);
  const lowQ95 = quantile(lowSorted, 0.95);
  const whiteSeverity = clamp((lowStreakNow - lowQ85) / Math.max(1, lowQ95 - lowQ85), 0, 1);
  if (whiteSeverity > 0) {
    const adjusted = [...finalBucketDist];
    const boost = 0.18 * whiteSeverity;
    adjusted[0] *= (1 + boost);
    adjusted[1] *= (1 + (boost * 0.55));
    adjusted[3] *= (1 - (boost * 0.35));
    adjusted[4] *= (1 - (boost * 0.5));
    finalBucketDist = normalizeDistribution(adjusted);
  }
  // === UPGRADE END ===

  const expectedFromBuckets = (dist) => {
    let out = 0;
    for (let i = 0; i < BUCKETS.length; i++) {
      out += dist[i] * bucketMidpoint(BUCKETS[i]);
    }
    return out;
  };

  const rawExpectedMultiplier =
    (weights.cluster * currentClusterStats.meanNextMultiplier) +
    (weights.pattern * (patternStats.meanNextMultiplier || expectedFromBuckets(patternStats.bucketDistribution))) +
    (weights.markov * expectedFromBuckets(markov.distribution)) +
    (weights.baseline * baselineStats.meanNextMultiplier);

  const expectedMedian = estimateQuantileFromDistribution(finalBucketDist, 0.5);
  const expectedP75 = estimateQuantileFromDistribution(finalBucketDist, 0.75);
  const expectedP90 = estimateQuantileFromDistribution(finalBucketDist, 0.9);
  const distExpectedMean = expectedFromBuckets(finalBucketDist);
  const expectedMultiplier = expectedMedian;

  const topBucketIdx = finalBucketDist.reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0);
  const topBucket = BUCKETS[topBucketIdx];
  const maxProb = finalBucketDist[topBucketIdx];
  const sharpness = 1 - entropy(finalBucketDist);
  const evidence = clamp((currentClusterStats.count + patternStats.count + markov.support) / 1600, 0, 1);
  const pressureEvidence = clamp(
    ((gapPressure[10]?.soft || 0) + (gapPressure[10]?.hard || 0) + (gapPressure[25]?.soft || 0) + (gapPressure[25]?.hard || 0)) / 2,
    0,
    1
  );
  const alignment = clamp(
    (
      cosineSimilarity(currentClusterStats.bucketDistribution, patternStats.bucketDistribution) +
      cosineSimilarity(currentClusterStats.bucketDistribution, markov.distribution) +
      cosineSimilarity(patternStats.bucketDistribution, markov.distribution)
    ) / 3,
    0,
    1
  );
  const confidence = clamp(
    0.15 + (0.42 * maxProb) + (0.18 * sharpness) + (0.1 * evidence) + (0.08 * alignment) + (0.07 * pressureEvidence),
    0.05,
    0.97
  );
  const confidenceBand = confidence >= 0.78 ? 'high' : confidence >= 0.6 ? 'medium' : 'low';

  const targetProbabilities = THRESHOLDS.map((threshold) => {
    const g = gapPressure[threshold] || { currentGap: 0, soft: 0, hard: 0 };
    const gapNow = g.currentGap;
    const p1FromDist = probabilityFromDistribution(finalBucketDist, threshold);
    const markovP1 = probabilityFromDistribution(markov.distribution, threshold);
    const pressureBoost = (0.08 * g.soft) + (0.16 * g.hard);

    const byHorizon = {};
    for (const h of HORIZONS) {
      const clusterP = currentClusterStats.thresholdProbabilities[threshold][h];
      const patternP = patternStats.thresholdProbabilities[threshold][h];
      const baselineP = baselineStats.thresholdProbabilities[threshold][h];
      const markovPh = clamp(1 - ((1 - markovP1) ** h), 0, 1);
      const modelBlend = clamp(
        (weights.cluster * clusterP) +
        (weights.pattern * patternP) +
        (weights.markov * markovPh) +
        (weights.baseline * baselineP),
        0,
        1
      );

      if (h === 1) {
        byHorizon[h] = clamp((0.6 * modelBlend) + (0.4 * p1FromDist) + (pressureBoost * 0.28), 0, 1);
      } else {
        const implied = clamp(1 - ((1 - byHorizon[1]) ** h), 0, 1);
        byHorizon[h] = clamp((0.72 * modelBlend) + (0.28 * implied) + (pressureBoost * (h === 3 ? 0.35 : 0.42)), 0, 1);
      }
    }

    const expectedGap = p1FromDist > 0.0001 ? roundNum(1 / p1FromDist, 2) : null;
    return {
      target: threshold,
      gapNow,
      p1: roundNum(byHorizon[1], 4),
      p3: roundNum(Math.max(byHorizon[3], byHorizon[1]), 4),
      p5: roundNum(Math.max(byHorizon[5], byHorizon[3], byHorizon[1]), 4),
      expectedGap,
      softGapPressure: roundNum(g.soft, 4),
      hardGapPressure: roundNum(g.hard, 4),
    };
  });

  const clusterMean = currentClusterStats.meanNextMultiplier || expectedFromBuckets(currentClusterStats.bucketDistribution);
  let clusterRegime = 'balanced';
  if (clusterMean < 2) clusterRegime = 'compression';
  else if (clusterMean < 5) clusterRegime = 'low-mid';
  else if (clusterMean < 10) clusterRegime = 'mid-volatility';
  else clusterRegime = 'expansion';

  const cashoutPlan = buildCashoutPlan(finalBucketDist, topBucket, confidence, gapPressure);

  const signals = buildSignals({
    lowStreak: streakLength(cleanRounds, cleanRounds.length - 1, m => m < 2),
    highStreak: streakLength(cleanRounds, cleanRounds.length - 1, m => m >= 10),
    patternCount: patternStats.count,
    clusterRegime,
    trendRegime: trendContext.regime,
    gapPressure,
    recommendedCashout: cashoutPlan,
    topBucket,
  });

  const similarPatterns = patternMatches.slice(0, 8).map((m, idx) => ({
    rank: idx + 1,
    startRoundId: m.startRoundId,
    endRoundId: m.endRoundId,
    nextRoundId: m.nextRoundId,
    nextMultiplier: roundNum(m.nextMult, 4),
    nextBucket: BUCKETS[m.nextBucket].label,
    similarity: roundNum(m.similarity, 4),
  }));

  return {
    model: 'cluster-pattern-hybrid-v7',
    generatedAt: new Date().toISOString(),
    asOfRound: cleanRounds[cleanRounds.length - 1].roundId,
    sampleSize: cleanRounds.length,
    expectedMultiplier: roundNum(expectedMultiplier, 4),
    expectedMedian: roundNum(expectedMedian, 4),
    expectedP75: roundNum(expectedP75, 4),
    expectedP90: roundNum(expectedP90, 4),
    predictedBucket: {
      ...topBucket,
      probability: roundNum(maxProb, 4),
      confidence: roundNum(confidence, 4),
      confidenceBand,
    },
    bucketProbabilities: BUCKETS.map((bucket, i) => ({
      ...bucket,
      probability: roundNum(finalBucketDist[i], 4),
    })),
    targetProbabilities,
    cashoutPlan,
    diagnostics: {
      training: {
        samples: samples.length,
        clusters: k,
        windowSize,
        patternWindow,
      },
      blendWeights: {
        cluster: roundNum(weights.cluster, 4),
        pattern: roundNum(weights.pattern, 4),
        markov: roundNum(weights.markov, 4),
        baseline: roundNum(weights.baseline, 4),
      },
      cluster: {
        id: currentClusterId,
        regime: clusterRegime,
        support: Math.round(currentClusterStats.count),
        meanNextMultiplier: roundNum(clusterMean, 4),
      },
      trend: {
        regime: trendContext.regime,
        trendScore: roundNum(trendContext.trendScore, 4),
        volatilityRatio: roundNum(trendContext.volRatio, 4),
        whiteClusterSeverity: roundNum(whiteSeverity, 4),
      },
      pattern: {
        matches: patternStats.count,
        avgSimilarity: roundNum(patternStats.avgSimilarity, 4),
      },
      markov: {
        mode: markov.mode,
        support: markov.support,
      },
      expected: {
        central: roundNum(expectedMultiplier, 4),
        median: roundNum(expectedMedian, 4),
        p75: roundNum(expectedP75, 4),
        p90: roundNum(expectedP90, 4),
        meanFromDistribution: roundNum(distExpectedMean, 4),
        meanFromEnsemble: roundNum(rawExpectedMultiplier, 4),
      },
      gapPressure: Object.fromEntries(
        THRESHOLDS.map((t) => [t, {
          gap: gapPressure[t]?.currentGap ?? 0,
          soft: roundNum(gapPressure[t]?.soft ?? 0, 4),
          hard: roundNum(gapPressure[t]?.hard ?? 0, 4),
        }])
      ),
      bucketBoosts,
    },
    similarPatterns,
    signals,
  };
}

function buildPredictionReport(rounds) {
  const lastRoundId = rounds?.length ? Number(rounds[rounds.length - 1].roundId) : 0;
  const key = `${rounds?.length || 0}:${lastRoundId}`;
  const now = Date.now();

  if (cache.key === key && (now - cache.createdAt) < CACHE_TTL_MS && cache.report) {
    return cache.report;
  }

  const report = computeReport(rounds || []);
  cache.key = key;
  cache.createdAt = now;
  cache.report = report;
  return report;
}


  return { buildPredictionReport };
})();

function buildUnifiedPredictionEngine({ rounds, existingLocks = {}, historyRows = [] } = {}) {
  const nextRoundReport = NextRoundEngine.buildPredictionReport(rounds || []);
  const lockedRangeReport = LockedEngine.computeLockedRangePredictions(rounds || [], existingLocks || {}, {
    historyRows: historyRows || [],
  });

  return {
    model: 'unified-v7',
    generatedAt: new Date().toISOString(),
    nextRoundReport,
    lockedRangeReport,
  };
}

module.exports = {
  TARGETS: LockedEngine.TARGETS,
  buildPredictionReport: NextRoundEngine.buildPredictionReport,
  computeLockedRangePredictions: LockedEngine.computeLockedRangePredictions,
  buildUnifiedPredictionEngine,
};
