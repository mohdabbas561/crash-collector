'use strict';

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const WINDOW_SPAN = {
  5: 3,
  10: 6,
  20: 10,
  50: 17,
  100: 25,
  500: 50,
  1000: 60,
};
const MAX_AHEAD = {
  5: 220,
  10: 360,
  20: 560,
  50: 1100,
  100: 2200,
  500: 7000,
  1000: 12000,
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

  const lowStreak = new Array(n).fill(0);
  const highStreak = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    lowStreak[i] = cleanRounds[i].multiplier < 2 ? 1 + (i > 0 ? lowStreak[i - 1] : 0) : 0;
    highStreak[i] = cleanRounds[i].multiplier >= 10 ? 1 + (i > 0 ? highStreak[i - 1] : 0) : 0;
  }

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
      mean: roundNum(avg, 3),
      sd: roundNum(sd, 3),
      q50: roundNum(sorted.length ? quantile(sorted, 0.5) : 0, 3),
      q75: roundNum(sorted.length ? quantile(sorted, 0.75) : 0, 3),
      q90: roundNum(sorted.length ? quantile(sorted, 0.9) : 0, 3),
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
    let regime = 'balanced';
    if (trend <= -0.04 && volRatio <= 1.02) regime = 'compression';
    else if (trend >= 0.035 && volRatio >= 1.02) regime = 'expansion';
    else if (volRatio >= 1.18) regime = 'chaotic';
    else if (trend <= -0.02) regime = 'soft-down';
    else if (trend >= 0.02) regime = 'soft-up';
    return { trend, volRatio, regime };
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
      lowStreak: lowStreak[idx],
      highStreak: highStreak[idx],
      seq,
    };
  }

  return {
    rounds: cleanRounds,
    n,
    gapMaps,
    nextHitMaps,
    hitRoundIds,
    gapStats,
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
  const maxAhead = MAX_AHEAD[target] || 2000;
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
  const keep = target <= 20 ? 700 : target <= 100 ? 520 : 360;
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

function buildWindow(pre, target, currentIdx, calibration = null) {
  const currentRound = pre.rounds[currentIdx].roundId;
  const currentState = pre.stateAt(currentIdx, target);
  const neighbors = collectNeighbors(pre, target, currentIdx, currentState);
  const stats = pre.gapStats[target] || { mean: 0, q50: 0, q75: 0, q90: 0 };
  const gapNow = pre.gapMaps[target][currentIdx];
  const pressure = gapPressure(gapNow, stats);

  let q20 = weightedQuantile(neighbors, 0.2);
  let q50 = weightedQuantile(neighbors, 0.5);
  let q80 = weightedQuantile(neighbors, 0.8);

  if (!neighbors.length) {
    q20 = Math.max(1, stats.q50 || 2);
    q50 = Math.max(q20 + 1, stats.q75 || (q20 + 2));
    q80 = Math.max(q50 + 1, stats.q90 || (q50 + 3));
  }

  const sumW = neighbors.reduce((s, n) => s + n.weight, 0);
  const sumW2 = neighbors.reduce((s, n) => s + (n.weight * n.weight), 0);
  const effN = sumW2 > 0 ? (sumW * sumW) / sumW2 : 0;
  const uncertainty = clamp((18 - Math.min(18, effN)) / 18, 0, 1);

  const hazard = hazardEta(target, currentState, stats, pressure);

  const neighborWeight = clamp(0.35 + (0.45 * clamp(effN / 9, 0, 1)), 0.35, 0.8);
  const hazardWeight = clamp(0.28 + (0.17 * uncertainty), 0.18, 0.5);
  const priorWeight = clamp(1 - neighborWeight - hazardWeight, 0.05, 0.25);

  const priorCenter = Math.max(1, stats.q50 || stats.q75 || hazard.q50 || q50 || 2);
  const blendCenter =
    (neighborWeight * q50) +
    (hazardWeight * hazard.q50) +
    (priorWeight * priorCenter);

  const reg = currentState.regime;
  let centerFactor = 1;
  centerFactor *= 1 - (0.16 * pressure.hard) - (0.08 * pressure.soft);
  if (reg === 'expansion') centerFactor *= target >= 20 ? 0.9 : 0.94;
  if (reg === 'compression') centerFactor *= target >= 20 ? 1.1 : 1.05;
  if (reg === 'chaotic') centerFactor *= target >= 50 ? 0.94 : 0.98;
  centerFactor *= 1 - (0.1 * clamp(currentState.trend, -0.8, 0.8));
  centerFactor = clamp(centerFactor, 0.68, 1.35);

  let centerAhead = Math.max(1, Math.round(blendCenter * centerFactor));
  if (calibration && Number.isFinite(calibration.shift) && calibration.sample >= 5) {
    centerAhead = Math.max(1, Math.round(centerAhead * (1 + calibration.shift)));
  }
  const windowSpan = WINDOW_SPAN[target] || 10;
  const halfLeft = Math.floor((windowSpan - 1) / 2);
  const maxLoAhead = Math.max(1, (MAX_AHEAD[target] || 3000) - windowSpan + 1);

  let loAhead = Math.max(1, centerAhead - halfLeft);
  loAhead = Math.min(loAhead, maxLoAhead);
  const hiAhead = loAhead + windowSpan - 1;

  const engineAgreement = clamp(
    1 - (Math.abs(q50 - hazard.q50) / Math.max(2, q50 + hazard.q50)),
    0,
    1
  );

  const confidence = clamp(
    0.2 +
    (0.35 * (1 - uncertainty)) +
    (0.2 * Math.min(1, neighbors.length / (target <= 20 ? 300 : 180))) +
    (0.17 * ((pressure.soft + pressure.hard) * 0.5)) +
    (0.08 * engineAgreement),
    0.08,
    0.95
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
      hazardQ20: roundNum(hazard.q20, 2),
      hazardQ50: roundNum(hazard.q50, 2),
      hazardQ80: roundNum(hazard.q80, 2),
      pHit1: roundNum(hazard.pHit1, 6),
      blendCenter: roundNum(blendCenter, 2),
      centerAhead: roundNum(centerAhead, 2),
      windowSpan,
      neighbors: neighbors.length,
      effN: roundNum(effN, 2),
      uncertainty: roundNum(uncertainty, 4),
      engineAgreement: roundNum(engineAgreement, 4),
      regime: reg,
      currentGap: gapNow,
      historicalGapMean: stats.mean || 0,
      historicalQ75: stats.q75 || 0,
      historicalQ90: stats.q90 || 0,
      softGapPressure: roundNum(pressure.soft, 4),
      hardGapPressure: roundNum(pressure.hard, 4),
      confidence: roundNum(confidence, 4),
      reason: pressure.hard > 0.45
        ? 'Hard-gap pressure active; center shifted earlier with fixed short window.'
        : reg === 'expansion'
          ? 'Expansion regime detected; fixed window centered using ensemble ETA.'
          : reg === 'compression'
            ? 'Compression regime detected; fixed window centered later from ensemble ETA.'
            : 'Fixed window centered from neighbor-pattern + hazard + trend ensemble.',
      calibrationShift: roundNum(calibration?.shift || 0, 4),
      calibrationSample: Number(calibration?.sample || 0),
    },
    confidence: roundNum(confidence, 4),
  };
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

function buildCalibrationMap(historyRows) {
  const out = {};
  for (const target of TARGETS) {
    const label = `${target}x`;
    const rows = (historyRows || [])
      .filter(r => String(r.target || '').toLowerCase() === label)
      .slice(0, 80);

    if (!rows.length) {
      out[target] = { shift: 0, sample: 0 };
      continue;
    }

    let weightedWin = 0;
    let weightedLoss = 0;
    let weightedEarly = 0;
    let totalW = 0;

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const w = 1 + ((rows.length - i) / rows.length); // newer rows get slightly higher weight
      totalW += w;
      if (row.outcome === 'win') weightedWin += w;
      else if (row.outcome === 'loss') weightedLoss += w;
      else if (row.outcome === 'early') weightedEarly += w;
    }

    const denom = weightedWin + weightedLoss;
    const winRate = denom > 0 ? weightedWin / denom : 0.5;
    const earlyRate = totalW > 0 ? weightedEarly / totalW : 0;
    const sampleFactor = clamp(rows.length / 24, 0, 1);

    // Negative shift => earlier window. Positive shift => later window.
    let shift = 0;
    shift += -0.34 * earlyRate;           // early means we were late, shift earlier
    shift += (0.5 - winRate) * 0.2;       // lower win nudges timing
    shift = clamp(shift * sampleFactor, -0.3, 0.3);

    out[target] = {
      shift: roundNum(shift, 4),
      sample: rows.length,
      winRate: roundNum(winRate, 4),
      earlyRate: roundNum(earlyRate, 4),
    };
  }
  return out;
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
  const calibration = buildCalibrationMap(options.historyRows || []);
  if (pre.n < 800) {
    return {
      model: 'range-lock-v2',
      generatedAt: new Date().toISOString(),
      asOfRound: pre.rounds[pre.n - 1]?.roundId || null,
      targets: [],
      locksToSave: {},
      resolvedHistory: [],
      summary: { pending: 0, windowOpen: 0, relocked: 0, sampleSize: pre.n },
      calibration,
      settings: { windowSpan: WINDOW_SPAN },
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
    model: 'range-lock-v2',
    generatedAt: new Date().toISOString(),
    asOfRound: currentRound,
    sampleSize: pre.n,
    targets: targetsOut,
    locksToSave,
    resolvedHistory,
    calibration,
    settings: { windowSpan: WINDOW_SPAN },
    summary: {
      pending: pendingCount,
      windowOpen: openCount,
      relocked: relockedCount,
      sampleSize: pre.n,
    },
  };
}

module.exports = { TARGETS, computeLockedRangePredictions };
