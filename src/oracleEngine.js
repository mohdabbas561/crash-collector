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

function quantile(sorted, p) {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + ((sorted[hi] - sorted[lo]) * (idx - lo));
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function kdeModeEfficient(sortedGaps) {
  if (!sortedGaps.length) return null;
  const n = sortedGaps.length;
  if (n === 1) return sortedGaps[0];
  const mean = sortedGaps.reduce((sum, gap) => sum + gap, 0) / n;
  const std = Math.sqrt(
    sortedGaps.reduce((sum, gap) => sum + ((gap - mean) ** 2), 0) / n
  ) || 1;
  const bandwidth = Math.max(0.5, 1.06 * std * (n ** -0.2));
  const lo = sortedGaps[0];
  const hi = sortedGaps[n - 1];
  const steps = Math.min(200, hi - lo + 1);
  const step = steps > 1 ? (hi - lo) / (steps - 1) : 0;
  let bestScore = -1;
  let bestX = mean;

  for (let i = 0; i < steps; i += 1) {
    const x = lo + (i * step);
    let score = 0;
    for (const gap of sortedGaps) {
      const z = (x - gap) / bandwidth;
      score += Math.exp(-0.5 * z * z);
    }
    if (score > bestScore) {
      bestScore = score;
      bestX = x;
    }
  }

  return Math.round(bestX);
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
  return Math.round((1 - (kmTable[to] / sFrom)) * 1000) / 10;
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

function computeOracleForecast(rounds, target) {
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
  const n = allGapsRaw.length;
  const recentN = Math.min(scanN, n);
  const recentGaps = allGapsRaw.slice(-recentN);
  const recentSorted = [...recentGaps].sort((a, b) => a - b);
  const last10 = allGapsRaw.slice(-Math.min(10, n));
  const weighted = [...recentGaps, ...last10].sort((a, b) => a - b);
  const last20 = allGapsRaw.slice(-Math.min(20, n));
  const last20Sorted = [...last20].sort((a, b) => a - b);
  const medAll = quantile(allGapsSorted, 50);
  const medLast20 = quantile(last20Sorted, 50);
  const regimeDrift = medAll > 0 ? Math.abs(medLast20 - medAll) / medAll : 0;
  const useRecentForCluster = regimeDrift > 0.4 && last20.length >= 10;
  const lastHit = hits[hits.length - 1];
  const roundsSince = nowId - lastHit.id;
  const med = Math.round(quantile(weighted, 50));
  const p10 = Math.round(quantile(recentSorted, 10));
  const p25 = Math.round(quantile(recentSorted, 25));
  const p75 = Math.round(quantile(recentSorted, 75));
  const p90 = Math.round(quantile(recentSorted, 90));
  const p90all = Math.round(quantile(allGapsSorted, 90));
  const p99all = Math.round(quantile(allGapsSorted, 99));
  const maxGap = allGapsSorted[n - 1];
  const minGap = allGapsSorted[0];
  const avgGap = Math.round(allGapsRaw.reduce((sum, gap) => sum + gap, 0) / n);
  const iqr = Math.max(1, p75 - p25);
  const halfWin = Math.floor(winSize / 2);
  const clusterSource = useRecentForCluster ? recentSorted : allGapsSorted;
  const clusterCenter = kdeModeEfficient(clusterSource);
  const isTooEarly = roundsSince < p10;
  const isOverdue = roundsSince > med;
  const isHardGap = roundsSince > p90all;
  const isExtreme = roundsSince > p99all;
  const survivingGaps = allGapsSorted.filter((gap) => gap > roundsSince);
  const openWindow = isExtreme && survivingGaps.length === 0;

  let predictedGap;
  let predBasis;
  let predMethod;

  if (openWindow) {
    predictedGap = roundsSince + halfWin + 1;
    predBasis = `extreme drought - beyond all ${n} gaps`;
    predMethod = 'extreme';
  } else if (!isOverdue) {
    predictedGap = clusterCenter ?? med;
    predBasis = `cluster (${useRecentForCluster ? 'recent' : 'full'}, ${n} gaps)`;
    predMethod = 'cluster';
  } else if (survivingGaps.length > 0) {
    const survivors = [...survivingGaps];
    predictedGap = survivors.length >= 4 ? Math.round(quantile(survivors, 25)) : survivors[0];
    predBasis = `survival (${survivors.length} gaps > ${roundsSince}r)`;
    predMethod = 'survival';
  } else {
    predictedGap = roundsSince + halfWin + 1;
    predBasis = 'no survivors - extending';
    predMethod = 'extend';
  }

  let predictedRound = lastHit.id + predictedGap;
  if (predictedRound <= nowId) {
    const stepsNeeded = Math.ceil((nowId + 1 - predictedRound) / Math.max(1, med));
    predictedGap += stepsNeeded * Math.max(1, med);
    predictedRound = lastHit.id + predictedGap;
  }

  const windowLo = predictedRound - halfWin;
  const windowHi = windowLo + winSize - 1;
  const droughtPct = n > 0
    ? Math.round((allGapsSorted.filter((gap) => gap <= roundsSince).length / n) * 100)
    : 0;
  const kmTable = buildKMTable(allGapsSorted);
  const pHit1 = kmProb(kmTable, roundsSince, 1);
  const pHit5 = kmProb(kmTable, roundsSince, 5);
  const pHit10 = kmProb(kmTable, roundsSince, 10);
  const pHit20 = kmProb(kmTable, roundsSince, 20);
  const roundsUntilWindowLo = Math.max(0, windowLo - nowId);
  const roundsUntilWindowHi = Math.max(0, windowHi - nowId);
  const pHitWindow = roundsUntilWindowHi <= 0
    ? 0
    : kmIntervalProb(kmTable, roundsSince, roundsUntilWindowLo, roundsUntilWindowHi);

  const winGapLo = predictedGap - halfWin;
  const winGapHi = winGapLo + winSize - 1;
  const hitsInWindow = allGapsSorted.filter((gap) => gap >= winGapLo && gap <= winGapHi).length;
  const baseConf = n > 0 ? Math.round((hitsInWindow / n) * 100) : 0;
  const recentHitsInWindow = recentSorted.filter((gap) => gap >= winGapLo && gap <= winGapHi).length;
  const recentWindowHitRate = recentSorted.length > 0
    ? Math.round((recentHitsInWindow / recentSorted.length) * 100)
    : baseConf;
  const blendedBaseConf = Math.round((baseConf * 0.65) + (recentWindowHitRate * 0.35));
  const inWindow = nowId >= windowLo && nowId <= windowHi;
  const proximityBonus = inWindow ? 15 : roundsUntilWindowLo <= 5 ? 10 : roundsUntilWindowLo <= 15 ? 5 : 0;
  const confPenalty = (isExtreme ? 35 : 0)
    + (isHardGap ? 18 : 0)
    + (isOverdue && !isHardGap ? 6 : 0)
    + (regimeDrift > 0.4 ? 8 : 0);
  const confidence = Math.max(4, Math.min(92, blendedBaseConf - confPenalty + proximityBonus));

  const nearWindowRounds = Math.max(1, Math.min(75, Math.max(winSize, Math.round(med * 0.35))));
  const verySoonThreshold = Math.max(2, Math.ceil(winSize * 0.5));
  const soonThreshold = Math.max(5, Math.ceil(winSize * 1.5));
  const warmThreshold = Math.max(10, Math.ceil(winSize * 3));
  const farThreshold = Math.max(20, Math.ceil(winSize * 6));
  const pHitNearWindow = kmProb(kmTable, roundsSince, nearWindowRounds);
  const proximityScore = inWindow
    ? 12
    : roundsUntilWindowLo <= verySoonThreshold
      ? 8
      : roundsUntilWindowLo <= soonThreshold
        ? 4
        : roundsUntilWindowLo <= warmThreshold
          ? 2
          : roundsUntilWindowLo <= farThreshold
            ? 1
            : 0;
  const droughtScore = isTooEarly
    ? -8
    : openWindow
      ? -15
      : isExtreme
        ? -10
        : isHardGap
          ? -6
          : isOverdue
            ? 4
            : clampNumber((droughtPct - 50) * 0.18, -4, 8);
  const regimeScore = regimeDrift <= 0.2 ? 3 : regimeDrift > 0.4 ? -6 : 0;
  const clusterScore = med > 0 && clusterCenter != null
    ? clampNumber((((med - Math.abs(clusterCenter - med)) / med) * 6) - 3, -4, 6)
    : 0;
  const supportScore = (
    (confidence * 0.45) +
    (pHitWindow * 0.35) +
    (pHitNearWindow * 0.2)
  );
  const chaseRaw = clampNumber(Math.round(
    supportScore +
    proximityScore +
    droughtScore +
    regimeScore +
    clusterScore
  ), 0, 100);

  const chaseSignal = chaseRaw >= 70
    ? 'CHASE'
    : chaseRaw >= 50
      ? 'WATCH'
      : chaseRaw >= 30
        ? 'WAIT'
        : 'SKIP';
  const chaseColor = chaseRaw >= 68
    ? '#39ff8a'
    : chaseRaw >= 44
      ? '#ffd250'
      : chaseRaw >= 24
        ? '#ff9f43'
        : '#ff4040';

  return {
    ...target,
    noData: false,
    nowId,
    hits: hits.length,
    lastHit,
    roundsSince,
    n,
    med,
    p10,
    p25,
    p75,
    p90,
    p90all,
    p99all,
    maxGap,
    minGap,
    avgGap,
    iqr,
    clusterCenter,
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
    regimeDrift: Math.round(regimeDrift * 100),
    droughtPct,
    confidence,
    chaseRaw,
    chaseSignal,
    chaseColor,
    pHit1,
    pHit5,
    pHit10,
    pHit20,
    pHitWindow,
    pHitNearWindow,
    nearWindowRounds,
    baseWindowHitRate: baseConf,
    recentWindowHitRate,
    roundsUntilWindowLo,
    roundsUntilWindowHi,
    inWindow,
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
  };
}

module.exports = {
  ORACLE_TARGETS,
  normalizeRounds,
  computeOracleForecast,
  makeOracleLock,
};
