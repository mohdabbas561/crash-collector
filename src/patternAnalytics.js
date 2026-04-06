'use strict';

const DAY_MS = 24 * 60 * 60 * 1000;

const WINDOW_OPTIONS = [
  { key: '10', label: '10 Rounds', size: 10 },
  { key: '20', label: '20 Rounds', size: 20 },
  { key: '50', label: '50 Rounds', size: 50 },
  { key: '250', label: '250 Rounds', size: 250 },
  { key: '500', label: '500 Rounds', size: 500 },
];

const WINDOW_MAP = new Map(WINDOW_OPTIONS.map((item) => [item.key, item]));
const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const DEFAULT_WINDOW_KEY = '10';
const DEFAULT_TARGET = 5;
const MIN_MATCHES = 8;
//hello

const DISTRIBUTION_BANDS = [
  { key: 'lt2', label: '<2x', min: -Infinity, max: 2 },
  { key: '2to5', label: '2x-5x', min: 2, max: 5 },
  { key: '5to10', label: '5x-10x', min: 5, max: 10 },
  { key: '10to20', label: '10x-20x', min: 10, max: 20 },
  { key: '20to50', label: '20x-50x', min: 20, max: 50 },
  { key: '50to100', label: '50x-100x', min: 50, max: 100 },
  { key: '100to250', label: '100x-250x', min: 100, max: 250 },
  { key: '250to500', label: '250x-500x', min: 250, max: 500 },
  { key: '500to1000', label: '500x-1000x', min: 500, max: 1000 },
  { key: 'gte1000', label: '1000x+', min: 1000, max: Infinity },
];

function safeNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function ratio(part, whole, fallback = 0) {
  return whole > 0 ? part / whole : fallback;
}

function quantile(sortedValues, q) {
  if (!sortedValues.length) return 0;
  const index = clamp((sortedValues.length - 1) * q, 0, sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sortedValues[lower];
  const weight = index - lower;
  return (sortedValues[lower] * (1 - weight)) + (sortedValues[upper] * weight);
}

function weightedAverage(items, valueGetter, weightGetter, fallback = 0) {
  let valueSum = 0;
  let weightSum = 0;
  for (const item of items) {
    const weight = Math.max(0, safeNumber(weightGetter(item), 0));
    if (weight <= 0) continue;
    valueSum += safeNumber(valueGetter(item), 0) * weight;
    weightSum += weight;
  }
  return weightSum > 0 ? valueSum / weightSum : fallback;
}

function weightedQuantile(items, valueGetter, weightGetter, q, fallback = 0) {
  const rows = items
    .map((item) => ({
      value: safeNumber(valueGetter(item), NaN),
      weight: Math.max(0, safeNumber(weightGetter(item), 0)),
    }))
    .filter((item) => Number.isFinite(item.value) && item.weight > 0)
    .sort((a, b) => a.value - b.value);

  if (!rows.length) return fallback;
  const totalWeight = rows.reduce((sum, item) => sum + item.weight, 0);
  const threshold = totalWeight * clamp(q, 0, 1);
  let seen = 0;
  for (const row of rows) {
    seen += row.weight;
    if (seen >= threshold) return row.value;
  }
  return rows[rows.length - 1].value;
}

function normalizePatternWindowKey(value) {
  const key = String(value || '').trim();
  if (WINDOW_MAP.has(key)) return key;
  const numeric = safeNumber(value, 10);
  const nearest = WINDOW_OPTIONS
    .slice()
    .sort((a, b) => Math.abs(a.size - numeric) - Math.abs(b.size - numeric))[0];
  return nearest?.key || DEFAULT_WINDOW_KEY;
}

function normalizePatternTarget(value) {
  const numeric = safeNumber(value, DEFAULT_TARGET);
  const nearest = TARGETS.slice().sort((a, b) => Math.abs(a - numeric) - Math.abs(b - numeric))[0];
  return nearest || DEFAULT_TARGET;
}

function labelForTarget(target) {
  return `${safeNumber(target, DEFAULT_TARGET)}x`;
}

function normalizeRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : [])
    .map((round) => ({
      roundId: safeNumber(round?.roundId ?? round?.round_id, 0),
      multiplier: safeNumber(round?.multiplier, NaN),
      timestamp: safeNumber(round?.timestamp, NaN),
    }))
    .filter((round) => Number.isFinite(round.multiplier) && round.multiplier > 0 && Number.isFinite(round.timestamp))
    .sort((a, b) => (
      a.roundId && b.roundId && a.roundId !== b.roundId
        ? a.roundId - b.roundId
        : a.timestamp - b.timestamp
    ));
}

function formatMultiplier(value) {
  const numeric = safeNumber(value, 0);
  if (!numeric) return '-';
  if (numeric >= 100) return `${numeric.toFixed(1)}x`;
  if (numeric >= 10) return `${numeric.toFixed(1)}x`;
  return `${numeric.toFixed(2)}x`;
}

function formatRoundRange(fromRoundId, toRoundId) {
  if (!fromRoundId && !toRoundId) return '-';
  if (!toRoundId || fromRoundId === toRoundId) return `#${fromRoundId}`;
  return `#${fromRoundId} - #${toRoundId}`;
}

function findBand(multiplier) {
  const value = safeNumber(multiplier, 0);
  return DISTRIBUTION_BANDS.find((band) => value >= band.min && value < band.max) || DISTRIBUTION_BANDS[DISTRIBUTION_BANDS.length - 1];
}

function bandIndex(multiplier) {
  return DISTRIBUTION_BANDS.findIndex((band) => multiplier >= band.min && multiplier < band.max);
}

function adaptiveTolerance(multiplier) {
  const value = Math.max(1, safeNumber(multiplier, 1));
  if (value < 2) return 0.6;
  if (value < 5) return 1.2;
  if (value < 10) return 2.2;
  if (value < 20) return 4.5;
  if (value < 50) return 8;
  if (value < 100) return 15;
  if (value < 250) return 35;
  if (value < 500) return 65;
  return 140;
}

function pointSimilarity(a, b) {
  const av = safeNumber(a, 1);
  const bv = safeNumber(b, 1);
  const tolerance = adaptiveTolerance((av + bv) / 2);
  const diff = Math.abs(av - bv);
  return clamp(1 - (diff / tolerance), 0, 1);
}

function bucketSimilarity(a, b) {
  const diff = Math.abs(bandIndex(a) - bandIndex(b));
  if (diff <= 0) return 1;
  if (diff === 1) return 0.72;
  if (diff === 2) return 0.4;
  return 0;
}

function stepDirection(valueA, valueB) {
  const diff = safeNumber(valueB, 0) - safeNumber(valueA, 0);
  if (Math.abs(diff) < 0.15) return 0;
  return diff > 0 ? 1 : -1;
}

function stepSimilarity(prevA, nextA, prevB, nextB) {
  const dirA = stepDirection(prevA, nextA);
  const dirB = stepDirection(prevB, nextB);
  const directionScore = dirA === dirB ? 1 : (dirA === 0 || dirB === 0 ? 0.55 : 0.15);
  const jumpA = Math.abs(safeNumber(nextA, 0) - safeNumber(prevA, 0));
  const jumpB = Math.abs(safeNumber(nextB, 0) - safeNumber(prevB, 0));
  const amplitudeScore = pointSimilarity(jumpA, jumpB);
  return (directionScore * 0.6) + (amplitudeScore * 0.4);
}

function buildBandHistogram(rounds) {
  const counts = DISTRIBUTION_BANDS.map(() => 0);
  for (const round of rounds) {
    const index = bandIndex(round.multiplier);
    counts[Math.max(0, index)] += 1;
  }
  return counts.map((count) => ratio(count, rounds.length));
}

function histogramSimilarity(refHistogram, candidateHistogram) {
  if (!refHistogram.length || !candidateHistogram.length) return 0;
  let delta = 0;
  for (let index = 0; index < refHistogram.length; index += 1) {
    delta += Math.abs((refHistogram[index] || 0) - (candidateHistogram[index] || 0));
  }
  return clamp(1 - (delta / 2), 0, 1);
}

function comparePatternSequence(referenceRounds, candidateRounds) {
  const length = Math.min(referenceRounds.length, candidateRounds.length);
  if (!length) {
    return {
      similarity: 0,
      similarityPct: 0,
      pointScore: 0,
      bucketScore: 0,
      stepScore: 0,
      histogramScore: 0,
    };
  }

  let pointSum = 0;
  let bucketSum = 0;
  let stepSum = 0;
  let stepCount = 0;

  for (let index = 0; index < length; index += 1) {
    const reference = referenceRounds[index].multiplier;
    const candidate = candidateRounds[index].multiplier;
    pointSum += pointSimilarity(reference, candidate);
    bucketSum += bucketSimilarity(reference, candidate);
    if (index > 0) {
      stepSum += stepSimilarity(
        referenceRounds[index - 1].multiplier,
        referenceRounds[index].multiplier,
        candidateRounds[index - 1].multiplier,
        candidateRounds[index].multiplier,
      );
      stepCount += 1;
    }
  }

  const pointScore = pointSum / length;
  const bucketScore = bucketSum / length;
  const stepScore = stepCount ? stepSum / stepCount : pointScore;
  const histogramScore = histogramSimilarity(buildBandHistogram(referenceRounds), buildBandHistogram(candidateRounds));

  const similarity = (pointScore * 0.45) + (bucketScore * 0.25) + (stepScore * 0.2) + (histogramScore * 0.1);
  return {
    similarity,
    similarityPct: Number((similarity * 100).toFixed(1)),
    pointScore: Number((pointScore * 100).toFixed(1)),
    bucketScore: Number((bucketScore * 100).toFixed(1)),
    stepScore: Number((stepScore * 100).toFixed(1)),
    histogramScore: Number((histogramScore * 100).toFixed(1)),
  };
}

function buildPatternGrid(rounds, focusTarget) {
  const items = Array.isArray(rounds) ? rounds : [];
  return {
    totalCount: items.length,
    cells: items.map((round, index) => {
      const multiplier = safeNumber(round.multiplier, 0);
      const band = findBand(multiplier);
      return {
        index: index + 1,
        roundId: round.roundId || null,
        multiplier,
        multiplierLabel: formatMultiplier(multiplier),
        bandLabel: band.label,
        hit: multiplier >= focusTarget,
        near: multiplier >= Math.max(2, focusTarget * 0.75),
      };
    }),
  };
}

function chooseMatchCount(windowSize, totalCandidates) {
  if (totalCandidates <= 0) return 0;
  const cap = windowSize <= 20 ? 80 : windowSize <= 50 ? 60 : windowSize <= 250 ? 40 : 28;
  return clamp(totalCandidates, Math.min(MIN_MATCHES, totalCandidates), cap);
}

function buildNextRangeLabel(matches, weightGetter) {
  if (!matches.length) return '-';
  const from = weightedQuantile(matches, (match) => match.nextRound.multiplier, weightGetter, 0.2, 0);
  const to = weightedQuantile(matches, (match) => match.nextRound.multiplier, weightGetter, 0.8, 0);
  if (!from && !to) return '-';
  return `${formatMultiplier(from)} - ${formatMultiplier(to)}`;
}

function playChanceThreshold(target) {
  if (target <= 5) return 0.55;
  if (target <= 10) return 0.4;
  if (target <= 20) return 0.28;
  if (target <= 50) return 0.18;
  if (target <= 100) return 0.12;
  if (target <= 500) return 0.035;
  return 0.015;
}

function edgeThreshold(target) {
  if (target <= 5) return 0.05;
  if (target <= 10) return 0.045;
  if (target <= 20) return 0.04;
  if (target <= 50) return 0.03;
  if (target <= 100) return 0.02;
  if (target <= 500) return 0.008;
  return 0.004;
}

function signalLabel(hitChance, baselineChance, similarityPct, target) {
  const edge = hitChance - baselineChance;
  const absoluteThreshold = playChanceThreshold(target);
  if (hitChance >= absoluteThreshold && similarityPct >= 62 && edge >= edgeThreshold(target)) return 'Strong';
  if ((hitChance >= absoluteThreshold * 0.9 || edge >= edgeThreshold(target)) && similarityPct >= 50) return 'Playable';
  if (similarityPct >= 42) return 'Weak';
  return 'Low Match';
}

function buildEmptyReport(windowConfig, focusTarget) {
  return {
    ok: true,
    generatedAt: Date.now(),
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label, size }) => ({ key, label, size })),
    availableTargets: TARGETS.map((target) => ({ value: target, label: labelForTarget(target) })),
    window: { key: windowConfig.key, label: windowConfig.label, size: windowConfig.size },
    dataset: { totalRounds: 0, spanDays: 0, latestRoundId: null },
    reference: {
      roundFrom: null,
      roundTo: null,
      grid: buildPatternGrid([], focusTarget),
      label: windowConfig.label,
    },
    prediction: {
      action: 'SKIP',
      signal: 'Low Match',
      nextRoundId: null,
      nextRoundIdLabel: '-',
      nextHitChance: 0,
      nextHitChancePercent: 0,
      baselineHitChance: 0,
      baselineHitChancePercent: 0,
      edgePercent: 0,
      averageSimilarityPct: 0,
      nextRangeLabel: '-',
      fallbackRangeLabel: '-',
      fallbackRoundIdLabel: '-',
      summary: 'Not enough stored rounds yet for pattern matching.',
    },
    matches: [],
    comparedPatterns: 0,
    usedMatches: 0,
  };
}

function buildPatternAnalyticsReport(rounds, options = {}) {
  const windowKey = normalizePatternWindowKey(options.windowKey ?? options.window);
  const focusTarget = normalizePatternTarget(options.focusTarget);
  const windowConfig = WINDOW_MAP.get(windowKey) || WINDOW_MAP.get(DEFAULT_WINDOW_KEY);
  const normalized = normalizeRounds(rounds);

  if (normalized.length < (windowConfig.size * 2) + 1) {
    return buildEmptyReport(windowConfig, focusTarget);
  }

  const latestRound = normalized[normalized.length - 1];
  const earliestRound = normalized[0];
  const referenceStartIndex = normalized.length - windowConfig.size;
  const referenceRounds = normalized.slice(referenceStartIndex);

  const candidates = [];
  for (let startIndex = 0; startIndex <= normalized.length - (windowConfig.size * 2) - 1; startIndex += 1) {
    const setupRounds = normalized.slice(startIndex, startIndex + windowConfig.size);
    const nextRound = normalized[startIndex + windowConfig.size];
    if (setupRounds.length !== windowConfig.size || !nextRound) continue;
    const match = comparePatternSequence(referenceRounds, setupRounds);
    candidates.push({
      startIndex,
      setupRounds,
      nextRound,
      similarity: match.similarity,
      similarityPct: match.similarityPct,
      pointScore: match.pointScore,
      bucketScore: match.bucketScore,
      stepScore: match.stepScore,
      histogramScore: match.histogramScore,
    });
  }

  if (candidates.length < MIN_MATCHES) {
    return buildEmptyReport(windowConfig, focusTarget);
  }

  const sortedCandidates = candidates
    .slice()
    .sort((a, b) => b.similarity - a.similarity || b.nextRound.roundId - a.nextRound.roundId);

  const usedMatches = sortedCandidates.slice(0, chooseMatchCount(windowConfig.size, sortedCandidates.length));
  const weightGetter = (match) => Math.max(0.001, Math.pow(match.similarity, 3));

  const baselineHitChance = ratio(
    candidates.filter((candidate) => candidate.nextRound.multiplier >= focusTarget).length,
    candidates.length,
  );

  const nextHitChance = weightedAverage(
    usedMatches,
    (match) => (match.nextRound.multiplier >= focusTarget ? 1 : 0),
    weightGetter,
    baselineHitChance,
  );

  const averageSimilarityPct = weightedAverage(
    usedMatches,
    (match) => match.similarityPct,
    weightGetter,
    0,
  );

  const nextRangeLabel = buildNextRangeLabel(usedMatches, weightGetter);
  const missMatches = usedMatches.filter((match) => match.nextRound.multiplier < focusTarget);
  const fallbackRangeLabel = missMatches.length ? buildNextRangeLabel(missMatches, weightGetter) : '-';
  const predictedNextRoundId = latestRound.roundId ? latestRound.roundId + 1 : null;
  const edgePercent = Number((((nextHitChance - baselineHitChance) * 100)).toFixed(1));
  const signal = signalLabel(nextHitChance, baselineHitChance, averageSimilarityPct, focusTarget);
  const action = (
    usedMatches.length >= MIN_MATCHES
    && averageSimilarityPct >= 46
    && (nextHitChance >= playChanceThreshold(focusTarget) || (nextHitChance - baselineHitChance) >= edgeThreshold(focusTarget))
  ) ? 'PLAY' : 'SKIP';

  const summary = action === 'PLAY'
    ? `Play ${labelForTarget(focusTarget)} for next round #${predictedNextRoundId || '-'}. The best pattern matches show ${labelForTarget(focusTarget)} on the very next round ${Number((nextHitChance * 100).toFixed(1))}% of the time with ${averageSimilarityPct.toFixed(1)}% pattern match.`
    : `Skip ${labelForTarget(focusTarget)} for next round #${predictedNextRoundId || '-'}. The best pattern matches show ${labelForTarget(focusTarget)} on the very next round ${Number((nextHitChance * 100).toFixed(1))}% of the time versus ${Number((baselineHitChance * 100).toFixed(1))}% normal.`;

  return {
    ok: true,
    generatedAt: Date.now(),
    asOfTimestamp: Date.now(),
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label, size }) => ({ key, label, size })),
    availableTargets: TARGETS.map((target) => ({ value: target, label: labelForTarget(target) })),
    window: { key: windowConfig.key, label: windowConfig.label, size: windowConfig.size },
    dataset: {
      totalRounds: normalized.length,
      latestRoundId: latestRound.roundId || null,
      startTimestamp: earliestRound.timestamp,
      endTimestamp: latestRound.timestamp,
      spanDays: ratio(latestRound.timestamp - earliestRound.timestamp, DAY_MS),
    },
    reference: {
      label: `${windowConfig.label} pattern`,
      roundFrom: referenceRounds[0]?.roundId || null,
      roundTo: referenceRounds[referenceRounds.length - 1]?.roundId || null,
      grid: buildPatternGrid(referenceRounds, focusTarget),
    },
    comparedPatterns: candidates.length,
    usedMatches: usedMatches.length,
    prediction: {
      action,
      signal,
      nextRoundId: predictedNextRoundId,
      nextRoundIdLabel: predictedNextRoundId ? `#${predictedNextRoundId}` : '-',
      nextHitChance,
      nextHitChancePercent: Number((nextHitChance * 100).toFixed(1)),
      baselineHitChance,
      baselineHitChancePercent: Number((baselineHitChance * 100).toFixed(1)),
      edgePercent,
      averageSimilarityPct: Number(averageSimilarityPct.toFixed(1)),
      nextRangeLabel,
      fallbackRangeLabel,
      fallbackRoundIdLabel: predictedNextRoundId ? `#${predictedNextRoundId}` : '-',
      summary,
    },
    matches: usedMatches.slice(0, 8).map((match, index) => ({
      rank: index + 1,
      similarityPct: Number(match.similarityPct.toFixed(1)),
      setupRoundFrom: match.setupRounds[0]?.roundId || null,
      setupRoundTo: match.setupRounds[match.setupRounds.length - 1]?.roundId || null,
      setupGrid: buildPatternGrid(match.setupRounds, focusTarget),
      nextRoundId: match.nextRound.roundId || null,
      nextMultiplier: match.nextRound.multiplier,
      nextMultiplierLabel: formatMultiplier(match.nextRound.multiplier),
      nextBandLabel: findBand(match.nextRound.multiplier).label,
      nextHit: match.nextRound.multiplier >= focusTarget,
      pointScore: match.pointScore,
      bucketScore: match.bucketScore,
      stepScore: match.stepScore,
      histogramScore: match.histogramScore,
    })),
  };
}

module.exports = {
  buildPatternAnalyticsReport,
  normalizePatternTarget,
  normalizePatternWindowKey,
};
