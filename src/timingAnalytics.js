'use strict';

const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

const WINDOW_OPTIONS = [
  { key: '5m', label: '5 Minutes', ms: 5 * MINUTE_MS },
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '1h', label: '1 Hour', ms: 1 * HOUR_MS },
  { key: '2h', label: '2 Hours', ms: 2 * HOUR_MS },
  { key: '5h', label: '5 Hours', ms: 5 * HOUR_MS },
];

const WINDOW_MAP = new Map(WINDOW_OPTIONS.map((item) => [item.key, item]));
const DEFAULT_WINDOW_KEY = '5m';

const DISTRIBUTION_BANDS = [
  { key: 'lt2', label: '<2x', min: -Infinity, max: 2 },
  { key: '2to5', label: '2x-5x', min: 2, max: 5 },
  { key: '5to10', label: '5x-10x', min: 5, max: 10 },
  { key: '10to20', label: '10x-20x', min: 10, max: 20 },
  { key: '20to50', label: '20x-50x', min: 20, max: 50 },
  { key: '50to100', label: '50x-100x', min: 50, max: 100 },
  { key: '100to500', label: '100x-500x', min: 100, max: 500 },
  { key: '500to1000', label: '500x-1000x', min: 500, max: 1000 },
  { key: 'gte1000', label: '1000x+', min: 1000, max: Infinity },
];

function normalizeTimingWindowKey(raw) {
  const key = String(raw || '').trim().toLowerCase();
  return WINDOW_MAP.has(key) ? key : DEFAULT_WINDOW_KEY;
}

function normalizeTimingTimeZone(raw) {
  const value = String(raw || '').trim();
  if (!value) return 'UTC';
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value }).format(new Date());
    return value;
  } catch {
    return 'UTC';
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function ratio(num, den, fallback = 0) {
  return den > 0 ? num / den : fallback;
}

function average(values) {
  if (!Array.isArray(values) || !values.length) return 0;
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
}

function weightedAverage(items, valueGetter, weightGetter, fallback = 0) {
  if (!Array.isArray(items) || !items.length) return fallback;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const item of items) {
    const value = Number(valueGetter(item));
    const weight = Math.max(0, Number(weightGetter(item)));
    if (!Number.isFinite(value) || !Number.isFinite(weight) || weight <= 0) continue;
    weightedSum += value * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : fallback;
}

function weightedQuantile(items, valueGetter, weightGetter, q, fallback = 0) {
  if (!Array.isArray(items) || !items.length) return fallback;
  const rows = items
    .map((item) => ({
      value: Number(valueGetter(item)),
      weight: Math.max(0, Number(weightGetter(item))),
    }))
    .filter((item) => Number.isFinite(item.value) && Number.isFinite(item.weight) && item.weight > 0)
    .sort((left, right) => left.value - right.value);

  if (!rows.length) return fallback;

  const target = clamp(Number(q) || 0, 0, 1);
  const totalWeight = rows.reduce((sum, item) => sum + item.weight, 0);
  if (totalWeight <= 0) return fallback;

  let cumulative = 0;
  for (const row of rows) {
    cumulative += row.weight;
    if ((cumulative / totalWeight) >= target) return row.value;
  }
  return rows[rows.length - 1].value;
}

function quantile(sortedValues, q) {
  if (!sortedValues.length) return 0;
  const pos = (sortedValues.length - 1) * clamp(q, 0, 1);
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return sortedValues[lower];
  const weight = pos - lower;
  return sortedValues[lower] + ((sortedValues[upper] - sortedValues[lower]) * weight);
}

function safeNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function formatMultiplier(value) {
  const numeric = safeNumber(value, null);
  if (numeric == null) return '-';
  if (numeric >= 100) return `${numeric.toFixed(1)}x`;
  if (numeric >= 10) return `${numeric.toFixed(2)}x`;
  return `${numeric.toFixed(2)}x`;
}

function formatMultiplierRange(minValue, maxValue) {
  const minNumeric = safeNumber(minValue, null);
  const maxNumeric = safeNumber(maxValue, null);
  if (minNumeric == null && maxNumeric == null) return '-';
  if (minNumeric == null) return formatMultiplier(maxNumeric);
  if (maxNumeric == null) return formatMultiplier(minNumeric);
  if (Math.abs(minNumeric - maxNumeric) < 0.01) return formatMultiplier(maxNumeric);
  return `${formatMultiplier(minNumeric)} - ${formatMultiplier(maxNumeric)}`;
}

function formatThresholdLabel(value) {
  const numeric = safeNumber(value, null);
  if (numeric == null) return '-';
  if (Math.abs(numeric - Math.round(numeric)) < 0.001) return `${Math.round(numeric)}x+`;
  return `${numeric.toFixed(2)}x+`;
}

function formatRoundRange(fromRoundId, toRoundId) {
  if (!fromRoundId && !toRoundId) return '-';
  if (!toRoundId || fromRoundId === toRoundId) return `#${fromRoundId}`;
  return `#${fromRoundId} - #${toRoundId}`;
}

function formatWindowTimeRange(startTimestamp, endTimestamp, timeZone, includeDate = false) {
  const safeStart = safeNumber(startTimestamp, 0);
  const safeEnd = safeNumber(endTimestamp, 0);
  if (!safeStart || !safeEnd) return '-';

  const timeFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit',
  });
  const dayFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    month: 'short',
    day: 'numeric',
  });

  const timeLabel = `${timeFormatter.format(new Date(safeStart))} - ${timeFormatter.format(new Date(safeEnd))}`;
  if (!includeDate) return timeLabel;
  return `${dayFormatter.format(new Date(safeStart))} | ${timeLabel}`;
}

function normalizeRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : [])
    .map((round) => ({
      roundId: safeNumber(round?.roundId ?? round?.round_id, 0),
      multiplier: safeNumber(round?.multiplier, NaN),
      timestamp: safeNumber(round?.timestamp, NaN),
    }))
    .filter((round) => Number.isFinite(round.multiplier) && Number.isFinite(round.timestamp) && round.timestamp > 0)
    .sort((left, right) => {
      if (left.timestamp !== right.timestamp) return left.timestamp - right.timestamp;
      return left.roundId - right.roundId;
    });
}

function distributionPct(summary, bandKey) {
  const items = Array.isArray(summary?.distribution) ? summary.distribution : [];
  const band = items.find((item) => item.key === bandKey);
  return safeNumber(band?.pct, 0);
}

function summarizeRounds(rounds) {
  const values = [];
  const distributionCounts = Object.fromEntries(DISTRIBUTION_BANDS.map((band) => [band.key, 0]));
  let sum = 0;
  let max = 0;

  for (const round of rounds) {
    const multiplier = safeNumber(round.multiplier, 0);
    values.push(multiplier);
    sum += multiplier;
    if (multiplier > max) max = multiplier;
    for (const band of DISTRIBUTION_BANDS) {
      if (multiplier >= band.min && multiplier < band.max) {
        distributionCounts[band.key] += 1;
        break;
      }
    }
  }

  values.sort((left, right) => left - right);
  const roundCount = rounds.length;

  return {
    roundCount,
    avgMultiplier: ratio(sum, roundCount),
    medianMultiplier: quantile(values, 0.5),
    p90Multiplier: quantile(values, 0.9),
    maxMultiplier: max || 0,
    distribution: DISTRIBUTION_BANDS.map((band) => ({
      key: band.key,
      label: band.label,
      count: distributionCounts[band.key],
      pct: ratio(distributionCounts[band.key], roundCount),
    })),
  };
}

function summarizePeakRound(rounds) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) {
    return {
      peakMultiplier: 0,
      peakRoundId: null,
      peakOffset: null,
    };
  }

  let bestIndex = 0;
  let bestRound = items[0];
  for (let index = 1; index < items.length; index += 1) {
    const round = items[index];
    if ((round.multiplier || 0) > (bestRound.multiplier || 0)) {
      bestRound = round;
      bestIndex = index;
    }
  }

  return {
    peakMultiplier: safeNumber(bestRound.multiplier, 0),
    peakRoundId: bestRound.roundId || null,
    peakOffset: bestIndex + 1,
  };
}

function summarizePeakRoundRange(rounds, startIndex, endIndexExclusive) {
  const safeStart = Math.max(0, Number(startIndex) || 0);
  const safeEnd = Math.min(Array.isArray(rounds) ? rounds.length : 0, Number(endIndexExclusive) || 0);
  if (safeEnd <= safeStart) {
    return {
      peakMultiplier: 0,
      peakRoundId: null,
      peakOffset: null,
    };
  }

  let bestIndex = safeStart;
  let bestRound = rounds[safeStart];
  for (let index = safeStart + 1; index < safeEnd; index += 1) {
    const round = rounds[index];
    if ((round.multiplier || 0) > (bestRound.multiplier || 0)) {
      bestRound = round;
      bestIndex = index;
    }
  }

  return {
    peakMultiplier: safeNumber(bestRound?.multiplier, 0),
    peakRoundId: bestRound?.roundId || null,
    peakOffset: (bestIndex - safeStart) + 1,
  };
}

function buildSequenceSample(rounds, maxSamples = 12) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) return [];
  if (items.length <= maxSamples) return items.map((round) => safeNumber(round.multiplier, 0));

  const sample = [];
  const divisor = Math.max(1, maxSamples - 1);
  for (let index = 0; index < maxSamples; index += 1) {
    const pickIndex = Math.round((index / divisor) * (items.length - 1));
    sample.push(safeNumber(items[pickIndex]?.multiplier, 0));
  }
  return sample;
}

function buildSequenceSampleRange(rounds, startIndex, endIndexExclusive, maxSamples = 12) {
  const safeStart = Math.max(0, Number(startIndex) || 0);
  const safeEnd = Math.min(Array.isArray(rounds) ? rounds.length : 0, Number(endIndexExclusive) || 0);
  const itemCount = Math.max(0, safeEnd - safeStart);
  if (!itemCount) return [];

  if (itemCount <= maxSamples) {
    const sample = [];
    for (let index = safeStart; index < safeEnd; index += 1) {
      sample.push(safeNumber(rounds[index]?.multiplier, 0));
    }
    return sample;
  }

  const sample = [];
  const divisor = Math.max(1, maxSamples - 1);
  for (let index = 0; index < maxSamples; index += 1) {
    const pickIndex = safeStart + Math.round((index / divisor) * (itemCount - 1));
    sample.push(safeNumber(rounds[pickIndex]?.multiplier, 0));
  }
  return sample;
}

function fuzzyMultiplierTolerance(value) {
  const numeric = Math.max(0, safeNumber(value, 0));
  if (numeric < 2) return 0.2;
  if (numeric < 5) return 0.5;
  if (numeric < 20) return 1.5;
  if (numeric < 50) return 3;
  if (numeric < 100) return 8;
  if (numeric < 500) return 25;
  return 100;
}

function fuzzyMultiplierDiff(a, b) {
  const left = Math.max(0, safeNumber(a, 0));
  const right = Math.max(0, safeNumber(b, 0));
  const tolerance = Math.max(fuzzyMultiplierTolerance(left), fuzzyMultiplierTolerance(right));
  const delta = Math.abs(left - right);
  if (delta <= tolerance) return 0;
  return (delta - tolerance) / Math.max(tolerance, Math.max(left, right) * 0.35, 1);
}

function multiplierClusterBucket(value) {
  const numeric = Math.max(0, safeNumber(value, 0));
  if (numeric < 1.25) return 0;
  if (numeric < 1.6) return 1;
  if (numeric < 2.1) return 2;
  if (numeric < 3) return 3;
  if (numeric < 5) return 4;
  if (numeric < 8) return 5;
  if (numeric < 12) return 6;
  if (numeric < 20) return 7;
  if (numeric < 35) return 8;
  if (numeric < 60) return 9;
  if (numeric < 100) return 10;
  if (numeric < 250) return 11;
  if (numeric < 500) return 12;
  return 13;
}

function buildClusterSequence(sequence) {
  return (Array.isArray(sequence) ? sequence : []).map(multiplierClusterBucket);
}

function averageSlice(values, startIndex, endIndexExclusive) {
  const items = Array.isArray(values) ? values : [];
  const safeStart = Math.max(0, Math.min(items.length, Number(startIndex) || 0));
  const safeEnd = Math.max(safeStart, Math.min(items.length, Number(endIndexExclusive) || 0));
  if (safeEnd <= safeStart) return 0;
  let sum = 0;
  for (let index = safeStart; index < safeEnd; index += 1) {
    sum += safeNumber(items[index], 0);
  }
  return sum / (safeEnd - safeStart);
}

function classifySpreadLabel(p25, p90, p50) {
  const median = Math.max(1, safeNumber(p50, 0));
  const spreadRatio = Math.max(0, safeNumber(p90, 0) - safeNumber(p25, 0)) / median;
  if (spreadRatio <= 0.75) return 'Tight';
  if (spreadRatio <= 1.8) return 'Balanced';
  return 'Wide';
}

function classifyConfidenceLabel(score) {
  const numeric = safeNumber(score, 0);
  if (numeric >= 82) return 'High';
  if (numeric >= 62) return 'Medium';
  return 'Low';
}

function buildZonePrediction(matches, coveragePct, latestRoundId, basisLabel) {
  const zoneMatches = Array.isArray(matches) ? matches : [];
  if (!zoneMatches.length) {
    return {
      confidencePct: Number(safeNumber(coveragePct, 0).toFixed(1)),
      confidenceLabel: classifyConfidenceLabel(coveragePct),
      roundIdFrom: null,
      roundIdTo: null,
      roundIdLabel: '-',
      offsetFrom: null,
      offsetTo: null,
      offsetLabel: 'No stable round range yet',
      basis: basisLabel,
      sampleCount: 0,
    };
  }

  const offsetFrom = Math.max(
    1,
    Math.round(weightedQuantile(zoneMatches, (item) => item.offset ?? item.nextPeakOffset, (item) => item.weight, 0.2, 1))
  );
  const offsetTo = Math.max(
    1,
    Math.round(weightedQuantile(zoneMatches, (item) => item.offset ?? item.nextPeakOffset, (item) => item.weight, 0.8, 1))
  );
  const roundIdFrom = latestRoundId ? latestRoundId + offsetFrom : null;
  const roundIdTo = latestRoundId ? latestRoundId + offsetTo : null;

  return {
    confidencePct: Number(safeNumber(coveragePct, 0).toFixed(1)),
    confidenceLabel: classifyConfidenceLabel(coveragePct),
    roundIdFrom,
    roundIdTo,
    roundIdLabel: formatRoundRange(roundIdFrom, roundIdTo),
    offsetFrom,
    offsetTo,
    offsetLabel: offsetFrom === offsetTo
      ? `around round ${offsetFrom} of the next window`
      : `around rounds ${offsetFrom}-${offsetTo} of the next window`,
    basis: basisLabel,
    sampleCount: zoneMatches.length,
  };
}

function buildMatchedNextRoundRows(rounds, matches) {
  const items = Array.isArray(rounds) ? rounds : [];
  const rows = [];

  for (const match of Array.isArray(matches) ? matches : []) {
    const start = Math.max(0, Number(match?.nextStartIndex) || 0);
    const end = Math.min(items.length, Number(match?.nextEndIndexExclusive) || 0);
    const nextRoundCount = Math.max(1, end - start);
    const perRoundWeight = safeNumber(match?.weight, 0) / nextRoundCount;

    for (let index = start; index < end; index += 1) {
      const round = items[index];
      rows.push({
        matchWeight: safeNumber(match?.weight, 0),
        weight: Math.max(perRoundWeight, 0.0001),
        multiplier: safeNumber(round?.multiplier, 0),
        roundId: round?.roundId || null,
        offset: (index - start) + 1,
      });
    }
  }

  return rows;
}

function buildMatchedNextFirstRoundRows(rounds, matches) {
  const items = Array.isArray(rounds) ? rounds : [];
  const rows = [];

  for (const match of Array.isArray(matches) ? matches : []) {
    const index = Math.max(0, Number(match?.nextStartIndex) || 0);
    if (index >= items.length) continue;
    const round = items[index];
    rows.push({
      weight: Math.max(safeNumber(match?.weight, 0), 0.0001),
      multiplier: safeNumber(round?.multiplier, 0),
      roundId: round?.roundId || null,
      offset: 1,
    });
  }

  return rows;
}

function buildPatternWindowSummary(rounds) {
  const summary = summarizeRounds(rounds);
  const sequence = buildSequenceSample(rounds, 12);
  return {
    ...summary,
    roundFromId: rounds[0]?.roundId ?? null,
    roundToId: rounds[rounds.length - 1]?.roundId ?? null,
    peak: summarizePeakRound(rounds),
    sequence,
    clusterSequence: buildClusterSequence(sequence),
    startAvg: averageSlice(sequence, 0, Math.max(1, Math.ceil(sequence.length / 3))),
    middleAvg: averageSlice(
      sequence,
      Math.max(0, Math.floor(sequence.length / 3)),
      Math.max(Math.floor(sequence.length / 3), Math.ceil((sequence.length * 2) / 3))
    ),
    endAvg: averageSlice(sequence, Math.max(0, Math.floor((sequence.length * 2) / 3)), sequence.length),
  };
}

function computePatternDistance(candidateSummary, referenceSummary) {
  if (!candidateSummary || !referenceSummary) return Number.POSITIVE_INFINITY;

  const distributionDistance = average(
    DISTRIBUTION_BANDS.map((band) => Math.abs(distributionPct(candidateSummary, band.key) - distributionPct(referenceSummary, band.key)))
  ) * 2.4;

  const scalarDistance = average([
    fuzzyMultiplierDiff(candidateSummary.avgMultiplier, referenceSummary.avgMultiplier),
    fuzzyMultiplierDiff(candidateSummary.medianMultiplier, referenceSummary.medianMultiplier),
    fuzzyMultiplierDiff(candidateSummary.p90Multiplier, referenceSummary.p90Multiplier),
    fuzzyMultiplierDiff(candidateSummary.maxMultiplier, referenceSummary.maxMultiplier),
  ]);

  const sequenceLength = Math.min(candidateSummary.sequence.length, referenceSummary.sequence.length);
  const sequenceDistance = sequenceLength
    ? average(
        Array.from({ length: sequenceLength }, (_, index) => (
          fuzzyMultiplierDiff(candidateSummary.sequence[index], referenceSummary.sequence[index])
        ))
      )
    : 1;

  const clusterLength = Math.min(
    Array.isArray(candidateSummary.clusterSequence) ? candidateSummary.clusterSequence.length : 0,
    Array.isArray(referenceSummary.clusterSequence) ? referenceSummary.clusterSequence.length : 0
  );
  const clusterDistance = clusterLength
    ? average(
        Array.from({ length: clusterLength }, (_, index) => (
          Math.abs(
            safeNumber(candidateSummary.clusterSequence[index], 0)
            - safeNumber(referenceSummary.clusterSequence[index], 0)
          ) / 3
        ))
      )
    : 1;

  const trendDistance = average([
    fuzzyMultiplierDiff(candidateSummary.startAvg, referenceSummary.startAvg),
    fuzzyMultiplierDiff(candidateSummary.middleAvg, referenceSummary.middleAvg),
    fuzzyMultiplierDiff(candidateSummary.endAvg, referenceSummary.endAvg),
  ]);

  const roundCountDistance = Math.abs(safeNumber(candidateSummary.roundCount, 0) - safeNumber(referenceSummary.roundCount, 0))
    / Math.max(6, safeNumber(referenceSummary.roundCount, 0), 1);

  return (
    (sequenceDistance * 0.34)
    + (clusterDistance * 0.24)
    + (scalarDistance * 0.18)
    + (trendDistance * 0.14)
    + (distributionDistance * 0.07)
    + (roundCountDistance * 0.03)
  );
}

function similarityPctFromDistance(distance) {
  return Number(clamp(Math.exp(-Math.max(0, safeNumber(distance, 0)) * 1.35) * 100, 0, 100).toFixed(1));
}

function serializePatternRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : []).map((round) => ({
    roundId: round.roundId || null,
    multiplier: safeNumber(round.multiplier, 0),
    label: formatMultiplier(round.multiplier),
  }));
}

function serializePatternRoundsRange(rounds, startIndex, endIndexExclusive) {
  const safeStart = Math.max(0, Number(startIndex) || 0);
  const safeEnd = Math.min(Array.isArray(rounds) ? rounds.length : 0, Number(endIndexExclusive) || 0);
  if (safeEnd <= safeStart) return [];
  return serializePatternRounds(rounds.slice(safeStart, safeEnd));
}

function summarizeRoundsRange(rounds, startIndex, endIndexExclusive) {
  const safeStart = Math.max(0, Number(startIndex) || 0);
  const safeEnd = Math.min(Array.isArray(rounds) ? rounds.length : 0, Number(endIndexExclusive) || 0);
  if (safeEnd <= safeStart) {
    return {
      roundCount: 0,
      avgMultiplier: 0,
      medianMultiplier: 0,
      p90Multiplier: 0,
      maxMultiplier: 0,
      distribution: DISTRIBUTION_BANDS.map((band) => ({
        key: band.key,
        label: band.label,
        count: 0,
        pct: 0,
      })),
    };
  }

  const values = [];
  const distributionCounts = Object.fromEntries(DISTRIBUTION_BANDS.map((band) => [band.key, 0]));
  let sum = 0;
  let max = 0;

  for (let index = safeStart; index < safeEnd; index += 1) {
    const multiplier = safeNumber(rounds[index]?.multiplier, 0);
    values.push(multiplier);
    sum += multiplier;
    if (multiplier > max) max = multiplier;
    for (const band of DISTRIBUTION_BANDS) {
      if (multiplier >= band.min && multiplier < band.max) {
        distributionCounts[band.key] += 1;
        break;
      }
    }
  }

  values.sort((left, right) => left - right);
  const roundCount = values.length;

  return {
    roundCount,
    avgMultiplier: ratio(sum, roundCount),
    medianMultiplier: quantile(values, 0.5),
    p90Multiplier: quantile(values, 0.9),
    maxMultiplier: max || 0,
    distribution: DISTRIBUTION_BANDS.map((band) => ({
      key: band.key,
      label: band.label,
      count: distributionCounts[band.key],
      pct: ratio(distributionCounts[band.key], roundCount),
    })),
  };
}

function buildRollingInputWindow(rounds, windowMs, timeZone) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) return null;

  const endIndex = items.length - 1;
  const endTimestamp = items[endIndex].timestamp;
  const startTimestamp = endTimestamp - windowMs;
  let startIndex = endIndex;
  while (startIndex > 0 && items[startIndex - 1].timestamp >= startTimestamp) {
    startIndex -= 1;
  }

  const windowRounds = items.slice(startIndex, endIndex + 1);
  const sequence = buildSequenceSampleRange(items, startIndex, endIndex + 1, 12);
  const summary = {
    ...summarizeRoundsRange(items, startIndex, endIndex + 1),
    roundFromId: items[startIndex]?.roundId ?? null,
    roundToId: items[endIndex]?.roundId ?? null,
    peak: summarizePeakRoundRange(items, startIndex, endIndex + 1),
    sequence,
    clusterSequence: buildClusterSequence(sequence),
    startAvg: averageSlice(sequence, 0, Math.max(1, Math.ceil(sequence.length / 3))),
    middleAvg: averageSlice(
      sequence,
      Math.max(0, Math.floor(sequence.length / 3)),
      Math.max(Math.floor(sequence.length / 3), Math.ceil((sequence.length * 2) / 3))
    ),
    endAvg: averageSlice(sequence, Math.max(0, Math.floor((sequence.length * 2) / 3)), sequence.length),
  };

  return {
    startIndex,
    endIndex,
    startTimestamp,
    endTimestamp,
    label: formatWindowTimeRange(startTimestamp, endTimestamp, timeZone),
    datedLabel: formatWindowTimeRange(startTimestamp, endTimestamp, timeZone, true),
    rounds: windowRounds,
    summary,
  };
}

function buildRangePatternReport(rounds, windowConfig, timeZone) {
  const currentInput = buildRollingInputWindow(rounds, windowConfig.ms, timeZone);
  if (!currentInput || currentInput.summary.roundCount < 2) {
    return {
      available: false,
      reason: 'Not enough recent rounds yet to build a pattern window.',
    };
  }

  const latestRound = rounds[rounds.length - 1];
  const latestRoundId = latestRound?.roundId || null;
  const candidateRows = [];
  const minimumInputRounds = Math.max(3, Math.min(12, Math.round(currentInput.summary.roundCount * 0.4)));

  let inputStart = 0;
  let nextEnd = 1;
  for (let anchorIndex = 0; anchorIndex < rounds.length - 1; anchorIndex += 1) {
    const anchorRound = rounds[anchorIndex];
    const anchorTimestamp = anchorRound.timestamp;
    if (anchorTimestamp >= currentInput.startTimestamp) break;

    while (inputStart <= anchorIndex && rounds[inputStart].timestamp < (anchorTimestamp - windowConfig.ms)) {
      inputStart += 1;
    }
    if (nextEnd < anchorIndex + 1) nextEnd = anchorIndex + 1;
    while (nextEnd < rounds.length && rounds[nextEnd].timestamp <= (anchorTimestamp + windowConfig.ms)) {
      nextEnd += 1;
    }

    const inputCount = (anchorIndex + 1) - inputStart;
    const nextCount = nextEnd - (anchorIndex + 1);
    if (inputCount < minimumInputRounds || nextCount <= 0) continue;

    const inputSequence = buildSequenceSampleRange(rounds, inputStart, anchorIndex + 1, 12);
    const inputSummary = {
      ...summarizeRoundsRange(rounds, inputStart, anchorIndex + 1),
      roundFromId: rounds[inputStart]?.roundId ?? null,
      roundToId: rounds[anchorIndex]?.roundId ?? null,
      peak: summarizePeakRoundRange(rounds, inputStart, anchorIndex + 1),
      sequence: inputSequence,
      clusterSequence: buildClusterSequence(inputSequence),
      startAvg: averageSlice(inputSequence, 0, Math.max(1, Math.ceil(inputSequence.length / 3))),
      middleAvg: averageSlice(
        inputSequence,
        Math.max(0, Math.floor(inputSequence.length / 3)),
        Math.max(Math.floor(inputSequence.length / 3), Math.ceil((inputSequence.length * 2) / 3))
      ),
      endAvg: averageSlice(inputSequence, Math.max(0, Math.floor((inputSequence.length * 2) / 3)), inputSequence.length),
    };
    const nextSummary = {
      ...summarizeRoundsRange(rounds, anchorIndex + 1, nextEnd),
      roundFromId: rounds[anchorIndex + 1]?.roundId ?? null,
      roundToId: rounds[nextEnd - 1]?.roundId ?? null,
      peak: summarizePeakRoundRange(rounds, anchorIndex + 1, nextEnd),
      sequence: buildSequenceSampleRange(rounds, anchorIndex + 1, nextEnd, 12),
    };
    const distance = computePatternDistance(inputSummary, currentInput.summary);
    const similarityPct = similarityPctFromDistance(distance);
    const daysAgo = ratio(currentInput.endTimestamp - anchorTimestamp, DAY_MS);
    const weight = (1 / (1 + (distance * 3.6))) * (1 / (1 + (daysAgo / 20)));

    candidateRows.push({
      anchorTimestamp,
      distance,
      similarityPct,
      weight,
      inputStartTimestamp: anchorTimestamp - windowConfig.ms,
      inputEndTimestamp: anchorTimestamp,
      inputStartIndex: inputStart,
      inputEndIndexExclusive: anchorIndex + 1,
      inputRoundFrom: inputSummary.roundFromId,
      inputRoundTo: inputSummary.roundToId,
      inputRoundCount: inputSummary.roundCount,
      nextStartTimestamp: anchorTimestamp,
      nextEndTimestamp: anchorTimestamp + windowConfig.ms,
      nextStartIndex: anchorIndex + 1,
      nextEndIndexExclusive: nextEnd,
      nextRoundFrom: nextSummary.roundFromId,
      nextRoundTo: nextSummary.roundToId,
      nextRoundCount: nextSummary.roundCount,
      nextPeakMultiplier: nextSummary.peak.peakMultiplier,
      nextPeakRoundId: nextSummary.peak.peakRoundId,
      nextPeakOffset: nextSummary.peak.peakOffset,
    });
  }

  if (candidateRows.length < 4) {
    return {
      available: false,
      reason: 'Not enough historical pattern matches yet for this window size.',
      currentInput,
    };
  }

  const usedMatches = candidateRows
    .sort((left, right) => left.distance - right.distance)
    .slice(0, clamp(Math.round(Math.sqrt(candidateRows.length)), 6, 12));

  const nextFirstRoundRows = buildMatchedNextFirstRoundRows(rounds, usedMatches);
  const matchedNextRoundRows = buildMatchedNextRoundRows(rounds, usedMatches);
  const nextRoundP25 = weightedQuantile(nextFirstRoundRows, (item) => item.multiplier, (item) => item.weight, 0.25, 0);
  const nextRoundP50 = weightedQuantile(nextFirstRoundRows, (item) => item.multiplier, (item) => item.weight, 0.5, 0);
  const nextRoundP75 = weightedQuantile(nextFirstRoundRows, (item) => item.multiplier, (item) => item.weight, 0.75, 0);
  const peakP25 = weightedQuantile(usedMatches, (item) => item.nextPeakMultiplier, (item) => item.weight, 0.25, 0);
  const peakP50 = weightedQuantile(usedMatches, (item) => item.nextPeakMultiplier, (item) => item.weight, 0.5, 0);
  const peakP75 = weightedQuantile(usedMatches, (item) => item.nextPeakMultiplier, (item) => item.weight, 0.75, 0);
  const peakP90 = weightedQuantile(usedMatches, (item) => item.nextPeakMultiplier, (item) => item.weight, 0.9, 0);
  const likelyZoneFrom = nextRoundP25;
  const likelyZoneTo = nextRoundP75;
  const stretchZoneFrom = peakP50;
  const stretchZoneTo = peakP75;
  const rareSpikeFrom = peakP90;
  const matchedSetupRounds = usedMatches.reduce((sum, item) => sum + safeNumber(item.inputRoundCount, 0), 0);
  const matchedNextRounds = usedMatches.reduce((sum, item) => sum + safeNumber(item.nextRoundCount, 0), 0);
  const averageSimilarityPct = weightedAverage(usedMatches, (item) => item.similarityPct, (item) => item.weight, 0);
  const nextRoundCoveragePct = weightedAverage(
    nextFirstRoundRows,
    (item) => (item.multiplier >= likelyZoneFrom && item.multiplier <= likelyZoneTo ? 100 : 0),
    (item) => item.weight,
    0
  );
  const likelyZoneCoveragePct = nextRoundCoveragePct;
  const belowLikelyPct = weightedAverage(
    nextFirstRoundRows,
    (item) => (item.multiplier < likelyZoneFrom ? 100 : 0),
    (item) => item.weight,
    0
  );
  const likelyZoneMatches = nextFirstRoundRows.filter(
    (item) => item.multiplier >= likelyZoneFrom && item.multiplier <= likelyZoneTo
  );
  const stretchZoneMatches = usedMatches.filter(
    (item) => item.nextPeakMultiplier >= stretchZoneFrom && item.nextPeakMultiplier <= stretchZoneTo
  );
  const rareSpikeMatches = usedMatches.filter(
    (item) => item.nextPeakMultiplier >= rareSpikeFrom
  );
  const HIT_CHANCE_THRESHOLDS = [2, 5, 10, 25, 50];
  const hitChances = HIT_CHANCE_THRESHOLDS.map((threshold) => {
    const chancePct = weightedAverage(
      usedMatches,
      (item) => (item.nextPeakMultiplier >= threshold ? 100 : 0),
      (item) => item.weight,
      0
    );
    return {
      threshold,
      label: formatThresholdLabel(threshold),
      chancePct: Number(chancePct.toFixed(1)),
    };
  });
  const stretchZoneCoveragePct = hitChances.find((item) => item.threshold === 10)?.chancePct || 0;
  const rareSpikeCoveragePct = hitChances.find((item) => item.threshold === 50)?.chancePct || 0;
  const likelyZonePrediction = buildZonePrediction(
    likelyZoneMatches,
    likelyZoneCoveragePct,
    latestRoundId,
    `Built from ${likelyZoneMatches.length || 0} matched first-round outcomes inside the next-round band.`
  );
  const stretchZonePrediction = buildZonePrediction(
    stretchZoneMatches,
    stretchZoneCoveragePct,
    latestRoundId,
    `Built from ${stretchZoneMatches.length || 0} matched next-window peaks in the upper band.`
  );
  const rareSpikePrediction = buildZonePrediction(
    rareSpikeMatches,
    rareSpikeCoveragePct,
    latestRoundId,
    `Built from ${rareSpikeMatches.length || 0} matched next-window peaks in the high tail.`
  );
  const predictedPeakOffsetFrom = likelyZonePrediction.offsetFrom;
  const predictedPeakOffsetTo = likelyZonePrediction.offsetTo;
  const predictedPeakRoundIdFrom = latestRoundId ? latestRoundId + predictedPeakOffsetFrom : null;
  const predictedPeakRoundIdTo = latestRoundId ? latestRoundId + predictedPeakOffsetTo : null;
  const spreadLabel = classifySpreadLabel(nextRoundP25, peakP90, peakP50);
  const spreadRatio = Math.max(0, peakP90 - nextRoundP25) / Math.max(1, peakP50);
  const confidencePct = clamp(
    (averageSimilarityPct * 0.52)
    + ((Math.min(usedMatches.length, 12) / 12) * 22)
    + (Math.max(0, 1 - Math.min(spreadRatio, 3.5) / 3.5) * 26),
    8,
    96
  );

  return {
    available: true,
    currentInput,
    nextWindowLabel: formatWindowTimeRange(currentInput.endTimestamp, currentInput.endTimestamp + windowConfig.ms, timeZone),
    nextWindowDatedLabel: formatWindowTimeRange(currentInput.endTimestamp, currentInput.endTimestamp + windowConfig.ms, timeZone, true),
    candidateCount: candidateRows.length,
    usedMatches: usedMatches.length,
    lookbackDaysUsed: ratio(currentInput.endTimestamp - rounds[0].timestamp, DAY_MS),
    averageSimilarityPct,
    roundsComputed: {
      currentInputRounds: currentInput.summary.roundCount,
      matchedSetupRounds,
      matchedNextRounds,
      avgSetupRoundsPerMatch: ratio(matchedSetupRounds, usedMatches.length),
      avgNextRoundsPerMatch: ratio(matchedNextRounds, usedMatches.length),
      totalComparedRounds: matchedSetupRounds + matchedNextRounds,
    },
    prediction: {
      rangeFrom: nextRoundP25,
      rangeTo: nextRoundP75,
      medianPeak: peakP50,
      rangeLabel: formatMultiplierRange(nextRoundP25, nextRoundP75),
      likelyZoneFrom,
      likelyZoneTo,
      likelyZoneLabel: formatMultiplierRange(likelyZoneFrom, likelyZoneTo),
      stretchZoneFrom,
      stretchZoneTo,
      stretchZoneLabel: formatMultiplierRange(stretchZoneFrom, stretchZoneTo),
      rareSpikeFrom,
      rareSpikeLabel: `${formatMultiplier(rareSpikeFrom)}+`,
      p25: nextRoundP25,
      p50: nextRoundP50,
      p75: nextRoundP75,
      p90: peakP90,
      p25Label: formatMultiplier(nextRoundP25),
      p50Label: formatMultiplier(nextRoundP50),
      p75Label: formatMultiplier(nextRoundP75),
      p90Label: formatMultiplier(peakP90),
      spreadLabel,
      spreadRatio: Number(spreadRatio.toFixed(2)),
      confidencePct: Number(confidencePct.toFixed(1)),
      confidenceLabel: classifyConfidenceLabel(confidencePct),
      nextRound: {
        from: nextRoundP25,
        to: nextRoundP75,
        median: nextRoundP50,
        label: formatMultiplierRange(nextRoundP25, nextRoundP75),
        medianLabel: formatMultiplier(nextRoundP50),
        roundId: latestRoundId ? latestRoundId + 1 : null,
        roundIdLabel: latestRoundId ? `#${latestRoundId + 1}` : '-',
        confidencePct: Number(likelyZoneCoveragePct.toFixed(1)),
        confidenceLabel: classifyConfidenceLabel(likelyZoneCoveragePct),
      },
      windowHitChances: hitChances,
      expectedPeak: {
        p50: peakP50,
        p75: peakP75,
        p90: peakP90,
        p50Label: formatMultiplier(peakP50),
        p75Label: formatMultiplier(peakP75),
        p90Label: formatMultiplier(peakP90),
        roundIdFrom: stretchZonePrediction.roundIdFrom,
        roundIdTo: rareSpikePrediction.roundIdTo || stretchZonePrediction.roundIdTo,
        roundIdLabel: formatRoundRange(
          stretchZonePrediction.roundIdFrom,
          rareSpikePrediction.roundIdTo || stretchZonePrediction.roundIdTo
        ),
        offsetLabel: stretchZonePrediction.offsetLabel || 'No stable peak range yet',
      },
      likelyZonePrediction,
      stretchZonePrediction,
      rareSpikePrediction,
      predictedPeakRoundIdFrom,
      predictedPeakRoundIdTo,
      predictedPeakRoundIdLabel: formatRoundRange(predictedPeakRoundIdFrom, predictedPeakRoundIdTo),
      predictedPeakRoundIdBasis: `Built from ${nextFirstRoundRows.length} matched next-round outcomes.`,
      predictedPeakOffsetFrom,
      predictedPeakOffsetTo,
      predictedPeakOffsetLabel: predictedPeakOffsetFrom === predictedPeakOffsetTo
        ? `around round ${predictedPeakOffsetFrom} of the next window`
        : `around rounds ${predictedPeakOffsetFrom}-${predictedPeakOffsetTo} of the next window`,
      summary: `Next matched first rounds usually land in ${formatMultiplierRange(nextRoundP25, nextRoundP75)}. Across the whole next window, ${formatThresholdLabel(5)} hit ${formatPercent(hitChances.find((item) => item.threshold === 5)?.chancePct || 0)}, ${formatThresholdLabel(10)} hit ${formatPercent(hitChances.find((item) => item.threshold === 10)?.chancePct || 0)}, and the expected peak sat near ${formatMultiplier(peakP50)}.`,
    },
    note: `Matched the latest ${windowConfig.label.toLowerCase()} pattern against ${candidateRows.length} historical patterns and kept the closest ${usedMatches.length}.`,
    support: {
      matchedPatterns: usedMatches.length,
      candidatePatterns: candidateRows.length,
      averageSimilarityPct,
      likelyZoneCoveragePct: Number(likelyZoneCoveragePct.toFixed(1)),
      stretchZoneCoveragePct: Number(stretchZoneCoveragePct.toFixed(1)),
      rareSpikeCoveragePct: Number(rareSpikeCoveragePct.toFixed(1)),
      belowLikelyPct: Number(belowLikelyPct.toFixed(1)),
      spreadLabel,
      summary: `Across matched history, the next round usually stayed in ${formatMultiplierRange(nextRoundP25, nextRoundP75)}, while the whole next window peaked near ${formatMultiplier(peakP50)} and reached ${formatThresholdLabel(10)} ${formatPercent(hitChances.find((item) => item.threshold === 10)?.chancePct || 0)} of the time.`,
    },
    honesty: {
      label: spreadLabel === 'Tight' ? 'Tighter Read' : spreadLabel === 'Balanced' ? 'Usable Read' : 'Wide Read',
      note: spreadLabel === 'Tight'
        ? 'The matched first rounds stayed compact and the next-window peaks agreed fairly well.'
        : spreadLabel === 'Balanced'
          ? 'The matched first rounds were usable, but the next-window peaks still had some spread.'
          : 'The next round is readable, but the full next-window peak still spreads out across matched history.',
    },
    examples: usedMatches.slice(0, 4).map((item, index) => ({
      rank: index + 1,
      similarityPct: item.similarityPct,
      inputLabel: formatWindowTimeRange(item.inputStartTimestamp, item.inputEndTimestamp, timeZone, true),
      inputRoundFrom: item.inputRoundFrom,
      inputRoundTo: item.inputRoundTo,
      inputRoundCount: item.inputRoundCount,
      inputRounds: serializePatternRoundsRange(rounds, item.inputStartIndex, item.inputEndIndexExclusive),
      nextLabel: formatWindowTimeRange(item.nextStartTimestamp, item.nextEndTimestamp, timeZone, true),
      nextRoundFrom: item.nextRoundFrom,
      nextRoundTo: item.nextRoundTo,
      nextRoundCount: item.nextRoundCount,
      nextRounds: serializePatternRoundsRange(rounds, item.nextStartIndex, item.nextEndIndexExclusive),
      nextPeakMultiplier: item.nextPeakMultiplier,
      nextPeakLabel: formatMultiplier(item.nextPeakMultiplier),
      nextPeakRoundId: item.nextPeakRoundId,
      nextPeakOffset: item.nextPeakOffset,
    })),
  };
}

function buildEmptyReport(windowConfig, timeZone) {
  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId: null,
    totalRounds: 0,
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    timeZone,
    dataset: {
      totalRounds: 0,
      startTimestamp: null,
      endTimestamp: null,
      spanDays: 0,
    },
    window: {
      key: windowConfig.key,
      label: windowConfig.label,
      ms: windowConfig.ms,
      inputLabel: '-',
      predictionLabel: '-',
    },
    currentPattern: {
      label: '-',
      roundCount: 0,
      roundFromId: null,
      roundToId: null,
      avgMultiplier: 0,
      medianMultiplier: 0,
      p90Multiplier: 0,
      maxMultiplier: 0,
      rounds: [],
    },
    prediction: {
      rangeFrom: null,
      rangeTo: null,
      medianPeak: null,
      rangeLabel: '-',
      likelyZoneFrom: null,
      likelyZoneTo: null,
      likelyZoneLabel: '-',
      stretchZoneFrom: null,
      stretchZoneTo: null,
      stretchZoneLabel: '-',
      rareSpikeFrom: null,
      rareSpikeLabel: '-',
      p25: null,
      p50: null,
      p75: null,
      p90: null,
      p25Label: '-',
      p50Label: '-',
      p75Label: '-',
      p90Label: '-',
      spreadLabel: 'Wide',
      spreadRatio: 0,
      confidencePct: 0,
      confidenceLabel: 'Low',
      nextRound: {
        from: null,
        to: null,
        median: null,
        label: '-',
        medianLabel: '-',
        roundId: null,
        roundIdLabel: '-',
        confidencePct: 0,
        confidenceLabel: 'Low',
      },
      windowHitChances: [
        { threshold: 2, label: '2x+', chancePct: 0 },
        { threshold: 5, label: '5x+', chancePct: 0 },
        { threshold: 10, label: '10x+', chancePct: 0 },
        { threshold: 25, label: '25x+', chancePct: 0 },
        { threshold: 50, label: '50x+', chancePct: 0 },
      ],
      expectedPeak: {
        p50: null,
        p75: null,
        p90: null,
        p50Label: '-',
        p75Label: '-',
        p90Label: '-',
        roundIdFrom: null,
        roundIdTo: null,
        roundIdLabel: '-',
        offsetLabel: 'No stable peak range yet',
      },
      likelyZonePrediction: {
        confidencePct: 0,
        confidenceLabel: 'Low',
        roundIdFrom: null,
        roundIdTo: null,
        roundIdLabel: '-',
        offsetFrom: null,
        offsetTo: null,
        offsetLabel: 'No stable round range yet',
        basis: '',
        sampleCount: 0,
      },
      stretchZonePrediction: {
        confidencePct: 0,
        confidenceLabel: 'Low',
        roundIdFrom: null,
        roundIdTo: null,
        roundIdLabel: '-',
        offsetFrom: null,
        offsetTo: null,
        offsetLabel: 'No stable round range yet',
        basis: '',
        sampleCount: 0,
      },
      rareSpikePrediction: {
        confidencePct: 0,
        confidenceLabel: 'Low',
        roundIdFrom: null,
        roundIdTo: null,
        roundIdLabel: '-',
        offsetFrom: null,
        offsetTo: null,
        offsetLabel: 'No stable round range yet',
        basis: '',
        sampleCount: 0,
      },
      predictedPeakRoundIdFrom: null,
      predictedPeakRoundIdTo: null,
      predictedPeakRoundIdLabel: '-',
      predictedPeakRoundIdBasis: '',
      predictedPeakOffsetFrom: null,
      predictedPeakOffsetTo: null,
      predictedPeakOffsetLabel: '-',
      summary: 'No rounds are stored yet, so there is no pattern prediction yet.',
    },
    patternMatches: {
      available: false,
      note: 'No rounds are stored yet.',
      candidateCount: 0,
      usedMatches: 0,
      lookbackDaysUsed: 0,
      averageSimilarityPct: 0,
      roundsComputed: {
        currentInputRounds: 0,
        matchedSetupRounds: 0,
        matchedNextRounds: 0,
        avgSetupRoundsPerMatch: 0,
        avgNextRoundsPerMatch: 0,
        totalComparedRounds: 0,
      },
      examples: [],
    },
    support: {
      matchedPatterns: 0,
      candidatePatterns: 0,
      averageSimilarityPct: 0,
      likelyZoneCoveragePct: 0,
      stretchZoneCoveragePct: 0,
      rareSpikeCoveragePct: 0,
      belowLikelyPct: 0,
      spreadLabel: 'Wide',
      summary: 'No support data yet.',
    },
    honesty: {
      label: 'Wide Read',
      note: 'No support data yet.',
    },
  };
}

function buildTimingAnalyticsReport(rounds, options = {}) {
  const windowKey = normalizeTimingWindowKey(options.windowKey);
  const timeZone = normalizeTimingTimeZone(options.timeZone);
  const windowConfig = WINDOW_MAP.get(windowKey);
  const normalizedRounds = normalizeRounds(rounds);

  if (!normalizedRounds.length) {
    return buildEmptyReport(windowConfig, timeZone);
  }

  const latestRound = normalizedRounds[normalizedRounds.length - 1];
  const earliestRound = normalizedRounds[0];
  const rangeReport = buildRangePatternReport(normalizedRounds, windowConfig, timeZone);

  if (!rangeReport.available) {
    const empty = buildEmptyReport(windowConfig, timeZone);
    return {
      ...empty,
      generatedAt: Date.now(),
      latestRoundId: latestRound.roundId || null,
      totalRounds: normalizedRounds.length,
      dataset: {
        totalRounds: normalizedRounds.length,
        startTimestamp: earliestRound.timestamp,
        endTimestamp: latestRound.timestamp,
        spanDays: ratio(latestRound.timestamp - earliestRound.timestamp, DAY_MS),
      },
      patternMatches: {
        ...empty.patternMatches,
        note: rangeReport.reason || empty.patternMatches.note,
      },
      prediction: {
        ...empty.prediction,
        summary: rangeReport.reason || empty.prediction.summary,
      },
      support: {
        ...empty.support,
      },
      honesty: {
        ...empty.honesty,
      },
    };
  }

  const currentInput = rangeReport.currentInput;

  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId: latestRound.roundId || null,
    totalRounds: normalizedRounds.length,
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    timeZone,
    dataset: {
      totalRounds: normalizedRounds.length,
      startTimestamp: earliestRound.timestamp,
      endTimestamp: latestRound.timestamp,
      spanDays: ratio(latestRound.timestamp - earliestRound.timestamp, DAY_MS),
    },
    window: {
      key: windowConfig.key,
      label: windowConfig.label,
      ms: windowConfig.ms,
      inputLabel: currentInput.label,
      inputDatedLabel: currentInput.datedLabel,
      predictionLabel: rangeReport.nextWindowLabel,
      predictionDatedLabel: rangeReport.nextWindowDatedLabel,
      inputStartTimestamp: currentInput.startTimestamp,
      inputEndTimestamp: currentInput.endTimestamp,
      predictionStartTimestamp: currentInput.endTimestamp,
      predictionEndTimestamp: currentInput.endTimestamp + windowConfig.ms,
    },
    currentPattern: {
      label: currentInput.label,
      roundCount: currentInput.summary.roundCount,
      roundFromId: currentInput.summary.roundFromId,
      roundToId: currentInput.summary.roundToId,
      avgMultiplier: currentInput.summary.avgMultiplier,
      medianMultiplier: currentInput.summary.medianMultiplier,
      p90Multiplier: currentInput.summary.p90Multiplier,
      maxMultiplier: currentInput.summary.maxMultiplier,
      rounds: serializePatternRounds(currentInput.rounds),
    },
    prediction: rangeReport.prediction,
    patternMatches: {
      available: true,
      note: rangeReport.note,
      candidateCount: rangeReport.candidateCount,
      usedMatches: rangeReport.usedMatches,
      lookbackDaysUsed: rangeReport.lookbackDaysUsed,
      averageSimilarityPct: rangeReport.averageSimilarityPct,
      roundsComputed: rangeReport.roundsComputed,
      examples: rangeReport.examples,
    },
    support: rangeReport.support,
    honesty: rangeReport.honesty,
  };
}

module.exports = {
  buildTimingAnalyticsReport,
  normalizeTimingWindowKey,
  normalizeTimingTimeZone,
};
