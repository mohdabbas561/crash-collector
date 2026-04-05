'use strict';

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

const WINDOW_OPTIONS = [
  { key: '5m',  label: '5 Minutes',  ms: 5  * MINUTE_MS },
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '1h',  label: '1 Hour',     ms: 1  * HOUR_MS   },
  { key: '2h',  label: '2 Hours',    ms: 2  * HOUR_MS   },
  { key: '5h',  label: '5 Hours',    ms: 5  * HOUR_MS   },
  { key: '12h', label: '12 Hours',   ms: 12 * HOUR_MS   },
  { key: '24h', label: '24 Hours',   ms: 24 * HOUR_MS   },
  { key: '3d',  label: '3 Days',     ms: 3  * DAY_MS    },
  { key: '7d',  label: '7 Days',     ms: 7  * DAY_MS    },
  { key: '10d', label: '10 Days',    ms: 10 * DAY_MS    },
  { key: '15d', label: '15 Days',    ms: 15 * DAY_MS    },
  { key: '30d', label: '30 Days',    ms: 30 * DAY_MS    },
];

const WINDOW_MAP = new Map(WINDOW_OPTIONS.map((item) => [item.key, item]));
const DEFAULT_WINDOW_KEY = '5m';

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const TARGET_SET = new Set(TARGETS);
const DEFAULT_TARGET = 5;

const DISTRIBUTION_BANDS = [
  { key: 'lt2',       label: '<2x',        min: -Infinity, max: 2,    color: '#ff5d73' },
  { key: '2to5',      label: '2x-5x',      min: 2,         max: 5,    color: '#ff9f43' },
  { key: '5to10',     label: '5x-10x',     min: 5,         max: 10,   color: '#ffd84d' },
  { key: '10to20',    label: '10x-20x',    min: 10,        max: 20,   color: '#9ef01a' },
  { key: '20to50',    label: '20x-50x',    min: 20,        max: 50,   color: '#22d3ee' },
  { key: '50to100',   label: '50x-100x',   min: 50,        max: 100,  color: '#38bdf8' },
  { key: '100to500',  label: '100x-500x',  min: 100,       max: 500,  color: '#f472b6' },
  { key: '500to1000', label: '500x-1000x', min: 500,       max: 1000, color: '#c084fc' },
  { key: 'gte1000',   label: '1000x+',     min: 1000,      max: Infinity, color: '#818cf8' },
];

const COOLDOWN_WINDOWS = [
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '20m', label: '20 Minutes', ms: 20 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '60m', label: '60 Minutes', ms: 60 * MINUTE_MS },
];

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const WEEKDAY_INDEX = WEEKDAYS.reduce((acc, label, index) => { acc[label] = index; return acc; }, {});

// ---------------------------------------------------------------------------
// STATISTICAL SIGNIFICANCE  (chi-square 1-df, one-tailed)
// ---------------------------------------------------------------------------
// Minimum sample thresholds before we trust a rate comparison.
// MIN_CELL = 5 is the standard Cochran condition for chi-square validity.
const MIN_CELL_COUNT = 5;

/**
 * Chi-square goodness-of-fit for a binary rate vs expected rate.
 * Returns { significant, pValue, chi2 }.
 * Uses a continuity-corrected (Yates) formula for small cells.
 */
function chiSquareTest(observed, total, expectedRate) {
  if (total < MIN_CELL_COUNT * 2 || expectedRate <= 0 || expectedRate >= 1) {
    return { significant: false, pValue: 1, chi2: 0 };
  }
  const expected = total * expectedRate;
  const expectedNeg = total * (1 - expectedRate);
  const observedNeg = total - observed;
  // Yates continuity correction
  const diff = Math.abs(observed - expected) - 0.5;
  if (diff <= 0) return { significant: false, pValue: 1, chi2: 0 };
  const chi2 = (diff * diff) / expected + (diff * diff) / expectedNeg;
  // p-value approximation for chi2 with df=1 (Wilson-Hilferty)
  const pValue = chiSquarePValue(chi2);
  return { significant: pValue < 0.05, pValue, chi2 };
}

/**
 * Approximation of the survival function of chi-squared(df=1).
 * Uses the complementary error function via a Horner series.
 */
function chiSquarePValue(chi2) {
  // P(X > chi2) for df=1  = erfc(sqrt(chi2/2))
  const x = Math.sqrt(chi2 / 2);
  return erfcApprox(x);
}

/**
 * Complementary error function approximation (Abramowitz & Stegun 7.1.26).
 * Max absolute error < 1.5e-7.
 */
function erfcApprox(x) {
  if (x < 0) return 2 - erfcApprox(-x);
  const t = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return poly * Math.exp(-x * x);
}

// ---------------------------------------------------------------------------
// NORMALISATION  (z-score per feature across a collection)
// ---------------------------------------------------------------------------

/**
 * Given an array of objects and a list of numeric feature keys,
 * returns a parallel array of z-score normalised feature vectors.
 * Features with zero std-dev are left as 0.
 */
function zScoreNormalize(items, featureKeys) {
  if (!items.length) return [];
  const means = {};
  const stds = {};
  for (const key of featureKeys) {
    const values = items.map((item) => safeNumber(item[key], 0));
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    means[key] = mean;
    stds[key] = Math.sqrt(variance) || 1; // avoid divide-by-zero
  }
  return items.map((item) => {
    const vec = {};
    for (const key of featureKeys) {
      vec[key] = (safeNumber(item[key], 0) - means[key]) / stds[key];
    }
    return vec;
  });
}

// ---------------------------------------------------------------------------
// DECAY  –  exponential time-decay weight
// ---------------------------------------------------------------------------

/**
 * Exponential decay weight.
 * halfLifeDays: age at which weight = 0.5.
 * Returns a value in (0, 1].
 */
function decayWeight(ageMs, halfLifeDays = 14) {
  const ageDays = ageMs / DAY_MS;
  return Math.pow(0.5, ageDays / halfLifeDays);
}

// ---------------------------------------------------------------------------
// NORMALISATION HELPERS
// ---------------------------------------------------------------------------

function normalizeTimingWindowKey(raw) {
  const key = String(raw || '').trim().toLowerCase();
  return WINDOW_MAP.has(key) ? key : DEFAULT_WINDOW_KEY;
}

function normalizeTimingTarget(raw) {
  const numeric = Number.parseInt(raw, 10);
  return TARGET_SET.has(numeric) ? numeric : DEFAULT_TARGET;
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

// ---------------------------------------------------------------------------
// MATH UTILITIES
// ---------------------------------------------------------------------------

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function ratio(num, den, fallback = 0) {
  return den > 0 ? num / den : fallback;
}

function average(values) {
  if (!values.length) return 0;
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
    .sort((a, b) => a.value - b.value);

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

function labelForTarget(target) { return `${target}x`; }

function pctString(value, digits = 1) {
  const numeric = safeNumber(value, null);
  if (numeric == null) return '-';
  return `${(numeric * 100).toFixed(digits)}%`;
}

function formatHourLabel(hour) {
  const safeHour = ((Number(hour) % 24) + 24) % 24;
  const suffix = safeHour >= 12 ? 'PM' : 'AM';
  const hour12 = safeHour % 12 || 12;
  return `${hour12}:00 ${suffix}`;
}

function formatClockMinute(minuteOfDay) {
  const total = ((Number(minuteOfDay) % 1440) + 1440) % 1440;
  const hour = Math.floor(total / 60);
  const minute = total % 60;
  const suffix = hour >= 12 ? 'PM' : 'AM';
  const hour12 = hour % 12 || 12;
  return `${hour12}:${String(minute).padStart(2, '0')} ${suffix}`;
}

function formatSlotLabel(startMinute, slotMinutes, mode) {
  if (mode === 'start-time') return `Starts ${formatClockMinute(startMinute)}`;
  if (slotMinutes >= 1440) return `All Day (${formatClockMinute(startMinute)} start)`;
  const endMinute = (startMinute + slotMinutes) % 1440;
  return `${formatClockMinute(startMinute)} - ${formatClockMinute(endMinute)}`;
}

function formatOccurrenceLabel(startMinute, slotMinutes, mode, dayOffset) {
  const base = formatSlotLabel(startMinute, slotMinutes, mode);
  if (!dayOffset) return base;
  return `${base} Tomorrow`;
}

function formatTimestampInTimeZone(timestamp, timeZone) {
  const numeric = safeNumber(timestamp, 0);
  if (!numeric) return '-';
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  }).format(new Date(numeric));
}

function summarizeRoundRange(rounds, focusTarget, limit = 8) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) return { fromRoundId: null, toRoundId: null, roundCount: 0, hitRoundIds: [], hitCount: 0 };
  const hitRoundIds = items
    .filter((round) => round.multiplier >= focusTarget)
    .map((round) => round.roundId)
    .slice(0, limit);
  return {
    fromRoundId: items[0]?.roundId ?? null,
    toRoundId: items[items.length - 1]?.roundId ?? null,
    roundCount: items.length,
    hitRoundIds,
    hitCount: items.filter((round) => round.multiplier >= focusTarget).length,
  };
}

function chooseSlotMinutes(windowMs) {
  if (windowMs > DAY_MS) return 60;
  const minutes = Math.round(windowMs / MINUTE_MS);
  return Math.max(5, minutes || 5);
}

// ---------------------------------------------------------------------------
// SIGNIFICANCE-AWARE LIFT CLASSIFIER
// Replaces the old arbitrary lift thresholds with significance-gated ones.
// ---------------------------------------------------------------------------

/**
 * Classifies lift with chi-square significance gate.
 * Only labels "green" or "red" if the deviation is statistically significant
 * at p < 0.05 AND sample size is adequate.
 */
function classifyLift(lift, sampleCount, observedHits, totalRounds, expectedRate) {
  // Legacy call without stat params — fall back to conservative neutral
  if (observedHits === undefined || totalRounds === undefined || expectedRate === undefined) {
    if (sampleCount < MIN_CELL_COUNT * 2) return { key: 'neutral', label: 'Insufficient Data', tone: 'neutral' };
    if (lift >= 1.15) return { key: 'green', label: 'Green Zone', tone: 'good' };
    if (lift <= 0.85) return { key: 'red', label: 'Red Zone', tone: 'bad' };
    return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
  }

  const { significant } = chiSquareTest(observedHits, totalRounds, expectedRate);
  if (!significant || sampleCount < MIN_CELL_COUNT) {
    return { key: 'neutral', label: 'Watch Zone', tone: 'neutral' };
  }
  if (lift >= 1.1) return { key: 'green', label: 'Green Zone', tone: 'good' };
  if (lift <= 0.9)  return { key: 'red',   label: 'Red Zone',  tone: 'bad'  };
  return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
}

function describeBand(score) {
  if (score >= 68) return { key: 'play',  label: 'PLAY WINDOW',  tone: 'good'    };
  if (score >= 50) return { key: 'wait',  label: 'WAIT / WATCH', tone: 'neutral' };
  return                  { key: 'skip',  label: 'SKIP WINDOW',  tone: 'bad'     };
}

function createTargetMap(initialValue) {
  const out = {};
  for (const target of TARGETS) {
    out[target] = typeof initialValue === 'function' ? initialValue(target) : initialValue;
  }
  return out;
}

// ---------------------------------------------------------------------------
// ZONED DATE HELPERS
// ---------------------------------------------------------------------------

function buildZonedPartsGetter(timeZone) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    weekday: 'short', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  });
  return (timestamp) => {
    const parts = {};
    for (const part of formatter.formatToParts(new Date(timestamp))) {
      if (part.type !== 'literal') parts[part.type] = part.value;
    }
    const weekday = parts.weekday || 'Mon';
    const hour = safeNumber(parts.hour, 0);
    const minute = safeNumber(parts.minute, 0);
    return {
      weekday,
      dayIndex: WEEKDAY_INDEX[weekday] ?? 0,
      hour,
      minute,
      minuteOfDay: (hour * 60) + minute,
      dateKey: `${parts.year || '1970'}-${parts.month || '01'}-${parts.day || '01'}`,
    };
  };
}

// ---------------------------------------------------------------------------
// ROUND NORMALISATION & BASIC SUMMARISATION
// ---------------------------------------------------------------------------

function normalizeRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : [])
    .map((round) => ({
      roundId:    safeNumber(round?.roundId ?? round?.round_id, 0),
      multiplier: safeNumber(round?.multiplier, NaN),
      timestamp:  safeNumber(round?.timestamp, NaN),
    }))
    .filter((round) => Number.isFinite(round.multiplier) && Number.isFinite(round.timestamp) && round.timestamp > 0)
    .sort((a, b) => {
      if (a.timestamp !== b.timestamp) return a.timestamp - b.timestamp;
      return a.roundId - b.roundId;
    });
}

function summarizeRounds(rounds, focusTarget) {
  const values = [];
  const hitCounts = createTargetMap(0);
  const distributionCounts = Object.fromEntries(DISTRIBUTION_BANDS.map((band) => [band.key, 0]));
  let sum = 0, max = 0, min = Infinity, lowCrashCount = 0, hugeHitCount = 0, megaHitCount = 0;

  for (const round of rounds) {
    const m = round.multiplier;
    values.push(m);
    sum += m;
    if (m > max) max = m;
    if (m < min) min = m;
    if (m < 2)    lowCrashCount += 1;
    if (m >= 100) hugeHitCount  += 1;
    if (m >= 500) megaHitCount  += 1;
    for (const target of TARGETS) { if (m >= target) hitCounts[target] += 1; }
    for (const band of DISTRIBUTION_BANDS) {
      if (m >= band.min && m < band.max) { distributionCounts[band.key] += 1; break; }
    }
  }

  values.sort((a, b) => a - b);
  const roundCount = rounds.length;
  const hitRates = createTargetMap((target) => ratio(hitCounts[target], roundCount));
  const distribution = DISTRIBUTION_BANDS.map((band) => ({
    key:   band.key,
    label: band.label,
    count: distributionCounts[band.key],
    pct:   ratio(distributionCounts[band.key], roundCount),
    color: band.color,
  }));

  return {
    roundCount,
    avgMultiplier:    ratio(sum, roundCount),
    medianMultiplier: quantile(values, 0.5),
    p90Multiplier:    quantile(values, 0.9),
    maxMultiplier:    max || 0,
    minMultiplier:    Number.isFinite(min) ? min : 0,
    focusHitCount:    hitCounts[focusTarget] || 0,
    focusHitRate:     hitRates[focusTarget]  || 0,
    lowCrashRate:     ratio(lowCrashCount, roundCount),
    hugeHitRate:      ratio(hugeHitCount,  roundCount),
    megaHitRate:      ratio(megaHitCount,  roundCount),
    hitCounts,
    hitRates,
    distribution,
  };
}

function distributionPct(summary, bandKey) {
  const items = Array.isArray(summary?.distribution) ? summary.distribution : [];
  const band = items.find((item) => item.key === bandKey);
  return safeNumber(band?.pct, 0);
}

// ---------------------------------------------------------------------------
// Z-SCORE NORMALISED SUMMARY DISTANCE
// Replaces hand-tuned magic multipliers with feature-scale-invariant distance.
// ---------------------------------------------------------------------------

/**
 * Build a feature vector from a round summary for distance computation.
 */
function summaryToFeatureVector(summary, focusTarget) {
  return {
    focusHitRate:  summary?.hitRates?.[focusTarget]  || 0,
    lowCrashRate:  summary?.lowCrashRate              || 0,
    hugeHitRate:   summary?.hugeHitRate               || 0,
    megaHitRate:   summary?.megaHitRate               || 0,
    avgMultiplier: summary?.avgMultiplier             || 0,
    maxMultiplier: summary?.maxMultiplier             || 0,
    // distribution spread
    distLt2:      distributionPct(summary, 'lt2'),
    dist2to5:     distributionPct(summary, '2to5'),
    dist5to10:    distributionPct(summary, '5to10'),
    dist10to20:   distributionPct(summary, '10to20'),
    dist20to50:   distributionPct(summary, '20to50'),
    dist50to100:  distributionPct(summary, '50to100'),
  };
}

const SUMMARY_FEATURE_KEYS = [
  'focusHitRate', 'lowCrashRate', 'hugeHitRate', 'megaHitRate',
  'avgMultiplier', 'maxMultiplier',
  'distLt2', 'dist2to5', 'dist5to10', 'dist10to20', 'dist20to50', 'dist50to100',
];

/**
 * Euclidean distance in z-score space.
 * Requires pre-computed normalised vectors (same basis).
 */
function normalizedDistance(vecA, vecB) {
  if (!vecA || !vecB) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (const key of SUMMARY_FEATURE_KEYS) {
    const diff = (vecA[key] || 0) - (vecB[key] || 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Build a normalised vector pool from an array of summary objects.
 * Returns { vectors: zNormVec[], rawVectors: featureVec[] }.
 */
function buildNormalizedVectorPool(summaries, focusTarget) {
  const rawVectors = summaries.map((s) => summaryToFeatureVector(s, focusTarget));
  const zVectors   = zScoreNormalize(rawVectors, SUMMARY_FEATURE_KEYS);
  return { rawVectors, zVectors };
}

// ---------------------------------------------------------------------------
// PROGRESS SPLIT
// ---------------------------------------------------------------------------

function splitWindowRoundsByProgress(window, progressRatio, slotMinutes) {
  const rounds = Array.isArray(window?.rounds) ? window.rounds : [];
  if (!rounds.length) return { elapsedRounds: [], remainingRounds: [], elapsedCount: 0 };
  const safeRatio = clamp(progressRatio, 0, 1);
  if (safeRatio <= 0)     return { elapsedRounds: [], remainingRounds: rounds.slice(), elapsedCount: 0 };
  if (safeRatio >= 0.999) return { elapsedRounds: rounds.slice(), remainingRounds: [], elapsedCount: rounds.length };

  const startTimestamp = safeNumber(window.firstTimestamp, rounds[0].timestamp);
  const cutoffTimestamp = startTimestamp + (slotMinutes * MINUTE_MS * safeRatio);
  let elapsedCount = rounds.findIndex((round) => round.timestamp >= cutoffTimestamp);
  if (elapsedCount < 0) elapsedCount = rounds.length;
  const expectedCount = clamp(Math.round(rounds.length * safeRatio), 0, rounds.length);
  if (elapsedCount === 0 && expectedCount > 0) elapsedCount = expectedCount;

  return {
    elapsedRounds:   rounds.slice(0, elapsedCount),
    remainingRounds: rounds.slice(elapsedCount),
    elapsedCount,
  };
}

// ---------------------------------------------------------------------------
// WINDOW COLLECTION SUMMARY
// ---------------------------------------------------------------------------

function summarizeWindowCollection(windows, focusTarget) {
  if (!windows.length) {
    return {
      windowCount: 0, roundCount: 0, focusHitRate: 0, focusAnyHitRate: 0,
      avgMultiplier: 0, lowCrashRate: 0, hugeHitRate: 0, megaHitRate: 0,
      avgPeakMultiplier: 0, perRoundHitRates: createTargetMap(0),
      windowAnyHitRates: createTargetMap(0),
    };
  }

  const totalHits      = createTargetMap(0);
  const windowsWithHit = createTargetMap(0);
  let totalRounds = 0, weightedAvgMultiplier = 0, weightedLowCrash = 0;
  let weightedHuge = 0, weightedMega = 0, peakSum = 0;

  for (const window of windows) {
    const summary = window.summary;
    totalRounds            += summary.roundCount;
    weightedAvgMultiplier  += summary.avgMultiplier * summary.roundCount;
    weightedLowCrash       += summary.lowCrashRate  * summary.roundCount;
    weightedHuge           += summary.hugeHitRate   * summary.roundCount;
    weightedMega           += summary.megaHitRate   * summary.roundCount;
    peakSum                += summary.maxMultiplier;
    for (const target of TARGETS) {
      totalHits[target]      += summary.hitCounts[target] || 0;
      if ((summary.hitCounts[target] || 0) > 0) windowsWithHit[target] += 1;
    }
  }

  return {
    windowCount:       windows.length,
    roundCount:        totalRounds,
    focusHitRate:      ratio(totalHits[focusTarget], totalRounds),
    focusAnyHitRate:   ratio(windowsWithHit[focusTarget], windows.length),
    avgMultiplier:     ratio(weightedAvgMultiplier, totalRounds),
    lowCrashRate:      ratio(weightedLowCrash, totalRounds),
    hugeHitRate:       ratio(weightedHuge, totalRounds),
    megaHitRate:       ratio(weightedMega, totalRounds),
    avgPeakMultiplier: ratio(peakSum, windows.length),
    perRoundHitRates:  createTargetMap((target) => ratio(totalHits[target], totalRounds)),
    windowAnyHitRates: createTargetMap((target) => ratio(windowsWithHit[target], windows.length)),
  };
}

// ---------------------------------------------------------------------------
// FIXED WINDOW SEGMENTATION
// ---------------------------------------------------------------------------

function segmentRoundsByWindow(rounds, windowMs, focusTarget) {
  const buckets = new Map();
  for (const round of rounds) {
    const bucketStart = Math.floor(round.timestamp / windowMs) * windowMs;
    const key = String(bucketStart);
    let entry = buckets.get(key);
    if (!entry) {
      entry = { startTimestamp: bucketStart, endTimestamp: bucketStart + windowMs, rounds: [] };
      buckets.set(key, entry);
    }
    entry.rounds.push(round);
  }
  return Array.from(buckets.values())
    .sort((a, b) => a.startTimestamp - b.startTimestamp)
    .map((window) => ({
      startTimestamp: window.startTimestamp,
      endTimestamp:   window.endTimestamp,
      rounds:         window.rounds,
      summary:        summarizeRounds(window.rounds, focusTarget),
    }));
}

// ---------------------------------------------------------------------------
// SLOT WINDOW BUILDER
// ---------------------------------------------------------------------------

function buildSlotWindows(rounds, slotMinutes, timeZone, focusTarget) {
  const getParts  = buildZonedPartsGetter(timeZone);
  const slotCount = Math.max(1, Math.floor(1440 / slotMinutes));
  const groups    = new Map();

  for (const round of rounds) {
    const parts      = getParts(round.timestamp);
    const slotIndex  = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, slotCount - 1);
    const slotStartMinute = slotIndex * slotMinutes;
    const key = `${parts.dateKey}|${slotIndex}`;
    let entry = groups.get(key);
    if (!entry) {
      entry = {
        key, dateKey: parts.dateKey, dayIndex: parts.dayIndex,
        slotIndex, slotStartMinute, firstTimestamp: round.timestamp, rounds: [],
      };
      groups.set(key, entry);
    }
    entry.rounds.push(round);
  }

  return Array.from(groups.values())
    .sort((a, b) => {
      if (a.dateKey !== b.dateKey) return a.dateKey.localeCompare(b.dateKey);
      return a.slotIndex - b.slotIndex;
    })
    .map((entry) => ({ ...entry, summary: summarizeRounds(entry.rounds, focusTarget) }));
}

// ---------------------------------------------------------------------------
// SLOT ANALYTICS  – significance-gated, decay-weighted
// ---------------------------------------------------------------------------

/**
 * sampleWeight now only scales confidence display, not the hit rate itself.
 * We gate on chi-square significance instead.
 */
function sampleWeight(sampleCount) {
  return clamp(sampleCount / 12, 0.3, 1);
}

function buildSlotAnalytics(slotWindows, slotMinutes, windowMs, focusTarget, latestTimestamp, timeZone) {
  const mode           = windowMs > DAY_MS ? 'start-time' : 'window';
  const slotCount      = Math.max(1, Math.floor(1440 / slotMinutes));
  const currentParts   = buildZonedPartsGetter(timeZone)(latestTimestamp);
  const currentSlotIndex = clamp(Math.floor(currentParts.minuteOfDay / slotMinutes), 0, slotCount - 1);

  const baselineByTarget  = {};
  const slotMapsByTarget  = {};
  const slotStatsByTarget = {};
  const items = [];
  const minSamples = Math.max(MIN_CELL_COUNT, Math.floor(Math.sqrt(Math.max(1, slotWindows.length)) / 2));

  for (const target of TARGETS) {
    let baselineHitWindows = 0;
    let baselineHits       = 0;
    let baselineRounds     = 0;
    const slotMap = new Map();

    for (const slotWindow of slotWindows) {
      const summary  = slotWindow.summary;
      const hitCount = summary.hitCounts[target] || 0;
      const ageMs    = latestTimestamp - slotWindow.firstTimestamp;
      const dw       = decayWeight(ageMs);          // exponential decay by age
      baselineRounds += summary.roundCount;
      baselineHits   += hitCount;
      if (hitCount > 0) baselineHitWindows += 1;

      let aggregate = slotMap.get(slotWindow.slotIndex);
      if (!aggregate) {
        aggregate = {
          slotIndex: slotWindow.slotIndex,
          startMinute: slotWindow.slotStartMinute,
          sampleCount: 0, totalRounds: 0, totalHits: 0,
          hitWindows: 0, peakSum: 0,
          weightedHitWindows: 0, weightedSamples: 0,
        };
        slotMap.set(slotWindow.slotIndex, aggregate);
      }
      aggregate.sampleCount       += 1;
      aggregate.totalRounds       += summary.roundCount;
      aggregate.totalHits         += hitCount;
      aggregate.peakSum           += summary.maxMultiplier;
      aggregate.weightedSamples   += dw;
      aggregate.weightedHitWindows += hitCount > 0 ? dw : 0;
      if (hitCount > 0) aggregate.hitWindows += 1;
    }

    const baselineAnyHitRate   = ratio(baselineHitWindows, slotWindows.length);
    const baselineRoundHitRate = ratio(baselineHits, baselineRounds);
    baselineByTarget[target]   = { anyHitRate: baselineAnyHitRate, roundHitRate: baselineRoundHitRate };
    slotMapsByTarget[target]   = slotMap;

    const slotStats = Array.from(slotMap.values())
      .map((slot) => {
        // Decay-weighted any-hit rate — more honest for recent trend
        const anyHitChance = slot.weightedSamples > 0
          ? slot.weightedHitWindows / slot.weightedSamples
          : ratio(slot.hitWindows, slot.sampleCount);
        const roundHitRate = ratio(slot.totalHits, slot.totalRounds);
        const lift         = ratio(anyHitChance, baselineAnyHitRate, 1);

        // Significance gate
        const sigTest      = chiSquareTest(slot.hitWindows, slot.sampleCount, baselineAnyHitRate);
        const classification = classifyLift(lift, slot.sampleCount, slot.hitWindows, slot.sampleCount, baselineAnyHitRate);

        return {
          slotIndex:          slot.slotIndex,
          startMinute:        slot.startMinute,
          label:              formatSlotLabel(slot.startMinute, slotMinutes, mode),
          anyHitChance,
          roundHitRate,
          lift,
          liftSignificant:    sigTest.significant,
          liftPValue:         Number(sigTest.pValue.toFixed(4)),
          sampleCount:        slot.sampleCount,
          avgPeakMultiplier:  ratio(slot.peakSum, slot.sampleCount),
          status:             classification.key,
          zoneLabel:          classification.label,
          tone:               classification.tone,
          // Score uses decay-weighted hit rate × significant lift bonus (no fake multipliers)
          score: anyHitChance * (sigTest.significant && lift > 1 ? lift : 1) * sampleWeight(slot.sampleCount),
        };
      })
      .sort((a, b) => a.slotIndex - b.slotIndex);

    slotStatsByTarget[target] = slotStats;

    const currentSlot = slotStats.find((slot) => slot.slotIndex === currentSlotIndex) || {
      slotIndex:         currentSlotIndex,
      startMinute:       currentSlotIndex * slotMinutes,
      label:             formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, mode),
      anyHitChance:      baselineAnyHitRate,
      roundHitRate:      baselineRoundHitRate,
      lift:              1,
      liftSignificant:   false,
      liftPValue:        1,
      sampleCount:       0,
      avgPeakMultiplier: 0,
      status:            'neutral',
      zoneLabel:         'Insufficient Data',
      tone:              'neutral',
      score:             baselineAnyHitRate,
    };

    const futureOptions = slotStats
      .filter((slot) => slot.sampleCount >= minSamples)
      .map((slot) => {
        const dayOffset   = slot.slotIndex > currentSlotIndex ? 0 : 1;
        const deltaSlots  = dayOffset === 0
          ? slot.slotIndex - currentSlotIndex
          : (slot.slotIndex + slotCount) - currentSlotIndex;
        return {
          ...slot,
          dayOffset,
          deltaSlots,
          occurrenceLabel: formatOccurrenceLabel(slot.startMinute, slotMinutes, mode, dayOffset),
        };
      })
      .filter((slot) => slot.deltaSlots > 0)
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return a.deltaSlots - b.deltaSlots;
      });

    const todayOptions = futureOptions
      .filter((slot) => slot.dayOffset === 0)
      .sort((a, b) => {
        const aStrong = (a.liftSignificant && a.lift >= 1.05) ? 1 : 0;
        const bStrong = (b.liftSignificant && b.lift >= 1.05) ? 1 : 0;
        if (bStrong !== aStrong) return bStrong - aStrong;
        if (a.deltaSlots !== b.deltaSlots) return a.deltaSlots - b.deltaSlots;
        return b.score - a.score;
      });

    const worstFuture = [...futureOptions]
      .sort((a, b) => {
        if (a.lift !== b.lift) return a.lift - b.lift;
        return a.deltaSlots - b.deltaSlots;
      })[0] || null;

    items.push({
      target,
      label:            labelForTarget(target),
      baselineAnyHitRate,
      baselineRoundHitRate,
      currentSlot,
      nextTodayWindow:  todayOptions[0]         || null,
      nextWindow:       futureOptions[0]        || null,
      todayWindows:     todayOptions.slice(0, 3),
      backups:          futureOptions.slice(1, 3),
      avoidWindow:      worstFuture,
      topSlots:         [...slotStats]
        .filter((slot) => slot.sampleCount >= minSamples && slot.liftSignificant)
        .sort((a, b) => {
          if (b.score !== a.score) return b.score - a.score;
          return a.slotIndex - b.slotIndex;
        })
        .slice(0, 3),
    });
  }

  return {
    timeZone, slotMinutes, slotMode: mode, currentSlotIndex, minSamples,
    baselineByTarget, slotStatsByTarget, slotMapsByTarget, items,
  };
}

// ---------------------------------------------------------------------------
// PATTERN MATCH REPORT  –  z-score distances, decay weights, no magic nums
// ---------------------------------------------------------------------------

function buildPatternMatchReport(slotWindows, previousSummary, currentSummary, baselineStats, focusTarget, latestTimestamp, timeZone, slotAnalytics) {
  if (!slotWindows.length) {
    return { available: false, examples: [], reason: 'Not enough stored slot history yet to build a past-slot pattern match.' };
  }

  const slotMinutes   = slotAnalytics.slotMinutes;
  const slotCount     = Math.max(1, Math.floor(1440 / slotMinutes));
  const slotMode      = slotAnalytics.slotMode;
  const getParts      = buildZonedPartsGetter(timeZone);
  const currentParts  = getParts(latestTimestamp);
  const currentSlotIndex  = slotAnalytics.currentSlotIndex;
  const currentKey    = `${currentParts.dateKey}|${currentSlotIndex}`;
  const lookbackMs    = 30 * DAY_MS;
  const lookbackStart = latestTimestamp - lookbackMs;
  const historySpanDays  = ratio(latestTimestamp - (slotWindows[0]?.firstTimestamp || latestTimestamp), DAY_MS);
  const lookbackDaysUsed = Math.min(30, Math.max(0, historySpanDays));

  const orderedWindows = [...slotWindows].sort((a, b) => {
    if (a.dateKey !== b.dateKey) return a.dateKey.localeCompare(b.dateKey);
    return a.slotIndex - b.slotIndex;
  });
  const indexByKey      = new Map(orderedWindows.map((window, index) => [window.key, index]));
  const currentPosition = indexByKey.get(currentKey);
  const previousWindow  = Number.isInteger(currentPosition) && currentPosition > 0 ? orderedWindows[currentPosition - 1] : null;
  const expectedPreviousSlotIndex = (currentSlotIndex - 1 + slotCount) % slotCount;
  const inputSlotLabel   = formatSlotLabel(expectedPreviousSlotIndex * slotMinutes, slotMinutes, slotMode);
  const currentSlotLabel = formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode);

  if (!Number.isInteger(currentPosition) || !previousWindow || previousWindow.slotIndex !== expectedPreviousSlotIndex) {
    return { available: false, examples: [], reason: 'The last closed slot is not available yet, so the current-slot prediction cannot be built.' };
  }

  // Build normalised vector pool for all candidate "previous" slots
  const poolCandidates = orderedWindows.filter((window) => (
    window.slotIndex === previousWindow.slotIndex
    && window.key !== previousWindow.key
    && window.firstTimestamp < latestTimestamp
    && window.firstTimestamp >= lookbackStart
  ));

  if (poolCandidates.length < 6) {
    return { available: false, examples: [], reason: 'Not enough matching past closed slots are stored yet to predict the current live window.' };
  }

  // Pre-build normalised vectors for candidate previous-slot summaries + reference
  const allSummaries  = [...poolCandidates.map((w) => w.summary), previousSummary];
  const { zVectors }  = buildNormalizedVectorPool(allSummaries, focusTarget);
  const referenceVec  = zVectors[zVectors.length - 1]; // last = previousSummary

  const currentMinuteProgress = clamp(
    (currentParts.minuteOfDay - currentSlotIndex * slotMinutes + 1) / Math.max(1, slotMinutes), 0, 1,
  );
  const liveEvidenceWeight = currentSummary.roundCount > 0
    ? clamp((currentMinuteProgress * 0.85) + clamp(currentSummary.roundCount / 35, 0, 0.25), 0.12, 0.68)
    : 0;
  const alreadyHitCurrentWindow = (currentSummary.hitCounts?.[focusTarget] || 0) > 0;

  // Build live-slot normalised vectors if we have live evidence
  let liveZVectors = null;
  if (liveEvidenceWeight > 0) {
    const liveSummaries = [...poolCandidates.map((_, i) => {
      // will fill in per-candidate below — placeholder
      return {};
    }), currentSummary];
    // We'll compute live distance per candidate inline; pre-build a single pool ref
    const livePoolSummaries = poolCandidates.map((candidate) => {
      const position = indexByKey.get(candidate.key);
      const matchedCurrentWindow = Number.isInteger(position) ? orderedWindows[position + 1] : null;
      if (!matchedCurrentWindow) return null;
      const split = splitWindowRoundsByProgress(matchedCurrentWindow, currentMinuteProgress, slotMinutes);
      return summarizeRounds(split.elapsedRounds, focusTarget);
    });
    const validLive = [...livePoolSummaries.filter(Boolean), currentSummary];
    const { zVectors: lz } = buildNormalizedVectorPool(validLive, focusTarget);
    liveZVectors = { summaries: livePoolSummaries, zVectors: lz, currentVec: lz[lz.length - 1] };
  }

  const sameSetupCandidates = poolCandidates
    .map((window, i) => {
      const position = indexByKey.get(window.key);
      const matchedCurrentWindow = Number.isInteger(position) ? orderedWindows[position + 1] : null;
      if (!matchedCurrentWindow) return null;
      const expectedCurrentSlotIndex = (window.slotIndex + 1) % slotCount;
      if (matchedCurrentWindow.slotIndex !== expectedCurrentSlotIndex) return null;

      const weekdayMatch      = matchedCurrentWindow.dayIndex === currentParts.dayIndex;
      const previousDistance  = normalizedDistance(zVectors[i], referenceVec);
      const split             = splitWindowRoundsByProgress(matchedCurrentWindow, currentMinuteProgress, slotMinutes);
      const matchedElapsedSummary   = summarizeRounds(split.elapsedRounds, focusTarget);
      const matchedRemainingSummary = summarizeRounds(split.remainingRounds, focusTarget);

      let liveDistance = 0;
      if (liveEvidenceWeight > 0 && liveZVectors && liveZVectors.summaries[i]) {
        const liveVecIdx = i; // same index
        const liveVec    = liveZVectors.zVectors[liveVecIdx];
        liveDistance     = liveVec ? normalizedDistance(liveVec, liveZVectors.currentVec) : 0;
      }

      const distance = (previousDistance + liveDistance * liveEvidenceWeight) / Math.max(1, 1 + liveEvidenceWeight);

      return {
        weekdayMatch, previousDistance, liveDistance,
        distance: distance + (weekdayMatch ? 0 : 0.18),
        matchedCurrentWindow, matchedElapsedSummary, matchedRemainingSummary,
        elapsedRounds: split.elapsedRounds, remainingRounds: split.remainingRounds,
        window,
      };
    })
    .filter(Boolean);

  if (sameSetupCandidates.length < 6) {
    return { available: false, examples: [], reason: 'Not enough matching past closed slots are stored yet to predict the current live window.' };
  }

  const sameWeekdayPool = sameSetupCandidates.filter((item) => item.weekdayMatch);
  const pool            = sameWeekdayPool.length >= 6 ? sameWeekdayPool : sameSetupCandidates;
  const matchMode       = sameWeekdayPool.length >= 6 ? 'same-weekday' : 'same-time';
  const sampleSize      = clamp(Math.round(pool.length * 0.45), 6, 18);
  const matches         = [...pool].sort((a, b) => a.distance - b.distance).slice(0, sampleSize);

  if (!matches.length) {
    return { available: false, examples: [], reason: 'No usable past closed-slot pattern was found for the current live window.' };
  }

  // Build match rows with decay-based recency weights
  const matchRows = matches.map((match) => {
    const ageMs         = latestTimestamp - match.window.firstTimestamp;
    const dw            = decayWeight(ageMs);  // exponential decay, half-life 14d
    // Distance-based closeness weight (normalised Euclidean, not magic multipliers)
    const closenessW    = 1 / (1 + match.distance);
    const liveW         = liveEvidenceWeight > 0 ? 1 / (1 + match.liveDistance) : 1;
    const weekdayW      = match.weekdayMatch ? 1.15 : 0.95;
    const weight        = closenessW * liveW * dw * weekdayW;
    const remainingRounds = Array.isArray(match.remainingRounds) ? match.remainingRounds : [];
    const firstRemainingHitIndex = remainingRounds.findIndex((round) => round.multiplier >= focusTarget);
    return { ...match, ageMs, weight, firstRemainingHitIndex };
  });

  // Aggregate stats
  let inputHistoryHits = 0, inputHistoryRounds = 0, inputHistoryAnyHit = 0;
  let remainingHistoryHits = 0, remainingHistoryRounds = 0, remainingHistoryAnyHit = 0;
  for (const item of sameSetupCandidates) {
    const is = item.window.summary;
    inputHistoryHits    += is.hitCounts[focusTarget] || 0;
    inputHistoryRounds  += is.roundCount;
    if ((is.hitCounts[focusTarget] || 0) > 0) inputHistoryAnyHit += 1;
    remainingHistoryHits   += item.matchedRemainingSummary.hitCounts[focusTarget] || 0;
    remainingHistoryRounds += item.matchedRemainingSummary.roundCount;
    if ((item.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0) remainingHistoryAnyHit += 1;
  }

  let currentWindowHits = 0, currentWindowRounds = 0, currentWindowAnyHit = 0, currentPeakSum = 0;
  let remainingWindowHits = 0, remainingWindowRounds = 0, remainingWindowAnyHit = 0;
  let sameWeekdayMatchCount = 0;
  for (const match of matchRows) {
    const s = match.matchedCurrentWindow.summary;
    currentWindowHits   += s.hitCounts[focusTarget] || 0;
    currentWindowRounds += s.roundCount;
    currentPeakSum      += s.maxMultiplier || 0;
    if ((s.hitCounts[focusTarget] || 0) > 0) currentWindowAnyHit += 1;
    const rs = match.matchedRemainingSummary;
    remainingWindowHits   += rs.hitCounts[focusTarget] || 0;
    remainingWindowRounds += rs.roundCount;
    if ((rs.hitCounts[focusTarget] || 0) > 0) remainingWindowAnyHit += 1;
    if (match.weekdayMatch) sameWeekdayMatchCount += 1;
  }

  const examples = matchRows.slice(0, 6).map((match, index) => {
    const inputRounds          = summarizeRoundRange(match.window.rounds, focusTarget);
    const matchedCurrentRounds = summarizeRoundRange(match.matchedCurrentWindow.rounds, focusTarget);
    const remainingRounds      = summarizeRoundRange(match.remainingRounds, focusTarget);
    return {
      rank:                       index + 1,
      weekdayMatch:               match.weekdayMatch,
      distance:                   Number(match.distance.toFixed(3)),
      weight:                     Number(match.weight.toFixed(3)),
      inputWindowLabel:           formatTimestampInTimeZone(match.window.firstTimestamp, timeZone),
      inputSlotLabel:             formatSlotLabel(match.window.slotStartMinute, slotMinutes, slotMode),
      inputRoundFrom:             inputRounds.fromRoundId,
      inputRoundTo:               inputRounds.toRoundId,
      inputRoundCount:            inputRounds.roundCount,
      inputHitRate:               match.window.summary.hitRates[focusTarget] || 0,
      matchedCurrentWindowLabel:  formatTimestampInTimeZone(match.matchedCurrentWindow.firstTimestamp, timeZone),
      matchedCurrentSlotLabel:    formatSlotLabel(match.matchedCurrentWindow.slotStartMinute, slotMinutes, slotMode),
      matchedCurrentRoundFrom:    matchedCurrentRounds.fromRoundId,
      matchedCurrentRoundTo:      matchedCurrentRounds.toRoundId,
      matchedCurrentRoundCount:   matchedCurrentRounds.roundCount,
      matchedCurrentHitCount:     matchedCurrentRounds.hitCount,
      matchedCurrentHitRoundIds:  matchedCurrentRounds.hitRoundIds,
      remainingRoundFrom:         remainingRounds.fromRoundId,
      remainingRoundTo:           remainingRounds.toRoundId,
      remainingRoundCount:        remainingRounds.roundCount,
      remainingHitCount:          remainingRounds.hitCount,
      remainingHitRoundIds:       remainingRounds.hitRoundIds,
      firstRemainingHitOffset:    match.firstRemainingHitIndex >= 0 ? match.firstRemainingHitIndex + 1 : null,
      matchedCurrentAnyHit:       (match.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0) > 0,
      remainingAnyHit:            (match.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0,
      matchedCurrentPeakMultiplier: match.matchedCurrentWindow.summary.maxMultiplier || 0,
    };
  });

  // Weighted rate estimates
  const inputHistoryAnyHitRate    = ratio(inputHistoryAnyHit, sameSetupCandidates.length);
  const inputHistoryPerRoundRate  = ratio(inputHistoryHits, inputHistoryRounds);
  const inputBaselineAnyHitRate   = baselineStats.windowAnyHitRates[focusTarget];
  const inputHistoryLift          = ratio(inputHistoryAnyHitRate, inputBaselineAnyHitRate, inputHistoryAnyHitRate > 0 ? 1.25 : 1);

  const currentWindowAnyHitRate = weightedAverage(
    matchRows,
    (match) => ((match.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0) > 0 ? 1 : 0),
    (match) => match.weight,
    ratio(currentWindowAnyHit, matches.length),
  );
  const currentWindowPerRoundHitRate = weightedAverage(
    matchRows,
    (match) => ratio(match.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0, match.matchedCurrentWindow.summary.roundCount),
    (match) => match.weight,
    ratio(currentWindowHits, currentWindowRounds),
  );
  const currentWindowLift        = ratio(currentWindowAnyHitRate, inputBaselineAnyHitRate, currentWindowAnyHitRate > 0 ? 1.25 : 1);

  const remainingBaselineAnyHitRate = ratio(remainingHistoryAnyHit, sameSetupCandidates.length);
  const remainingBaselinePerRoundRate = ratio(remainingHistoryHits, remainingHistoryRounds);
  const remainingAnyHitRate     = weightedAverage(
    matchRows,
    (match) => ((match.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0 ? 1 : 0),
    (match) => match.weight,
    ratio(remainingWindowAnyHit, matches.length),
  );
  const remainingPerRoundHitRate = weightedAverage(
    matchRows,
    (match) => ratio(match.matchedRemainingSummary.hitCounts[focusTarget] || 0, match.matchedRemainingSummary.roundCount),
    (match) => match.weight,
    ratio(remainingWindowHits, remainingWindowRounds),
  );
  const remainingLift = ratio(remainingAnyHitRate, remainingBaselineAnyHitRate, remainingAnyHitRate > 0 ? 1.25 : 1);

  // Chi-square significance on remaining rate vs baseline
  const remainingSigTest = chiSquareTest(remainingWindowAnyHit, matches.length, remainingBaselineAnyHitRate);
  const currentWindowSigTest = chiSquareTest(currentWindowAnyHit, matches.length, inputBaselineAnyHitRate);

  const weightedAvgPeakMultiplier = weightedAverage(
    matchRows,
    (match) => match.matchedCurrentWindow.summary.maxMultiplier || 0,
    (match) => match.weight,
    ratio(currentPeakSum, matches.length),
  );
  const currentWindowTone = classifyLift(currentWindowLift, matches.length, currentWindowAnyHit, matches.length, inputBaselineAnyHitRate).tone;
  const remainingWindowTone = classifyLift(remainingLift, matches.length, remainingWindowAnyHit, matches.length, remainingBaselineAnyHitRate).tone;

  const remainingHitMatches = matchRows.filter((match) => match.firstRemainingHitIndex >= 0);
  const expectedRoundRange = remainingHitMatches.length >= 2
    ? {
        hitMatchCount: remainingHitMatches.length,
        firstRemainingHitOffsetFrom: Math.max(1, Math.round(weightedQuantile(remainingHitMatches, (m) => m.firstRemainingHitIndex + 1, (m) => m.weight, 0.2, 1))),
        firstRemainingHitOffsetTo:   Math.max(1, Math.round(weightedQuantile(remainingHitMatches, (m) => m.firstRemainingHitIndex + 1, (m) => m.weight, 0.8, 1))),
      }
    : null;

  return {
    available: true,
    matchMode,
    lookbackDaysUsed,
    inputSlotLabel,
    currentSlotLabel,
    currentSlotIndex,
    currentWindowLabel: currentSlotLabel,
    candidateCount: sameSetupCandidates.length,
    usedMatches: matches.length,
    sameWeekdayMatches: sameWeekdayMatchCount,
    note: `Built from ${matches.length} matched ${matchMode === 'same-weekday' ? 'same-weekday' : 'same-time'} closed ${inputSlotLabel} setups from the last ${lookbackDaysUsed.toFixed(1)} days to judge the live ${currentSlotLabel} window.`,
    examples,
    averageMatchWeight: Number(weightedAverage(matchRows, (m) => m.weight, () => 1, 0).toFixed(3)),
    averageDistance:    Number(weightedAverage(matchRows, (m) => m.distance, (m) => m.weight, 0).toFixed(3)),
    expectedRoundRange,
    progress: {
      ratio:       currentMinuteProgress,
      roundsSeen:  currentSummary.roundCount || 0,
      alreadyHit:  alreadyHitCurrentWindow,
    },
    significance: {
      currentWindowSignificant:  currentWindowSigTest.significant,
      currentWindowPValue:       Number(currentWindowSigTest.pValue.toFixed(4)),
      remainingSignificant:      remainingSigTest.significant,
      remainingPValue:           Number(remainingSigTest.pValue.toFixed(4)),
    },
    inputHistory: {
      anyHitRate:        inputHistoryAnyHitRate,
      perRoundHitRate:   inputHistoryPerRoundRate,
      lift:              inputHistoryLift,
      sampleCount:       sameSetupCandidates.length,
      weekdaySampleCount: sameWeekdayPool.length,
    },
    currentWindow: {
      occurrenceLabel:   currentSlotLabel,
      anyHitRate:        currentWindowAnyHitRate,
      perRoundHitRate:   currentWindowPerRoundHitRate,
      avgPeakMultiplier: weightedAvgPeakMultiplier,
      lift:              currentWindowLift,
      liftSignificant:   currentWindowSigTest.significant,
      tone:              currentWindowTone,
      label: currentWindowLift >= 1.08 ? 'Stronger Than Normal' : currentWindowLift <= 0.94 ? 'Weaker Than Normal' : 'Near Normal',
    },
    remainingWindow: {
      occurrenceLabel:        currentSlotLabel,
      anyHitRate:             remainingAnyHitRate,
      perRoundHitRate:        remainingPerRoundHitRate,
      baselineAnyHitRate:     remainingBaselineAnyHitRate,
      baselinePerRoundHitRate: remainingBaselinePerRoundRate,
      lift:                   remainingLift,
      liftSignificant:        remainingSigTest.significant,
      tone:                   remainingWindowTone,
      label: remainingLift >= 1.08 ? 'Stronger Than Normal' : remainingLift <= 0.94 ? 'Weaker Than Normal' : 'Near Normal',
    },
  };
}

// ---------------------------------------------------------------------------
// PATTERN PREDICTION  –  significance-gated decisions
// ---------------------------------------------------------------------------

function buildPatternPrediction({ focusTarget, windowLabel, latestRoundId, currentSummary, baselineStats, slotAnalytics, patternMatch }) {
  const slotItem                  = slotAnalytics.items.find((item) => item.target === focusTarget) || null;
  const currentSlot               = slotItem?.currentSlot || null;
  const currentHitRate            = currentSummary.hitRates[focusTarget] || 0;
  const baselineHitRate           = baselineStats.perRoundHitRates[focusTarget] || 0;
  const baselineCurrentWindowHitRate = baselineStats.windowAnyHitRates[focusTarget] || 0;
  const currentLift               = ratio(currentHitRate, baselineHitRate, baselineHitRate > 0 ? 1 : 0);
  const slotLift                  = safeNumber(currentSlot?.lift, 1);
  const slotSignificant           = Boolean(currentSlot?.liftSignificant);
  const isLowTarget               = focusTarget <= 20;
  const isMidTarget               = focusTarget > 20 && focusTarget <= 100;
  const expectedCurrentHits       = baselineHitRate * safeNumber(currentSummary.roundCount, 0);
  const currentEvidenceWeight     = clamp(expectedCurrentHits / (isLowTarget ? 10 : isMidTarget ? 4 : 2.5), 0.12, 1);
  const effectiveCurrentLift      = 1 + ((currentLift - 1) * currentEvidenceWeight);

  if (!patternMatch?.available) {
    return {
      action: 'WAIT',
      tone: 'neutral',
      confidence: 0,
      confidenceLabel: 'Low',
      dataQuality: 'insufficient',
      predictsLabel: `Current ${windowLabel}`,
      inputLabel: `Closed Previous ${windowLabel}`,
      inputSlotLabel: '-',
      currentSlotLabel: currentSlot?.label || '-',
      currentHitRate, baselineHitRate, baselineCurrentWindowHitRate,
      currentLift, effectiveCurrentLift, currentEvidenceWeight,
      currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
      currentWindowHitRate: 0, currentWindowLift: 1,
      remainingHitRate: 0, remainingLift: 1, baselineRemainingHitRate: 0,
      matchedWindows: 0, sameWeekdayMatches: 0, lookbackDaysUsed: 0,
      alreadyHitInCurrentWindow: (currentSummary.hitCounts?.[focusTarget] || 0) > 0,
      hitsSoFar: currentSummary.hitCounts?.[focusTarget] || 0,
      expectedRoundIdFrom: null, expectedRoundIdTo: null,
      expectedRoundIdLabel: '-', expectedRoundIdBasis: '',
      summary: `Not enough matching closed ${windowLabel.toLowerCase()} history yet to judge the current live ${windowLabel.toLowerCase()} for ${labelForTarget(focusTarget)}.`,
      reasons: [
        `Live ${windowLabel.toLowerCase()} hit rate so far: ${pctString(currentHitRate)} versus ${pctString(baselineHitRate)} per-round baseline.`,
        `The last closed ${windowLabel.toLowerCase()} does not have enough historical matches yet.`,
      ],
    };
  }

  const currentWindowHitRate      = safeNumber(patternMatch.currentWindow?.anyHitRate, 0);
  const currentWindowLift         = safeNumber(patternMatch.currentWindow?.lift, 1);
  const currentWindowSignificant  = Boolean(patternMatch.currentWindow?.liftSignificant);
  const remainingHitRate          = safeNumber(patternMatch.remainingWindow?.anyHitRate, 0);
  const baselineRemainingHitRate  = safeNumber(patternMatch.remainingWindow?.baselineAnyHitRate, 0);
  const remainingLift             = safeNumber(patternMatch.remainingWindow?.lift, 1);
  const remainingSignificant      = Boolean(patternMatch.remainingWindow?.liftSignificant);
  const currentWindowEdge         = currentWindowHitRate - baselineCurrentWindowHitRate;
  const matchedWindows            = safeNumber(patternMatch.usedMatches, 0);
  const sameWeekdayMatches        = safeNumber(patternMatch.sameWeekdayMatches, 0);
  const lookbackDaysUsed          = safeNumber(patternMatch.lookbackDaysUsed, 0);
  const safeLatestRoundId         = safeNumber(latestRoundId, 0);
  const alreadyHitInCurrentWindow = Boolean(patternMatch.progress?.alreadyHit || (currentSummary.hitCounts?.[focusTarget] || 0) > 0);
  const hitsSoFar                 = currentSummary.hitCounts?.[focusTarget] || 0;

  // Confidence derived from sample count, significance, and weekday match proportion.
  // No fake precision: we bucket into Low / Medium / High.
  const rawConfidence = clamp(
    (Math.min(matchedWindows, 18) / 18) * 0.50
    + (remainingSignificant ? 0.25 : 0)
    + (currentWindowSignificant ? 0.15 : 0)
    + (Math.min(sameWeekdayMatches, 8) / 8) * 0.10,
    0, 1,
  );
  const confidenceLabel = rawConfidence >= 0.70 ? 'High' : rawConfidence >= 0.45 ? 'Medium' : 'Low';

  // Data quality flag
  const dataQuality = rawConfidence >= 0.70 ? 'good' : rawConfidence >= 0.45 ? 'moderate' : 'limited';

  // Decision logic — significance-gated
  const remainingEdge = remainingHitRate - baselineRemainingHitRate;
  let action = 'WATCH NOW';
  let tone   = 'neutral';
  let summary = `Closed ${patternMatch.inputSlotLabel} matches say the live ${patternMatch.currentSlotLabel} window is near normal right now for ${labelForTarget(focusTarget)}.`;

  if (
    !alreadyHitInCurrentWindow
    && remainingSignificant
    && remainingLift >= 1.1
    && remainingEdge >= 0
    && (currentWindowSignificant ? currentWindowLift >= 1 : true)
    && (slotSignificant ? slotLift >= 1 : true)
  ) {
    action  = 'PLAY NOW';
    tone    = 'good';
    summary = `Significantly stronger remaining-window pattern (p<0.05, lift ${remainingLift.toFixed(2)}x) supports ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window.`;
  } else if (
    !alreadyHitInCurrentWindow
    && (
      (remainingLift >= 1.05 && matchedWindows >= 8)
      || (currentWindowSignificant && currentWindowLift >= 1.08)
      || currentWindowEdge >= 0.03
      || effectiveCurrentLift >= 1.02
    )
  ) {
    action  = 'WATCH NOW';
    tone    = 'good';
    summary = `Pattern leans positive for ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window but significance is not yet confirmed — watch, do not commit fully.`;
  } else if (
    alreadyHitInCurrentWindow
    && remainingSignificant
    && remainingLift <= 0.9
  ) {
    action  = 'SKIP NOW';
    tone    = 'bad';
    summary = `A ${labelForTarget(focusTarget)} hit has already landed and the significantly weaker remaining pattern (p<0.05) says another is unlikely soon.`;
  } else if (
    remainingSignificant
    && remainingLift <= 0.9
    && currentWindowSignificant
    && currentWindowLift <= 0.9
  ) {
    action  = 'SKIP NOW';
    tone    = 'bad';
    summary = `Both current-window and remaining-window matched patterns are significantly below normal — skip ${labelForTarget(focusTarget)} for now.`;
  } else if (remainingLift >= 1 || currentWindowLift >= 1 || effectiveCurrentLift >= 1 || slotLift >= 1) {
    action  = 'WATCH NOW';
    tone    = 'neutral';
    summary = `Mixed signals in the live ${patternMatch.currentSlotLabel} window — no significant edge confirmed yet for ${labelForTarget(focusTarget)}.`;
  } else {
    action  = 'WAIT';
    tone    = 'neutral';
    summary = `Pattern and live read are both below baseline without significance — better to wait for a clearer window for ${labelForTarget(focusTarget)}.`;
  }

  const expectedRoundIdFrom = patternMatch.expectedRoundRange && safeLatestRoundId > 0
    ? safeLatestRoundId + patternMatch.expectedRoundRange.firstRemainingHitOffsetFrom
    : null;
  const expectedRoundIdTo = patternMatch.expectedRoundRange && safeLatestRoundId > 0
    ? safeLatestRoundId + patternMatch.expectedRoundRange.firstRemainingHitOffsetTo
    : null;
  const expectedRoundIdLabel  = (expectedRoundIdFrom && expectedRoundIdTo) ? `#${expectedRoundIdFrom} - #${expectedRoundIdTo}` : '-';
  const expectedRoundIdBasis  = patternMatch.expectedRoundRange
    ? `Based on ${patternMatch.expectedRoundRange.hitMatchCount} matched current-window hits still ahead from this point.`
    : '';

  return {
    action, tone,
    confidence:      rawConfidence,
    confidenceLabel,
    dataQuality,
    predictsLabel:   `Current ${windowLabel}`,
    inputLabel:      `Closed Previous ${windowLabel}`,
    inputSlotLabel:  patternMatch.inputSlotLabel,
    currentSlotLabel: patternMatch.currentSlotLabel,
    currentHitRate, baselineHitRate, baselineCurrentWindowHitRate,
    currentLift, effectiveCurrentLift, currentEvidenceWeight,
    currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
    currentWindowHitRate, currentWindowLift,
    remainingHitRate, remainingLift, baselineRemainingHitRate,
    matchedWindows, sameWeekdayMatches, lookbackDaysUsed,
    alreadyHitInCurrentWindow, hitsSoFar,
    expectedRoundIdFrom, expectedRoundIdTo, expectedRoundIdLabel, expectedRoundIdBasis,
    summary,
    reasons: [
      `Input: the last closed ${windowLabel.toLowerCase()} is ${patternMatch.inputSlotLabel}, live window is ${patternMatch.currentSlotLabel}.`,
      `Matched ${matchedWindows} past setups over ${lookbackDaysUsed.toFixed(1)} days using z-score normalised distance with exponential decay weighting.`,
      `Full live-window history hit ${labelForTarget(focusTarget)} ${pctString(currentWindowHitRate)} vs ${pctString(baselineCurrentWindowHitRate)} baseline${currentWindowSignificant ? ' (significant p<0.05)' : ' (not significant)'}.`,
      `Remaining window matched history: ${pctString(remainingHitRate)} vs ${pctString(baselineRemainingHitRate)} baseline${remainingSignificant ? ' (significant p<0.05)' : ' (not significant)'}.`,
      alreadyHitInCurrentWindow
        ? `${labelForTarget(focusTarget)} has already hit ${hitsSoFar} time(s) in the current live window.`
        : `${labelForTarget(focusTarget)} has not hit yet in the current live window.`,
      expectedRoundIdBasis ? `Expected ${labelForTarget(focusTarget)} around rounds ${expectedRoundIdLabel}. ${expectedRoundIdBasis}` : null,
    ].filter(Boolean),
  };
}

// ---------------------------------------------------------------------------
// HOURLY HISTORY
// ---------------------------------------------------------------------------

function buildHourlyHistory(rounds, timeZone, baselinePerRoundRates) {
  const getParts = buildZonedPartsGetter(timeZone);
  const buckets  = Array.from({ length: 24 }, (_, hour) => ({
    hour, roundCount: 0, sumMultiplier: 0, hits: createTargetMap(0),
  }));

  for (const round of rounds) {
    const hour   = getParts(round.timestamp).hour;
    const bucket = buckets[hour];
    bucket.roundCount    += 1;
    bucket.sumMultiplier += round.multiplier;
    for (const target of TARGETS) { if (round.multiplier >= target) bucket.hits[target] += 1; }
  }

  const rows = buckets.map((bucket) => {
    const targetRates = createTargetMap((target) => ratio(bucket.hits[target], bucket.roundCount));
    let bestTarget = '-', bestLift = -Infinity;
    for (const target of TARGETS) {
      const lift = ratio(targetRates[target], baselinePerRoundRates[target], 1);
      if (lift > bestLift) { bestLift = lift; bestTarget = labelForTarget(target); }
    }
    return {
      hour:          bucket.hour,
      label:         formatHourLabel(bucket.hour),
      roundCount:    bucket.roundCount,
      avgMultiplier: ratio(bucket.sumMultiplier, bucket.roundCount),
      targetRates,
      bestTarget,
    };
  });

  const bestHours = TARGETS.map((target) => {
    const sorted = [...rows].filter((r) => r.roundCount > 0).sort((a, b) => b.targetRates[target] - a.targetRates[target]);
    const best = sorted[0], worst = sorted[sorted.length - 1];
    return {
      target, label: labelForTarget(target),
      bestHour:    best?.hour  ?? null, bestLabel:  best?.label  || '-', bestHitRate:  best?.targetRates?.[target]  || 0,
      worstHour:   worst?.hour ?? null, worstLabel: worst?.label || '-', worstHitRate: worst?.targetRates?.[target] || 0,
    };
  });

  return { timeZone, rows, bestHours };
}

// ---------------------------------------------------------------------------
// HEATMAP
// ---------------------------------------------------------------------------

function buildHeatmap(rounds, timeZone, focusTarget, baselinePerRoundRates) {
  const getParts = buildZonedPartsGetter(timeZone);
  const cells = Array.from({ length: 7 }, (_, dayIndex) => (
    Array.from({ length: 24 }, (_, hour) => ({
      dayIndex, dayLabel: WEEKDAYS[dayIndex], hour, hourLabel: formatHourLabel(hour),
      roundCount: 0, hitCount: 0, hitRate: 0, lift: 1, tone: 'neutral',
    }))
  ));

  for (const round of rounds) {
    const parts = getParts(round.timestamp);
    const cell  = cells[parts.dayIndex][parts.hour];
    cell.roundCount += 1;
    if (round.multiplier >= focusTarget) cell.hitCount += 1;
  }

  const flat = [];
  const baselineRate = baselinePerRoundRates[focusTarget] || 0;
  for (const row of cells) {
    for (const cell of row) {
      cell.hitRate = ratio(cell.hitCount, cell.roundCount);
      cell.lift    = ratio(cell.hitRate, baselineRate, 1);
      // Significance-gated tone for heatmap
      const sig  = chiSquareTest(cell.hitCount, cell.roundCount, baselineRate);
      cell.tone  = sig.significant
        ? (cell.lift >= 1.1 ? 'good' : cell.lift <= 0.9 ? 'bad' : 'neutral')
        : 'neutral';
      flat.push(cell);
    }
  }

  const ranked   = flat.filter((cell) => cell.roundCount >= MIN_CELL_COUNT);
  const strongest = [...ranked].sort((a, b) => b.lift !== a.lift ? b.lift - a.lift : b.hitRate - a.hitRate).slice(0, 3)
    .map((c) => ({ label: `${c.dayLabel} ${c.hourLabel}`, hitRate: c.hitRate, lift: c.lift, roundCount: c.roundCount }));
  const weakest   = [...ranked].sort((a, b) => a.lift !== b.lift ? a.lift - b.lift : a.hitRate - b.hitRate).slice(0, 3)
    .map((c) => ({ label: `${c.dayLabel} ${c.hourLabel}`, hitRate: c.hitRate, lift: c.lift, roundCount: c.roundCount }));

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    days: WEEKDAYS,
    hours: Array.from({ length: 24 }, (_, hour) => ({ value: hour, label: formatHourLabel(hour) })),
    cells, strongest, weakest,
  };
}

// ---------------------------------------------------------------------------
// LAST HIT MAP
// ---------------------------------------------------------------------------

function buildLastHitMap(rounds) {
  const lastHits = createTargetMap(() => ({ roundId: null, timestamp: null, multiplier: null }));
  for (let index = rounds.length - 1; index >= 0; index -= 1) {
    const round = rounds[index];
    for (const target of TARGETS) {
      if (lastHits[target].roundId == null && round.multiplier >= target) {
        lastHits[target] = { roundId: round.roundId, timestamp: round.timestamp, multiplier: round.multiplier };
      }
    }
  }
  return lastHits;
}

// ---------------------------------------------------------------------------
// COOLDOWN REPORT  –  significance-gated
// ---------------------------------------------------------------------------

function buildCooldownReport(rounds, baselinePerRoundRates, latestTimestamp, lastHits) {
  const prefixes = {};
  for (const target of TARGETS) {
    const prefix = new Array(rounds.length + 1).fill(0);
    for (let i = 0; i < rounds.length; i += 1) {
      prefix[i + 1] = prefix[i] + (rounds[i].multiplier >= target ? 1 : 0);
    }
    prefixes[target] = prefix;
  }

  const items = [];
  for (const target of TARGETS) {
    const eventIndices = [];
    for (let index = 0; index < rounds.length; index += 1) {
      if (rounds[index].multiplier >= target) eventIndices.push(index);
    }

    const horizons = COOLDOWN_WINDOWS.map((window) => {
      if (!eventIndices.length) {
        return {
          key: window.key, label: window.label, anyHitRate: 0, perRoundHitRate: 0,
          baselinePerRoundRate: baselinePerRoundRates[target] || 0,
          lift: 1, sampleCount: 0, status: 'neutral', significant: false, pValue: 1,
        };
      }

      let totalHits = 0, totalRounds = 0, totalAnyHit = 0, right = 0;
      for (const index of eventIndices) {
        if (right < index + 1) right = index + 1;
        while (right < rounds.length && rounds[right].timestamp <= (rounds[index].timestamp + window.ms)) right += 1;
        const roundsInRange = Math.max(0, right - (index + 1));
        const hitsInRange   = prefixes[target][right] - prefixes[target][index + 1];
        totalRounds += roundsInRange;
        totalHits   += hitsInRange;
        if (hitsInRange > 0) totalAnyHit += 1;
      }

      const anyHitRate        = ratio(totalAnyHit, eventIndices.length);
      const perRoundHitRate   = ratio(totalHits, totalRounds);
      const baselineRate      = baselinePerRoundRates[target] || 0;
      const lift              = ratio(perRoundHitRate, baselineRate, 1);
      const sigTest           = chiSquareTest(totalHits, totalRounds, baselineRate);
      const classification    = classifyLift(lift, eventIndices.length, totalHits, totalRounds, baselineRate);

      return {
        key: window.key, label: window.label, anyHitRate, perRoundHitRate,
        baselinePerRoundRate: baselineRate, lift,
        sampleCount: eventIndices.length, status: classification.key,
        significant: sigTest.significant, pValue: Number(sigTest.pValue.toFixed(4)),
      };
    });

    const recentHit = lastHits[target];
    const ageMs     = recentHit?.timestamp ? latestTimestamp - recentHit.timestamp : null;
    let recentPressure = null;
    if (ageMs != null) {
      const activeIndex  = COOLDOWN_WINDOWS.findIndex((window) => ageMs <= window.ms);
      const activeWindow = activeIndex >= 0 ? horizons[activeIndex] : null;
      if (activeWindow) {
        recentPressure = {
          ageMs,
          activeWindow: activeWindow.label,
          lift:         activeWindow.lift,
          significant:  activeWindow.significant,
          status:       activeWindow.status,
          note: activeWindow.significant
            ? (activeWindow.lift < 1
              ? `${labelForTarget(target)} significantly cools in the ${activeWindow.label.toLowerCase()} after it lands (p<0.05).`
              : `${labelForTarget(target)} significantly stays active in the ${activeWindow.label.toLowerCase()} after it lands (p<0.05).`)
            : `${labelForTarget(target)} cooldown pattern after a hit is not statistically significant in the ${activeWindow.label.toLowerCase()}.`,
        };
      }
    }

    items.push({
      target, label: labelForTarget(target),
      lastHitRoundId:    recentHit?.roundId    || null,
      lastHitTimestamp:  recentHit?.timestamp  || null,
      ageMs, horizons, recentPressure,
    });
  }
  return items;
}

// ---------------------------------------------------------------------------
// TARGET READINESS  –  significance-gated scores
// ---------------------------------------------------------------------------

function buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, latestTimestamp) {
  return TARGETS.map((target) => {
    const baselineRate  = baselineStats.perRoundHitRates[target] || 0;
    const currentRate   = currentSummary.hitRates[target] || 0;
    const slotItem      = slotAnalytics.items.find((item) => item.target === target);
    const slotLift      = slotItem?.currentSlot?.lift || 1;
    const slotSig       = Boolean(slotItem?.currentSlot?.liftSignificant);
    const rateLift      = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
    const cooldown      = cooldowns.find((item) => item.target === target);

    // Cooldown penalty only if the cooldown effect is statistically significant
    let cooldownPenalty = 0;
    if (cooldown?.recentPressure && cooldown.recentPressure.significant && cooldown.recentPressure.lift < 0.95) {
      cooldownPenalty = (0.95 - cooldown.recentPressure.lift) * 25;
    }

    // Score: slot lift only contributes if significant
    const effectiveSlotLift = slotSig ? slotLift : 1;
    const score = clamp(
      50
      + clamp((rateLift - 1) * 38, -20, 22)
      + clamp((effectiveSlotLift - 1) * 22, -12, 14)
      - cooldownPenalty,
      0, 100,
    );

    let status = 'neutral', label = 'Neutral';
    if (score >= 64) { status = 'ready'; label = 'Ready'; }
    else if (score <= 42) { status = 'avoid'; label = 'Avoid'; }

    let reason = `${labelForTarget(target)} is close to its usual pace.`;
    if (rateLift >= 1.12 && slotSig && slotLift >= 1.05) reason = `${labelForTarget(target)} is running above normal and this time block has a significantly supportive pattern.`;
    else if (rateLift <= 0.9)                             reason = `${labelForTarget(target)} is landing below its usual rate in the current window.`;
    else if (slotSig && slotLift <= 0.9)                  reason = `${labelForTarget(target)} is in a historically and significantly weak time block right now.`;
    else if (cooldown?.recentPressure?.significant && cooldown.recentPressure.lift < 0.95) reason = `${labelForTarget(target)} is in a statistically significant post-hit cooldown zone.`;

    return {
      target, label: labelForTarget(target), score, status, statusLabel: label,
      currentHitRate: currentRate, baselineHitRate: baselineRate,
      currentLift: rateLift, slotLift, slotSignificant: slotSig,
      reason, lastUpdatedAt: latestTimestamp,
    };
  });
}

// ---------------------------------------------------------------------------
// REGIME DETECTION  –  significance-gated
// ---------------------------------------------------------------------------

function buildRegime(focusTarget, currentSummary, baselineStats, cooldowns) {
  const focusLift     = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const hugeLift      = ratio(currentSummary.hugeHitRate,  baselineStats.hugeHitRate,  1);
  const megaLift      = ratio(currentSummary.megaHitRate,  baselineStats.megaHitRate,  1);
  const lowCrashLift  = ratio(currentSummary.lowCrashRate, baselineStats.lowCrashRate, 1);
  const recent100     = cooldowns.find((item) => item.target === 100)?.recentPressure;
  const recent500     = cooldowns.find((item) => item.target === 500)?.recentPressure;

  if (
    (recent500?.significant && recent500.lift < 0.95)
    || (recent100?.significant && recent100.lift < 0.95 && megaLift < 1)
  ) {
    return {
      key: 'post-spike-cooldown', label: 'Post-Spike Cooldown', tone: 'bad',
      description: 'A recent high spike has a statistically significant cooling effect — chasing is riskier right now.',
    };
  }
  if (megaLift >= 1.35 || hugeLift >= 1.25) {
    return {
      key: 'spike-mode', label: 'Spike Mode', tone: 'good',
      description: 'High multipliers are arriving above their historical pace.',
    };
  }
  if (lowCrashLift >= 1.15 && focusLift <= 0.95) {
    return {
      key: 'low-mode', label: 'Low Mode', tone: 'bad',
      description: 'Short crashes are stacking above normal and the selected target is lagging.',
    };
  }
  if (focusLift >= 1.12 && lowCrashLift <= 1) {
    return {
      key: 'target-friendly', label: 'Target-Friendly', tone: 'good',
      description: `Current flow is leaning toward ${labelForTarget(focusTarget)} more than the historical norm.`,
    };
  }
  return {
    key: 'balanced', label: 'Balanced Mode', tone: 'neutral',
    description: 'The board is close to its historical mix.',
  };
}

// ---------------------------------------------------------------------------
// DECISION  –  significance-gated, decay-aware
// ---------------------------------------------------------------------------

function buildDecision({ focusTarget, windowLabel, currentSummary, baselineStats, slotAnalytics, cooldowns, readiness }) {
  const focusReadiness = readiness.find((item) => item.target === focusTarget);
  const focusSlot      = slotAnalytics.items.find((item) => item.target === focusTarget)?.currentSlot;
  const cooldown       = cooldowns.find((item) => item.target === focusTarget);
  const baselineRate   = baselineStats.perRoundHitRates[focusTarget] || 0;
  const currentRate    = currentSummary.hitRates[focusTarget] || 0;
  const rateLift       = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
  const avgLift        = ratio(currentSummary.avgMultiplier, baselineStats.avgMultiplier, baselineStats.avgMultiplier > 0 ? 1 : 0);
  const lowCrashRelief = baselineStats.lowCrashRate > 0
    ? (baselineStats.lowCrashRate - currentSummary.lowCrashRate) / baselineStats.lowCrashRate
    : 0;
  const slotLift       = focusSlot?.lift || 1;
  const slotSig        = Boolean(focusSlot?.liftSignificant);

  // Cooldown penalty only when statistically significant
  const cooldownPenalty = (cooldown?.recentPressure?.significant && cooldown.recentPressure.lift < 0.95)
    ? (0.95 - cooldown.recentPressure.lift) * 24
    : 0;

  const effectiveSlotLift = slotSig ? slotLift : 1;

  const score = clamp(
    50
    + clamp((rateLift - 1) * 32, -18, 22)
    + clamp((effectiveSlotLift - 1) * 20, -12, 14)
    + clamp((avgLift - 1) * 12, -8, 8)
    + clamp(lowCrashRelief * 18, -10, 10)
    - cooldownPenalty,
    0, 100,
  );

  const band = describeBand(score);
  const zone = classifyLift(slotLift, focusSlot?.sampleCount || 0, focusSlot?.hitWindows, focusSlot?.sampleCount, baselineStats.windowAnyHitRates[focusTarget]);

  const playReasons = [], skipReasons = [];
  if (rateLift >= 1.1)                      playReasons.push(`${labelForTarget(focusTarget)} hit rate is above its historical baseline for this window.`);
  if (slotSig && slotLift >= 1.1)           playReasons.push(`This local time block has a significantly stronger pattern for ${labelForTarget(focusTarget)} (p<0.05).`);
  if (currentSummary.lowCrashRate <= baselineStats.lowCrashRate * 0.92) playReasons.push('Low crashes are lighter than the historical norm.');
  if (avgLift >= 1.08)                      playReasons.push('Average multiplier is running hotter than the historical baseline.');

  if (rateLift <= 0.92)                     skipReasons.push(`${labelForTarget(focusTarget)} is underperforming versus its historical hit rate.`);
  if (slotSig && slotLift <= 0.9)           skipReasons.push('This time block has a significantly weak historical pattern for the selected target.');
  if (currentSummary.lowCrashRate >= baselineStats.lowCrashRate * 1.1) skipReasons.push('Low crashes are stacking above the historical norm.');
  if (cooldown?.recentPressure?.significant && cooldown.recentPressure.lift < 0.95) skipReasons.push(cooldown.recentPressure.note);

  if (!playReasons.length) playReasons.push(`No significant positive signal for ${labelForTarget(focusTarget)} right now.`);
  if (!skipReasons.length) skipReasons.push('No statistically significant red flag at this time.');

  const regime = buildRegime(focusTarget, currentSummary, baselineStats, cooldowns);

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    score, band: band.key, label: band.label, tone: band.tone, windowLabel,
    zone: {
      key: zone.key, label: zone.label, tone: zone.tone, lift: slotLift, significant: slotSig,
      anyHitRate:        focusSlot?.anyHitChance || 0,
      baselineAnyHitRate: slotAnalytics.baselineByTarget[focusTarget]?.anyHitRate || 0,
      currentSlotLabel:  focusSlot?.label || '-',
    },
    regime, readiness: focusReadiness || null, playReasons, skipReasons,
    summary: band.key === 'play'
      ? `${labelForTarget(focusTarget)} is in a stronger-than-historical ${windowLabel.toLowerCase()} with a supportive time block.`
      : band.key === 'skip'
        ? `${labelForTarget(focusTarget)} is running below its historical norm for this ${windowLabel.toLowerCase()}.`
        : `${labelForTarget(focusTarget)} is mixed — wait-and-watch rather than a clear entry point.`,
  };
}

// ---------------------------------------------------------------------------
// WINDOW SCORE HELPER
// ---------------------------------------------------------------------------

function scoreWindowDecision(windowSummary, baselineStats, slotStat) {
  const scoreByTarget = {};
  for (const target of TARGETS) {
    const rateLift = ratio(windowSummary.hitRates[target], baselineStats.perRoundHitRates[target], baselineStats.perRoundHitRates[target] > 0 ? 1 : 0);
    const slotLift = slotStat?.liftSignificant ? (slotStat?.lift || 1) : 1;
    scoreByTarget[target] = clamp(50 + clamp((rateLift - 1) * 32, -18, 22) + clamp((slotLift - 1) * 20, -12, 14), 0, 100);
  }
  return scoreByTarget;
}

// ---------------------------------------------------------------------------
// STABILITY
// ---------------------------------------------------------------------------

function buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget) {
  const getParts     = buildZonedPartsGetter(timeZone);
  const recent       = completedWindows.slice(-8);
  if (!recent.length) {
    return { score: 0, label: 'Low Stability', status: 'unstable', flipCount: 0, windowsChecked: 0, message: 'Not enough completed windows yet to judge signal stability.', bands: [] };
  }

  const slotMinutes  = slotAnalytics.slotMinutes;
  const slotStats    = slotAnalytics.slotStatsByTarget[focusTarget] || [];
  const slotStatsMap = new Map(slotStats.map((slot) => [slot.slotIndex, slot]));
  const bands        = recent.map((window) => {
    const parts      = getParts(window.startTimestamp);
    const slotIndex  = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, Math.max(0, Math.floor(1440 / slotMinutes) - 1));
    const score      = scoreWindowDecision(window.summary, baselineStats, slotStatsMap.get(slotIndex))[focusTarget];
    const band       = describeBand(score);
    return { startTimestamp: window.startTimestamp, score, band: band.key, label: band.label };
  });

  let flipCount = 0;
  for (let index = 1; index < bands.length; index += 1) {
    if (bands[index].band !== bands[index - 1].band) flipCount += 1;
  }

  const counts = bands.reduce((acc, item) => { acc[item.band] = (acc[item.band] || 0) + 1; return acc; }, {});
  const dominantCount = Math.max(...Object.values(counts));
  const consistency   = ratio(dominantCount, bands.length);
  const score         = Math.round(clamp((consistency * 70) + ((1 - ratio(flipCount, Math.max(1, bands.length - 1))) * 30), 0, 100));

  let status = 'mixed', label = 'Medium Stability', message = 'Signal is moving around — keep position size smaller.';
  if (score >= 72) { status = 'stable'; label = 'High Stability'; message = 'Same decision band across several windows — signal is steadier.'; }
  else if (score <= 46) { status = 'unstable'; label = 'Low Stability'; message = 'Signal has been flipping a lot — treat it as weak.'; }

  return { score, label, status, flipCount, windowsChecked: bands.length, message, bands };
}

// ---------------------------------------------------------------------------
// BACKTEST  –  walk-forward on significance-approved slots only
// ---------------------------------------------------------------------------

function buildBacktest(slotWindows, slotAnalytics, focusTarget) {
  const focusItem = slotAnalytics.items.find((item) => item.target === focusTarget);
  if (!focusItem) {
    return {
      focusTarget, focusTargetLabel: labelForTarget(focusTarget),
      summary: 'Not enough time-slot history yet for a backtest.',
      allWindows:   { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 },
      greenWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 },
      redWindows:   { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 },
    };
  }

  // Only use slots whose lift is statistically significant for green/red labelling
  const greenSlots = new Set(
    (focusItem.topSlots || [])
      .filter((slot) => slot.liftSignificant && slot.lift >= 1.1 && slot.sampleCount >= slotAnalytics.minSamples)
      .map((slot) => slot.slotIndex),
  );
  const redSlots = new Set(
    (slotAnalytics.slotStatsByTarget[focusTarget] || [])
      .filter((slot) => slot.liftSignificant && slot.lift <= 0.9 && slot.sampleCount >= slotAnalytics.minSamples)
      .map((slot) => slot.slotIndex),
  );

  const summarize = (predicate) => {
    const selected = slotWindows.filter(predicate);
    if (!selected.length) return { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 };
    let hitWindows = 0, peakSum = 0;
    for (const slotWindow of selected) {
      if ((slotWindow.summary.hitCounts[focusTarget] || 0) > 0) hitWindows += 1;
      peakSum += slotWindow.summary.maxMultiplier;
    }
    return { count: selected.length, anyHitRate: ratio(hitWindows, selected.length), avgPeakMultiplier: ratio(peakSum, selected.length) };
  };

  const allWindows   = summarize(() => true);
  const greenWindows = summarize((w) => greenSlots.has(w.slotIndex));
  const redWindows   = summarize((w) => redSlots.has(w.slotIndex));

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    summary: greenWindows.count > 0
      ? `Significantly strong (p<0.05) slots hit ${labelForTarget(focusTarget)} ${pctString(greenWindows.anyHitRate)} vs ${pctString(allWindows.anyHitRate)} across all slots.`
      : 'No slots with statistically significant strength yet for a reliable backtest.',
    allWindows,
    greenWindows: { ...greenWindows, lift: ratio(greenWindows.anyHitRate, allWindows.anyHitRate, 1) },
    redWindows:   { ...redWindows,   lift: ratio(redWindows.anyHitRate,   allWindows.anyHitRate, 1) },
  };
}

// ---------------------------------------------------------------------------
// COMPARISON
// ---------------------------------------------------------------------------

function buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday) {
  const currentLift  = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const currentWindow = bestWindowsToday.items.find((item) => item.target === focusTarget);
  const zoneLabel    = currentWindow?.currentSlot?.zoneLabel || decision.zone.label;
  let message = `${labelForTarget(focusTarget)} is running close to its historical pace.`;
  if (currentLift >= 1.12)      message = `${labelForTarget(focusTarget)} is hotter than normal, time block is ${zoneLabel.toLowerCase()}.`;
  else if (currentLift <= 0.9)  message = `${labelForTarget(focusTarget)} is colder than normal, time block leans ${zoneLabel.toLowerCase()}.`;
  else                          message = `${labelForTarget(focusTarget)} is mixed — use zone, cooldown, and stability panels together.`;
  return { band: decision.band, label: decision.label, message };
}

// ---------------------------------------------------------------------------
// TARGET CARDS
// ---------------------------------------------------------------------------

function buildTargetCards(currentSummary, baselineStats) {
  return TARGETS.map((target) => {
    const currentHitRate  = currentSummary.hitRates[target] || 0;
    const baselineHitRate = baselineStats.perRoundHitRates[target] || 0;
    return {
      target, label: labelForTarget(target), currentHitRate, baselineHitRate,
      delta: currentHitRate - baselineHitRate,
      lift: ratio(currentHitRate, baselineHitRate, baselineHitRate > 0 ? 1 : 0),
    };
  });
}

// ---------------------------------------------------------------------------
// OUTLOOK  –  z-score normalised nearest-neighbour
// ---------------------------------------------------------------------------

function buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) {
  const usable = completedWindows.filter((_, index) => index < completedWindows.length - 1);
  if (usable.length < 10) {
    return { available: false, reason: 'Not enough completed windows yet to build a next-window outlook.' };
  }

  // Build normalised vector pool for all usable windows + current
  const allSummaries = [...usable.map((w) => w.summary), currentSummary];
  const { zVectors } = buildNormalizedVectorPool(allSummaries, focusTarget);
  const currentVec   = zVectors[zVectors.length - 1];

  const candidates = usable
    .map((window, index) => {
      const nextWindow = completedWindows[index + 1];
      if (!nextWindow) return null;
      const distance = normalizedDistance(zVectors[index], currentVec);
      return { distance, nextWindow };
    })
    .filter(Boolean)
    .sort((a, b) => a.distance - b.distance);

  const sampleSize = clamp(Math.round(candidates.length * 0.18), 12, 60);
  const matches    = candidates.slice(0, sampleSize);
  if (!matches.length) return { available: false, reason: 'No close historical match was found for the current window shape.' };

  const candidateRows = TARGETS.map((target) => {
    let windowHits = 0, totalHits = 0, totalRounds = 0, peakSum = 0;
    for (const match of matches) {
      const summary = match.nextWindow.summary;
      totalHits    += summary.hitCounts[target] || 0;
      totalRounds  += summary.roundCount;
      peakSum      += summary.maxMultiplier;
      if ((summary.hitCounts[target] || 0) > 0) windowHits += 1;
    }
    const anyHitRate         = ratio(windowHits, matches.length);
    const perRoundHitRate    = ratio(totalHits, totalRounds);
    const baselineAnyHitRate = baselineStats.windowAnyHitRates[target] || 0;
    const lift               = ratio(anyHitRate, baselineAnyHitRate, baselineAnyHitRate > 0 ? 1 : 0);
    const sigTest            = chiSquareTest(windowHits, matches.length, baselineAnyHitRate);
    const rewardWeight       = 1 + (Math.log10(target) / 2);
    const score              = anyHitRate * (sigTest.significant && lift > 1 ? lift : 1) * rewardWeight;
    return {
      target, label: labelForTarget(target), anyHitRate, perRoundHitRate,
      avgPeakMultiplier: ratio(peakSum, matches.length),
      expectedHits: Math.round(ratio(totalHits, matches.length) * 100) / 100,
      baselineAnyHitRate, lift, liftSignificant: sigTest.significant,
      liftPValue: Number(sigTest.pValue.toFixed(4)), score,
      style: anyHitRate >= 0.55 ? 'Frequent' : anyHitRate >= 0.3 ? 'Balanced' : 'Long-shot',
    };
  }).sort((a, b) => b.score - a.score);

  const recommendation = candidateRows[0] || null;
  const focusCandidate = candidateRows.find((item) => item.target === focusTarget) || null;
  const distanceSpread = average(matches.map((item) => item.distance));
  const confidence     = clamp((sampleWeight(matches.length) * 0.55) + (1 / (1 + distanceSpread)) * 0.45, 0, 1);

  return {
    available: true, basedOnMatches: matches.length, confidence,
    note: `Built from ${matches.length} completed windows most similar to the current one (z-score Euclidean distance).`,
    recommendation, focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    focusCandidate, candidates: candidateRows,
  };
}

// ---------------------------------------------------------------------------
// BEST WINDOWS TODAY
// ---------------------------------------------------------------------------

function buildBestWindowsToday(slotAnalytics, focusTarget) {
  return {
    slotMode: slotAnalytics.slotMode,
    slotMinutes: slotAnalytics.slotMinutes,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    note: slotAnalytics.slotMode === 'start-time'
      ? 'For multi-day windows, this ranks the best starting times rather than same-day end times.'
      : 'These are the best recurring local-time windows from the stored dataset (significance-filtered).',
    items: slotAnalytics.items.map((item) => ({
      target: item.target, label: item.label, currentSlot: item.currentSlot,
      nextTodayWindow: item.nextTodayWindow, nextWindow: item.nextWindow,
      todayWindows: item.todayWindows, backups: item.backups,
      avoidWindow: item.avoidWindow, bestWindow: item.topSlots[0] || null, topWindows: item.topSlots,
    })),
  };
}

// ---------------------------------------------------------------------------
// EMPTY REPORT  (unchanged structure, updated strings)
// ---------------------------------------------------------------------------

function buildEmptyReport(windowConfig, focusTarget, timeZone) {
  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId: null,
    totalRounds: 0,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets: TARGETS.map((value) => ({ value, label: labelForTarget(value) })),
    timeZone,
    dataset: { totalRounds: 0, startTimestamp: null, endTimestamp: null, spanDays: 0 },
    window: { key: windowConfig.key, label: windowConfig.label, ms: windowConfig.ms, startTimestamp: null, endTimestamp: null },
    baseline: summarizeWindowCollection([], focusTarget),
    previousWindow: summarizeRounds([], focusTarget),
    currentWindow:  summarizeRounds([], focusTarget),
    patternPrediction: {
      action: 'WAIT FOR MORE DATA', tone: 'neutral', confidence: 0, confidenceLabel: 'Low', dataQuality: 'insufficient',
      predictsLabel: `Current ${windowConfig.label}`, inputLabel: `Closed Previous ${windowConfig.label}`,
      inputSlotLabel: '-', currentSlotLabel: '-',
      currentHitRate: 0, baselineHitRate: 0, baselineCurrentWindowHitRate: 0,
      currentLift: 1, effectiveCurrentLift: 1, currentEvidenceWeight: 0, currentSlotChance: 0,
      currentWindowHitRate: 0, currentWindowLift: 1, remainingHitRate: 0, remainingLift: 1, baselineRemainingHitRate: 0,
      matchedWindows: 0, sameWeekdayMatches: 0, lookbackDaysUsed: 0,
      alreadyHitInCurrentWindow: false, hitsSoFar: 0,
      expectedRoundIdFrom: null, expectedRoundIdTo: null, expectedRoundIdLabel: '-', expectedRoundIdBasis: '',
      summary: 'No rounds are stored yet, so there is no timing prediction.',
      reasons: [],
    },
    comparison: { band: 'wait', label: 'WAIT / WATCH', message: 'No rounds are stored yet.' },
    targetCards: [],
    decision: null,
    targetReadiness: [],
    recommendationStability: null,
    bestWindowsToday: { items: [], slotMode: 'window', slotMinutes: chooseSlotMinutes(windowConfig.ms), note: '' },
    patternMatch: { available: false, examples: [], reason: 'No rounds are stored yet.', inputSlotLabel: '-', currentSlotLabel: '-' },
    cooldowns: [],
    backtest: null,
    hourlyHistory: { timeZone, rows: [], bestHours: [] },
    dayHourHeatmap: {
      focusTarget, focusTargetLabel: labelForTarget(focusTarget),
      days: WEEKDAYS,
      hours: Array.from({ length: 24 }, (_, hour) => ({ value: hour, label: formatHourLabel(hour) })),
      cells: [], strongest: [], weakest: [],
    },
    outlook: null,
  };
}

// ---------------------------------------------------------------------------
// MAIN ENTRY POINT
// ---------------------------------------------------------------------------

function buildTimingAnalyticsReport(rounds, options = {}) {
  const windowKey      = normalizeTimingWindowKey(options.windowKey);
  const focusTarget    = normalizeTimingTarget(options.focusTarget);
  const timeZone       = normalizeTimingTimeZone(options.timeZone);
  const includeOutlook = Boolean(options.includeOutlook);
  const windowConfig   = WINDOW_MAP.get(windowKey);
  const normalizedRounds = normalizeRounds(rounds);

  if (!normalizedRounds.length) return buildEmptyReport(windowConfig, focusTarget, timeZone);

  const latestRound   = normalizedRounds[normalizedRounds.length - 1];
  const earliestRound = normalizedRounds[0];
  const slotMinutes   = chooseSlotMinutes(windowConfig.ms);
  const slotWindows   = buildSlotWindows(normalizedRounds, slotMinutes, timeZone, focusTarget);
  const slotAnalytics = buildSlotAnalytics(slotWindows, slotMinutes, windowConfig.ms, focusTarget, latestRound.timestamp, timeZone);

  const currentParts    = buildZonedPartsGetter(timeZone)(latestRound.timestamp);
  const currentKey      = `${currentParts.dateKey}|${slotAnalytics.currentSlotIndex}`;
  const orderedSlotWindows = [...slotWindows].sort((a, b) => {
    if (a.dateKey !== b.dateKey) return a.dateKey.localeCompare(b.dateKey);
    return a.slotIndex - b.slotIndex;
  });
  const slotIndexByKey   = new Map(orderedSlotWindows.map((window, index) => [window.key, index]));
  const currentSlotPosition = slotIndexByKey.get(currentKey);
  const currentSlotWindow   = Number.isInteger(currentSlotPosition)
    ? orderedSlotWindows[currentSlotPosition]
    : (orderedSlotWindows[orderedSlotWindows.length - 1] || null);
  const previousSlotWindow  = Number.isInteger(currentSlotPosition) && currentSlotPosition > 0
    ? orderedSlotWindows[currentSlotPosition - 1]
    : null;

  const currentSummary  = currentSlotWindow  ? currentSlotWindow.summary  : summarizeRounds([], focusTarget);
  const previousSummary = previousSlotWindow ? previousSlotWindow.summary : summarizeRounds([], focusTarget);

  const fixedWindows       = segmentRoundsByWindow(normalizedRounds, windowConfig.ms, focusTarget);
  const completedWindows   = fixedWindows.slice(0, -1);
  const completedSlotWindows = slotWindows.filter((window) => window.key !== currentKey);
  const baselineStats      = completedSlotWindows.length
    ? summarizeWindowCollection(completedSlotWindows, focusTarget)
    : summarizeWindowCollection(slotWindows, focusTarget);

  const bestWindowsToday       = buildBestWindowsToday(slotAnalytics, focusTarget);
  const lastHits               = buildLastHitMap(normalizedRounds);
  const cooldowns              = buildCooldownReport(normalizedRounds, baselineStats.perRoundHitRates, latestRound.timestamp, lastHits);
  const targetReadiness        = buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, latestRound.timestamp);
  const decision               = buildDecision({ focusTarget, windowLabel: windowConfig.label, currentSummary, baselineStats, slotAnalytics, cooldowns, readiness: targetReadiness });
  const recommendationStability = buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget);
  const hourlyHistory          = buildHourlyHistory(normalizedRounds, timeZone, baselineStats.perRoundHitRates);
  const dayHourHeatmap         = buildHeatmap(normalizedRounds, timeZone, focusTarget, baselineStats.perRoundHitRates);
  const backtest               = buildBacktest(slotWindows, slotAnalytics, focusTarget);
  const comparison             = buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday);
  const targetCards            = buildTargetCards(currentSummary, baselineStats);
  const patternMatch           = buildPatternMatchReport(slotWindows, previousSummary, currentSummary, baselineStats, focusTarget, latestRound.timestamp, timeZone, slotAnalytics);
  const patternPrediction      = buildPatternPrediction({ focusTarget, windowLabel: windowConfig.label, latestRoundId: latestRound.roundId || null, currentSummary, baselineStats, slotAnalytics, patternMatch });
  const outlook                = includeOutlook ? buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) : null;

  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId:     latestRound.roundId || null,
    totalRounds:       normalizedRounds.length,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    availableWindows:  WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets:  TARGETS.map((value) => ({ value, label: labelForTarget(value) })),
    timeZone,
    dataset: {
      totalRounds:    normalizedRounds.length,
      startTimestamp: earliestRound.timestamp,
      endTimestamp:   latestRound.timestamp,
      spanDays:       ratio(latestRound.timestamp - earliestRound.timestamp, DAY_MS),
    },
    window: {
      key: windowConfig.key, label: windowConfig.label, ms: windowConfig.ms,
      startTimestamp: currentSlotWindow?.firstTimestamp || (latestRound.timestamp - windowConfig.ms),
      endTimestamp:   latestRound.timestamp,
    },
    baseline: baselineStats,
    previousWindow: previousSummary,
    currentWindow:  currentSummary,
    patternPrediction, comparison, targetCards, decision,
    targetReadiness, recommendationStability, bestWindowsToday, patternMatch,
    cooldowns, backtest, hourlyHistory, dayHourHeatmap, outlook,
  };
}

module.exports = {
  buildTimingAnalyticsReport,
  normalizeTimingWindowKey,
  normalizeTimingTarget,
  normalizeTimingTimeZone,
};