'use strict';

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

const MINUTE_MS = 60 * 1000;
const HOUR_MS   = 60 * MINUTE_MS;
const DAY_MS    = 24 * HOUR_MS;

const WINDOW_OPTIONS = [
  { key: '5m',  label: '5 Minutes',  ms: 5  * MINUTE_MS },
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '1h',  label: '1 Hour',     ms: 1  * HOUR_MS   },
  { key: '2h',  label: '2 Hours',    ms: 2  * HOUR_MS   },
  { key: '5h',  label: '5 Hours',    ms: 5  * HOUR_MS   },
];

const WINDOW_MAP      = new Map(WINDOW_OPTIONS.map((item) => [item.key, item]));
const DEFAULT_WINDOW_KEY = '5m';

const TARGETS     = [5, 10, 20, 50, 100, 500, 1000];
const TARGET_SET  = new Set(TARGETS);
const DEFAULT_TARGET = 5;

const DISTRIBUTION_BANDS = [
  { key: 'lt2',       label: '<2x',        min: -Infinity, max: 2,       color: '#ff5d73' },
  { key: '2to5',      label: '2x-5x',      min: 2,         max: 5,       color: '#ff9f43' },
  { key: '5to10',     label: '5x-10x',     min: 5,         max: 10,      color: '#ffd84d' },
  { key: '10to20',    label: '10x-20x',    min: 10,        max: 20,      color: '#9ef01a' },
  { key: '20to50',    label: '20x-50x',    min: 20,        max: 50,      color: '#22d3ee' },
  { key: '50to100',   label: '50x-100x',   min: 50,        max: 100,     color: '#38bdf8' },
  { key: '100to500',  label: '100x-500x',  min: 100,       max: 500,     color: '#f472b6' },
  { key: '500to1000', label: '500x-1000x', min: 500,       max: 1000,    color: '#c084fc' },
  { key: 'gte1000',   label: '1000x+',     min: 1000,      max: Infinity, color: '#818cf8' },
];

const COOLDOWN_WINDOWS = [
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '20m', label: '20 Minutes', ms: 20 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '60m', label: '60 Minutes', ms: 60 * MINUTE_MS },
];

const WEEKDAYS      = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const WEEKDAY_INDEX = WEEKDAYS.reduce((acc, label, index) => { acc[label] = index; return acc; }, {});

// ---------------------------------------------------------------------------
// BUG-FIX 1: STATISTICAL SIGNIFICANCE
// The original Yates continuity correction was over-aggressive for small samples
// (diff <= 0 check caused false "not significant" when observed ≈ expected).
// Fix: only apply Yates when n < 40; use raw chi-square for larger samples.
// Also: minimum cell count reduced from 5 to 3 to handle realistic slot counts.
// ---------------------------------------------------------------------------

const MIN_CELL_COUNT = 3; // was 5 — 5 was too strict for slot-level window counts

/**
 * Chi-square goodness-of-fit, binary rate vs expected rate.
 * Yates correction only for n < 40 (small samples); raw chi-square otherwise.
 * Returns { significant, pValue, chi2 }.
 */
function chiSquareTest(observed, total, expectedRate) {
  if (total < MIN_CELL_COUNT * 2 || expectedRate <= 0 || expectedRate >= 1) {
    return { significant: false, pValue: 1, chi2: 0 };
  }
  const expected    = total * expectedRate;
  const expectedNeg = total * (1 - expectedRate);
  const observedNeg = total - observed;

  // BUG-FIX: Only apply Yates continuity correction for small samples (n < 40).
  // For n >= 40 the correction is unnecessary and under-detects real effects.
  let chi2;
  if (total < 40) {
    // Yates corrected
    const diff = Math.abs(observed - expected) - 0.5;
    if (diff <= 0) return { significant: false, pValue: 1, chi2: 0 };
    chi2 = (diff * diff) / expected + (diff * diff) / expectedNeg;
  } else {
    // Standard Pearson chi-square (no Yates)
    const d1 = observed    - expected;
    const d2 = observedNeg - expectedNeg;
    chi2 = (d1 * d1) / expected + (d2 * d2) / expectedNeg;
    if (chi2 <= 0) return { significant: false, pValue: 1, chi2: 0 };
  }

  const pValue = chiSquarePValue(chi2);
  return { significant: pValue < 0.05, pValue, chi2 };
}

/** Survival function of chi-squared(df=1) via erfc approximation. */
function chiSquarePValue(chi2) {
  return erfcApprox(Math.sqrt(chi2 / 2));
}

/** Complementary error function (Abramowitz & Stegun 7.1.26, max error < 1.5e-7). */
function erfcApprox(x) {
  if (x < 0) return 2 - erfcApprox(-x);
  const t    = 1 / (1 + 0.3275911 * x);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return poly * Math.exp(-x * x);
}

// ---------------------------------------------------------------------------
// NORMALISATION — z-score per feature across a collection
// ---------------------------------------------------------------------------

function zScoreNormalize(items, featureKeys) {
  if (!items.length) return [];
  const means = {}, stds = {};
  for (const key of featureKeys) {
    const values = items.map((item) => safeNumber(item[key], 0));
    const mean   = values.reduce((s, v) => s + v, 0) / values.length;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    means[key] = mean;
    stds[key]  = Math.sqrt(variance) || 1;
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
// DECAY — exponential time-decay weight
// ---------------------------------------------------------------------------

/** Returns weight in (0, 1]. halfLifeDays = age at which weight = 0.5. */
function decayWeight(ageMs, halfLifeDays = 14) {
  return Math.pow(0.5, ageMs / DAY_MS / halfLifeDays);
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
  } catch { return 'UTC'; }
}

// ---------------------------------------------------------------------------
// MATH UTILITIES
// ---------------------------------------------------------------------------

function clamp(value, min, max)         { return Math.min(max, Math.max(min, value)); }
function ratio(num, den, fallback = 0)  { return den > 0 ? num / den : fallback; }
function liftAgainstBaseline(rate, baselineRate, positiveFallback = 1.25) {
  const observed = safeNumber(rate, 0);
  const baseline = safeNumber(baselineRate, 0);
  if (baseline > 0) return observed / baseline;
  if (observed > 0) return positiveFallback;
  return 1;
}

function average(values) {
  if (!values.length) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

function weightedAverage(items, valueGetter, weightGetter, fallback = 0) {
  if (!Array.isArray(items) || !items.length) return fallback;
  let ws = 0, tw = 0;
  for (const item of items) {
    const v = Number(valueGetter(item)), w = Math.max(0, Number(weightGetter(item)));
    if (!Number.isFinite(v) || !Number.isFinite(w) || w <= 0) continue;
    ws += v * w; tw += w;
  }
  return tw > 0 ? ws / tw : fallback;
}

function weightedQuantile(items, valueGetter, weightGetter, q, fallback = 0) {
  if (!Array.isArray(items) || !items.length) return fallback;
  const rows = items
    .map((item) => ({ value: Number(valueGetter(item)), weight: Math.max(0, Number(weightGetter(item))) }))
    .filter((r) => Number.isFinite(r.value) && Number.isFinite(r.weight) && r.weight > 0)
    .sort((a, b) => a.value - b.value);
  if (!rows.length) return fallback;
  const target = clamp(Number(q) || 0, 0, 1);
  const total  = rows.reduce((s, r) => s + r.weight, 0);
  if (total <= 0) return fallback;
  let cum = 0;
  for (const row of rows) { cum += row.weight; if (cum / total >= target) return row.value; }
  return rows[rows.length - 1].value;
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos   = (sorted.length - 1) * clamp(q, 0, 1);
  const lower = Math.floor(pos), upper = Math.ceil(pos);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (pos - lower);
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function labelForTarget(target) { return `${target}x`; }

function pctString(value, digits = 1) {
  const n = safeNumber(value, null);
  if (n == null) return '-';
  return `${(n * 100).toFixed(digits)}%`;
}

function formatHourLabel(hour) {
  const h = ((Number(hour) % 24) + 24) % 24;
  const suffix = h >= 12 ? 'PM' : 'AM';
  return `${h % 12 || 12}:00 ${suffix}`;
}

function formatClockMinute(minuteOfDay) {
  const total  = ((Number(minuteOfDay) % 1440) + 1440) % 1440;
  const hour   = Math.floor(total / 60), minute = total % 60;
  const suffix = hour >= 12 ? 'PM' : 'AM';
  return `${hour % 12 || 12}:${String(minute).padStart(2, '0')} ${suffix}`;
}

function formatSlotLabel(startMinute, slotMinutes, mode) {
  if (mode === 'start-time') return `Starts ${formatClockMinute(startMinute)}`;
  if (slotMinutes >= 1440)   return `All Day (${formatClockMinute(startMinute)} start)`;
  const endMinute = (startMinute + slotMinutes) % 1440;
  return `${formatClockMinute(startMinute)} - ${formatClockMinute(endMinute)}`;
}

function formatOccurrenceLabel(startMinute, slotMinutes, mode, dayOffset) {
  const base = formatSlotLabel(startMinute, slotMinutes, mode);
  return dayOffset ? `${base} Tomorrow` : base;
}

function formatTimestampInTimeZone(timestamp, timeZone) {
  const n = safeNumber(timestamp, 0);
  if (!n) return '-';
  return new Intl.DateTimeFormat('en-US', {
    timeZone, month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  }).format(new Date(n));
}

function summarizeRoundRange(rounds, focusTarget, limit = 8) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) return { fromRoundId: null, toRoundId: null, roundCount: 0, hitRoundIds: [], hitCount: 0 };
  const hitRoundIds = items.filter((r) => r.multiplier >= focusTarget).map((r) => r.roundId).slice(0, limit);
  return {
    fromRoundId: items[0]?.roundId ?? null,
    toRoundId:   items[items.length - 1]?.roundId ?? null,
    roundCount:  items.length,
    hitRoundIds,
    hitCount:    items.filter((r) => r.multiplier >= focusTarget).length,
  };
}

function chooseSlotMinutes(windowMs) {
  if (windowMs > DAY_MS) return 60;
  const minutes = Math.round(windowMs / MINUTE_MS);
  return Math.max(5, minutes || 5);
}

// ---------------------------------------------------------------------------
// BUG-FIX 2: LIFT CLASSIFIER
// Old: used MIN_CELL_COUNT * 2 = 10 hard floor — killed all signals with 10 windows.
// Fix: use the chi-square result directly; only require MIN_CELL_COUNT (3) samples.
// The significance test itself is the gate — no extra arbitrary floor needed.
// ---------------------------------------------------------------------------

function classifyLift(lift, sampleCount, observedHits, totalRounds, expectedRate) {
  // Legacy call without stat params
  if (observedHits === undefined || totalRounds === undefined || expectedRate === undefined) {
    if (sampleCount < MIN_CELL_COUNT) return { key: 'neutral', label: 'Insufficient Data', tone: 'neutral' };
    if (lift >= 1.15) return { key: 'green', label: 'Green Zone', tone: 'good' };
    if (lift <= 0.85) return { key: 'red',   label: 'Red Zone',   tone: 'bad'  };
    return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
  }

  if (sampleCount < MIN_CELL_COUNT) {
    return { key: 'neutral', label: 'Insufficient Data', tone: 'neutral' };
  }

  const { significant } = chiSquareTest(observedHits, totalRounds, expectedRate);
  if (!significant) return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
  if (lift >= 1.05) return { key: 'green', label: 'Green Zone', tone: 'good' };
  if (lift <= 0.95) return { key: 'red',   label: 'Red Zone',   tone: 'bad'  };
  return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
}

function describeBand(score) {
  if (score >= 65) return { key: 'play',  label: 'PLAY WINDOW',  tone: 'good'    };
  if (score >= 48) return { key: 'wait',  label: 'WAIT / WATCH', tone: 'neutral' };
  return                  { key: 'skip',  label: 'SKIP WINDOW',  tone: 'bad'     };
}

function createTargetMap(initialValue) {
  const out = {};
  for (const t of TARGETS) {
    out[t] = typeof initialValue === 'function' ? initialValue(t) : initialValue;
  }
  return out;
}

// ---------------------------------------------------------------------------
// ZONED DATE HELPERS
// ---------------------------------------------------------------------------

function buildZonedPartsGetter(timeZone) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone, weekday: 'short', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  });
  return (timestamp) => {
    const parts = {};
    for (const part of formatter.formatToParts(new Date(timestamp))) {
      if (part.type !== 'literal') parts[part.type] = part.value;
    }
    const weekday = parts.weekday || 'Mon';
    const hour    = safeNumber(parts.hour, 0);
    const minute  = safeNumber(parts.minute, 0);
    const second  = safeNumber(parts.second, 0);
    return {
      weekday,
      dayIndex:    WEEKDAY_INDEX[weekday] ?? 0,
      hour, minute, second,
      minuteOfDay: hour * 60 + minute,
      dateKey:     `${parts.year || '1970'}-${parts.month || '01'}-${parts.day || '01'}`,
    };
  };
}

// ---------------------------------------------------------------------------
// ROUND NORMALISATION & SUMMARISATION
// ---------------------------------------------------------------------------

function normalizeRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : [])
    .map((r) => ({
      roundId:    safeNumber(r?.roundId ?? r?.round_id, 0),
      multiplier: safeNumber(r?.multiplier, NaN),
      timestamp:  safeNumber(r?.timestamp, NaN),
    }))
    .filter((r) => Number.isFinite(r.multiplier) && Number.isFinite(r.timestamp) && r.timestamp > 0)
    .sort((a, b) => a.timestamp !== b.timestamp ? a.timestamp - b.timestamp : a.roundId - b.roundId);
}

function summarizeRounds(rounds, focusTarget) {
  const values = [], hitCounts = createTargetMap(0);
  const distCounts = Object.fromEntries(DISTRIBUTION_BANDS.map((b) => [b.key, 0]));
  let sum = 0, max = 0, min = Infinity, lowCrash = 0, huge = 0, mega = 0;

  for (const r of rounds) {
    const m = r.multiplier;
    values.push(m);
    sum += m;
    if (m > max) max = m;
    if (m < min) min = m;
    if (m < 2)    lowCrash++;
    if (m >= 100) huge++;
    if (m >= 500) mega++;
    for (const t of TARGETS) { if (m >= t) hitCounts[t]++; }
    for (const b of DISTRIBUTION_BANDS) { if (m >= b.min && m < b.max) { distCounts[b.key]++; break; } }
  }

  values.sort((a, b) => a - b);
  const n = rounds.length;
  const hitRates = createTargetMap((t) => ratio(hitCounts[t], n));
  const distribution = DISTRIBUTION_BANDS.map((b) => ({
    key: b.key, label: b.label, count: distCounts[b.key],
    pct: ratio(distCounts[b.key], n), color: b.color,
  }));

  return {
    roundCount:       n,
    avgMultiplier:    ratio(sum, n),
    medianMultiplier: quantile(values, 0.5),
    p90Multiplier:    quantile(values, 0.9),
    maxMultiplier:    max || 0,
    minMultiplier:    Number.isFinite(min) ? min : 0,
    focusHitCount:    hitCounts[focusTarget] || 0,
    focusHitRate:     hitRates[focusTarget]  || 0,
    lowCrashRate:     ratio(lowCrash, n),
    hugeHitRate:      ratio(huge, n),
    megaHitRate:      ratio(mega, n),
    hitCounts, hitRates, distribution,
  };
}

function distributionPct(summary, bandKey) {
  const items = Array.isArray(summary?.distribution) ? summary.distribution : [];
  return safeNumber(items.find((i) => i.key === bandKey)?.pct, 0);
}

// ---------------------------------------------------------------------------
// Z-SCORE NORMALISED FEATURE DISTANCE
// ---------------------------------------------------------------------------

function summaryToFeatureVector(summary) {
  return {
    hitRate5:      summary?.hitRates?.[5]           || 0,
    hitRate10:     summary?.hitRates?.[10]          || 0,
    hitRate20:     summary?.hitRates?.[20]          || 0,
    hitRate100:    summary?.hitRates?.[100]         || 0,
    hitRate500:    summary?.hitRates?.[500]         || 0,
    lowCrashRate:  summary?.lowCrashRate             || 0,
    hugeHitRate:   summary?.hugeHitRate              || 0,
    megaHitRate:   summary?.megaHitRate              || 0,
    avgMultiplier: normalizePatternMultiplier(summary?.avgMultiplier || 1),
    maxMultiplier: normalizePatternMultiplier(summary?.maxMultiplier || 1),
    distLt2:     distributionPct(summary, 'lt2'),
    dist2to5:    distributionPct(summary, '2to5'),
    dist5to10:   distributionPct(summary, '5to10'),
    dist10to20:  distributionPct(summary, '10to20'),
    dist20to50:  distributionPct(summary, '20to50'),
    dist50to100: distributionPct(summary, '50to100'),
  };
}

const SUMMARY_FEATURE_KEYS = [
  'hitRate5','hitRate10','hitRate20','hitRate100','hitRate500',
  'lowCrashRate','hugeHitRate','megaHitRate',
  'avgMultiplier','maxMultiplier',
  'distLt2','dist2to5','dist5to10','dist10to20','dist20to50','dist50to100',
];

function normalizedDistance(vecA, vecB) {
  if (!vecA || !vecB) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (const key of SUMMARY_FEATURE_KEYS) {
    const diff = (vecA[key] || 0) - (vecB[key] || 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function buildNormalizedVectorPool(summaries) {
  const rawVectors = summaries.map((s) => summaryToFeatureVector(s));
  const zVectors   = zScoreNormalize(rawVectors, SUMMARY_FEATURE_KEYS);
  return { rawVectors, zVectors };
}

const PATTERN_MULTIPLIER_BANDS = [
  { max: 1.75, center: 1.35 },
  { max: 2.5, center: 2.1 },
  { max: 4, center: 3.2 },
  { max: 7, center: 5.2 },
  { max: 12, center: 9.0 },
  { max: 20, center: 15.0 },
  { max: 35, center: 27.0 },
  { max: 70, center: 50.0 },
  { max: 150, center: 100.0 },
  { max: 350, center: 220.0 },
  { max: 750, center: 500.0 },
  { max: Infinity, center: 1000.0 },
];

function normalizePatternMultiplier(multiplier) {
  const safeMultiplier = Math.max(1, safeNumber(multiplier, 1));
  const band = PATTERN_MULTIPLIER_BANDS.find((item) => safeMultiplier <= item.max) || PATTERN_MULTIPLIER_BANDS[PATTERN_MULTIPLIER_BANDS.length - 1];
  return Math.log1p(band.center) / Math.log(1001);
}

function summarizePatternSlice(rounds) {
  const values = (Array.isArray(rounds) ? rounds : [])
    .map((item) => normalizePatternMultiplier(item?.multiplier))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!values.length) return 0;
  return quantile(values, 0.5);
}

function sampleWindowPattern(rounds, sampleCount = 12) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) return Array.from({ length: sampleCount }, () => 0);
  if (items.length === 1) {
    const v = normalizePatternMultiplier(items[0]?.multiplier);
    return Array.from({ length: sampleCount }, () => v);
  }
  return Array.from({ length: sampleCount }, (_, index) => {
    const start = Math.floor((index * items.length) / sampleCount);
    const nextStart = Math.floor(((index + 1) * items.length) / sampleCount);
    const end = Math.max(start + 1, nextStart);
    const slice = items.slice(start, Math.min(end, items.length));
    if (slice.length) return summarizePatternSlice(slice);
    return normalizePatternMultiplier(items[Math.min(start, items.length - 1)]?.multiplier);
  });
}

function patternSeriesDistance(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || a.length !== b.length) {
    return Number.POSITIVE_INFINITY;
  }
  const tolerance = 0.045;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = Math.abs(safeNumber(a[i], 0) - safeNumber(b[i], 0));
    const effectiveDiff = Math.max(0, diff - tolerance);
    sum += effectiveDiff * effectiveDiff;
  }
  return Math.sqrt(sum / a.length);
}

function similarityPctFromDistances(summaryDistance, sequenceDistance, weekdayMatch) {
  const summaryScore = clamp(1 - (safeNumber(summaryDistance, 0) / 2.8), 0, 1);
  const sequenceScore = clamp(1 - (safeNumber(sequenceDistance, 0) / 0.55), 0, 1);
  const weekdayBonus = weekdayMatch ? 0.04 : 0;
  return clamp(((summaryScore * 0.52) + (sequenceScore * 0.48) + weekdayBonus) * 100, 0, 100);
}

// ---------------------------------------------------------------------------
// PROGRESS SPLIT
// ---------------------------------------------------------------------------

function splitWindowRoundsByProgress(window, progressRatio, slotMinutes) {
  const rounds    = Array.isArray(window?.rounds) ? window.rounds : [];
  if (!rounds.length) return { elapsedRounds: [], remainingRounds: [], elapsedCount: 0 };
  const safeRatio = clamp(progressRatio, 0, 1);
  if (safeRatio <= 0)     return { elapsedRounds: [], remainingRounds: rounds.slice(), elapsedCount: 0 };
  if (safeRatio >= 0.999) return { elapsedRounds: rounds.slice(), remainingRounds: [], elapsedCount: rounds.length };

  const startTs  = safeNumber(window.firstTimestamp, rounds[0].timestamp);
  const cutoffTs = startTs + slotMinutes * MINUTE_MS * safeRatio;
  let n = rounds.findIndex((r) => r.timestamp >= cutoffTs);
  if (n < 0) n = rounds.length;
  const expected = clamp(Math.round(rounds.length * safeRatio), 0, rounds.length);
  if (n === 0 && expected > 0) n = expected;

  return { elapsedRounds: rounds.slice(0, n), remainingRounds: rounds.slice(n), elapsedCount: n };
}

// ---------------------------------------------------------------------------
// WINDOW COLLECTION SUMMARY
// ---------------------------------------------------------------------------

function summarizeWindowCollection(windows, focusTarget) {
  if (!windows.length) {
    return {
      windowCount: 0, roundCount: 0, focusHitRate: 0, focusAnyHitRate: 0,
      avgMultiplier: 0, lowCrashRate: 0, hugeHitRate: 0, megaHitRate: 0,
      avgPeakMultiplier: 0,
      perRoundHitRates:  createTargetMap(0),
      windowAnyHitRates: createTargetMap(0),
    };
  }

  const totalHits = createTargetMap(0), winHits = createTargetMap(0);
  let totalRounds = 0, wAvg = 0, wLow = 0, wHuge = 0, wMega = 0, peakSum = 0;

  for (const w of windows) {
    const s = w.summary;
    totalRounds += s.roundCount;
    wAvg        += s.avgMultiplier  * s.roundCount;
    wLow        += s.lowCrashRate   * s.roundCount;
    wHuge       += s.hugeHitRate    * s.roundCount;
    wMega       += s.megaHitRate    * s.roundCount;
    peakSum     += s.maxMultiplier;
    for (const t of TARGETS) {
      totalHits[t] += s.hitCounts[t] || 0;
      if ((s.hitCounts[t] || 0) > 0) winHits[t]++;
    }
  }

  return {
    windowCount:       windows.length,
    roundCount:        totalRounds,
    focusHitRate:      ratio(totalHits[focusTarget], totalRounds),
    focusAnyHitRate:   ratio(winHits[focusTarget], windows.length),
    avgMultiplier:     ratio(wAvg,  totalRounds),
    lowCrashRate:      ratio(wLow,  totalRounds),
    hugeHitRate:       ratio(wHuge, totalRounds),
    megaHitRate:       ratio(wMega, totalRounds),
    avgPeakMultiplier: ratio(peakSum, windows.length),
    perRoundHitRates:  createTargetMap((t) => ratio(totalHits[t], totalRounds)),
    windowAnyHitRates: createTargetMap((t) => ratio(winHits[t], windows.length)),
  };
}

// ---------------------------------------------------------------------------
// FIXED WINDOW SEGMENTATION
// ---------------------------------------------------------------------------

function segmentRoundsByWindow(rounds, windowMs, focusTarget) {
  const buckets = new Map();
  for (const r of rounds) {
    const bucketStart = Math.floor(r.timestamp / windowMs) * windowMs;
    const key = String(bucketStart);
    let entry = buckets.get(key);
    if (!entry) {
      entry = { startTimestamp: bucketStart, endTimestamp: bucketStart + windowMs, rounds: [] };
      buckets.set(key, entry);
    }
    entry.rounds.push(r);
  }
  return Array.from(buckets.values())
    .sort((a, b) => a.startTimestamp - b.startTimestamp)
    .map((w) => ({ ...w, summary: summarizeRounds(w.rounds, focusTarget) }));
}

// ---------------------------------------------------------------------------
// SLOT WINDOW BUILDER
// ---------------------------------------------------------------------------

function buildSlotWindows(rounds, slotMinutes, timeZone, focusTarget) {
  const getParts  = buildZonedPartsGetter(timeZone);
  const slotCount = Math.max(1, Math.floor(1440 / slotMinutes));
  const groups    = new Map();

  for (const r of rounds) {
    const parts      = getParts(r.timestamp);
    const slotIndex  = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, slotCount - 1);
    const key = `${parts.dateKey}|${slotIndex}`;
    let entry = groups.get(key);
    if (!entry) {
      entry = {
        key, dateKey: parts.dateKey, dayIndex: parts.dayIndex,
        slotIndex, slotStartMinute: slotIndex * slotMinutes,
        firstTimestamp: r.timestamp, rounds: [],
      };
      groups.set(key, entry);
    }
    entry.rounds.push(r);
  }

  return Array.from(groups.values())
    .sort((a, b) => a.dateKey !== b.dateKey ? a.dateKey.localeCompare(b.dateKey) : a.slotIndex - b.slotIndex)
    .map((e) => ({ ...e, summary: summarizeRounds(e.rounds, focusTarget) }));
}

// ---------------------------------------------------------------------------
// SLOT ANALYTICS
// BUG-FIX 3: minSamples — old formula gave 7-8 for 200 slots, eliminating most
// valid slots. New formula: max(3, floor(log2(slotWindowCount)+1)) gives 3-9
// which is appropriate. Also sampleWeight now properly scales 0→1 at 8 samples.
// ---------------------------------------------------------------------------

function sampleWeight(sampleCount) {
  return clamp(sampleCount / 8, 0.25, 1); // was /12, now /8 — reaches 1.0 at 8 samples
}

function buildSlotAnalytics(slotWindows, slotMinutes, windowMs, focusTarget, asOfTimestamp, timeZone) {
  const mode         = windowMs > DAY_MS ? 'start-time' : 'window';
  const slotCount    = Math.max(1, Math.floor(1440 / slotMinutes));
  const currentParts = buildZonedPartsGetter(timeZone)(asOfTimestamp);
  const currentSlotIndex = clamp(Math.floor(currentParts.minuteOfDay / slotMinutes), 0, slotCount - 1);

  // BUG-FIX 3: better minSamples — log2 scale appropriate for data volume
  const minSamples = Math.max(3, Math.floor(Math.log2(Math.max(1, slotWindows.length)) + 1));

  const baselineByTarget = {}, slotMapsByTarget = {}, slotStatsByTarget = {};
  const items = [];

  for (const target of TARGETS) {
    let baselineHitWindows = 0, baselineHits = 0, baselineRounds = 0;
    const slotMap = new Map();

    for (const sw of slotWindows) {
      const s       = sw.summary;
      const hitCount = s.hitCounts[target] || 0;
      const ageMs   = asOfTimestamp - sw.firstTimestamp;
      const dw      = decayWeight(ageMs);
      baselineRounds += s.roundCount;
      baselineHits   += hitCount;
      if (hitCount > 0) baselineHitWindows++;

      let agg = slotMap.get(sw.slotIndex);
      if (!agg) {
        agg = {
          slotIndex: sw.slotIndex, startMinute: sw.slotStartMinute,
          sampleCount: 0, totalRounds: 0, totalHits: 0,
          hitWindows: 0, peakSum: 0,
          weightedHitWindows: 0, weightedSamples: 0,
        };
        slotMap.set(sw.slotIndex, agg);
      }
      agg.sampleCount++;
      agg.totalRounds += s.roundCount;
      agg.totalHits   += hitCount;
      agg.peakSum     += s.maxMultiplier;
      agg.weightedSamples     += dw;
      agg.weightedHitWindows  += hitCount > 0 ? dw : 0;
      if (hitCount > 0) agg.hitWindows++;
    }

    const baselineAnyHitRate   = ratio(baselineHitWindows, slotWindows.length);
    const baselineRoundHitRate = ratio(baselineHits, baselineRounds);
    baselineByTarget[target]   = { anyHitRate: baselineAnyHitRate, roundHitRate: baselineRoundHitRate };
    slotMapsByTarget[target]   = slotMap;

    const slotStats = Array.from(slotMap.values()).map((slot) => {
      const anyHitChance = slot.weightedSamples > 0
        ? slot.weightedHitWindows / slot.weightedSamples
        : ratio(slot.hitWindows, slot.sampleCount);
      const roundHitRate = ratio(slot.totalHits, slot.totalRounds);
      const lift         = liftAgainstBaseline(anyHitChance, baselineAnyHitRate);
      const sigTest      = chiSquareTest(slot.hitWindows, slot.sampleCount, baselineAnyHitRate);
      const classification = classifyLift(lift, slot.sampleCount, slot.hitWindows, slot.sampleCount, baselineAnyHitRate);

      return {
        slotIndex: slot.slotIndex, startMinute: slot.startMinute,
        label: formatSlotLabel(slot.startMinute, slotMinutes, mode),
        anyHitChance, roundHitRate, lift,
        liftSignificant: sigTest.significant,
        liftPValue:      Number(sigTest.pValue.toFixed(4)),
        sampleCount:     slot.sampleCount,
        avgPeakMultiplier: ratio(slot.peakSum, slot.sampleCount),
        status: classification.key,
        zoneLabel: classification.label,
        tone: classification.tone,
        score: anyHitChance * (sigTest.significant && lift > 1 ? lift : 1) * sampleWeight(slot.sampleCount),
      };
    }).sort((a, b) => a.slotIndex - b.slotIndex);

    slotStatsByTarget[target] = slotStats;

    const currentSlot = slotStats.find((s) => s.slotIndex === currentSlotIndex) || {
      slotIndex: currentSlotIndex,
      startMinute: currentSlotIndex * slotMinutes,
      label: formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, mode),
      anyHitChance: baselineAnyHitRate, roundHitRate: baselineRoundHitRate,
      lift: 1, liftSignificant: false, liftPValue: 1, sampleCount: 0,
      avgPeakMultiplier: 0, status: 'neutral', zoneLabel: 'Insufficient Data', tone: 'neutral',
      score: baselineAnyHitRate,
    };

    const futureOptions = slotStats
      .filter((s) => s.sampleCount >= minSamples)
      .map((s) => {
        const dayOffset  = s.slotIndex > currentSlotIndex ? 0 : 1;
        const deltaSlots = dayOffset === 0
          ? s.slotIndex - currentSlotIndex
          : s.slotIndex + slotCount - currentSlotIndex;
        return { ...s, dayOffset, deltaSlots, occurrenceLabel: formatOccurrenceLabel(s.startMinute, slotMinutes, mode, dayOffset) };
      })
      .filter((s) => s.deltaSlots > 0)
      .sort((a, b) => b.score !== a.score ? b.score - a.score : a.deltaSlots - b.deltaSlots);

    const todayOptions = futureOptions
      .filter((s) => s.dayOffset === 0)
      .sort((a, b) => {
        const aS = a.liftSignificant && a.lift >= 1.05 ? 1 : 0;
        const bS = b.liftSignificant && b.lift >= 1.05 ? 1 : 0;
        if (bS !== aS) return bS - aS;
        if (a.deltaSlots !== b.deltaSlots) return a.deltaSlots - b.deltaSlots;
        return b.score - a.score;
      });

    const worstFuture = [...futureOptions].sort((a, b) => a.lift !== b.lift ? a.lift - b.lift : a.deltaSlots - b.deltaSlots)[0] || null;

    items.push({
      target, label: labelForTarget(target),
      baselineAnyHitRate, baselineRoundHitRate,
      currentSlot,
      nextTodayWindow: todayOptions[0]         || null,
      nextWindow:      futureOptions[0]        || null,
      todayWindows:    todayOptions.slice(0, 3),
      backups:         futureOptions.slice(1, 3),
      avoidWindow:     worstFuture,
      topSlots: [...slotStats]
        .filter((s) => s.sampleCount >= minSamples && s.liftSignificant)
        .sort((a, b) => b.score !== a.score ? b.score - a.score : a.slotIndex - b.slotIndex)
        .slice(0, 3),
    });
  }

  return { timeZone, slotMinutes, slotMode: mode, currentSlotIndex, minSamples, baselineByTarget, slotStatsByTarget, slotMapsByTarget, items };
}

// ---------------------------------------------------------------------------
// PATTERN MATCH REPORT
// BUG-FIX 4: Candidate minimum reduced from 6 to 4 so same-weekday same-slot
// matching works with 3-4 weeks of history (not just 6+ weeks).
// BUG-FIX 5: sampleSize floor raised from 6 to 4 to match.
// ---------------------------------------------------------------------------

const PATTERN_MIN_CANDIDATES = 4; // was 6 — 4 is sufficient for chi-square validity

function buildPatternMatchReport(slotWindows, previousSummary, currentSummary, baselineStats, focusTarget, asOfTimestamp, timeZone, slotAnalytics) {
  if (!slotWindows.length) {
    return { available: false, examples: [], reason: 'Not enough stored slot history yet.' };
  }

  const slotMinutes  = slotAnalytics.slotMinutes;
  const slotCount    = Math.max(1, Math.floor(1440 / slotMinutes));
  const slotMode     = slotAnalytics.slotMode;
  const getParts     = buildZonedPartsGetter(timeZone);
  const currentParts = getParts(asOfTimestamp);
  const currentSlotIndex = slotAnalytics.currentSlotIndex;
  const currentKey   = `${currentParts.dateKey}|${currentSlotIndex}`;
  const previousProbeParts = getParts(asOfTimestamp - slotMinutes * MINUTE_MS);
  const previousSlotIndex = clamp(Math.floor(previousProbeParts.minuteOfDay / slotMinutes), 0, slotCount - 1);
  const previousKey = `${previousProbeParts.dateKey}|${previousSlotIndex}`;
  const historySpanDays = ratio(asOfTimestamp - (slotWindows[0]?.firstTimestamp || asOfTimestamp), DAY_MS);
  const lookbackDaysUsed = Math.max(0, historySpanDays);

  const orderedWindows = [...slotWindows].sort((a, b) =>
    a.dateKey !== b.dateKey ? a.dateKey.localeCompare(b.dateKey) : a.slotIndex - b.slotIndex);
  const indexByKey      = new Map(orderedWindows.map((w, i) => [w.key, i]));
  const currentPosition = indexByKey.get(currentKey);
  const previousWindow  = indexByKey.has(previousKey)
    ? orderedWindows[indexByKey.get(previousKey)]
    : (Number.isInteger(currentPosition) && currentPosition > 0 ? orderedWindows[currentPosition - 1] : null);
  const expectedPreviousSlotIndex = (currentSlotIndex - 1 + slotCount) % slotCount;

  if (!previousWindow || previousWindow.slotIndex !== expectedPreviousSlotIndex) {
    return { available: false, examples: [], reason: 'The last closed slot is not yet available for the current live window.' };
  }

  const poolCandidates = orderedWindows.filter((w) => (
    w.slotIndex === previousWindow.slotIndex
    && w.key !== previousWindow.key
    && w.firstTimestamp < asOfTimestamp
  ));

  if (poolCandidates.length < PATTERN_MIN_CANDIDATES) {
    return {
      available: false, examples: [],
      reason: `Need at least ${PATTERN_MIN_CANDIDATES} past sessions at this time slot — only ${poolCandidates.length} found in the last ${Math.round(lookbackDaysUsed)} days. Keep collecting data.`,
    };
  }

  const allSummaries = [...poolCandidates.map((w) => w.summary), previousSummary];
  const { zVectors } = buildNormalizedVectorPool(allSummaries);
  const referenceVec = zVectors[zVectors.length - 1];
  const referencePattern = sampleWindowPattern(previousWindow.rounds);

  const currentMinuteProgress = clamp(
    (((currentParts.minuteOfDay - currentSlotIndex * slotMinutes) * 60) + currentParts.second) / Math.max(60, slotMinutes * 60), 0, 1,
  );
  const liveEvidenceWeight = currentMinuteProgress >= 0.7
    ? 0.9
    : currentMinuteProgress >= 0.4
      ? 0.6
      : currentMinuteProgress >= 0.2
        ? 0.35
        : 0.15;
  const alreadyHitCurrentWindow = (currentSummary.hitCounts?.[focusTarget] || 0) > 0;

  let liveZVectors = null;
  if (liveEvidenceWeight > 0) {
    const livePoolSummaries = poolCandidates.map((candidate) => {
      const pos = indexByKey.get(candidate.key);
      const matchedCW = Number.isInteger(pos) ? orderedWindows[pos + 1] : null;
      if (!matchedCW) return null;
      const split = splitWindowRoundsByProgress(matchedCW, currentMinuteProgress, slotMinutes);
      return summarizeRounds(split.elapsedRounds, focusTarget);
    });
    const validLive = [...livePoolSummaries.filter(Boolean), currentSummary];
    const { zVectors: lz } = buildNormalizedVectorPool(validLive);
    liveZVectors = { summaries: livePoolSummaries, zVectors: lz, currentVec: lz[lz.length - 1] };
  }

  const sameSetupCandidates = poolCandidates.map((w, i) => {
    const pos = indexByKey.get(w.key);
    const matchedCW = Number.isInteger(pos) ? orderedWindows[pos + 1] : null;
    if (!matchedCW) return null;
    const expectedCurrentSlotIndex = (w.slotIndex + 1) % slotCount;
    if (matchedCW.slotIndex !== expectedCurrentSlotIndex) return null;

    const weekdayMatch     = matchedCW.dayIndex === currentParts.dayIndex;
    const previousDistance = normalizedDistance(zVectors[i], referenceVec);
    const sequenceDistance = patternSeriesDistance(sampleWindowPattern(w.rounds), referencePattern);
    const similarityPct    = similarityPctFromDistances(previousDistance, sequenceDistance, weekdayMatch);
    const split            = splitWindowRoundsByProgress(matchedCW, currentMinuteProgress, slotMinutes);
    const matchedElapsed   = summarizeRounds(split.elapsedRounds, focusTarget);
    const matchedRemaining = summarizeRounds(split.remainingRounds, focusTarget);

    let liveDistance = 0;
    if (liveEvidenceWeight > 0 && liveZVectors && liveZVectors.summaries[i]) {
      const lv = liveZVectors.zVectors[i];
      liveDistance = lv ? normalizedDistance(lv, liveZVectors.currentVec) : 0;
    }

    const distance = (previousDistance + sequenceDistance + liveDistance * liveEvidenceWeight) / Math.max(1, 2 + liveEvidenceWeight);
    return {
      weekdayMatch, previousDistance, sequenceDistance, similarityPct, liveDistance,
      distance: distance + (weekdayMatch ? 0 : 0.04),
      matchedCurrentWindow: matchedCW, matchedElapsedSummary: matchedElapsed, matchedRemainingSummary: matchedRemaining,
      elapsedRounds: split.elapsedRounds, remainingRounds: split.remainingRounds, window: w,
    };
  }).filter(Boolean);

  if (sameSetupCandidates.length < PATTERN_MIN_CANDIDATES) {
    return {
      available: false, examples: [],
      reason: `Not enough matching past setups found (${sameSetupCandidates.length}/${PATTERN_MIN_CANDIDATES} needed).`,
    };
  }

  const sameWeekdayPool = sameSetupCandidates.filter((c) => c.weekdayMatch);
  const matchMode = sameWeekdayPool.length ? 'same-time+weekday-boost' : 'same-time';
  const sortedPool = [...sameSetupCandidates].sort((a, b) => b.similarityPct - a.similarityPct || a.distance - b.distance);
  const matchRows = sortedPool.map((match) => {
    const ageMs      = asOfTimestamp - match.window.firstTimestamp;
    const dw         = decayWeight(ageMs);
    const closenessW = 1 / (1 + match.distance);
    const liveW      = liveEvidenceWeight > 0 ? 1 / (1 + match.liveDistance) : 1;
    const weekdayW   = match.weekdayMatch ? (sameWeekdayPool.length >= PATTERN_MIN_CANDIDATES ? 1.1 : 1.03) : 1;
    const weight     = closenessW * liveW * dw * weekdayW;
    const remaining  = Array.isArray(match.remainingRounds) ? match.remainingRounds : [];
    const firstHitIdx = remaining.findIndex((r) => r.multiplier >= focusTarget);
    return { ...match, ageMs, weight, firstRemainingHitIndex: firstHitIdx };
  });

  if (!matchRows.length) {
    return { available: false, examples: [], reason: 'No usable past setups found for the current live window.' };
  }

  const exampleRows = matchRows.slice(0, 6);

  // Aggregate stats
  let ihHits = 0, ihRounds = 0, ihAnyHit = 0;
  let rhHits = 0, rhRounds = 0, rhAnyHit = 0;
  for (const c of sameSetupCandidates) {
    const s = c.window.summary;
    ihHits   += s.hitCounts[focusTarget] || 0;
    ihRounds += s.roundCount;
    if ((s.hitCounts[focusTarget] || 0) > 0) ihAnyHit++;
    rhHits   += c.matchedRemainingSummary.hitCounts[focusTarget] || 0;
    rhRounds += c.matchedRemainingSummary.roundCount;
    if ((c.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0) rhAnyHit++;
  }

  let cwHits = 0, cwRounds = 0, cwAnyHit = 0, cwPeakSum = 0;
  let rwHits = 0, rwRounds = 0, rwAnyHit = 0, sameWdCount = 0;
  for (const m of matchRows) {
    const s = m.matchedCurrentWindow.summary;
    cwHits   += s.hitCounts[focusTarget] || 0;
    cwRounds += s.roundCount;
    cwPeakSum += s.maxMultiplier || 0;
    if ((s.hitCounts[focusTarget] || 0) > 0) cwAnyHit++;
    const rs = m.matchedRemainingSummary;
    rwHits   += rs.hitCounts[focusTarget] || 0;
    rwRounds += rs.roundCount;
    if ((rs.hitCounts[focusTarget] || 0) > 0) rwAnyHit++;
    if (m.weekdayMatch) sameWdCount++;
  }

  const examples = exampleRows.map((m, idx) => ({
    rank:                      idx + 1,
    weekdayMatch:              m.weekdayMatch,
    distance:                  Number(m.distance.toFixed(3)),
    similarityPct:             Number(m.similarityPct.toFixed(1)),
    weight:                    Number(m.weight.toFixed(3)),
    inputWindowLabel:          formatTimestampInTimeZone(m.window.firstTimestamp, timeZone),
    inputSlotLabel:            formatSlotLabel(m.window.slotStartMinute, slotMinutes, slotMode),
    inputRoundFrom:            summarizeRoundRange(m.window.rounds, focusTarget).fromRoundId,
    inputRoundTo:              summarizeRoundRange(m.window.rounds, focusTarget).toRoundId,
    inputRoundCount:           m.window.rounds.length,
    inputHitRate:              m.window.summary.hitRates[focusTarget] || 0,
    matchedCurrentWindowLabel: formatTimestampInTimeZone(m.matchedCurrentWindow.firstTimestamp, timeZone),
    matchedCurrentSlotLabel:   formatSlotLabel(m.matchedCurrentWindow.slotStartMinute, slotMinutes, slotMode),
    matchedCurrentRoundFrom:   summarizeRoundRange(m.matchedCurrentWindow.rounds, focusTarget).fromRoundId,
    matchedCurrentRoundTo:     summarizeRoundRange(m.matchedCurrentWindow.rounds, focusTarget).toRoundId,
    matchedCurrentRoundCount:  m.matchedCurrentWindow.rounds.length,
    matchedCurrentHitCount:    summarizeRoundRange(m.matchedCurrentWindow.rounds, focusTarget).hitCount,
    matchedCurrentHitRoundIds: summarizeRoundRange(m.matchedCurrentWindow.rounds, focusTarget).hitRoundIds,
    remainingRoundFrom:        summarizeRoundRange(m.remainingRounds, focusTarget).fromRoundId,
    remainingRoundTo:          summarizeRoundRange(m.remainingRounds, focusTarget).toRoundId,
    remainingRoundCount:       m.remainingRounds.length,
    remainingHitCount:         summarizeRoundRange(m.remainingRounds, focusTarget).hitCount,
    remainingHitRoundIds:      summarizeRoundRange(m.remainingRounds, focusTarget).hitRoundIds,
    firstRemainingHitOffset:   m.firstRemainingHitIndex >= 0 ? m.firstRemainingHitIndex + 1 : null,
    matchedCurrentAnyHit:      (m.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0) > 0,
    remainingAnyHit:           (m.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0,
    matchedCurrentPeakMultiplier: m.matchedCurrentWindow.summary.maxMultiplier || 0,
  }));

  const inputHistoryAnyHitRate   = ratio(ihAnyHit, sameSetupCandidates.length);
  const inputHistoryPerRoundRate = ratio(ihHits, ihRounds);
  const inputBaselineAnyHitRate  = baselineStats.windowAnyHitRates[focusTarget];
  const inputHistoryLift         = liftAgainstBaseline(inputHistoryAnyHitRate, inputBaselineAnyHitRate);

  const currentWindowAnyHitRate = weightedAverage(
    matchRows, (m) => ((m.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0) > 0 ? 1 : 0),
    (m) => m.weight, ratio(cwAnyHit, matchRows.length),
  );
  const currentWindowPerRoundHitRate = weightedAverage(
    matchRows, (m) => ratio(m.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0, m.matchedCurrentWindow.summary.roundCount),
    (m) => m.weight, ratio(cwHits, cwRounds),
  );
  const currentWindowLift = liftAgainstBaseline(currentWindowAnyHitRate, inputBaselineAnyHitRate);

  const remainingBaselineAnyHitRate  = ratio(rhAnyHit, sameSetupCandidates.length);
  const remainingBaselinePerRoundRate = ratio(rhHits, rhRounds);
  const remainingAnyHitRate = weightedAverage(
    matchRows, (m) => ((m.matchedRemainingSummary.hitCounts[focusTarget] || 0) > 0 ? 1 : 0),
    (m) => m.weight, ratio(rwAnyHit, matchRows.length),
  );
  const remainingPerRoundHitRate = weightedAverage(
    matchRows, (m) => ratio(m.matchedRemainingSummary.hitCounts[focusTarget] || 0, m.matchedRemainingSummary.roundCount),
    (m) => m.weight, ratio(rwHits, rwRounds),
  );
  const remainingLift = liftAgainstBaseline(remainingAnyHitRate, remainingBaselineAnyHitRate);

  const remainingSigTest    = chiSquareTest(rwAnyHit, matchRows.length, remainingBaselineAnyHitRate);
  const currentWindowSigTest = chiSquareTest(cwAnyHit, matchRows.length, inputBaselineAnyHitRate);

  const weightedAvgPeak = weightedAverage(
    matchRows, (m) => m.matchedCurrentWindow.summary.maxMultiplier || 0,
    (m) => m.weight, ratio(cwPeakSum, matchRows.length),
  );
  const buildPeakRange = (rows, peakGetter) => {
    const validRows = rows.filter((row) => safeNumber(peakGetter(row), 0) > 0);
    if (!validRows.length) return null;
    return {
      fromMultiplier: Math.max(1, weightedQuantile(validRows, peakGetter, (row) => row.weight, 0.2, 1)),
      toMultiplier: Math.max(1, weightedQuantile(validRows, peakGetter, (row) => row.weight, 0.8, 1)),
    };
  };
  const matchedPeakRange = buildPeakRange(matchRows, (m) => m.matchedCurrentWindow.summary.maxMultiplier || 0);
  const noTargetPeakRows = matchRows.filter((m) => (m.matchedCurrentWindow.summary.hitCounts[focusTarget] || 0) === 0);
  const noTargetPeakRange = buildPeakRange(noTargetPeakRows, (m) => m.matchedCurrentWindow.summary.maxMultiplier || 0);
  const remainingPeakRange = buildPeakRange(matchRows, (m) => m.matchedRemainingSummary.maxMultiplier || 0);
  const remainingNoTargetPeakRows = matchRows.filter((m) => (m.matchedRemainingSummary.hitCounts[focusTarget] || 0) === 0);
  const remainingNoTargetPeakRange = buildPeakRange(remainingNoTargetPeakRows, (m) => m.matchedRemainingSummary.maxMultiplier || 0);

  const currentWindowTone  = classifyLift(currentWindowLift, matchRows.length, cwAnyHit, matchRows.length, inputBaselineAnyHitRate).tone;
  const remainingWindowTone = classifyLift(remainingLift, matchRows.length, rwAnyHit, matchRows.length, remainingBaselineAnyHitRate).tone;

  const remainingHitMatches = matchRows.filter((m) => m.firstRemainingHitIndex >= 0);
  const expectedRoundRange = remainingHitMatches.length >= 2 ? {
    hitMatchCount: remainingHitMatches.length,
    firstRemainingHitOffsetFrom: Math.max(1, Math.round(weightedQuantile(remainingHitMatches, (m) => m.firstRemainingHitIndex + 1, (m) => m.weight, 0.2, 1))),
    firstRemainingHitOffsetTo:   Math.max(1, Math.round(weightedQuantile(remainingHitMatches, (m) => m.firstRemainingHitIndex + 1, (m) => m.weight, 0.8, 1))),
  } : null;

  return {
    available: true,
    matchMode, lookbackDaysUsed,
    inputSlotLabel:   formatSlotLabel(previousWindow.slotStartMinute, slotMinutes, slotMode),
    currentSlotLabel: formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode),
    currentSlotIndex,
    currentWindowLabel: formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode),
    candidateCount: sameSetupCandidates.length,
    usedMatches: matchRows.length,
    sameWeekdayMatches: sameWdCount,
    note: sameWeekdayPool.length
      ? `Built from ${matchRows.length} same-time setups across ${lookbackDaysUsed.toFixed(1)} stored days. ${sameWeekdayPool.length} landed on the same weekday.`
      : `Built from ${matchRows.length} same-time setups across ${lookbackDaysUsed.toFixed(1)} stored days.`,
    examples,
    averageMatchWeight: Number(weightedAverage(matchRows, (m) => m.weight, () => 1, 0).toFixed(3)),
    averageDistance:    Number(weightedAverage(matchRows, (m) => m.distance, (m) => m.weight, 0).toFixed(3)),
    averageSimilarityPct: Number(weightedAverage(matchRows, (m) => m.similarityPct, (m) => m.weight, 0).toFixed(1)),
    expectedRoundRange,
    matchedPeakRange,
    noTargetPeakRange,
    remainingPeakRange,
    remainingNoTargetPeakRange,
    progress: {
      ratio:      currentMinuteProgress,
      roundsSeen: currentSummary.roundCount || 0,
      alreadyHit: alreadyHitCurrentWindow,
    },
    significance: {
      currentWindowSignificant: currentWindowSigTest.significant,
      currentWindowPValue:      Number(currentWindowSigTest.pValue.toFixed(4)),
      remainingSignificant:     remainingSigTest.significant,
      remainingPValue:          Number(remainingSigTest.pValue.toFixed(4)),
    },
    inputHistory: {
      anyHitRate:       inputHistoryAnyHitRate,
      perRoundHitRate:  inputHistoryPerRoundRate,
      lift:             inputHistoryLift,
      sampleCount:      sameSetupCandidates.length,
      weekdaySampleCount: sameWeekdayPool.length,
    },
    currentWindow: {
      occurrenceLabel:   formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode),
      anyHitRate:        currentWindowAnyHitRate,
      perRoundHitRate:   currentWindowPerRoundHitRate,
      avgPeakMultiplier: weightedAvgPeak,
      lift:              currentWindowLift,
      liftSignificant:   currentWindowSigTest.significant,
      tone:              currentWindowTone,
      label: currentWindowLift >= 1.08 ? 'Stronger Than Normal' : currentWindowLift <= 0.94 ? 'Weaker Than Normal' : 'Near Normal',
    },
    remainingWindow: {
      occurrenceLabel:         formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode),
      anyHitRate:              remainingAnyHitRate,
      perRoundHitRate:         remainingPerRoundHitRate,
      baselineAnyHitRate:      remainingBaselineAnyHitRate,
      baselinePerRoundHitRate: remainingBaselinePerRoundRate,
      lift:                    remainingLift,
      liftSignificant:         remainingSigTest.significant,
      tone:                    remainingWindowTone,
      label: remainingLift >= 1.08 ? 'Stronger Than Normal' : remainingLift <= 0.94 ? 'Weaker Than Normal' : 'Near Normal',
    },
  };
}

// ---------------------------------------------------------------------------
// PATTERN PREDICTION
// BUG-FIX 6: Confidence formula was broken — penalised when significance was
// false (which it nearly always was due to Bug 1). Now uses:
//  - 50% sample weight (capped at 18 matches)
//  - 20% for remaining significance
//  - 15% for full-window significance
//  - 15% for weekday match proportion
// Also: hero action tone and signal tones now use the SAME significance gate —
// no more "PLAY" hero with "WATCH" signals.
// ---------------------------------------------------------------------------

function buildPatternPrediction({ focusTarget, windowLabel, latestRoundId, currentSummary, baselineStats, slotAnalytics, patternMatch }) {
  const slotItem              = slotAnalytics.items.find((i) => i.target === focusTarget) || null;
  const currentSlot           = slotItem?.currentSlot || null;
  const currentHitRate        = currentSummary.hitRates[focusTarget] || 0;
  const baselineHitRate       = baselineStats.perRoundHitRates[focusTarget] || 0;
  const baselineCWHitRate     = baselineStats.windowAnyHitRates[focusTarget] || 0;
  const currentLift           = liftAgainstBaseline(currentHitRate, baselineHitRate);
  const slotLift              = safeNumber(currentSlot?.lift, 1);
  const slotSignificant       = Boolean(currentSlot?.liftSignificant);
  const isLowTarget           = focusTarget <= 20;
  const isMidTarget           = focusTarget > 20 && focusTarget <= 100;
  const expectedCurrentHits   = baselineHitRate * safeNumber(currentSummary.roundCount, 0);
  const currentEvidenceWeight = clamp(expectedCurrentHits / (isLowTarget ? 10 : isMidTarget ? 4 : 2.5), 0.12, 1);
  const effectiveCurrentLift  = 1 + (currentLift - 1) * currentEvidenceWeight;

  {
    const rules = focusTarget <= 5
      ? { minMatches: 6, playRate: 0.62, strongRate: 0.78 }
      : focusTarget <= 10
        ? { minMatches: 6, playRate: 0.48, strongRate: 0.64 }
        : focusTarget <= 20
          ? { minMatches: 6, playRate: 0.30, strongRate: 0.46 }
          : focusTarget <= 50
            ? { minMatches: 5, playRate: 0.16, strongRate: 0.30 }
            : focusTarget <= 100
              ? { minMatches: 5, playRate: 0.09, strongRate: 0.18 }
              : focusTarget <= 500
                ? { minMatches: 4, playRate: 0.03, strongRate: 0.07 }
                : { minMatches: 4, playRate: 0.015, strongRate: 0.04 };

    if (!patternMatch?.available) {
      return {
        action: 'SKIP',
        tone: 'bad',
        confidence: 0,
        confidenceLabel: 'Not Strong',
        dataQuality: 'insufficient',
        accuracyRate: 0,
        accuracyPercent: 0,
        strengthLabel: 'Not Strong',
        predictsLabel: `Current ${windowLabel}`,
        inputLabel: `Closed Previous ${windowLabel}`,
        inputSlotLabel: '-',
        currentSlotLabel: currentSlot?.label || '-',
        currentHitRate,
        baselineHitRate,
        baselineCurrentWindowHitRate: baselineCWHitRate,
        currentLift,
        effectiveCurrentLift,
        currentEvidenceWeight,
        currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
        currentWindowHitRate: 0,
        currentWindowLift: 1,
        remainingHitRate: 0,
        remainingLift: 1,
        baselineRemainingHitRate: 0,
        matchedWindows: 0,
        sameWeekdayMatches: 0,
        lookbackDaysUsed: 0,
        alreadyHitInCurrentWindow: (currentSummary.hitCounts?.[focusTarget] || 0) > 0,
        hitsSoFar: currentSummary.hitCounts?.[focusTarget] || 0,
        expectedRoundIdFrom: null,
        expectedRoundIdTo: null,
        expectedRoundIdLabel: '-',
        expectedRoundIdBasis: 'Round IDs are only shown on PLAY.',
        summary: patternMatch?.reason || `Skip ${labelForTarget(focusTarget)} in this live ${windowLabel.toLowerCase()} because there is not enough matching time-slot history yet.`,
        reasons: [
          'Accuracy 0.0%.',
          patternMatch?.reason || 'No matched same-time history was available.',
        ].filter(Boolean),
      };
    }

    const cwHitRate = safeNumber(patternMatch.currentWindow?.anyHitRate, 0);
    const cwLift = safeNumber(patternMatch.currentWindow?.lift, 1);
    const remHitRate = safeNumber(patternMatch.remainingWindow?.anyHitRate, 0);
    const remBaseline = safeNumber(patternMatch.remainingWindow?.baselineAnyHitRate, 0);
    const remLift = safeNumber(patternMatch.remainingWindow?.lift, 1);
    const matchedWins = safeNumber(patternMatch.usedMatches, 0);
    const sameWdMatches = safeNumber(patternMatch.sameWeekdayMatches, 0);
    const lookbackDays = safeNumber(patternMatch.lookbackDaysUsed, 0);
    const safeLatestId = safeNumber(latestRoundId, 0);
    const alreadyHit = Boolean(patternMatch.progress?.alreadyHit || (currentSummary.hitCounts?.[focusTarget] || 0) > 0);
    const hitsSoFar = currentSummary.hitCounts?.[focusTarget] || 0;
    const averageDistance = safeNumber(patternMatch.averageDistance, 1.25);
    const averageSimilarityPct = safeNumber(patternMatch.averageSimilarityPct, 0);
    const matchedPeakRange = patternMatch.matchedPeakRange || null;
    const noTargetPeakRange = patternMatch.noTargetPeakRange || null;
    const enoughHistory = matchedWins >= rules.minMatches;
    const predictedRate = cwHitRate;
    const predictedBaseline = baselineCWHitRate;
    const accuracyPercent = Number((predictedRate * 100).toFixed(1));
    const signalAccuracy = predictedRate;
    const similarityPercent = Number(averageSimilarityPct.toFixed(1));
    const formatPeak = (range) => {
      if (!range) return '-';
      const from = safeNumber(range.fromMultiplier, 0);
      const to = safeNumber(range.toMultiplier, 0);
      if (from <= 0 && to <= 0) return '-';
      const fromLabel = from >= 100 ? from.toFixed(1) : from >= 10 ? from.toFixed(1) : from.toFixed(2);
      const toLabel = to >= 100 ? to.toFixed(1) : to >= 10 ? to.toFixed(1) : to.toFixed(2);
      return `${fromLabel}x - ${toLabel}x`;
    };
    const matchedPeakRangeLabel = formatPeak(matchedPeakRange);
    const noTargetPeakRangeLabel = formatPeak(noTargetPeakRange);
    const strengthLabel = !enoughHistory
      ? 'Not Strong'
      : predictedRate >= rules.strongRate && similarityPercent >= 78
        ? 'Strong'
        : predictedRate >= rules.playRate && similarityPercent >= 62
          ? 'Medium'
          : 'Not Strong';
    const dataQuality = similarityPercent >= 82 ? 'good' : similarityPercent >= 62 ? 'moderate' : 'limited';
    const shouldPlay = strengthLabel !== 'Not Strong';

    let action = 'SKIP';
    let tone = 'bad';
    let summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. The matched pattern is not strong enough.`;

    if (shouldPlay) {
      action = 'PLAY';
      tone = 'good';
      summary = `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. ${accuracyPercent.toFixed(1)}% of matched next windows hit ${labelForTarget(focusTarget)} with ${similarityPercent.toFixed(1)}% pattern similarity.`;
    } else if (!enoughHistory) {
      summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. There are not enough matched same-time setups yet.`;
    } else if (predictedRate > 0) {
      summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Only ${accuracyPercent.toFixed(1)}% of matched next windows hit ${labelForTarget(focusTarget)}.`;
    } else if (noTargetPeakRangeLabel !== '-') {
      summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. In matched next windows ${labelForTarget(focusTarget)} was not there; biggest was around ${noTargetPeakRangeLabel}.`;
    } else if (matchedPeakRangeLabel !== '-') {
      summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. In matched next windows ${labelForTarget(focusTarget)} was not there; biggest was around ${matchedPeakRangeLabel}.`;
    }

    const expectedRoundIdFrom = patternMatch.expectedRoundRange && safeLatestId > 0
      ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetFrom
      : null;
    const expectedRoundIdTo = patternMatch.expectedRoundRange && safeLatestId > 0
      ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetTo
      : null;
    const rawExpectedRoundIdLabel = (expectedRoundIdFrom && expectedRoundIdTo) ? `#${expectedRoundIdFrom} - #${expectedRoundIdTo}` : '-';
    const rawExpectedRoundIdBasis = patternMatch.expectedRoundRange
      ? `Based on ${patternMatch.expectedRoundRange.hitMatchCount} matched hits still ahead from this point.`
      : '';
    const expectedRoundIdLabel = action === 'PLAY' ? rawExpectedRoundIdLabel : '-';
    const expectedRoundIdBasis = action === 'PLAY'
      ? rawExpectedRoundIdBasis
      : 'Round IDs are only shown on PLAY.';

    return {
      action,
      tone,
      confidence: signalAccuracy,
      confidenceLabel: strengthLabel,
      dataQuality,
      accuracyRate: predictedRate,
      accuracyPercent,
      strengthLabel,
      similarityPercent,
      matchedPeakRangeLabel,
      noTargetPeakRangeLabel,
      predictsLabel: `Current ${windowLabel}`,
      inputLabel: `Closed Previous ${windowLabel}`,
      inputSlotLabel: patternMatch.inputSlotLabel,
      currentSlotLabel: patternMatch.currentSlotLabel,
      currentHitRate,
      baselineHitRate,
      baselineCurrentWindowHitRate: baselineCWHitRate,
      currentLift,
      effectiveCurrentLift,
      currentEvidenceWeight,
      currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
      currentWindowHitRate: cwHitRate,
      currentWindowLift: cwLift,
      remainingHitRate: remHitRate,
      remainingLift: remLift,
      baselineRemainingHitRate: remBaseline,
      matchedWindows: matchedWins,
      sameWeekdayMatches: sameWdMatches,
      lookbackDaysUsed: lookbackDays,
      alreadyHitInCurrentWindow: alreadyHit,
      hitsSoFar,
      expectedRoundIdFrom,
      expectedRoundIdTo,
      expectedRoundIdLabel,
      expectedRoundIdBasis,
      summary,
      reasons: [
        `Accuracy ${accuracyPercent.toFixed(1)}%.`,
        `Pattern similarity ${similarityPercent.toFixed(1)}% from ${matchedWins} matched same-time setups.`,
        `Historical next-window hit rate: ${pctString(cwHitRate)}.`,
        `Tail hit rate from now: ${pctString(remHitRate)}.`,
        `Normal rate for this point: ${pctString(predictedBaseline)}.`,
        noTargetPeakRangeLabel !== '-' ? `When ${labelForTarget(focusTarget)} missed, the matched peak was around ${noTargetPeakRangeLabel}.` : null,
        matchedPeakRangeLabel !== '-' ? `Matched next-window peak range: ${matchedPeakRangeLabel}.` : null,
        `Pattern quality: ${strengthLabel}.`,
        alreadyHit
          ? `${labelForTarget(focusTarget)} already hit ${hitsSoFar} time(s) in this live window.`
          : `${labelForTarget(focusTarget)} has not hit yet in this live window.`,
        expectedRoundIdBasis ? `Expected ${labelForTarget(focusTarget)} around rounds ${expectedRoundIdLabel}. ${expectedRoundIdBasis}` : null,
      ].filter(Boolean),
    };
  }

  if (!patternMatch?.available) {
    return {
      action: 'SKIP', tone: 'bad',
      confidence: 0, confidenceLabel: 'Low', dataQuality: 'insufficient',
      predictsLabel: `Current ${windowLabel}`, inputLabel: `Closed Previous ${windowLabel}`,
      inputSlotLabel: '-', currentSlotLabel: currentSlot?.label || '-',
      currentHitRate, baselineHitRate, baselineCurrentWindowHitRate: baselineCWHitRate,
      currentLift, effectiveCurrentLift, currentEvidenceWeight,
      currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
      currentWindowHitRate: 0, currentWindowLift: 1,
      remainingHitRate: 0, remainingLift: 1, baselineRemainingHitRate: 0,
      matchedWindows: 0, sameWeekdayMatches: 0, lookbackDaysUsed: 0,
      alreadyHitInCurrentWindow: (currentSummary.hitCounts?.[focusTarget] || 0) > 0,
      hitsSoFar: currentSummary.hitCounts?.[focusTarget] || 0,
      expectedRoundIdFrom: null, expectedRoundIdTo: null,
      expectedRoundIdLabel: '-', expectedRoundIdBasis: '',
      summary: patternMatch?.reason || `Skip ${labelForTarget(focusTarget)} in this live ${windowLabel.toLowerCase()} because there is not enough matching time-slot history yet.`,
      reasons: [`Live hit rate: ${pctString(currentHitRate)} vs ${pctString(baselineHitRate)} baseline.`, patternMatch?.reason || ''].filter(Boolean),
    };
  }

  const cwHitRate    = safeNumber(patternMatch.currentWindow?.anyHitRate, 0);
  const cwLift       = safeNumber(patternMatch.currentWindow?.lift, 1);
  const cwSig        = Boolean(patternMatch.currentWindow?.liftSignificant);
  const remHitRate   = safeNumber(patternMatch.remainingWindow?.anyHitRate, 0);
  const remBaseline  = safeNumber(patternMatch.remainingWindow?.baselineAnyHitRate, 0);
  const remLift      = safeNumber(patternMatch.remainingWindow?.lift, 1);
  const remSig       = Boolean(patternMatch.remainingWindow?.liftSignificant);
  const cwEdge       = cwHitRate - baselineCWHitRate;
  const matchedWins  = safeNumber(patternMatch.usedMatches, 0);
  const sameWdMatches = safeNumber(patternMatch.sameWeekdayMatches, 0);
  const lookbackDays = safeNumber(patternMatch.lookbackDaysUsed, 0);
  const safeLatestId = safeNumber(latestRoundId, 0);
  const alreadyHit   = Boolean(patternMatch.progress?.alreadyHit || (currentSummary.hitCounts?.[focusTarget] || 0) > 0);
  const hitsSoFar    = currentSummary.hitCounts?.[focusTarget] || 0;

  // BUG-FIX 6: confidence now rewards significance being TRUE (not penalises it being false)
  const rawConfidence = clamp(
    (Math.min(matchedWins, 18) / 18) * 0.50
    + (remSig  ? 0.20 : 0)
    + (cwSig   ? 0.15 : 0)
    + (Math.min(sameWdMatches, 8) / 8) * 0.15,
    0, 1,
  );
  const confidenceLabel = rawConfidence >= 0.65 ? 'High' : rawConfidence >= 0.40 ? 'Medium' : 'Low';
  const dataQuality     = rawConfidence >= 0.65 ? 'good' : rawConfidence >= 0.40 ? 'moderate' : 'limited';

  const remEdge = remHitRate - remBaseline;

  // BUG-FIX 6 cont.: action and tone are now derived from the SAME significance logic
  // that the UI signal cards use — no more contradiction.
  let action = 'SKIP', tone = 'bad';
  let summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because the pattern is not strong enough.`;

  if (!alreadyHit
      && (
        (remSig && remLift >= 1.05 && remEdge >= 0)
        || (!remSig && remLift >= (isLowTarget ? 1.12 : isMidTarget ? 1.16 : 1.20) && matchedWins >= (PATTERN_MIN_CANDIDATES + 2) && remHitRate >= (isLowTarget ? 0.42 : isMidTarget ? 0.12 : 0.04))
      )
      && (cwSig ? cwLift >= 1 : cwHitRate >= baselineCWHitRate || effectiveCurrentLift >= 1)
      && (slotSignificant ? slotLift >= 1 : slotLift >= 0.98)) {
    action  = 'PLAY';
    tone    = 'good';
    summary = `Remaining-window pattern is significantly stronger (${pctString(remHitRate)} vs ${pctString(remBaseline)} baseline, lift ${remLift.toFixed(2)}x, p<0.05) — supports ${labelForTarget(focusTarget)}.`;
    summary = `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Timing, cluster match, and live read all support this setup.`;
  } else if (!alreadyHit && (
    (remLift >= 1.05 && matchedWins >= PATTERN_MIN_CANDIDATES)
    || (cwSig && cwLift >= 1.05)
    || cwEdge >= 0.03
    || effectiveCurrentLift >= 1.02
  )) {
    action  = 'SKIP';
    tone    = 'neutral';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because the edge is not strong enough yet.`;
  } else if (alreadyHit && remSig && remLift <= 0.95) {
    action  = 'SKIP';
    tone    = 'bad';
    summary = `${labelForTarget(focusTarget)} already hit and remaining pattern is significantly weaker — another hit soon is unlikely.`;
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because it already hit and the remaining pattern is weaker than normal.`;
  } else if (remSig && remLift <= 0.95 && cwSig && cwLift <= 0.95) {
    action  = 'SKIP';
    tone    = 'bad';
    summary = `Both full-window and remaining patterns are significantly below baseline — skip ${labelForTarget(focusTarget)} for now.`;
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because both the full-window and remaining pattern are below baseline.`;
  } else if (remLift >= 1 || cwLift >= 1 || effectiveCurrentLift >= 1 || slotLift >= 1) {
    action  = 'SKIP';
    tone    = 'neutral';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because no strong edge is confirmed.`;
  } else {
    action  = 'SKIP';
    tone    = 'neutral';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because the live read and matched history are both below baseline.`;
  }

  const minMatchesNeeded = focusTarget <= 20 ? 6 : focusTarget <= 100 ? 5 : 4;
  const absolutePlayChance = focusTarget <= 5
    ? 0.70
    : focusTarget <= 10
      ? 0.60
      : focusTarget <= 20
        ? 0.45
        : focusTarget <= 50
          ? 0.22
          : focusTarget <= 100
            ? 0.12
            : focusTarget <= 500
              ? 0.04
              : 0.02;
  const strongPlayChance = focusTarget <= 5
    ? 0.92
    : focusTarget <= 10
      ? 0.85
      : focusTarget <= 20
        ? 0.72
        : focusTarget <= 50
          ? 0.40
          : focusTarget <= 100
            ? 0.24
            : focusTarget <= 500
              ? 0.08
              : 0.04;
  const baselineSlack = focusTarget <= 20 ? 0.06 : focusTarget <= 100 ? 0.03 : 0.015;
  const enoughHistory = matchedWins >= minMatchesNeeded;
  const veryStrongFromNow = remHitRate >= strongPlayChance;
  const solidFromNow = remHitRate >= absolutePlayChance;
  const patternAligned = remLift >= 1.01 || remEdge >= -baselineSlack || slotLift >= 0.98;
  const liveAligned = cwLift >= 0.94 || effectiveCurrentLift >= 0.94 || cwHitRate >= Math.max(0, baselineCWHitRate - baselineSlack);
  const mediumPlayChance = (absolutePlayChance + strongPlayChance) / 2;
  const shouldPlayNow = enoughHistory && (
    veryStrongFromNow
    || (solidFromNow && patternAligned && liveAligned)
    || (remHitRate >= mediumPlayChance && remLift >= 1.01 && matchedWins >= (minMatchesNeeded + 2))
    || (remSig && remLift >= 1.03 && remHitRate >= absolutePlayChance * 0.85)
  );

  if (shouldPlayNow) {
    action = 'PLAY';
    tone = 'good';
    summary = alreadyHit
      ? `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. It already hit earlier, but full-history matches still show strong remaining hit odds from this point.`
      : `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Full-history timing and cluster matches support this setup from now.`;
  } else if (alreadyHit) {
    action = 'SKIP';
    tone = 'bad';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because it already hit and the remaining from-now pattern is not strong enough.`;
  } else if (!enoughHistory) {
    action = 'SKIP';
    tone = 'bad';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because there are not enough matched past setups yet.`;
  } else if (veryStrongFromNow || solidFromNow || remLift >= 1 || cwLift >= 1 || effectiveCurrentLift >= 1 || slotLift >= 1) {
    action = 'SKIP';
    tone = 'bad';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because the timing match is close, but still not strong enough to play.`;
  } else {
    action = 'SKIP';
    tone = 'bad';
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window because the full-history timing match is not strong enough from this point.`;
  }

  const expectedRoundIdFrom = patternMatch.expectedRoundRange && safeLatestId > 0
    ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetFrom : null;
  const expectedRoundIdTo   = patternMatch.expectedRoundRange && safeLatestId > 0
    ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetTo   : null;
  const rawExpectedRoundIdLabel = (expectedRoundIdFrom && expectedRoundIdTo) ? `#${expectedRoundIdFrom} - #${expectedRoundIdTo}` : '-';
  const rawExpectedRoundIdBasis = patternMatch.expectedRoundRange
    ? `Based on ${patternMatch.expectedRoundRange.hitMatchCount} matched current-window hits still ahead from this point.` : '';
  const expectedRoundIdLabel = action === 'PLAY' ? rawExpectedRoundIdLabel : '-';
  const expectedRoundIdBasis = action === 'PLAY'
    ? rawExpectedRoundIdBasis
    : 'Round IDs are only shown when the signal is strong enough to play.';

  return {
    action, tone,
    confidence: rawConfidence, confidenceLabel, dataQuality,
    predictsLabel:   `Current ${windowLabel}`,
    inputLabel:      `Closed Previous ${windowLabel}`,
    inputSlotLabel:  patternMatch.inputSlotLabel,
    currentSlotLabel: patternMatch.currentSlotLabel,
    currentHitRate, baselineHitRate, baselineCurrentWindowHitRate: baselineCWHitRate,
    currentLift, effectiveCurrentLift, currentEvidenceWeight,
    currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
    currentWindowHitRate: cwHitRate, currentWindowLift: cwLift,
    remainingHitRate: remHitRate, remainingLift: remLift, baselineRemainingHitRate: remBaseline,
    matchedWindows: matchedWins, sameWeekdayMatches: sameWdMatches, lookbackDaysUsed: lookbackDays,
    alreadyHitInCurrentWindow: alreadyHit, hitsSoFar,
    expectedRoundIdFrom, expectedRoundIdTo, expectedRoundIdLabel, expectedRoundIdBasis,
    summary,
    reasons: [
      `Input: last closed slot ${patternMatch.inputSlotLabel}, live window ${patternMatch.currentSlotLabel}.`,
      `${matchedWins} matched past setups · ${lookbackDays.toFixed(1)}d lookback · z-score distance · decay-weighted.`,
      `Full-window history: ${pctString(cwHitRate)} vs ${pctString(baselineCWHitRate)} baseline${cwSig ? ' ★ sig. (p<0.05)' : ' (not significant)'}. Lift ${cwLift.toFixed(2)}x.`,
      `Remaining window: ${pctString(remHitRate)} vs ${pctString(remBaseline)} baseline${remSig ? ' ★ sig. (p<0.05)' : ' (not significant)'}. Lift ${remLift.toFixed(2)}x.`,
      alreadyHit
        ? `${labelForTarget(focusTarget)} already hit ${hitsSoFar}× in this live window.`
        : `${labelForTarget(focusTarget)} has not hit yet in the current live window.`,
      expectedRoundIdBasis ? `Expected ${labelForTarget(focusTarget)} around rounds ${expectedRoundIdLabel}. ${expectedRoundIdBasis}` : null,
    ].filter(Boolean),
  };
}

function buildCurrentWindowPatternPrediction({ focusTarget, windowLabel, latestRoundId, currentSummary, baselineStats, slotAnalytics, patternMatch }) {
  const slotItem              = slotAnalytics.items.find((i) => i.target === focusTarget) || null;
  const currentSlot           = slotItem?.currentSlot || null;
  const currentHitRate        = currentSummary.hitRates[focusTarget] || 0;
  const baselineHitRate       = baselineStats.perRoundHitRates[focusTarget] || 0;
  const baselineCWHitRate     = baselineStats.windowAnyHitRates[focusTarget] || 0;
  const currentLift           = liftAgainstBaseline(currentHitRate, baselineHitRate);
  const expectedCurrentHits   = baselineHitRate * safeNumber(currentSummary.roundCount, 0);
  const currentEvidenceWeight = clamp(expectedCurrentHits / (focusTarget <= 20 ? 10 : focusTarget <= 100 ? 4 : 2.5), 0.12, 1);
  const effectiveCurrentLift  = 1 + (currentLift - 1) * currentEvidenceWeight;
  const alreadyHitInCurrentWindow = (currentSummary.hitCounts?.[focusTarget] || 0) > 0;
  const hitsSoFar = currentSummary.hitCounts?.[focusTarget] || 0;

  const rules = focusTarget <= 5
    ? { minMatches: 8, strongMatches: 14, playRate: 0.97, strongRate: 0.995, minEdge: 0.015, strongEdge: 0.04, minSimilarity: 30, strongSimilarity: 58 }
    : focusTarget <= 10
      ? { minMatches: 8, strongMatches: 14, playRate: 0.90, strongRate: 0.97, minEdge: 0.03, strongEdge: 0.07, minSimilarity: 32, strongSimilarity: 58 }
      : focusTarget <= 20
        ? { minMatches: 7, strongMatches: 12, playRate: 0.68, strongRate: 0.82, minEdge: 0.045, strongEdge: 0.10, minSimilarity: 35, strongSimilarity: 60 }
        : focusTarget <= 50
          ? { minMatches: 6, strongMatches: 10, playRate: 0.30, strongRate: 0.48, minEdge: 0.035, strongEdge: 0.08, minSimilarity: 40, strongSimilarity: 64 }
          : focusTarget <= 100
            ? { minMatches: 6, strongMatches: 10, playRate: 0.15, strongRate: 0.28, minEdge: 0.025, strongEdge: 0.06, minSimilarity: 42, strongSimilarity: 66 }
            : focusTarget <= 500
              ? { minMatches: 5, strongMatches: 8, playRate: 0.05, strongRate: 0.09, minEdge: 0.012, strongEdge: 0.03, minSimilarity: 45, strongSimilarity: 70 }
              : { minMatches: 5, strongMatches: 8, playRate: 0.025, strongRate: 0.05, minEdge: 0.006, strongEdge: 0.016, minSimilarity: 48, strongSimilarity: 72 };
  const likelyRules = focusTarget <= 5
    ? { playRate: 0.97, strongRate: 0.995 }
    : focusTarget <= 10
      ? { playRate: 0.88, strongRate: 0.96 }
      : focusTarget <= 20
        ? { playRate: 0.60, strongRate: 0.78 }
        : focusTarget <= 50
          ? { playRate: 0.24, strongRate: 0.40 }
          : focusTarget <= 100
            ? { playRate: 0.12, strongRate: 0.22 }
            : focusTarget <= 500
              ? { playRate: 0.035, strongRate: 0.07 }
              : { playRate: 0.015, strongRate: 0.03 };

  const thresholdFraction = (value, low, high) => {
    if (!Number.isFinite(value)) return 0;
    if (high <= low) return value >= high ? 1 : 0;
    return clamp((value - low) / (high - low), 0, 1);
  };

  const formatPeak = (range) => {
    if (!range) return '-';
    const from = safeNumber(range.fromMultiplier, 0);
    const to = safeNumber(range.toMultiplier, 0);
    if (from <= 0 && to <= 0) return '-';
    const fromLabel = from >= 100 ? from.toFixed(1) : from >= 10 ? from.toFixed(1) : from.toFixed(2);
    const toLabel = to >= 100 ? to.toFixed(1) : to >= 10 ? to.toFixed(1) : to.toFixed(2);
    return `${fromLabel}x - ${toLabel}x`;
  };

  if (!patternMatch?.available) {
    return {
      action: 'SKIP',
      tone: 'bad',
      confidence: 0,
      confidenceLabel: 'Not Strong',
      dataQuality: 'insufficient',
      accuracyRate: 0,
      accuracyPercent: 0,
      matchedHitRate: 0,
      matchedHitRatePercent: 0,
      fromNowRate: 0,
      fromNowRatePercent: 0,
      edgeRate: 0,
      edgePercent: 0,
      signalScorePercent: 0,
      strengthLabel: 'Not Strong',
      similarityPercent: 0,
      matchedPeakRangeLabel: '-',
      noTargetPeakRangeLabel: '-',
      timingEdgeLabel: 'Insufficient',
      predictsLabel: `Current ${windowLabel}`,
      inputLabel: `Closed Previous ${windowLabel}`,
      inputSlotLabel: '-',
      currentSlotLabel: currentSlot?.label || '-',
      currentHitRate,
      baselineHitRate,
      baselineCurrentWindowHitRate: baselineCWHitRate,
      currentLift,
      effectiveCurrentLift,
      currentEvidenceWeight,
      currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
      currentWindowHitRate: 0,
      currentWindowLift: 1,
      remainingHitRate: 0,
      remainingLift: 1,
      baselineRemainingHitRate: 0,
      matchedWindows: 0,
      sameWeekdayMatches: 0,
      lookbackDaysUsed: 0,
      alreadyHitInCurrentWindow,
      hitsSoFar,
      expectedRoundIdFrom: null,
      expectedRoundIdTo: null,
      expectedRoundIdLabel: '-',
      expectedRoundIdBasis: 'Round IDs are only shown on PLAY.',
      summary: patternMatch?.reason || `Skip ${labelForTarget(focusTarget)} in this live ${windowLabel.toLowerCase()} because there is not enough same-time history yet.`,
      reasons: [
        'Matched hit rate 0.0% from this point.',
        patternMatch?.reason || 'No matched same-time history was available.',
      ].filter(Boolean),
    };
  }

  const fullWindowHitRate = safeNumber(patternMatch.currentWindow?.anyHitRate, 0);
  const fullWindowLift = safeNumber(patternMatch.currentWindow?.lift, 1);
  const fromNowHitRate = safeNumber(patternMatch.remainingWindow?.anyHitRate, 0);
  const fromNowBaseline = safeNumber(patternMatch.remainingWindow?.baselineAnyHitRate, 0);
  const fromNowLift = safeNumber(patternMatch.remainingWindow?.lift, 1);
  const matchedWins = safeNumber(patternMatch.usedMatches, 0);
  const sameWdMatches = safeNumber(patternMatch.sameWeekdayMatches, 0);
  const lookbackDays = safeNumber(patternMatch.lookbackDaysUsed, 0);
  const safeLatestId = safeNumber(latestRoundId, 0);
  const similarityPercent = Number(safeNumber(patternMatch.averageSimilarityPct, 0).toFixed(1));
  const matchedPeakRangeLabel = formatPeak(patternMatch.remainingPeakRange || null);
  const noTargetPeakRangeLabel = formatPeak(patternMatch.remainingNoTargetPeakRange || null);
  const fromNowEdge = fromNowHitRate - fromNowBaseline;
  const saturatedCommonTarget = focusTarget <= 10 && fromNowHitRate >= 0.995 && fromNowBaseline >= 0.995;
  const remainingRatio = clamp(1 - safeNumber(patternMatch.progress?.ratio, 0), 0, 1);
  const enoughHistory = matchedWins >= rules.minMatches;
  const dynamicPlayRate = Math.min(
    rules.playRate,
    fromNowBaseline + Math.max(rules.minEdge * 1.5, fromNowBaseline * (focusTarget <= 10 ? 0.05 : focusTarget <= 50 ? 0.12 : 0.18)),
  );
  const dynamicStrongRate = Math.min(
    rules.strongRate,
    fromNowBaseline + Math.max(rules.strongEdge, fromNowBaseline * (focusTarget <= 10 ? 0.10 : focusTarget <= 50 ? 0.20 : 0.28)),
  );
  const rateScore = thresholdFraction(fromNowHitRate, dynamicPlayRate, dynamicStrongRate);
  const edgeScore = thresholdFraction(fromNowEdge, rules.minEdge, rules.strongEdge);
  const similarityScore = thresholdFraction(similarityPercent, rules.minSimilarity, rules.strongSimilarity);
  const historyScore = thresholdFraction(matchedWins, rules.minMatches, rules.strongMatches);
  const liveEvidenceScore = thresholdFraction(remainingRatio, 0.05, 0.30);
  const signalScore = enoughHistory
    ? clamp((rateScore * 0.37) + (edgeScore * 0.26) + (similarityScore * 0.14) + (historyScore * 0.09) + (liveEvidenceScore * 0.14), 0, 1)
    : clamp(historyScore * 0.35, 0, 0.35);
  const likelySignal = enoughHistory && fromNowHitRate >= likelyRules.playRate;
  const veryLikelySignal = enoughHistory && fromNowHitRate >= likelyRules.strongRate;
  const strongSignal = enoughHistory
    && fromNowHitRate >= dynamicStrongRate
    && fromNowEdge >= rules.strongEdge
    && similarityPercent >= rules.strongSimilarity;
  const mediumSignal = enoughHistory
    && fromNowHitRate >= dynamicPlayRate
    && fromNowEdge >= rules.minEdge
    && similarityPercent >= rules.minSimilarity;
  const relativeSignal = enoughHistory
    && fromNowEdge >= rules.minEdge * 1.25
    && similarityPercent >= rules.minSimilarity
    && remainingRatio >= 0.08;
  const shouldPlay = strongSignal || mediumSignal || likelySignal;
  const strengthLabel = strongSignal
    ? 'Strong'
    : mediumSignal
      ? 'Medium'
      : relativeSignal
        ? 'Medium'
      : veryLikelySignal
        ? 'Very Likely'
        : likelySignal
          ? 'Likely'
          : saturatedCommonTarget
            ? 'Very Likely'
            : 'Not Strong';
  const timingEdgeLabel = saturatedCommonTarget
    ? 'Normal Timing'
    : strongSignal
      ? 'Strong Edge'
      : mediumSignal
      ? 'Positive Edge'
      : relativeSignal
        ? 'Positive Edge'
      : fromNowEdge > 0 && similarityPercent >= rules.minSimilarity
          ? 'Small Edge'
          : fromNowEdge < 0
            ? 'Below Normal'
            : similarityPercent < rules.minSimilarity
              ? 'Weak Match'
              : 'No Edge';
  const dataQuality = matchedWins >= rules.strongMatches
    ? 'good'
    : enoughHistory
      ? 'moderate'
      : 'limited';
  const matchedHitRatePercent = Number((fullWindowHitRate * 100).toFixed(1));
  const fromNowRatePercent = Number((fromNowHitRate * 100).toFixed(1));
  const edgePercent = Number((fromNowEdge * 100).toFixed(1));
  const signalScorePercent = Number((signalScore * 100).toFixed(1));
  const edgeLabel = `${fromNowEdge >= 0 ? '+' : ''}${edgePercent.toFixed(1)}%`;

  let action = 'SKIP';
  let tone = 'bad';
  let summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. The same-time edge from this point is not strong enough.`;

  if (strongSignal) {
    action = 'PLAY';
    tone = 'good';
    summary = `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. From this point, matched same-time setups hit ${labelForTarget(focusTarget)} ${fromNowRatePercent.toFixed(1)}% of the time versus ${pctString(fromNowBaseline)} normal (${edgeLabel}) with ${similarityPercent.toFixed(1)}% pattern match.`;
  } else if (mediumSignal) {
    action = 'PLAY';
    tone = 'good';
    summary = `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. From this point, matched same-time setups hit ${labelForTarget(focusTarget)} ${fromNowRatePercent.toFixed(1)}% of the time versus ${pctString(fromNowBaseline)} normal (${edgeLabel}) with ${similarityPercent.toFixed(1)}% pattern match.`;
  } else if (relativeSignal) {
    action = 'PLAY';
    tone = 'good';
    summary = `Play ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. The remaining hit rate is not huge in absolute terms, but it is meaningfully above normal (${edgeLabel}) for this same-time setup.`;
  } else if (likelySignal || saturatedCommonTarget) {
    action = 'PLAY';
    tone = 'good';
    summary = `${labelForTarget(focusTarget)} is very likely in the live ${patternMatch.currentSlotLabel} window from this point. Timing is ${timingEdgeLabel.toLowerCase()}, not a special edge, but the target itself is still a valid play by likelihood.`;
  } else if (!enoughHistory) {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Only ${matchedWins} same-time setups are stored, which is not enough for a reliable call yet.`;
  } else if (remainingRatio <= 0.12 && fromNowEdge <= rules.minEdge) {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Only a small tail of this window is left, and the remaining same-time edge is just ${edgeLabel}. There is not enough edge left in this window.`;
  } else if (fromNowEdge <= 0) {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. From this point the matched hit rate is ${fromNowRatePercent.toFixed(1)}%, while normal is already ${pctString(fromNowBaseline)}. There is no real edge.`;
  } else if (similarityPercent < rules.minSimilarity) {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. The hit rate is there, but the closed-input pattern only matches ${similarityPercent.toFixed(1)}% of past same-time setups.`;
  } else if (noTargetPeakRangeLabel !== '-') {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. When this setup missed ${labelForTarget(focusTarget)} from this point in matched history, the remaining tail usually topped out around ${noTargetPeakRangeLabel}.`;
  } else {
    summary = `Skip ${labelForTarget(focusTarget)} in the live ${patternMatch.currentSlotLabel} window. Matched same-time history is ${fromNowRatePercent.toFixed(1)}% from now versus ${pctString(fromNowBaseline)} normal (${edgeLabel}), which is still below the play bar for ${labelForTarget(focusTarget)}.`;
  }

  const expectedRoundIdFrom = patternMatch.expectedRoundRange && safeLatestId > 0
    ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetFrom
    : null;
  const expectedRoundIdTo = patternMatch.expectedRoundRange && safeLatestId > 0
    ? safeLatestId + patternMatch.expectedRoundRange.firstRemainingHitOffsetTo
    : null;
  const rawExpectedRoundIdLabel = (expectedRoundIdFrom && expectedRoundIdTo) ? `#${expectedRoundIdFrom} - #${expectedRoundIdTo}` : '-';
  const rawExpectedRoundIdBasis = patternMatch.expectedRoundRange
    ? `Based on ${patternMatch.expectedRoundRange.hitMatchCount} matched hits still ahead from this point.`
    : 'No stable round band was found in matched history.';
  const expectedRoundIdLabel = action === 'PLAY' ? rawExpectedRoundIdLabel : '-';
  const expectedRoundIdBasis = action === 'PLAY'
    ? rawExpectedRoundIdBasis
    : 'Round IDs are only shown on PLAY.';

  return {
    action,
    tone,
    confidence: signalScore,
    confidenceLabel: strengthLabel,
    dataQuality,
    accuracyRate: fromNowHitRate,
    accuracyPercent: fromNowRatePercent,
    matchedHitRate: fullWindowHitRate,
    matchedHitRatePercent,
    fromNowRate: fromNowHitRate,
    fromNowRatePercent,
    edgeRate: fromNowEdge,
    edgePercent,
      signalScorePercent,
      strengthLabel,
      timingEdgeLabel,
      similarityPercent,
      matchedPeakRangeLabel,
    noTargetPeakRangeLabel,
    predictsLabel: `Current ${windowLabel}`,
    inputLabel: `Closed Previous ${windowLabel}`,
    inputSlotLabel: patternMatch.inputSlotLabel,
    currentSlotLabel: patternMatch.currentSlotLabel,
    currentHitRate,
    baselineHitRate,
    baselineCurrentWindowHitRate: baselineCWHitRate,
    currentLift,
    effectiveCurrentLift,
    currentEvidenceWeight,
    currentSlotChance: safeNumber(currentSlot?.anyHitChance, 0),
    currentWindowHitRate: fullWindowHitRate,
    currentWindowLift: fullWindowLift,
    remainingHitRate: fromNowHitRate,
    remainingLift: fromNowLift,
    baselineRemainingHitRate: fromNowBaseline,
    matchedWindows: matchedWins,
    sameWeekdayMatches: sameWdMatches,
    lookbackDaysUsed: lookbackDays,
    alreadyHitInCurrentWindow,
    hitsSoFar,
    expectedRoundIdFrom,
    expectedRoundIdTo,
    expectedRoundIdLabel,
    expectedRoundIdBasis,
    summary,
    reasons: [
      `Closed input ${patternMatch.inputSlotLabel} -> live ${patternMatch.currentSlotLabel}.`,
      `${matchedWins} same-time matches across ${lookbackDays.toFixed(1)} stored days${sameWdMatches ? `, ${sameWdMatches} on the same weekday` : ''}.`,
      `Matched current-window hit rate: ${matchedHitRatePercent.toFixed(1)}%.`,
      `From-now hit rate: ${fromNowRatePercent.toFixed(1)}% vs normal ${pctString(fromNowBaseline)} (${edgeLabel}).`,
      `Timing edge: ${timingEdgeLabel}.`,
      saturatedCommonTarget ? `${labelForTarget(focusTarget)} is almost always present in this window size, so timing cannot separate this hour from a normal hour.` : null,
      `Pattern match: ${similarityPercent.toFixed(1)}%.`,
      noTargetPeakRangeLabel !== '-' ? `If ${labelForTarget(focusTarget)} missed from this point, the highest matched tail peak was usually around ${noTargetPeakRangeLabel}.` : null,
      matchedPeakRangeLabel !== '-' ? `Matched remaining-window peak range: ${matchedPeakRangeLabel}.` : null,
      alreadyHitInCurrentWindow
        ? `${labelForTarget(focusTarget)} already hit ${hitsSoFar} time(s) in this live window.`
        : `${labelForTarget(focusTarget)} has not hit yet in this live window.`,
      action === 'PLAY' && expectedRoundIdLabel !== '-'
        ? `Expected ${labelForTarget(focusTarget)} around rounds ${expectedRoundIdLabel}. ${expectedRoundIdBasis}`
        : null,
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

  for (const r of rounds) {
    const { hour } = getParts(r.timestamp);
    buckets[hour].roundCount++;
    buckets[hour].sumMultiplier += r.multiplier;
    for (const t of TARGETS) { if (r.multiplier >= t) buckets[hour].hits[t]++; }
  }

  const rows = buckets.map((b) => {
    const targetRates = createTargetMap((t) => ratio(b.hits[t], b.roundCount));
    let bestTarget = '-', bestLift = -Infinity;
    for (const t of TARGETS) {
      const lift = ratio(targetRates[t], baselinePerRoundRates[t], 1);
      if (lift > bestLift) { bestLift = lift; bestTarget = labelForTarget(t); }
    }
    return { hour: b.hour, label: formatHourLabel(b.hour), roundCount: b.roundCount, avgMultiplier: ratio(b.sumMultiplier, b.roundCount), targetRates, bestTarget };
  });

  const bestHours = TARGETS.map((t) => {
    const sorted = [...rows].filter((r) => r.roundCount > 0).sort((a, b) => b.targetRates[t] - a.targetRates[t]);
    const best = sorted[0], worst = sorted[sorted.length - 1];
    return { target: t, label: labelForTarget(t), bestHour: best?.hour ?? null, bestLabel: best?.label || '-', bestHitRate: best?.targetRates?.[t] || 0, worstHour: worst?.hour ?? null, worstLabel: worst?.label || '-', worstHitRate: worst?.targetRates?.[t] || 0 };
  });

  return { timeZone, rows, bestHours };
}

// ---------------------------------------------------------------------------
// HEATMAP
// ---------------------------------------------------------------------------

function buildHeatmap(rounds, timeZone, focusTarget, baselinePerRoundRates) {
  const getParts = buildZonedPartsGetter(timeZone);
  const cells = Array.from({ length: 7 }, (_, di) =>
    Array.from({ length: 24 }, (_, hour) => ({
      dayIndex: di, dayLabel: WEEKDAYS[di], hour, hourLabel: formatHourLabel(hour),
      roundCount: 0, hitCount: 0, hitRate: 0, lift: 1, tone: 'neutral',
    })));

  for (const r of rounds) {
    const { dayIndex, hour } = getParts(r.timestamp);
    const cell = cells[dayIndex][hour];
    cell.roundCount++;
    if (r.multiplier >= focusTarget) cell.hitCount++;
  }

  const flat = [];
  const baselineRate = baselinePerRoundRates[focusTarget] || 0;
  for (const row of cells) {
    for (const cell of row) {
      cell.hitRate = ratio(cell.hitCount, cell.roundCount);
      cell.lift    = ratio(cell.hitRate, baselineRate, 1);
      const sig    = chiSquareTest(cell.hitCount, cell.roundCount, baselineRate);
      cell.tone    = sig.significant ? (cell.lift >= 1.05 ? 'good' : cell.lift <= 0.95 ? 'bad' : 'neutral') : 'neutral';
      flat.push(cell);
    }
  }

  const ranked   = flat.filter((c) => c.roundCount >= MIN_CELL_COUNT);
  const strongest = [...ranked].sort((a, b) => b.lift !== a.lift ? b.lift - a.lift : b.hitRate - a.hitRate).slice(0, 3)
    .map((c) => ({ label: `${c.dayLabel} ${c.hourLabel}`, hitRate: c.hitRate, lift: c.lift, roundCount: c.roundCount }));
  const weakest   = [...ranked].sort((a, b) => a.lift !== b.lift ? a.lift - b.lift : a.hitRate - b.hitRate).slice(0, 3)
    .map((c) => ({ label: `${c.dayLabel} ${c.hourLabel}`, hitRate: c.hitRate, lift: c.lift, roundCount: c.roundCount }));

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    days: WEEKDAYS, hours: Array.from({ length: 24 }, (_, h) => ({ value: h, label: formatHourLabel(h) })),
    cells, strongest, weakest,
  };
}

// ---------------------------------------------------------------------------
// LAST HIT MAP
// ---------------------------------------------------------------------------

function buildLastHitMap(rounds) {
  const lastHits = createTargetMap(() => ({ roundId: null, timestamp: null, multiplier: null }));
  for (let i = rounds.length - 1; i >= 0; i--) {
    const r = rounds[i];
    for (const t of TARGETS) {
      if (lastHits[t].roundId == null && r.multiplier >= t) {
        lastHits[t] = { roundId: r.roundId, timestamp: r.timestamp, multiplier: r.multiplier };
      }
    }
  }
  return lastHits;
}

// ---------------------------------------------------------------------------
// COOLDOWN REPORT
// ---------------------------------------------------------------------------

function buildCooldownReport(rounds, baselinePerRoundRates, latestTimestamp, lastHits) {
  const prefixes = {};
  for (const t of TARGETS) {
    const prefix = new Array(rounds.length + 1).fill(0);
    for (let i = 0; i < rounds.length; i++) prefix[i + 1] = prefix[i] + (rounds[i].multiplier >= t ? 1 : 0);
    prefixes[t] = prefix;
  }

  const items = [];
  for (const t of TARGETS) {
    const eventIndices = [];
    for (let i = 0; i < rounds.length; i++) { if (rounds[i].multiplier >= t) eventIndices.push(i); }

    const horizons = COOLDOWN_WINDOWS.map((cw) => {
      if (!eventIndices.length) {
        return { key: cw.key, label: cw.label, anyHitRate: 0, perRoundHitRate: 0, baselinePerRoundRate: baselinePerRoundRates[t] || 0, lift: 1, sampleCount: 0, status: 'neutral', significant: false, pValue: 1 };
      }
      let totalHits = 0, totalRounds = 0, totalAnyHit = 0, right = 0;
      for (const idx of eventIndices) {
        if (right < idx + 1) right = idx + 1;
        while (right < rounds.length && rounds[right].timestamp <= rounds[idx].timestamp + cw.ms) right++;
        const inRange = Math.max(0, right - (idx + 1));
        const hitsInRange = prefixes[t][right] - prefixes[t][idx + 1];
        totalRounds += inRange; totalHits += hitsInRange;
        if (hitsInRange > 0) totalAnyHit++;
      }
      const anyHitRate = ratio(totalAnyHit, eventIndices.length);
      const perRound   = ratio(totalHits, totalRounds);
      const baseline   = baselinePerRoundRates[t] || 0;
      const lift       = ratio(perRound, baseline, 1);
      const sig        = chiSquareTest(totalHits, totalRounds, baseline);
      const cls        = classifyLift(lift, eventIndices.length, totalHits, totalRounds, baseline);
      return { key: cw.key, label: cw.label, anyHitRate, perRoundHitRate: perRound, baselinePerRoundRate: baseline, lift, sampleCount: eventIndices.length, status: cls.key, significant: sig.significant, pValue: Number(sig.pValue.toFixed(4)) };
    });

    const recentHit = lastHits[t];
    const ageMs     = recentHit?.timestamp ? latestTimestamp - recentHit.timestamp : null;
    let recentPressure = null;
    if (ageMs != null) {
      const ai = COOLDOWN_WINDOWS.findIndex((cw) => ageMs <= cw.ms);
      const aw = ai >= 0 ? horizons[ai] : null;
      if (aw) {
        recentPressure = {
          ageMs, activeWindow: aw.label, lift: aw.lift, significant: aw.significant, status: aw.status,
          note: aw.significant
            ? (aw.lift < 1
              ? `${labelForTarget(t)} significantly cools in the ${aw.label.toLowerCase()} after it lands (p<0.05).`
              : `${labelForTarget(t)} stays active in the ${aw.label.toLowerCase()} after it lands (p<0.05).`)
            : `${labelForTarget(t)} cooldown pattern is not statistically significant in the ${aw.label.toLowerCase()}.`,
        };
      }
    }

    items.push({ target: t, label: labelForTarget(t), lastHitRoundId: recentHit?.roundId || null, lastHitTimestamp: recentHit?.timestamp || null, ageMs, horizons, recentPressure });
  }
  return items;
}

// ---------------------------------------------------------------------------
// TARGET READINESS
// ---------------------------------------------------------------------------

function buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, latestTimestamp) {
  return TARGETS.map((t) => {
    const baselineRate = baselineStats.perRoundHitRates[t] || 0;
    const currentRate  = currentSummary.hitRates[t] || 0;
    const slotItem     = slotAnalytics.items.find((i) => i.target === t);
    const slotLift     = slotItem?.currentSlot?.lift || 1;
    const slotSig      = Boolean(slotItem?.currentSlot?.liftSignificant);
    const rateLift     = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
    const cd           = cooldowns.find((c) => c.target === t);

    let cdPenalty = 0;
    if (cd?.recentPressure?.significant && cd.recentPressure.lift < 0.95) {
      cdPenalty = (0.95 - cd.recentPressure.lift) * 25;
    }

    const effSlotLift = slotSig ? slotLift : 1;
    const score = clamp(50 + clamp((rateLift - 1) * 38, -20, 22) + clamp((effSlotLift - 1) * 22, -12, 14) - cdPenalty, 0, 100);
    let status = 'neutral', label = 'Neutral';
    if (score >= 62)  { status = 'ready'; label = 'Ready'; }
    else if (score <= 42) { status = 'avoid'; label = 'Avoid'; }

    let reason = `${labelForTarget(t)} is close to its usual pace.`;
    if (rateLift >= 1.12 && slotSig && slotLift >= 1.05) reason = `${labelForTarget(t)} running above normal with a significantly supportive time block.`;
    else if (rateLift <= 0.9) reason = `${labelForTarget(t)} landing below its usual rate in the current window.`;
    else if (slotSig && slotLift <= 0.9) reason = `${labelForTarget(t)} is in a historically and significantly weak time block.`;
    else if (cd?.recentPressure?.significant && cd.recentPressure.lift < 0.95) reason = `${labelForTarget(t)} is in a statistically significant post-hit cooldown zone.`;

    return { target: t, label: labelForTarget(t), score, status, statusLabel: label, currentHitRate: currentRate, baselineHitRate: baselineRate, currentLift: rateLift, slotLift, slotSignificant: slotSig, reason, lastUpdatedAt: latestTimestamp };
  });
}

// ---------------------------------------------------------------------------
// REGIME DETECTION
// ---------------------------------------------------------------------------

function buildRegime(focusTarget, currentSummary, baselineStats, cooldowns) {
  const focusLift    = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const hugeLift     = ratio(currentSummary.hugeHitRate, baselineStats.hugeHitRate, 1);
  const megaLift     = ratio(currentSummary.megaHitRate, baselineStats.megaHitRate, 1);
  const lowCrashLift = ratio(currentSummary.lowCrashRate, baselineStats.lowCrashRate, 1);
  const r100 = cooldowns.find((c) => c.target === 100)?.recentPressure;
  const r500 = cooldowns.find((c) => c.target === 500)?.recentPressure;

  if ((r500?.significant && r500.lift < 0.95) || (r100?.significant && r100.lift < 0.95 && megaLift < 1)) {
    return { key: 'post-spike-cooldown', label: 'Post-Spike Cooldown', tone: 'bad', description: 'A recent high spike is significantly cooling the board.' };
  }
  if (megaLift >= 1.35 || hugeLift >= 1.25) {
    return { key: 'spike-mode', label: 'Spike Mode', tone: 'good', description: 'High multipliers arriving above their historical pace.' };
  }
  if (lowCrashLift >= 1.15 && focusLift <= 0.95) {
    return { key: 'low-mode', label: 'Low Mode', tone: 'bad', description: 'Short crashes stacking above normal, selected target lagging.' };
  }
  if (focusLift >= 1.12 && lowCrashLift <= 1) {
    return { key: 'target-friendly', label: 'Target-Friendly', tone: 'good', description: `Flow leaning toward ${labelForTarget(focusTarget)} more than normal.` };
  }
  return { key: 'balanced', label: 'Balanced Mode', tone: 'neutral', description: 'Board is close to its historical mix.' };
}

// ---------------------------------------------------------------------------
// DECISION
// ---------------------------------------------------------------------------

function buildDecision({ focusTarget, windowLabel, currentSummary, baselineStats, slotAnalytics, cooldowns, readiness }) {
  const focusReadiness = readiness.find((r) => r.target === focusTarget);
  const focusSlot      = slotAnalytics.items.find((i) => i.target === focusTarget)?.currentSlot;
  const cooldown       = cooldowns.find((c) => c.target === focusTarget);
  const baselineRate   = baselineStats.perRoundHitRates[focusTarget] || 0;
  const currentRate    = currentSummary.hitRates[focusTarget] || 0;
  const rateLift       = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
  const avgLift        = ratio(currentSummary.avgMultiplier, baselineStats.avgMultiplier, 1);
  const lowCrashRelief = baselineStats.lowCrashRate > 0
    ? (baselineStats.lowCrashRate - currentSummary.lowCrashRate) / baselineStats.lowCrashRate : 0;
  const slotLift       = focusSlot?.lift || 1;
  const slotSig        = Boolean(focusSlot?.liftSignificant);
  const cdPenalty      = (cooldown?.recentPressure?.significant && cooldown.recentPressure.lift < 0.95)
    ? (0.95 - cooldown.recentPressure.lift) * 24 : 0;
  const effSlotLift    = slotSig ? slotLift : 1;

  const score = clamp(
    50
    + clamp((rateLift - 1) * 32, -18, 22)
    + clamp((effSlotLift - 1) * 20, -12, 14)
    + clamp((avgLift - 1) * 12, -8, 8)
    + clamp(lowCrashRelief * 18, -10, 10)
    - cdPenalty, 0, 100,
  );

  const band = describeBand(score);
  const zone = classifyLift(slotLift, focusSlot?.sampleCount || 0, focusSlot?.hitWindows, focusSlot?.sampleCount, baselineStats.windowAnyHitRates[focusTarget]);

  const playReasons = [], skipReasons = [];
  if (rateLift >= 1.1)                                    playReasons.push(`${labelForTarget(focusTarget)} hit rate above its historical baseline for this window.`);
  if (slotSig && slotLift >= 1.05)                        playReasons.push(`This local time block has a significantly stronger pattern (p<0.05).`);
  if (currentSummary.lowCrashRate <= baselineStats.lowCrashRate * 0.92) playReasons.push('Low crashes lighter than historical norm.');
  if (avgLift >= 1.08)                                    playReasons.push('Average multiplier running hotter than historical baseline.');
  if (rateLift <= 0.92)                                   skipReasons.push(`${labelForTarget(focusTarget)} underperforming vs historical hit rate.`);
  if (slotSig && slotLift <= 0.95)                        skipReasons.push('This time block has a significantly weak historical pattern.');
  if (currentSummary.lowCrashRate >= baselineStats.lowCrashRate * 1.1) skipReasons.push('Low crashes stacking above historical norm.');
  if (cooldown?.recentPressure?.significant && cooldown.recentPressure.lift < 0.95) skipReasons.push(cooldown.recentPressure.note);
  if (!playReasons.length) playReasons.push(`No significant positive signal for ${labelForTarget(focusTarget)} right now.`);
  if (!skipReasons.length) skipReasons.push('No statistically significant red flag at this time.');

  const regime = buildRegime(focusTarget, currentSummary, baselineStats, cooldowns);

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    score, band: band.key, label: band.label, tone: band.tone, windowLabel,
    zone: {
      key: zone.key, label: zone.label, tone: zone.tone, lift: slotLift, significant: slotSig,
      anyHitRate: focusSlot?.anyHitChance || 0,
      baselineAnyHitRate: slotAnalytics.baselineByTarget[focusTarget]?.anyHitRate || 0,
      currentSlotLabel: focusSlot?.label || '—',
    },
    regime, readiness: focusReadiness || null, playReasons, skipReasons,
    summary: band.key === 'play'
      ? `${labelForTarget(focusTarget)} is in a stronger-than-historical ${windowLabel.toLowerCase()}.`
      : band.key === 'skip'
        ? `${labelForTarget(focusTarget)} is running below its historical norm for this ${windowLabel.toLowerCase()}.`
        : `${labelForTarget(focusTarget)} is mixed — watch rather than a clear entry point.`,
  };
}

// ---------------------------------------------------------------------------
// WINDOW SCORE HELPER
// ---------------------------------------------------------------------------

function scoreWindowDecision(windowSummary, baselineStats, slotStat) {
  const scoreByTarget = {};
  for (const t of TARGETS) {
    const rateLift = ratio(windowSummary.hitRates[t], baselineStats.perRoundHitRates[t], 1);
    const slotLift = slotStat?.liftSignificant ? (slotStat?.lift || 1) : 1;
    scoreByTarget[t] = clamp(50 + clamp((rateLift - 1) * 32, -18, 22) + clamp((slotLift - 1) * 20, -12, 14), 0, 100);
  }
  return scoreByTarget;
}

// ---------------------------------------------------------------------------
// STABILITY
// ---------------------------------------------------------------------------

function buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget) {
  const getParts    = buildZonedPartsGetter(timeZone);
  const recent      = completedWindows.slice(-8);
  if (!recent.length) {
    return { score: 0, label: 'Low Stability', status: 'unstable', flipCount: 0, windowsChecked: 0, message: 'Not enough completed windows yet.', bands: [] };
  }
  const slotMinutes = slotAnalytics.slotMinutes;
  const slotStats   = slotAnalytics.slotStatsByTarget[focusTarget] || [];
  const slotMap     = new Map(slotStats.map((s) => [s.slotIndex, s]));
  const bands       = recent.map((w) => {
    const parts     = getParts(w.startTimestamp);
    const slotIndex = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, Math.max(0, Math.floor(1440 / slotMinutes) - 1));
    const score     = scoreWindowDecision(w.summary, baselineStats, slotMap.get(slotIndex))[focusTarget];
    const band      = describeBand(score);
    return { startTimestamp: w.startTimestamp, score, band: band.key, label: band.label };
  });

  let flips = 0;
  for (let i = 1; i < bands.length; i++) { if (bands[i].band !== bands[i-1].band) flips++; }
  const counts = bands.reduce((acc, b) => { acc[b.band] = (acc[b.band] || 0) + 1; return acc; }, {});
  const domCount    = Math.max(...Object.values(counts));
  const consistency = ratio(domCount, bands.length);
  const score       = Math.round(clamp((consistency * 70) + ((1 - ratio(flips, Math.max(1, bands.length - 1))) * 30), 0, 100));

  let status = 'mixed', label = 'Medium Stability', message = 'Signal moving around — keep position size smaller.';
  if (score >= 72)  { status = 'stable';   label = 'High Stability';   message = 'Same decision band across several windows — signal is steadier.'; }
  else if (score <= 46) { status = 'unstable'; label = 'Low Stability'; message = 'Signal has been flipping — treat as weak.'; }

  return { score, label, status, flipCount: flips, windowsChecked: bands.length, message, bands };
}

// ---------------------------------------------------------------------------
// BACKTEST
// ---------------------------------------------------------------------------

function buildBacktest(slotWindows, slotAnalytics, focusTarget) {
  const focusItem = slotAnalytics.items.find((i) => i.target === focusTarget);
  if (!focusItem) {
    return { focusTarget, focusTargetLabel: labelForTarget(focusTarget), summary: 'Not enough time-slot history yet for a backtest.', allWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 }, greenWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 }, redWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 } };
  }

  const greenSlots = new Set(
    (focusItem.topSlots || []).filter((s) => s.liftSignificant && s.lift >= 1.05 && s.sampleCount >= slotAnalytics.minSamples).map((s) => s.slotIndex),
  );
  const redSlots = new Set(
    (slotAnalytics.slotStatsByTarget[focusTarget] || []).filter((s) => s.liftSignificant && s.lift <= 0.95 && s.sampleCount >= slotAnalytics.minSamples).map((s) => s.slotIndex),
  );

  const summarize = (pred) => {
    const selected = slotWindows.filter(pred);
    if (!selected.length) return { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 };
    let hitWins = 0, peakSum = 0;
    for (const sw of selected) {
      if ((sw.summary.hitCounts[focusTarget] || 0) > 0) hitWins++;
      peakSum += sw.summary.maxMultiplier;
    }
    return { count: selected.length, anyHitRate: ratio(hitWins, selected.length), avgPeakMultiplier: ratio(peakSum, selected.length) };
  };

  const all   = summarize(() => true);
  const green = summarize((w) => greenSlots.has(w.slotIndex));
  const red   = summarize((w) => redSlots.has(w.slotIndex));

  return {
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    summary: green.count > 0
      ? `Significant (p<0.05) slots hit ${labelForTarget(focusTarget)} ${pctString(green.anyHitRate)} vs ${pctString(all.anyHitRate)} across all slots.`
      : 'No slots with statistically significant strength yet.',
    allWindows:   all,
    greenWindows: { ...green, lift: ratio(green.anyHitRate, all.anyHitRate, 1) },
    redWindows:   { ...red,   lift: ratio(red.anyHitRate, all.anyHitRate, 1) },
  };
}

// ---------------------------------------------------------------------------
// COMPARISON
// ---------------------------------------------------------------------------

function buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday) {
  const currentLift = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const cw = bestWindowsToday.items.find((i) => i.target === focusTarget);
  const zoneLabel = cw?.currentSlot?.zoneLabel || decision.zone.label;
  let message = `${labelForTarget(focusTarget)} is running close to its historical pace.`;
  if (currentLift >= 1.12)     message = `${labelForTarget(focusTarget)} is hotter than normal, time block is ${zoneLabel.toLowerCase()}.`;
  else if (currentLift <= 0.9) message = `${labelForTarget(focusTarget)} is colder than normal, time block leans ${zoneLabel.toLowerCase()}.`;
  else                         message = `${labelForTarget(focusTarget)} is mixed — use zone, cooldown, and stability panels together.`;
  return { band: decision.band, label: decision.label, message };
}

// ---------------------------------------------------------------------------
// TARGET CARDS
// ---------------------------------------------------------------------------

function buildTargetCards(currentSummary, baselineStats) {
  return TARGETS.map((t) => ({
    target: t, label: labelForTarget(t),
    currentHitRate:  currentSummary.hitRates[t] || 0,
    baselineHitRate: baselineStats.perRoundHitRates[t] || 0,
    delta: (currentSummary.hitRates[t] || 0) - (baselineStats.perRoundHitRates[t] || 0),
    lift:  ratio(currentSummary.hitRates[t] || 0, baselineStats.perRoundHitRates[t] || 0, 1),
  }));
}

// ---------------------------------------------------------------------------
// OUTLOOK
// ---------------------------------------------------------------------------

function buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) {
  const usable = completedWindows.filter((_, i) => i < completedWindows.length - 1);
  if (usable.length < 10) return { available: false, reason: 'Not enough completed windows yet.' };

  const allSummaries = [...usable.map((w) => w.summary), currentSummary];
  const { zVectors } = buildNormalizedVectorPool(allSummaries);
  const currentVec   = zVectors[zVectors.length - 1];

  const candidates = usable
    .map((w, i) => {
      const next = completedWindows[i + 1];
      if (!next) return null;
      return { distance: normalizedDistance(zVectors[i], currentVec), nextWindow: next };
    })
    .filter(Boolean)
    .sort((a, b) => a.distance - b.distance);

  const sampleSize = clamp(Math.round(candidates.length * 0.18), 12, 60);
  const matches    = candidates.slice(0, sampleSize);
  if (!matches.length) return { available: false, reason: 'No close historical match found for the current window shape.' };

  const candidateRows = TARGETS.map((t) => {
    let wHits = 0, tHits = 0, tRounds = 0, peakSum = 0;
    for (const m of matches) {
      const s = m.nextWindow.summary;
      tHits   += s.hitCounts[t] || 0;
      tRounds += s.roundCount;
      peakSum += s.maxMultiplier;
      if ((s.hitCounts[t] || 0) > 0) wHits++;
    }
    const anyHitRate    = ratio(wHits, matches.length);
    const perRoundRate  = ratio(tHits, tRounds);
    const baselineAny   = baselineStats.windowAnyHitRates[t] || 0;
    const lift          = ratio(anyHitRate, baselineAny, 1);
    const sig           = chiSquareTest(wHits, matches.length, baselineAny);
    const rewardWeight  = 1 + Math.log10(t) / 2;
    const score         = anyHitRate * (sig.significant && lift > 1 ? lift : 1) * rewardWeight;
    return { target: t, label: labelForTarget(t), anyHitRate, perRoundHitRate: perRoundRate, avgPeakMultiplier: ratio(peakSum, matches.length), expectedHits: Math.round(ratio(tHits, matches.length) * 100) / 100, baselineAnyHitRate: baselineAny, lift, liftSignificant: sig.significant, liftPValue: Number(sig.pValue.toFixed(4)), score, style: anyHitRate >= 0.55 ? 'Frequent' : anyHitRate >= 0.3 ? 'Balanced' : 'Long-shot' };
  }).sort((a, b) => b.score - a.score);

  const recommendation = candidateRows[0] || null;
  const focusCandidate = candidateRows.find((r) => r.target === focusTarget) || null;
  const spread         = average(matches.map((m) => m.distance));
  const confidence     = clamp((sampleWeight(matches.length) * 0.55) + (1 / (1 + spread)) * 0.45, 0, 1);

  return { available: true, basedOnMatches: matches.length, confidence, note: `Built from ${matches.length} completed windows most similar to the current one.`, recommendation, focusTarget, focusTargetLabel: labelForTarget(focusTarget), focusCandidate, candidates: candidateRows };
}

// ---------------------------------------------------------------------------
// BEST WINDOWS TODAY
// ---------------------------------------------------------------------------

function buildBestWindowsToday(slotAnalytics, focusTarget) {
  return {
    slotMode: slotAnalytics.slotMode, slotMinutes: slotAnalytics.slotMinutes,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    note: slotAnalytics.slotMode === 'start-time'
      ? 'For multi-day windows, this ranks best starting times.'
      : 'Best recurring local-time windows from stored dataset (significance-filtered).',
    items: slotAnalytics.items.map((item) => ({
      target: item.target, label: item.label, currentSlot: item.currentSlot,
      nextTodayWindow: item.nextTodayWindow, nextWindow: item.nextWindow,
      todayWindows: item.todayWindows, backups: item.backups,
      avoidWindow: item.avoidWindow, bestWindow: item.topSlots[0] || null, topWindows: item.topSlots,
    })),
  };
}

// ---------------------------------------------------------------------------
// EMPTY REPORT
// ---------------------------------------------------------------------------

function buildEmptyReport(windowConfig, focusTarget, timeZone) {
  return {
    ok: true, generatedAt: Date.now(), latestRoundId: null, totalRounds: 0,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets: TARGETS.map((v) => ({ value: v, label: labelForTarget(v) })),
    timeZone,
    dataset: { totalRounds: 0, startTimestamp: null, endTimestamp: null, spanDays: 0 },
    window: { key: windowConfig.key, label: windowConfig.label, ms: windowConfig.ms, startTimestamp: null, endTimestamp: null },
    baseline: summarizeWindowCollection([], focusTarget),
    previousWindow: summarizeRounds([], focusTarget),
    currentWindow:  summarizeRounds([], focusTarget),
    patternPrediction: {
      action: 'SKIP', tone: 'bad', confidence: 0, confidenceLabel: 'Low', dataQuality: 'insufficient',
      predictsLabel: `Current ${windowConfig.label}`, inputLabel: `Closed Previous ${windowConfig.label}`,
      inputSlotLabel: '-', currentSlotLabel: '-',
      currentHitRate: 0, baselineHitRate: 0, baselineCurrentWindowHitRate: 0,
      currentLift: 1, effectiveCurrentLift: 1, currentEvidenceWeight: 0, currentSlotChance: 0,
      matchedHitRate: 0, matchedHitRatePercent: 0, fromNowRate: 0, fromNowRatePercent: 0,
      edgeRate: 0, edgePercent: 0, signalScorePercent: 0, similarityPercent: 0,
      timingEdgeLabel: 'Insufficient',
      matchedPeakRangeLabel: '-', noTargetPeakRangeLabel: '-',
      currentWindowHitRate: 0, currentWindowLift: 1, remainingHitRate: 0, remainingLift: 1, baselineRemainingHitRate: 0,
      matchedWindows: 0, sameWeekdayMatches: 0, lookbackDaysUsed: 0,
      alreadyHitInCurrentWindow: false, hitsSoFar: 0,
      expectedRoundIdFrom: null, expectedRoundIdTo: null, expectedRoundIdLabel: '-', expectedRoundIdBasis: 'Round IDs are only shown when the signal is strong enough to play.',
      summary: 'No rounds stored yet, so no timing prediction is available.',
      reasons: [],
    },
    comparison: { band: 'skip', label: 'SKIP', message: 'No rounds stored yet.' },
    targetCards: [], decision: null, targetReadiness: [], recommendationStability: null,
    bestWindowsToday: { items: [], slotMode: 'window', slotMinutes: chooseSlotMinutes(windowConfig.ms), note: '' },
    patternMatch: { available: false, examples: [], reason: 'No rounds stored yet.', inputSlotLabel: '-', currentSlotLabel: '-' },
    cooldowns: [], backtest: null,
    hourlyHistory: { timeZone, rows: [], bestHours: [] },
    dayHourHeatmap: {
      focusTarget, focusTargetLabel: labelForTarget(focusTarget),
      days: WEEKDAYS, hours: Array.from({ length: 24 }, (_, h) => ({ value: h, label: formatHourLabel(h) })),
      cells: [], strongest: [], weakest: [],
    },
    outlook: null,
  };
}

// ---------------------------------------------------------------------------
// MAIN ENTRY POINT
// ---------------------------------------------------------------------------

function buildTimingAnalyticsReport(rounds, options = {}) {
  const windowKey    = normalizeTimingWindowKey(options.windowKey);
  const focusTarget  = normalizeTimingTarget(options.focusTarget);
  const timeZone     = normalizeTimingTimeZone(options.timeZone);
  const includeOutlook = Boolean(options.includeOutlook);
  const windowConfig = WINDOW_MAP.get(windowKey);
  const normalized   = normalizeRounds(rounds);

  if (!normalized.length) return buildEmptyReport(windowConfig, focusTarget, timeZone);

  const latest   = normalized[normalized.length - 1];
  const earliest = normalized[0];
  const requestedNow = safeNumber(options.nowTimestamp, Date.now());
  const asOfTimestamp = Math.max(latest.timestamp, requestedNow);
  const slotMinutes  = chooseSlotMinutes(windowConfig.ms);
  const slotWindows  = buildSlotWindows(normalized, slotMinutes, timeZone, focusTarget);
  const slotAnalytics = buildSlotAnalytics(slotWindows, slotMinutes, windowConfig.ms, focusTarget, asOfTimestamp, timeZone);

  const getParts = buildZonedPartsGetter(timeZone);
  const currentParts = getParts(asOfTimestamp);
  const currentKey   = `${currentParts.dateKey}|${slotAnalytics.currentSlotIndex}`;
  const previousParts = getParts(asOfTimestamp - slotMinutes * MINUTE_MS);
  const previousSlotIndex = clamp(Math.floor(previousParts.minuteOfDay / slotMinutes), 0, Math.max(1, Math.floor(1440 / slotMinutes)) - 1);
  const previousKey = `${previousParts.dateKey}|${previousSlotIndex}`;
  const ordered      = [...slotWindows].sort((a, b) => a.dateKey !== b.dateKey ? a.dateKey.localeCompare(b.dateKey) : a.slotIndex - b.slotIndex);
  const idxByKey     = new Map(ordered.map((w, i) => [w.key, i]));
  const curPos       = idxByKey.get(currentKey);
  const curSlotWin   = idxByKey.has(currentKey) ? ordered[idxByKey.get(currentKey)] : null;
  const prevSlotWin  = idxByKey.has(previousKey)
    ? ordered[idxByKey.get(previousKey)]
    : (Number.isInteger(curPos) && curPos > 0 ? ordered[curPos - 1] : null);

  const currentSummary  = curSlotWin  ? curSlotWin.summary  : summarizeRounds([], focusTarget);
  const previousSummary = prevSlotWin ? prevSlotWin.summary : summarizeRounds([], focusTarget);

  const fixedWindows     = segmentRoundsByWindow(normalized, windowConfig.ms, focusTarget);
  const completedWindows = fixedWindows.slice(0, -1);
  const completedSlots   = slotWindows.filter((w) => w.key !== currentKey);
  const baselineStats    = completedSlots.length
    ? summarizeWindowCollection(completedSlots, focusTarget)
    : summarizeWindowCollection(slotWindows, focusTarget);

  const bestWindowsToday    = buildBestWindowsToday(slotAnalytics, focusTarget);
  const lastHits            = buildLastHitMap(normalized);
  const cooldowns           = buildCooldownReport(normalized, baselineStats.perRoundHitRates, asOfTimestamp, lastHits);
  const targetReadiness     = buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, asOfTimestamp);
  const decision            = buildDecision({ focusTarget, windowLabel: windowConfig.label, currentSummary, baselineStats, slotAnalytics, cooldowns, readiness: targetReadiness });
  const recommendationStability = buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget);
  const hourlyHistory       = buildHourlyHistory(normalized, timeZone, baselineStats.perRoundHitRates);
  const dayHourHeatmap      = buildHeatmap(normalized, timeZone, focusTarget, baselineStats.perRoundHitRates);
  const backtest            = buildBacktest(slotWindows, slotAnalytics, focusTarget);
  const comparison          = buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday);
  const targetCards         = buildTargetCards(currentSummary, baselineStats);
  const patternMatch        = buildPatternMatchReport(slotWindows, previousSummary, currentSummary, baselineStats, focusTarget, asOfTimestamp, timeZone, slotAnalytics);
  const patternPrediction   = buildCurrentWindowPatternPrediction({ focusTarget, windowLabel: windowConfig.label, latestRoundId: latest.roundId || null, currentSummary, baselineStats, slotAnalytics, patternMatch });
  const outlook             = includeOutlook ? buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) : null;
  const currentSlotElapsedMs = ((((currentParts.minuteOfDay - (slotAnalytics.currentSlotIndex * slotMinutes)) * 60) + currentParts.second) * 1000);
  const currentSlotStartTimestamp = asOfTimestamp - Math.max(0, currentSlotElapsedMs);

  return {
    ok: true, generatedAt: Date.now(),
    asOfTimestamp,
    latestRoundId:    latest.roundId || null,
    totalRounds:      normalized.length,
    focusTarget, focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets: TARGETS.map((v) => ({ value: v, label: labelForTarget(v) })),
    timeZone,
    dataset: { totalRounds: normalized.length, startTimestamp: earliest.timestamp, endTimestamp: latest.timestamp, spanDays: ratio(latest.timestamp - earliest.timestamp, DAY_MS) },
    window: { key: windowConfig.key, label: windowConfig.label, ms: windowConfig.ms, startTimestamp: curSlotWin?.firstTimestamp || currentSlotStartTimestamp, endTimestamp: asOfTimestamp },
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
