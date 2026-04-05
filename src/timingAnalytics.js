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
  { key: '12h', label: '12 Hours', ms: 12 * HOUR_MS },
  { key: '24h', label: '24 Hours', ms: 24 * HOUR_MS },
  { key: '3d', label: '3 Days', ms: 3 * DAY_MS },
  { key: '7d', label: '7 Days', ms: 7 * DAY_MS },
  { key: '10d', label: '10 Days', ms: 10 * DAY_MS },
  { key: '15d', label: '15 Days', ms: 15 * DAY_MS },
  { key: '30d', label: '30 Days', ms: 30 * DAY_MS },
];

const WINDOW_MAP = new Map(WINDOW_OPTIONS.map((item) => [item.key, item]));
const DEFAULT_WINDOW_KEY = '5m';

const TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const TARGET_SET = new Set(TARGETS);
const DEFAULT_TARGET = 5;

const DISTRIBUTION_BANDS = [
  { key: 'lt2', label: '<2x', min: -Infinity, max: 2, color: '#ff5d73' },
  { key: '2to5', label: '2x-5x', min: 2, max: 5, color: '#ff9f43' },
  { key: '5to10', label: '5x-10x', min: 5, max: 10, color: '#ffd84d' },
  { key: '10to20', label: '10x-20x', min: 10, max: 20, color: '#9ef01a' },
  { key: '20to50', label: '20x-50x', min: 20, max: 50, color: '#22d3ee' },
  { key: '50to100', label: '50x-100x', min: 50, max: 100, color: '#38bdf8' },
  { key: '100to500', label: '100x-500x', min: 100, max: 500, color: '#f472b6' },
  { key: '500to1000', label: '500x-1000x', min: 500, max: 1000, color: '#c084fc' },
  { key: 'gte1000', label: '1000x+', min: 1000, max: Infinity, color: '#818cf8' },
];

const COOLDOWN_WINDOWS = [
  { key: '10m', label: '10 Minutes', ms: 10 * MINUTE_MS },
  { key: '20m', label: '20 Minutes', ms: 20 * MINUTE_MS },
  { key: '30m', label: '30 Minutes', ms: 30 * MINUTE_MS },
  { key: '60m', label: '60 Minutes', ms: 60 * MINUTE_MS },
];

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const WEEKDAY_INDEX = WEEKDAYS.reduce((acc, label, index) => {
  acc[label] = index;
  return acc;
}, {});

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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function circularDistance(a, b, size) {
  const safeSize = Math.max(1, Number(size) || 1);
  const diff = Math.abs(Number(a) - Number(b));
  return Math.min(diff, safeSize - diff);
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

function labelForTarget(target) {
  return `${target}x`;
}

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
  if (mode === 'start-time') {
    return `Starts ${formatClockMinute(startMinute)}`;
  }
  if (slotMinutes >= 1440) {
    return `All Day (${formatClockMinute(startMinute)} start)`;
  }
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
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(numeric));
}

function summarizeRoundRange(rounds, focusTarget, limit = 8) {
  const items = Array.isArray(rounds) ? rounds : [];
  if (!items.length) {
    return {
      fromRoundId: null,
      toRoundId: null,
      roundCount: 0,
      hitRoundIds: [],
      hitCount: 0,
    };
  }

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

function sampleWeight(sampleCount) {
  return clamp(sampleCount / 12, 0.3, 1);
}

function describeBand(score) {
  if (score >= 68) return { key: 'play', label: 'PLAY WINDOW', tone: 'good' };
  if (score >= 50) return { key: 'wait', label: 'WAIT / WATCH', tone: 'neutral' };
  return { key: 'skip', label: 'SKIP WINDOW', tone: 'bad' };
}

function classifyLift(lift, sampleCount) {
  if (!Number.isFinite(lift) || sampleCount < 3) {
    return { key: 'neutral', label: 'Watch Zone', tone: 'neutral' };
  }
  if (lift >= 1.12) return { key: 'green', label: 'Green Zone', tone: 'good' };
  if (lift <= 0.9) return { key: 'red', label: 'Red Zone', tone: 'bad' };
  return { key: 'watch', label: 'Watch Zone', tone: 'neutral' };
}

function createTargetMap(initialValue) {
  const out = {};
  for (const target of TARGETS) {
    out[target] = typeof initialValue === 'function' ? initialValue(target) : initialValue;
  }
  return out;
}

function buildZonedPartsGetter(timeZone) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    weekday: 'short',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
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

function normalizeRounds(rounds) {
  return (Array.isArray(rounds) ? rounds : [])
    .map((round) => ({
      roundId: safeNumber(round?.roundId ?? round?.round_id, 0),
      multiplier: safeNumber(round?.multiplier, NaN),
      timestamp: safeNumber(round?.timestamp, NaN),
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
  let sum = 0;
  let max = 0;
  let min = Infinity;
  let lowCrashCount = 0;
  let hugeHitCount = 0;
  let megaHitCount = 0;

  for (const round of rounds) {
    const multiplier = round.multiplier;
    values.push(multiplier);
    sum += multiplier;
    if (multiplier > max) max = multiplier;
    if (multiplier < min) min = multiplier;
    if (multiplier < 2) lowCrashCount += 1;
    if (multiplier >= 100) hugeHitCount += 1;
    if (multiplier >= 500) megaHitCount += 1;
    for (const target of TARGETS) {
      if (multiplier >= target) hitCounts[target] += 1;
    }
    for (const band of DISTRIBUTION_BANDS) {
      if (multiplier >= band.min && multiplier < band.max) {
        distributionCounts[band.key] += 1;
        break;
      }
    }
  }

  values.sort((a, b) => a - b);
  const roundCount = rounds.length;
  const hitRates = createTargetMap((target) => ratio(hitCounts[target], roundCount));
  const distribution = DISTRIBUTION_BANDS.map((band) => ({
    key: band.key,
    label: band.label,
    count: distributionCounts[band.key],
    pct: ratio(distributionCounts[band.key], roundCount),
    color: band.color,
  }));

  return {
    roundCount,
    avgMultiplier: ratio(sum, roundCount),
    medianMultiplier: quantile(values, 0.5),
    p90Multiplier: quantile(values, 0.9),
    maxMultiplier: max || 0,
    minMultiplier: Number.isFinite(min) ? min : 0,
    focusHitCount: hitCounts[focusTarget] || 0,
    focusHitRate: hitRates[focusTarget] || 0,
    lowCrashRate: ratio(lowCrashCount, roundCount),
    hugeHitRate: ratio(hugeHitCount, roundCount),
    megaHitRate: ratio(megaHitCount, roundCount),
    hitCounts,
    hitRates,
    distribution,
  };
}

function summarizeWindowCollection(windows, focusTarget) {
  if (!windows.length) {
    return {
      windowCount: 0,
      roundCount: 0,
      focusHitRate: 0,
      focusAnyHitRate: 0,
      avgMultiplier: 0,
      lowCrashRate: 0,
      hugeHitRate: 0,
      megaHitRate: 0,
      avgPeakMultiplier: 0,
      perRoundHitRates: createTargetMap(0),
      windowAnyHitRates: createTargetMap(0),
    };
  }

  const totalHits = createTargetMap(0);
  const windowsWithHit = createTargetMap(0);
  let totalRounds = 0;
  let weightedAvgMultiplier = 0;
  let weightedLowCrash = 0;
  let weightedHuge = 0;
  let weightedMega = 0;
  let peakSum = 0;

  for (const window of windows) {
    const summary = window.summary;
    totalRounds += summary.roundCount;
    weightedAvgMultiplier += summary.avgMultiplier * summary.roundCount;
    weightedLowCrash += summary.lowCrashRate * summary.roundCount;
    weightedHuge += summary.hugeHitRate * summary.roundCount;
    weightedMega += summary.megaHitRate * summary.roundCount;
    peakSum += summary.maxMultiplier;
    for (const target of TARGETS) {
      totalHits[target] += summary.hitCounts[target] || 0;
      if ((summary.hitCounts[target] || 0) > 0) windowsWithHit[target] += 1;
    }
  }

  return {
    windowCount: windows.length,
    roundCount: totalRounds,
    focusHitRate: ratio(totalHits[focusTarget], totalRounds),
    focusAnyHitRate: ratio(windowsWithHit[focusTarget], windows.length),
    avgMultiplier: ratio(weightedAvgMultiplier, totalRounds),
    lowCrashRate: ratio(weightedLowCrash, totalRounds),
    hugeHitRate: ratio(weightedHuge, totalRounds),
    megaHitRate: ratio(weightedMega, totalRounds),
    avgPeakMultiplier: ratio(peakSum, windows.length),
    perRoundHitRates: createTargetMap((target) => ratio(totalHits[target], totalRounds)),
    windowAnyHitRates: createTargetMap((target) => ratio(windowsWithHit[target], windows.length)),
  };
}

function segmentRoundsByWindow(rounds, windowMs, focusTarget) {
  const buckets = new Map();
  for (const round of rounds) {
    const bucketStart = Math.floor(round.timestamp / windowMs) * windowMs;
    const key = String(bucketStart);
    let entry = buckets.get(key);
    if (!entry) {
      entry = {
        startTimestamp: bucketStart,
        endTimestamp: bucketStart + windowMs,
        rounds: [],
      };
      buckets.set(key, entry);
    }
    entry.rounds.push(round);
  }

  return Array.from(buckets.values())
    .sort((a, b) => a.startTimestamp - b.startTimestamp)
    .map((window) => ({
      startTimestamp: window.startTimestamp,
      endTimestamp: window.endTimestamp,
      rounds: window.rounds,
      summary: summarizeRounds(window.rounds, focusTarget),
    }));
}

function buildSlotWindows(rounds, slotMinutes, timeZone, focusTarget) {
  const getParts = buildZonedPartsGetter(timeZone);
  const slotCount = Math.max(1, Math.floor(1440 / slotMinutes));
  const groups = new Map();

  for (const round of rounds) {
    const parts = getParts(round.timestamp);
    const slotIndex = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, slotCount - 1);
    const slotStartMinute = slotIndex * slotMinutes;
    const key = `${parts.dateKey}|${slotIndex}`;
    let entry = groups.get(key);
    if (!entry) {
      entry = {
        key,
        dateKey: parts.dateKey,
        dayIndex: parts.dayIndex,
        slotIndex,
        slotStartMinute,
        firstTimestamp: round.timestamp,
        rounds: [],
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
    .map((entry) => ({
      ...entry,
      summary: summarizeRounds(entry.rounds, focusTarget),
    }));
}

function buildSlotAnalytics(slotWindows, slotMinutes, windowMs, focusTarget, latestTimestamp, timeZone) {
  const mode = windowMs > DAY_MS ? 'start-time' : 'window';
  const slotCount = Math.max(1, Math.floor(1440 / slotMinutes));
  const currentParts = buildZonedPartsGetter(timeZone)(latestTimestamp);
  const currentSlotIndex = clamp(Math.floor(currentParts.minuteOfDay / slotMinutes), 0, slotCount - 1);

  const baselineByTarget = {};
  const slotMapsByTarget = {};
  const slotStatsByTarget = {};
  const items = [];
  const minSamples = Math.max(3, Math.floor(Math.sqrt(Math.max(1, slotWindows.length)) / 2));

  for (const target of TARGETS) {
    let baselineHitWindows = 0;
    let baselineHits = 0;
    let baselineRounds = 0;
    const slotMap = new Map();

    for (const slotWindow of slotWindows) {
      const summary = slotWindow.summary;
      const hitCount = summary.hitCounts[target] || 0;
      baselineRounds += summary.roundCount;
      baselineHits += hitCount;
      if (hitCount > 0) baselineHitWindows += 1;

      let aggregate = slotMap.get(slotWindow.slotIndex);
      if (!aggregate) {
        aggregate = {
          slotIndex: slotWindow.slotIndex,
          startMinute: slotWindow.slotStartMinute,
          sampleCount: 0,
          totalRounds: 0,
          totalHits: 0,
          hitWindows: 0,
          peakSum: 0,
        };
        slotMap.set(slotWindow.slotIndex, aggregate);
      }

      aggregate.sampleCount += 1;
      aggregate.totalRounds += summary.roundCount;
      aggregate.totalHits += hitCount;
      aggregate.peakSum += summary.maxMultiplier;
      if (hitCount > 0) aggregate.hitWindows += 1;
    }

    const baselineAnyHitRate = ratio(baselineHitWindows, slotWindows.length);
    const baselineRoundHitRate = ratio(baselineHits, baselineRounds);
    baselineByTarget[target] = {
      anyHitRate: baselineAnyHitRate,
      roundHitRate: baselineRoundHitRate,
    };
    slotMapsByTarget[target] = slotMap;

    const slotStats = Array.from(slotMap.values())
      .map((slot) => {
        const anyHitChance = ratio(slot.hitWindows, slot.sampleCount);
        const roundHitRate = ratio(slot.totalHits, slot.totalRounds);
        const lift = ratio(anyHitChance, baselineAnyHitRate, 1);
        const classification = classifyLift(lift, slot.sampleCount);
        return {
          slotIndex: slot.slotIndex,
          startMinute: slot.startMinute,
          label: formatSlotLabel(slot.startMinute, slotMinutes, mode),
          anyHitChance,
          roundHitRate,
          lift,
          sampleCount: slot.sampleCount,
          avgPeakMultiplier: ratio(slot.peakSum, slot.sampleCount),
          status: classification.key,
          zoneLabel: classification.label,
          tone: classification.tone,
          score: anyHitChance * Math.max(0.25, lift) * sampleWeight(slot.sampleCount),
        };
      })
      .sort((a, b) => a.slotIndex - b.slotIndex);

    slotStatsByTarget[target] = slotStats;

    const currentSlot = slotStats.find((slot) => slot.slotIndex === currentSlotIndex) || {
      slotIndex: currentSlotIndex,
      startMinute: currentSlotIndex * slotMinutes,
      label: formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, mode),
      anyHitChance: baselineAnyHitRate,
      roundHitRate: baselineRoundHitRate,
      lift: 1,
      sampleCount: 0,
      avgPeakMultiplier: 0,
      status: 'neutral',
      zoneLabel: 'Watch Zone',
      tone: 'neutral',
      score: baselineAnyHitRate,
    };

    const futureOptions = slotStats
      .filter((slot) => slot.sampleCount >= minSamples)
      .map((slot) => {
        const dayOffset = slot.slotIndex > currentSlotIndex ? 0 : 1;
        const deltaSlots = dayOffset === 0
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
        const aStrong = (a.lift >= 1.05 || a.score >= currentSlot.score * 1.08) ? 1 : 0;
        const bStrong = (b.lift >= 1.05 || b.score >= currentSlot.score * 1.08) ? 1 : 0;
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
      label: labelForTarget(target),
      baselineAnyHitRate,
      baselineRoundHitRate,
      currentSlot,
      nextTodayWindow: todayOptions[0] || null,
      nextWindow: futureOptions[0] || null,
      todayWindows: todayOptions.slice(0, 3),
      backups: futureOptions.slice(1, 3),
      avoidWindow: worstFuture,
      topSlots: [...slotStats]
        .filter((slot) => slot.sampleCount >= minSamples)
        .sort((a, b) => {
          if (b.score !== a.score) return b.score - a.score;
          return a.slotIndex - b.slotIndex;
        })
        .slice(0, 3),
    });
  }

  return {
    timeZone,
    slotMinutes,
    slotMode: mode,
    currentSlotIndex,
    minSamples,
    baselineByTarget,
    slotStatsByTarget,
    slotMapsByTarget,
    items,
  };
}

function buildPatternMatchReport(slotWindows, currentSummary, baselineStats, focusTarget, latestTimestamp, timeZone, slotAnalytics) {
  if (!slotWindows.length) {
    return {
      available: false,
      examples: [],
      reason: 'Not enough stored slot history yet to build a same-time pattern match.',
    };
  }

  const slotMinutes = slotAnalytics.slotMinutes;
  const slotCount = Math.max(1, Math.floor(1440 / slotMinutes));
  const slotMode = slotAnalytics.slotMode;
  const getParts = buildZonedPartsGetter(timeZone);
  const currentParts = getParts(latestTimestamp);
  const currentSlotIndex = slotAnalytics.currentSlotIndex;
  const currentSlotLabel = formatSlotLabel(currentSlotIndex * slotMinutes, slotMinutes, slotMode);
  const currentKey = `${currentParts.dateKey}|${currentSlotIndex}`;
  const lookbackMs = 30 * DAY_MS;
  const lookbackStart = latestTimestamp - lookbackMs;
  const historySpanDays = ratio(latestTimestamp - (slotWindows[0]?.firstTimestamp || latestTimestamp), DAY_MS);
  const lookbackDaysUsed = Math.min(30, Math.max(0, historySpanDays));
  const orderedWindows = [...slotWindows].sort((a, b) => {
    if (a.dateKey !== b.dateKey) return a.dateKey.localeCompare(b.dateKey);
    return a.slotIndex - b.slotIndex;
  });
  const indexByKey = new Map(orderedWindows.map((window, index) => [window.key, index]));

  const currentVector = {
    focusHitRate: currentSummary.hitRates[focusTarget] || 0,
    lowCrashRate: currentSummary.lowCrashRate || 0,
    hugeHitRate: currentSummary.hugeHitRate || 0,
    avgMultiplier: currentSummary.avgMultiplier || 0,
    maxMultiplier: currentSummary.maxMultiplier || 0,
  };

  const sameTimeCandidates = orderedWindows
    .filter((window) => (
      window.slotIndex === currentSlotIndex
      && window.key !== currentKey
      && window.firstTimestamp < latestTimestamp
      && window.firstTimestamp >= lookbackStart
    ))
    .map((window) => {
      const position = indexByKey.get(window.key);
      const nextWindow = Number.isInteger(position) ? orderedWindows[position + 1] : null;
      if (!nextWindow) return null;
      const expectedNextSlotIndex = (window.slotIndex + 1) % slotCount;
      if (nextWindow.slotIndex !== expectedNextSlotIndex) return null;
      const weekdayMatch = window.dayIndex === currentParts.dayIndex;
      const summary = window.summary;
      const shapeDistance =
        Math.abs((summary.hitRates[focusTarget] || 0) - currentVector.focusHitRate) * 5.2
        + Math.abs((summary.lowCrashRate || 0) - currentVector.lowCrashRate) * 2.8
        + Math.abs((summary.hugeHitRate || 0) - currentVector.hugeHitRate) * 2
        + Math.abs((summary.avgMultiplier || 0) - currentVector.avgMultiplier) / Math.max(2, currentVector.avgMultiplier || 2)
        + Math.abs((summary.maxMultiplier || 0) - currentVector.maxMultiplier) / Math.max(20, currentVector.maxMultiplier || 20);

      return {
        weekdayMatch,
        distance: shapeDistance + (weekdayMatch ? 0 : 0.22),
        nextWindow,
        window,
      };
    })
    .filter(Boolean);

  if (sameTimeCandidates.length < 6) {
    return {
      available: false,
      examples: [],
      reason: 'Not enough same-time windows in the available history span to build a prediction.',
    };
  }

  const sameWeekdayPool = sameTimeCandidates.filter((item) => item.weekdayMatch);
  const pool = sameWeekdayPool.length >= 6 ? sameWeekdayPool : sameTimeCandidates;
  const matchMode = sameWeekdayPool.length >= 6 ? 'same-weekday' : 'same-time';
  const sampleSize = clamp(Math.round(pool.length * 0.45), 6, 18);
  const matches = [...pool]
    .sort((a, b) => a.distance - b.distance)
    .slice(0, sampleSize);

  if (!matches.length) {
    return {
      available: false,
      examples: [],
      reason: 'No same-time pattern match was found for the current window.',
    };
  }

  const matchRows = matches.map((match) => {
    const daysAgo = ratio(latestTimestamp - match.window.firstTimestamp, DAY_MS);
    const distanceWeight = 1 / (1 + (match.distance * 3.2));
    const recencyWeight = 1 / (1 + (daysAgo / 9));
    const weekdayWeight = match.weekdayMatch ? 1.14 : 0.96;
    const weight = distanceWeight * recencyWeight * weekdayWeight;
    const nextRounds = Array.isArray(match.nextWindow.rounds) ? match.nextWindow.rounds : [];
    const firstHitIndex = nextRounds.findIndex((round) => round.multiplier >= focusTarget);
    const hitOffsets = nextRounds
      .map((round, index) => ({ round, index }))
      .filter((entry) => entry.round.multiplier >= focusTarget)
      .map((entry) => entry.index);

    return {
      ...match,
      daysAgo,
      weight,
      firstHitIndex,
      hitOffsets,
    };
  });

  let slotHistoryHits = 0;
  let slotHistoryRounds = 0;
  let slotHistoryAnyHit = 0;
  for (const item of sameTimeCandidates) {
    const summary = item.window.summary;
    slotHistoryHits += summary.hitCounts[focusTarget] || 0;
    slotHistoryRounds += summary.roundCount;
    if ((summary.hitCounts[focusTarget] || 0) > 0) slotHistoryAnyHit += 1;
  }

  let nextWindowHits = 0;
  let nextWindowRounds = 0;
  let nextWindowAnyHit = 0;
  let nextPeakSum = 0;
  let sameWeekdayMatches = 0;

  for (const match of matchRows) {
    const summary = match.nextWindow.summary;
    nextWindowHits += summary.hitCounts[focusTarget] || 0;
    nextWindowRounds += summary.roundCount;
    nextPeakSum += summary.maxMultiplier || 0;
    if ((summary.hitCounts[focusTarget] || 0) > 0) nextWindowAnyHit += 1;
    if (match.weekdayMatch) sameWeekdayMatches += 1;
  }

  const examples = matchRows.slice(0, 6).map((match, index) => {
    const currentRounds = summarizeRoundRange(match.window.rounds, focusTarget);
    const nextRounds = summarizeRoundRange(match.nextWindow.rounds, focusTarget);
    return {
      rank: index + 1,
      weekdayMatch: match.weekdayMatch,
      distance: Number(match.distance.toFixed(3)),
      weight: Number(match.weight.toFixed(3)),
      currentWindowLabel: formatTimestampInTimeZone(match.window.firstTimestamp, timeZone),
      currentSlotLabel: formatSlotLabel(match.window.slotStartMinute, slotMinutes, slotMode),
      currentRoundFrom: currentRounds.fromRoundId,
      currentRoundTo: currentRounds.toRoundId,
      currentRoundCount: currentRounds.roundCount,
      currentHitRate: match.window.summary.hitRates[focusTarget] || 0,
      nextWindowLabel: formatTimestampInTimeZone(match.nextWindow.firstTimestamp, timeZone),
      nextSlotLabel: formatSlotLabel(match.nextWindow.slotStartMinute, slotMinutes, slotMode),
      nextRoundFrom: nextRounds.fromRoundId,
      nextRoundTo: nextRounds.toRoundId,
      nextRoundCount: nextRounds.roundCount,
      nextHitCount: nextRounds.hitCount,
      nextHitRoundIds: nextRounds.hitRoundIds,
      firstHitOffset: match.firstHitIndex >= 0 ? match.firstHitIndex + 1 : null,
      nextAnyHit: (match.nextWindow.summary.hitCounts[focusTarget] || 0) > 0,
      nextPeakMultiplier: match.nextWindow.summary.maxMultiplier || 0,
    };
  });

  const slotHistoryAnyHitRate = ratio(slotHistoryAnyHit, sameTimeCandidates.length);
  const slotHistoryPerRoundRate = ratio(slotHistoryHits, slotHistoryRounds);
  const slotHistoryLift = ratio(
    slotHistoryAnyHitRate,
    baselineStats.windowAnyHitRates[focusTarget],
    baselineStats.windowAnyHitRates[focusTarget] > 0 ? 1 : 0
  );
  const nextAnyHitRate = weightedAverage(
    matchRows,
    (match) => ((match.nextWindow.summary.hitCounts[focusTarget] || 0) > 0 ? 1 : 0),
    (match) => match.weight,
    ratio(nextWindowAnyHit, matches.length)
  );
  const nextPerRoundHitRate = weightedAverage(
    matchRows,
    (match) => ratio(match.nextWindow.summary.hitCounts[focusTarget] || 0, match.nextWindow.summary.roundCount),
    (match) => match.weight,
    ratio(nextWindowHits, nextWindowRounds)
  );
  const nextLift = ratio(nextAnyHitRate, baselineStats.windowAnyHitRates[focusTarget], baselineStats.windowAnyHitRates[focusTarget] > 0 ? 1 : 0);
  const weightedAvgPeakMultiplier = weightedAverage(
    matchRows,
    (match) => match.nextWindow.summary.maxMultiplier || 0,
    (match) => match.weight,
    ratio(nextPeakSum, matches.length)
  );
  const nextWindowStartMinute = ((currentSlotIndex + 1) % slotCount) * slotMinutes;
  const nextWindowDayOffset = currentSlotIndex + 1 >= slotCount ? 1 : 0;
  const nextWindowTone = classifyLift(nextLift, matches.length).tone;
  const hitMatches = matchRows.filter((match) => match.firstHitIndex >= 0);
  const averageRoundsPerWindow = weightedAverage(
    matchRows,
    (match) => match.nextWindow.summary.roundCount,
    (match) => match.weight,
    average(matchRows.map((match) => match.nextWindow.summary.roundCount))
  );
  const currentSlotStartMinute = currentSlotIndex * slotMinutes;
  const currentMinuteProgress = clamp((currentParts.minuteOfDay - currentSlotStartMinute + 1) / Math.max(1, slotMinutes), 0, 1);
  const estimatedRoundsUntilNextWindow = Math.max(0, Math.round(averageRoundsPerWindow * (1 - currentMinuteProgress)));
  const expectedRoundRange = hitMatches.length >= 2
    ? {
        hitMatchCount: hitMatches.length,
        estimatedRoundsUntilNextWindow,
        firstHitOffsetFrom: Math.max(1, Math.round(weightedQuantile(hitMatches, (match) => match.firstHitIndex + 1, (match) => match.weight, 0.2, 1))),
        firstHitOffsetTo: Math.max(1, Math.round(weightedQuantile(hitMatches, (match) => match.firstHitIndex + 1, (match) => match.weight, 0.8, 1))),
      }
    : null;

  return {
    available: true,
    matchMode,
    lookbackDaysUsed,
    currentSlotLabel,
    currentSlotIndex,
    candidateCount: sameTimeCandidates.length,
    usedMatches: matches.length,
    sameWeekdayMatches,
    note: `Built from ${matches.length} matched ${matchMode === 'same-weekday' ? 'same-weekday' : 'same-time'} windows from the last ${lookbackDaysUsed.toFixed(1)} days for ${currentSlotLabel}.`,
    examples,
    averageMatchWeight: Number(weightedAverage(matchRows, (match) => match.weight, () => 1, 0).toFixed(3)),
    expectedRoundRange,
    currentSlotHistory: {
      anyHitRate: slotHistoryAnyHitRate,
      perRoundHitRate: slotHistoryPerRoundRate,
      lift: slotHistoryLift,
      sampleCount: sameTimeCandidates.length,
      weekdaySampleCount: sameWeekdayPool.length,
    },
    nextWindow: {
      occurrenceLabel: formatOccurrenceLabel(nextWindowStartMinute, slotMinutes, slotMode, nextWindowDayOffset),
      dayOffset: nextWindowDayOffset,
      anyHitRate: nextAnyHitRate,
      perRoundHitRate: nextPerRoundHitRate,
      avgPeakMultiplier: weightedAvgPeakMultiplier,
      lift: nextLift,
      tone: nextWindowTone,
      label: nextLift >= 1.08 ? 'Stronger Than Normal'
        : nextLift <= 0.94 ? 'Weaker Than Normal'
          : 'Near Normal',
    },
  };
}

function buildPatternPrediction({
  focusTarget,
  windowLabel,
  latestRoundId,
  currentSummary,
  baselineStats,
  slotAnalytics,
  patternMatch,
}) {
  const slotItem = slotAnalytics.items.find((item) => item.target === focusTarget) || null;
  const currentSlot = slotItem?.currentSlot || null;
  const currentHitRate = currentSummary.hitRates[focusTarget] || 0;
  const baselineHitRate = baselineStats.perRoundHitRates[focusTarget] || 0;
  const baselineNextWindowHitRate = baselineStats.windowAnyHitRates[focusTarget] || 0;
  const currentLift = ratio(currentHitRate, baselineHitRate, baselineHitRate > 0 ? 1 : 0);
  const slotLift = safeNumber(currentSlot?.lift, 1);
  const currentSlotChance = safeNumber(currentSlot?.anyHitChance, 0);
  const isLowTarget = focusTarget <= 20;
  const isMidTarget = focusTarget > 20 && focusTarget <= 100;
  const expectedCurrentHits = baselineHitRate * safeNumber(currentSummary.roundCount, 0);
  const currentEvidenceWeight = clamp(
    expectedCurrentHits / (isLowTarget ? 10 : isMidTarget ? 4 : 2.5),
    0.12,
    1
  );
  const effectiveCurrentLift = 1 + ((currentLift - 1) * currentEvidenceWeight);

  if (!patternMatch?.available) {
    return {
      action: currentLift >= 1.03 ? 'WATCH' : 'WAIT',
      tone: currentLift >= 1.05 ? 'neutral' : 'bad',
      confidence: 0.24,
      confidenceLabel: 'Low',
      predictsLabel: `Next ${windowLabel}`,
      inputLabel: `Current ${windowLabel}`,
      currentSlotLabel: currentSlot?.label || '-',
      nextWindowLabel: '-',
      currentHitRate,
      baselineHitRate,
      baselineNextWindowHitRate,
      currentLift,
      effectiveCurrentLift,
      currentEvidenceWeight,
      currentSlotChance,
      nextWindowHitRate: 0,
      nextWindowLift: 1,
      matchedWindows: 0,
      sameWeekdayMatches: 0,
      lookbackDaysUsed: 0,
      expectedRoundIdFrom: null,
      expectedRoundIdTo: null,
      expectedRoundIdLabel: '-',
      expectedRoundIdBasis: '',
      summary: `Not enough same-time history yet to predict the next ${windowLabel.toLowerCase()} for ${labelForTarget(focusTarget)}.`,
      reasons: [
        `Current ${windowLabel.toLowerCase()} hit rate: ${pctString(currentHitRate)} versus ${pctString(baselineHitRate)} baseline.`,
        'Same-time pattern matcher does not have enough completed windows yet.',
      ],
    };
  }

  const nextWindowHitRate = safeNumber(patternMatch.nextWindow?.anyHitRate, 0);
  const nextWindowLift = safeNumber(patternMatch.nextWindow?.lift, 1);
  const nextWindowEdge = nextWindowHitRate - baselineNextWindowHitRate;
  const matchedWindows = safeNumber(patternMatch.usedMatches, 0);
  const sameWeekdayMatches = safeNumber(patternMatch.sameWeekdayMatches, 0);
  const lookbackDaysUsed = safeNumber(patternMatch.lookbackDaysUsed, 0);
  const safeLatestRoundId = safeNumber(latestRoundId, 0);
  const confidence = clamp(
    0.3
      + (Math.min(matchedWindows, 18) / 18) * 0.35
      + (Math.min(Math.abs(nextWindowLift - 1), 0.22) / 0.22) * 0.22
      + (Math.min(sameWeekdayMatches, 8) / 8) * 0.09,
    0.18,
    0.96
  );
  const confidenceLabel = confidence >= 0.74 ? 'High' : confidence >= 0.52 ? 'Medium' : 'Low';

  const playLiftThreshold = isLowTarget ? 1.02 : isMidTarget ? 1.06 : 1.12;
  const playEdgeThreshold = isLowTarget ? 0.02 : isMidTarget ? 0.05 : 0.03;
  const skipLiftThreshold = isLowTarget ? 0.88 : isMidTarget ? 0.92 : 0.94;
  const weakCurrentThreshold = isLowTarget ? 0.84 : isMidTarget ? 0.88 : 0.9;
  const solidCurrentThreshold = isLowTarget ? 0.94 : isMidTarget ? 0.92 : 0.9;
  const strongAbsoluteHitRate = isLowTarget ? 0.52 : isMidTarget ? 0.22 : 0.08;
  const permissiveCurrentThreshold = isLowTarget ? 0.85 : isMidTarget ? 0.88 : 0.9;

  let action = 'WATCH';
  let tone = 'neutral';
  let summary = `This predicts the next ${windowLabel.toLowerCase()}, not the current one. Matched ${patternMatch.currentSlotLabel} history says the next ${windowLabel.toLowerCase()} is close to normal for ${labelForTarget(focusTarget)}.`;

  if (
    ((nextWindowLift >= playLiftThreshold && nextWindowEdge >= playEdgeThreshold) || (nextWindowHitRate >= strongAbsoluteHitRate && nextWindowLift >= (isLowTarget ? 1.05 : 1.01)))
    && effectiveCurrentLift >= solidCurrentThreshold
  ) {
    action = 'PLAY';
    tone = 'good';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. The current ${windowLabel.toLowerCase()} is holding up, and matched ${patternMatch.currentSlotLabel} history says the next ${windowLabel.toLowerCase()} is good for ${labelForTarget(focusTarget)}.`;
  } else if (
    nextWindowHitRate >= strongAbsoluteHitRate
    && nextWindowLift >= (isLowTarget ? 1.05 : playLiftThreshold)
    && effectiveCurrentLift >= permissiveCurrentThreshold
  ) {
    action = 'PLAY';
    tone = 'good';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. Even though the current ${windowLabel.toLowerCase()} is only average, matched ${patternMatch.currentSlotLabel} history is strong enough to play the next ${windowLabel.toLowerCase()} for ${labelForTarget(focusTarget)}.`;
  } else if (nextWindowLift <= skipLiftThreshold && effectiveCurrentLift <= weakCurrentThreshold && slotLift <= 0.95) {
    action = 'SKIP';
    tone = 'bad';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. Current results are weak, and matched ${patternMatch.currentSlotLabel} history says the next ${windowLabel.toLowerCase()} is below normal for ${labelForTarget(focusTarget)}.`;
  } else if (nextWindowLift >= Math.max(1.0, playLiftThreshold - 0.03) || nextWindowEdge >= (playEdgeThreshold * 0.5) || effectiveCurrentLift >= 1.02 || slotLift >= 1) {
    action = 'WATCH';
    tone = 'good';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. Matched ${patternMatch.currentSlotLabel} history leans positive, but not strong enough yet for a full play call on ${labelForTarget(focusTarget)}.`;
  } else if (effectiveCurrentLift >= 1.08 && nextWindowLift >= 1) {
    action = 'WATCH';
    tone = 'neutral';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. The current ${windowLabel.toLowerCase()} is running hotter than normal, but matched next-window history is only average for ${labelForTarget(focusTarget)}.`;
  } else {
    action = 'WAIT';
    tone = 'neutral';
    summary = `This predicts the next ${windowLabel.toLowerCase()}. Current results and matched ${patternMatch.currentSlotLabel} history are mixed, so it is better to wait for a clearer edge on ${labelForTarget(focusTarget)}.`;
  }

  const expectedRoundIdFrom = patternMatch.expectedRoundRange && safeLatestRoundId > 0
    ? safeLatestRoundId + patternMatch.expectedRoundRange.estimatedRoundsUntilNextWindow + patternMatch.expectedRoundRange.firstHitOffsetFrom
    : null;
  const expectedRoundIdTo = patternMatch.expectedRoundRange && safeLatestRoundId > 0
    ? safeLatestRoundId + patternMatch.expectedRoundRange.estimatedRoundsUntilNextWindow + patternMatch.expectedRoundRange.firstHitOffsetTo
    : null;
  const expectedRoundIdLabel = (expectedRoundIdFrom && expectedRoundIdTo)
    ? `#${expectedRoundIdFrom} - #${expectedRoundIdTo}`
    : '-';
  const expectedRoundIdBasis = patternMatch.expectedRoundRange
    ? `Based on ${patternMatch.expectedRoundRange.hitMatchCount} matched next-window hits.`
    : '';

  return {
    action,
    tone,
    confidence,
    confidenceLabel,
    predictsLabel: `Next ${windowLabel}`,
    inputLabel: `Current ${windowLabel}`,
    currentSlotLabel: patternMatch.currentSlotLabel,
    nextWindowLabel: patternMatch.nextWindow?.occurrenceLabel || '-',
    currentHitRate,
    baselineHitRate,
    baselineNextWindowHitRate,
    currentLift,
    effectiveCurrentLift,
    currentEvidenceWeight,
    currentSlotChance,
    nextWindowHitRate,
    nextWindowLift,
    matchedWindows,
    sameWeekdayMatches,
    lookbackDaysUsed,
    expectedRoundIdFrom,
    expectedRoundIdTo,
    expectedRoundIdLabel,
    expectedRoundIdBasis,
    summary,
    reasons: [
      `Input: current ${windowLabel.toLowerCase()} hit rate for ${labelForTarget(focusTarget)} is ${pctString(currentHitRate)} versus ${pctString(baselineHitRate)} per-round baseline.`,
      `${patternMatch.currentSlotLabel} matched ${matchedWindows} past windows over ${lookbackDaysUsed.toFixed(1)} days with same-time weighting.`,
      `Prediction: the next ${windowLabel.toLowerCase()} hit ${labelForTarget(focusTarget)} ${pctString(nextWindowHitRate)} of the time in those matches versus ${pctString(baselineNextWindowHitRate)} normal next-window baseline.`,
      expectedRoundIdBasis ? `Expected ${labelForTarget(focusTarget)} around rounds ${expectedRoundIdLabel}. ${expectedRoundIdBasis}` : null,
    ].filter(Boolean),
  };
}

function buildHourlyHistory(rounds, timeZone, baselinePerRoundRates) {
  const getParts = buildZonedPartsGetter(timeZone);
  const buckets = Array.from({ length: 24 }, (_, hour) => ({
    hour,
    roundCount: 0,
    sumMultiplier: 0,
    hits: createTargetMap(0),
  }));

  for (const round of rounds) {
    const hour = getParts(round.timestamp).hour;
    const bucket = buckets[hour];
    bucket.roundCount += 1;
    bucket.sumMultiplier += round.multiplier;
    for (const target of TARGETS) {
      if (round.multiplier >= target) bucket.hits[target] += 1;
    }
  }

  const rows = buckets.map((bucket) => {
    const targetRates = createTargetMap((target) => ratio(bucket.hits[target], bucket.roundCount));
    let bestTarget = '-';
    let bestLift = -Infinity;
    for (const target of TARGETS) {
      const lift = ratio(targetRates[target], baselinePerRoundRates[target], 1);
      if (lift > bestLift) {
        bestLift = lift;
        bestTarget = labelForTarget(target);
      }
    }
    return {
      hour: bucket.hour,
      label: formatHourLabel(bucket.hour),
      roundCount: bucket.roundCount,
      avgMultiplier: ratio(bucket.sumMultiplier, bucket.roundCount),
      targetRates,
      bestTarget,
    };
  });

  const bestHours = TARGETS.map((target) => {
    const sorted = [...rows]
      .filter((row) => row.roundCount > 0)
      .sort((a, b) => b.targetRates[target] - a.targetRates[target]);
    const best = sorted[0];
    const worst = sorted[sorted.length - 1];
    return {
      target,
      label: labelForTarget(target),
      bestHour: best?.hour ?? null,
      bestLabel: best?.label || '-',
      bestHitRate: best?.targetRates?.[target] || 0,
      worstHour: worst?.hour ?? null,
      worstLabel: worst?.label || '-',
      worstHitRate: worst?.targetRates?.[target] || 0,
    };
  });

  return {
    timeZone,
    rows,
    bestHours,
  };
}

function buildHeatmap(rounds, timeZone, focusTarget, baselinePerRoundRates) {
  const getParts = buildZonedPartsGetter(timeZone);
  const cells = Array.from({ length: 7 }, (_, dayIndex) => (
    Array.from({ length: 24 }, (_, hour) => ({
      dayIndex,
      dayLabel: WEEKDAYS[dayIndex],
      hour,
      hourLabel: formatHourLabel(hour),
      roundCount: 0,
      hitCount: 0,
      hitRate: 0,
      lift: 1,
      tone: 'neutral',
    }))
  ));

  for (const round of rounds) {
    const parts = getParts(round.timestamp);
    const cell = cells[parts.dayIndex][parts.hour];
    cell.roundCount += 1;
    if (round.multiplier >= focusTarget) cell.hitCount += 1;
  }

  const flat = [];
  for (const row of cells) {
    for (const cell of row) {
      cell.hitRate = ratio(cell.hitCount, cell.roundCount);
      cell.lift = ratio(cell.hitRate, baselinePerRoundRates[focusTarget], 1);
      cell.tone = classifyLift(cell.lift, Math.ceil(cell.roundCount / 20)).tone;
      flat.push(cell);
    }
  }

  const ranked = flat.filter((cell) => cell.roundCount > 0);
  const strongest = [...ranked]
    .sort((a, b) => {
      if (b.lift !== a.lift) return b.lift - a.lift;
      return b.hitRate - a.hitRate;
    })
    .slice(0, 3)
    .map((cell) => ({
      label: `${cell.dayLabel} ${cell.hourLabel}`,
      hitRate: cell.hitRate,
      lift: cell.lift,
      roundCount: cell.roundCount,
    }));
  const weakest = [...ranked]
    .sort((a, b) => {
      if (a.lift !== b.lift) return a.lift - b.lift;
      return a.hitRate - b.hitRate;
    })
    .slice(0, 3)
    .map((cell) => ({
      label: `${cell.dayLabel} ${cell.hourLabel}`,
      hitRate: cell.hitRate,
      lift: cell.lift,
      roundCount: cell.roundCount,
    }));

  return {
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    days: WEEKDAYS,
    hours: Array.from({ length: 24 }, (_, hour) => ({
      value: hour,
      label: formatHourLabel(hour),
    })),
    cells,
    strongest,
    weakest,
  };
}

function buildLastHitMap(rounds) {
  const lastHits = createTargetMap(() => ({ roundId: null, timestamp: null, multiplier: null }));
  for (let index = rounds.length - 1; index >= 0; index -= 1) {
    const round = rounds[index];
    for (const target of TARGETS) {
      if (lastHits[target].roundId == null && round.multiplier >= target) {
        lastHits[target] = {
          roundId: round.roundId,
          timestamp: round.timestamp,
          multiplier: round.multiplier,
        };
      }
    }
  }
  return lastHits;
}

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
          key: window.key,
          label: window.label,
          anyHitRate: 0,
          perRoundHitRate: 0,
          baselinePerRoundRate: baselinePerRoundRates[target] || 0,
          lift: 1,
          sampleCount: 0,
          status: 'neutral',
        };
      }

      let totalHits = 0;
      let totalRounds = 0;
      let totalAnyHit = 0;
      let right = 0;

      for (const index of eventIndices) {
        if (right < index + 1) right = index + 1;
        while (right < rounds.length && rounds[right].timestamp <= (rounds[index].timestamp + window.ms)) {
          right += 1;
        }
        const roundsInRange = Math.max(0, right - (index + 1));
        const hitsInRange = prefixes[target][right] - prefixes[target][index + 1];
        totalRounds += roundsInRange;
        totalHits += hitsInRange;
        if (hitsInRange > 0) totalAnyHit += 1;
      }

      const anyHitRate = ratio(totalAnyHit, eventIndices.length);
      const perRoundHitRate = ratio(totalHits, totalRounds);
      const lift = ratio(perRoundHitRate, baselinePerRoundRates[target], 1);
      const status = classifyLift(lift, eventIndices.length).key;
      return {
        key: window.key,
        label: window.label,
        anyHitRate,
        perRoundHitRate,
        baselinePerRoundRate: baselinePerRoundRates[target] || 0,
        lift,
        sampleCount: eventIndices.length,
        status,
      };
    });

    const recentHit = lastHits[target];
    const ageMs = recentHit?.timestamp ? latestTimestamp - recentHit.timestamp : null;
    let recentPressure = null;
    if (ageMs != null) {
      const activeIndex = COOLDOWN_WINDOWS.findIndex((window) => ageMs <= window.ms);
      const activeWindow = activeIndex >= 0 ? horizons[activeIndex] : null;
      if (activeWindow) {
        recentPressure = {
          ageMs,
          activeWindow: activeWindow.label,
          lift: activeWindow.lift,
          status: activeWindow.status,
          note: activeWindow.lift < 1
            ? `${labelForTarget(target)} usually cools down in the ${activeWindow.label.toLowerCase()} after it lands.`
            : `${labelForTarget(target)} often stays active in the ${activeWindow.label.toLowerCase()} after it lands.`,
        };
      }
    }

    items.push({
      target,
      label: labelForTarget(target),
      lastHitRoundId: recentHit?.roundId || null,
      lastHitTimestamp: recentHit?.timestamp || null,
      ageMs,
      horizons,
      recentPressure,
    });
  }

  return items;
}

function buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, latestTimestamp) {
  return TARGETS.map((target) => {
    const baselineRate = baselineStats.perRoundHitRates[target] || 0;
    const currentRate = currentSummary.hitRates[target] || 0;
    const slotItem = slotAnalytics.items.find((item) => item.target === target);
    const slotLift = slotItem?.currentSlot?.lift || 1;
    const rateLift = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
    const cooldown = cooldowns.find((item) => item.target === target);
    let cooldownPenalty = 0;
    if (cooldown?.recentPressure && cooldown.recentPressure.lift < 0.95) {
      cooldownPenalty = (0.95 - cooldown.recentPressure.lift) * 25;
    }

    const score = clamp(
      50
      + clamp((rateLift - 1) * 38, -20, 22)
      + clamp((slotLift - 1) * 22, -12, 14)
      - cooldownPenalty,
      0,
      100
    );

    let status = 'neutral';
    let label = 'Neutral';
    if (score >= 64) {
      status = 'ready';
      label = 'Ready';
    } else if (score <= 42) {
      status = 'avoid';
      label = 'Avoid';
    }

    let reason = `${labelForTarget(target)} is close to its usual pace.`;
    if (rateLift >= 1.12 && slotLift >= 1.05) {
      reason = `${labelForTarget(target)} is running above normal and this time block is supportive.`;
    } else if (rateLift <= 0.9) {
      reason = `${labelForTarget(target)} is landing below its usual rate in the current window.`;
    } else if (slotLift <= 0.9) {
      reason = `${labelForTarget(target)} is in a historically weak time block right now.`;
    } else if (cooldown?.recentPressure && cooldown.recentPressure.lift < 0.95) {
      reason = `${labelForTarget(target)} is in a post-hit cooldown zone right now.`;
    }

    return {
      target,
      label: labelForTarget(target),
      score,
      status,
      statusLabel: label,
      currentHitRate: currentRate,
      baselineHitRate: baselineRate,
      currentLift: rateLift,
      slotLift,
      reason,
      lastUpdatedAt: latestTimestamp,
    };
  });
}

function buildRegime(focusTarget, currentSummary, baselineStats, cooldowns) {
  const focusLift = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const hugeLift = ratio(currentSummary.hugeHitRate, baselineStats.hugeHitRate, 1);
  const megaLift = ratio(currentSummary.megaHitRate, baselineStats.megaHitRate, 1);
  const lowCrashLift = ratio(currentSummary.lowCrashRate, baselineStats.lowCrashRate, 1);
  const recent100 = cooldowns.find((item) => item.target === 100)?.recentPressure;
  const recent500 = cooldowns.find((item) => item.target === 500)?.recentPressure;

  if ((recent500 && recent500.lift < 0.95) || (recent100 && recent100.lift < 0.95 && megaLift < 1)) {
    return {
      key: 'post-spike-cooldown',
      label: 'Post-Spike Cooldown',
      tone: 'bad',
      description: 'A recent high spike usually cools the board down for a while, so chasing is riskier right now.',
    };
  }

  if (megaLift >= 1.35 || hugeLift >= 1.25) {
    return {
      key: 'spike-mode',
      label: 'Spike Mode',
      tone: 'good',
      description: 'High multipliers are arriving above their normal pace, so aggressive targets have better support.',
    };
  }

  if (lowCrashLift >= 1.15 && focusLift <= 0.95) {
    return {
      key: 'low-mode',
      label: 'Low Mode',
      tone: 'bad',
      description: 'Short crashes are stacking up and the selected target is lagging, so this is a tougher stretch.',
    };
  }

  if (focusLift >= 1.12 && lowCrashLift <= 1) {
    return {
      key: 'target-friendly',
      label: 'Target-Friendly',
      tone: 'good',
      description: `The current flow is leaning toward ${labelForTarget(focusTarget)} more than usual.`,
    };
  }

  return {
    key: 'balanced',
    label: 'Balanced Mode',
    tone: 'neutral',
    description: 'The board is close to its normal mix, so time-of-day strength matters more than raw momentum.',
  };
}

function buildDecision({
  focusTarget,
  windowLabel,
  currentSummary,
  baselineStats,
  slotAnalytics,
  cooldowns,
  readiness,
}) {
  const focusReadiness = readiness.find((item) => item.target === focusTarget);
  const focusSlot = slotAnalytics.items.find((item) => item.target === focusTarget)?.currentSlot;
  const cooldown = cooldowns.find((item) => item.target === focusTarget);
  const baselineRate = baselineStats.perRoundHitRates[focusTarget] || 0;
  const currentRate = currentSummary.hitRates[focusTarget] || 0;
  const rateLift = ratio(currentRate, baselineRate, baselineRate > 0 ? 1 : 0);
  const avgLift = ratio(currentSummary.avgMultiplier, baselineStats.avgMultiplier, baselineStats.avgMultiplier > 0 ? 1 : 0);
  const lowCrashRelief = baselineStats.lowCrashRate > 0
    ? (baselineStats.lowCrashRate - currentSummary.lowCrashRate) / baselineStats.lowCrashRate
    : 0;
  const slotLift = focusSlot?.lift || 1;
  const cooldownPenalty = cooldown?.recentPressure && cooldown.recentPressure.lift < 0.95
    ? (0.95 - cooldown.recentPressure.lift) * 24
    : 0;

  const score = clamp(
    50
    + clamp((rateLift - 1) * 32, -18, 22)
    + clamp((slotLift - 1) * 20, -12, 14)
    + clamp((avgLift - 1) * 12, -8, 8)
    + clamp(lowCrashRelief * 18, -10, 10)
    - cooldownPenalty,
    0,
    100
  );

  const band = describeBand(score);
  const zone = classifyLift(slotLift, focusSlot?.sampleCount || 0);
  const playReasons = [];
  const skipReasons = [];

  if (rateLift >= 1.1) playReasons.push(`${labelForTarget(focusTarget)} hit rate is above its normal baseline for this window.`);
  if (slotLift >= 1.1) playReasons.push(`This local time block is historically stronger than usual for ${labelForTarget(focusTarget)}.`);
  if (currentSummary.lowCrashRate <= baselineStats.lowCrashRate * 0.92) playReasons.push('Low crashes are lighter than usual, which supports safer chase conditions.');
  if (avgLift >= 1.08) playReasons.push('Average multiplier in the current window is running hotter than normal.');

  if (rateLift <= 0.92) skipReasons.push(`${labelForTarget(focusTarget)} is underperforming versus its usual hit rate.`);
  if (slotLift <= 0.9) skipReasons.push('This time block is historically weak for the selected target.');
  if (currentSummary.lowCrashRate >= baselineStats.lowCrashRate * 1.1) skipReasons.push('Low crashes are stacking above normal.');
  if (cooldown?.recentPressure && cooldown.recentPressure.lift < 0.95) skipReasons.push(cooldown.recentPressure.note);

  if (!playReasons.length) playReasons.push(`Nothing exceptional is boosting ${labelForTarget(focusTarget)} right now.`);
  if (!skipReasons.length) skipReasons.push('No major red flag is standing out right now.');

  const regime = buildRegime(focusTarget, currentSummary, baselineStats, cooldowns);

  return {
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    score,
    band: band.key,
    label: band.label,
    tone: band.tone,
    windowLabel,
    zone: {
      key: zone.key,
      label: zone.label,
      tone: zone.tone,
      lift: slotLift,
      anyHitRate: focusSlot?.anyHitChance || 0,
      baselineAnyHitRate: slotAnalytics.baselineByTarget[focusTarget]?.anyHitRate || 0,
      currentSlotLabel: focusSlot?.label || '-',
    },
    regime,
    readiness: focusReadiness || null,
    playReasons,
    skipReasons,
    summary: band.key === 'play'
      ? `${labelForTarget(focusTarget)} is in a stronger-than-usual ${windowLabel.toLowerCase()} and this local time block is supportive.`
      : band.key === 'skip'
        ? `${labelForTarget(focusTarget)} is running colder than normal for this ${windowLabel.toLowerCase()}, so it is a better skip window.`
        : `${labelForTarget(focusTarget)} is mixed right now, so this is more of a wait-and-watch window than a clear chase spot.`,
  };
}

function scoreWindowDecision(windowSummary, baselineStats, slotStat) {
  const scoreByTarget = {};
  for (const target of TARGETS) {
    const rateLift = ratio(windowSummary.hitRates[target], baselineStats.perRoundHitRates[target], baselineStats.perRoundHitRates[target] > 0 ? 1 : 0);
    const slotLift = slotStat?.lift || 1;
    scoreByTarget[target] = clamp(
      50
      + clamp((rateLift - 1) * 32, -18, 22)
      + clamp((slotLift - 1) * 20, -12, 14),
      0,
      100
    );
  }
  return scoreByTarget;
}

function buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget) {
  const getParts = buildZonedPartsGetter(timeZone);
  const recent = completedWindows.slice(-8);
  if (!recent.length) {
    return {
      score: 0,
      label: 'Low Stability',
      status: 'unstable',
      flipCount: 0,
      windowsChecked: 0,
      message: 'Not enough completed windows yet to judge signal stability.',
      bands: [],
    };
  }

  const slotMinutes = slotAnalytics.slotMinutes;
  const slotStats = slotAnalytics.slotStatsByTarget[focusTarget] || [];
  const slotStatsMap = new Map(slotStats.map((slot) => [slot.slotIndex, slot]));
  const bands = recent.map((window) => {
    const parts = getParts(window.startTimestamp);
    const slotIndex = clamp(Math.floor(parts.minuteOfDay / slotMinutes), 0, Math.max(0, Math.floor(1440 / slotMinutes) - 1));
    const score = scoreWindowDecision(window.summary, baselineStats, slotStatsMap.get(slotIndex))[focusTarget];
    const band = describeBand(score);
    return {
      startTimestamp: window.startTimestamp,
      score,
      band: band.key,
      label: band.label,
    };
  });

  let flipCount = 0;
  for (let index = 1; index < bands.length; index += 1) {
    if (bands[index].band !== bands[index - 1].band) flipCount += 1;
  }

  const counts = bands.reduce((acc, item) => {
    acc[item.band] = (acc[item.band] || 0) + 1;
    return acc;
  }, {});
  const dominantCount = Math.max(...Object.values(counts));
  const consistency = ratio(dominantCount, bands.length);
  const score = Math.round(clamp((consistency * 70) + ((1 - ratio(flipCount, Math.max(1, bands.length - 1))) * 30), 0, 100));

  let status = 'mixed';
  let label = 'Medium Stability';
  let message = 'The recommendation is moving around, so keep position size smaller.';
  if (score >= 72) {
    status = 'stable';
    label = 'High Stability';
    message = 'The same decision band has held across several completed windows, so the signal is steadier.';
  } else if (score <= 46) {
    status = 'unstable';
    label = 'Low Stability';
    message = 'The signal has been flipping a lot between windows, so treat it as weak.';
  }

  return {
    score,
    label,
    status,
    flipCount,
    windowsChecked: bands.length,
    message,
    bands,
  };
}

function buildBacktest(slotWindows, slotAnalytics, focusTarget) {
  const focusItem = slotAnalytics.items.find((item) => item.target === focusTarget);
  if (!focusItem) {
    return {
      focusTarget,
      focusTargetLabel: labelForTarget(focusTarget),
      summary: 'Not enough time-slot history yet for a backtest.',
      allWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 },
      greenWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 },
      redWindows: { count: 0, anyHitRate: 0, avgPeakMultiplier: 0, lift: 1 },
    };
  }

  const greenSlots = new Set(
    (focusItem.topSlots || [])
      .filter((slot) => slot.lift >= 1.08 && slot.sampleCount >= slotAnalytics.minSamples)
      .map((slot) => slot.slotIndex)
  );
  const redSlots = new Set(
    (slotAnalytics.slotStatsByTarget[focusTarget] || [])
      .filter((slot) => slot.lift <= 0.92 && slot.sampleCount >= slotAnalytics.minSamples)
      .map((slot) => slot.slotIndex)
  );

  const summarize = (predicate) => {
    const selected = slotWindows.filter(predicate);
    if (!selected.length) return { count: 0, anyHitRate: 0, avgPeakMultiplier: 0 };
    let hitWindows = 0;
    let peakSum = 0;
    for (const slotWindow of selected) {
      if ((slotWindow.summary.hitCounts[focusTarget] || 0) > 0) hitWindows += 1;
      peakSum += slotWindow.summary.maxMultiplier;
    }
    return {
      count: selected.length,
      anyHitRate: ratio(hitWindows, selected.length),
      avgPeakMultiplier: ratio(peakSum, selected.length),
    };
  };

  const allWindows = summarize(() => true);
  const greenWindows = summarize((slotWindow) => greenSlots.has(slotWindow.slotIndex));
  const redWindows = summarize((slotWindow) => redSlots.has(slotWindow.slotIndex));

  return {
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    summary: greenWindows.count > 0
      ? `Green windows hit ${labelForTarget(focusTarget)} ${pctString(greenWindows.anyHitRate)} of the time versus ${pctString(allWindows.anyHitRate)} across all windows.`
      : 'There are not enough strong time slots yet to run a useful green-window backtest.',
    allWindows,
    greenWindows: {
      ...greenWindows,
      lift: ratio(greenWindows.anyHitRate, allWindows.anyHitRate, 1),
    },
    redWindows: {
      ...redWindows,
      lift: ratio(redWindows.anyHitRate, allWindows.anyHitRate, 1),
    },
  };
}

function buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday) {
  const currentLift = ratio(currentSummary.hitRates[focusTarget], baselineStats.perRoundHitRates[focusTarget], 1);
  const currentWindow = bestWindowsToday.items.find((item) => item.target === focusTarget);
  const zoneLabel = currentWindow?.currentSlot?.zoneLabel || decision.zone.label;
  let message = `${labelForTarget(focusTarget)} is running close to its usual pace.`;
  if (currentLift >= 1.12) {
    message = `${labelForTarget(focusTarget)} is hotter than normal, and the current time block is ${zoneLabel.toLowerCase()}.`;
  } else if (currentLift <= 0.9) {
    message = `${labelForTarget(focusTarget)} is colder than normal, and the current time block leans ${zoneLabel.toLowerCase()}.`;
  } else {
    message = `${labelForTarget(focusTarget)} is mixed right now, so use the zone, cooldown, and stability panels together.`;
  }
  return {
    band: decision.band,
    label: decision.label,
    message,
  };
}

function buildTargetCards(currentSummary, baselineStats) {
  return TARGETS.map((target) => {
    const currentHitRate = currentSummary.hitRates[target] || 0;
    const baselineHitRate = baselineStats.perRoundHitRates[target] || 0;
    return {
      target,
      label: labelForTarget(target),
      currentHitRate,
      baselineHitRate,
      delta: currentHitRate - baselineHitRate,
      lift: ratio(currentHitRate, baselineHitRate, baselineHitRate > 0 ? 1 : 0),
    };
  });
}

function buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) {
  const usable = completedWindows.filter((_, index) => index < completedWindows.length - 1);
  if (usable.length < 10) {
    return {
      available: false,
      reason: 'Not enough completed windows yet to build a next-window outlook.',
    };
  }

  const currentVector = {
    focusHitRate: currentSummary.hitRates[focusTarget] || 0,
    lowCrashRate: currentSummary.lowCrashRate || 0,
    hugeHitRate: currentSummary.hugeHitRate || 0,
    avgMultiplier: currentSummary.avgMultiplier || 0,
    maxMultiplier: currentSummary.maxMultiplier || 0,
  };

  const candidates = usable
    .map((window, index) => {
      const nextWindow = completedWindows[index + 1];
      if (!nextWindow) return null;
      const summary = window.summary;
      const distance =
        Math.abs((summary.hitRates[focusTarget] || 0) - currentVector.focusHitRate) * 4.5
        + Math.abs((summary.lowCrashRate || 0) - currentVector.lowCrashRate) * 2.6
        + Math.abs((summary.hugeHitRate || 0) - currentVector.hugeHitRate) * 1.8
        + Math.abs((summary.avgMultiplier || 0) - currentVector.avgMultiplier) / Math.max(2, currentVector.avgMultiplier || 2)
        + Math.abs((summary.maxMultiplier || 0) - currentVector.maxMultiplier) / Math.max(20, currentVector.maxMultiplier || 20);
      return {
        distance,
        nextWindow,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.distance - b.distance);

  const sampleSize = clamp(Math.round(candidates.length * 0.18), 12, 60);
  const matches = candidates.slice(0, sampleSize);
  if (!matches.length) {
    return {
      available: false,
      reason: 'No close historical match was found for the current window shape.',
    };
  }

  const candidateRows = TARGETS.map((target) => {
    let windowHits = 0;
    let totalHits = 0;
    let totalRounds = 0;
    let peakSum = 0;
    for (const match of matches) {
      const summary = match.nextWindow.summary;
      totalHits += summary.hitCounts[target] || 0;
      totalRounds += summary.roundCount;
      peakSum += summary.maxMultiplier;
      if ((summary.hitCounts[target] || 0) > 0) windowHits += 1;
    }
    const anyHitRate = ratio(windowHits, matches.length);
    const perRoundHitRate = ratio(totalHits, totalRounds);
    const baselineAnyHitRate = baselineStats.windowAnyHitRates[target] || 0;
    const lift = ratio(anyHitRate, baselineAnyHitRate, baselineAnyHitRate > 0 ? 1 : 0);
    const rewardWeight = 1 + (Math.log10(target) / 2);
    const score = anyHitRate * Math.max(0.3, lift) * rewardWeight;
    return {
      target,
      label: labelForTarget(target),
      anyHitRate,
      perRoundHitRate,
      avgPeakMultiplier: ratio(peakSum, matches.length),
      expectedHits: Math.round(ratio(totalHits, matches.length) * 100) / 100,
      baselineAnyHitRate,
      lift,
      score,
      style: anyHitRate >= 0.55 ? 'Frequent' : anyHitRate >= 0.3 ? 'Balanced' : 'Long-shot',
    };
  }).sort((a, b) => b.score - a.score);

  const recommendation = candidateRows[0] || null;
  const focusCandidate = candidateRows.find((item) => item.target === focusTarget) || null;
  const distanceSpread = average(matches.map((item) => item.distance));
  const confidence = clamp(
    (sampleWeight(matches.length) * 0.55)
    + (1 / (1 + distanceSpread)) * 0.45,
    0,
    1
  );

  return {
    available: true,
    basedOnMatches: matches.length,
    confidence,
    note: `Built from ${matches.length} completed windows that looked most similar to the current one.`,
    recommendation,
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    focusCandidate,
    candidates: candidateRows,
  };
}

function buildBestWindowsToday(slotAnalytics, focusTarget) {
  return {
    slotMode: slotAnalytics.slotMode,
    slotMinutes: slotAnalytics.slotMinutes,
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    note: slotAnalytics.slotMode === 'start-time'
      ? 'For multi-day windows, this ranks the best starting times rather than same-day end times.'
      : 'These are the best recurring local-time windows from the stored dataset.',
    items: slotAnalytics.items.map((item) => ({
      target: item.target,
      label: item.label,
      currentSlot: item.currentSlot,
      nextTodayWindow: item.nextTodayWindow,
      nextWindow: item.nextWindow,
      todayWindows: item.todayWindows,
      backups: item.backups,
      avoidWindow: item.avoidWindow,
      bestWindow: item.topSlots[0] || null,
      topWindows: item.topSlots,
    })),
  };
}

function buildEmptyReport(windowConfig, focusTarget, timeZone) {
  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId: null,
    totalRounds: 0,
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets: TARGETS.map((value) => ({ value, label: labelForTarget(value) })),
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
      startTimestamp: null,
      endTimestamp: null,
    },
    baseline: summarizeWindowCollection([], focusTarget),
    currentWindow: summarizeRounds([], focusTarget),
    patternPrediction: {
      action: 'WAIT FOR MORE DATA',
      tone: 'neutral',
      confidence: 0,
      confidenceLabel: 'Low',
      predictsLabel: `Next ${windowConfig.label}`,
      inputLabel: `Current ${windowConfig.label}`,
      currentSlotLabel: '-',
      nextWindowLabel: '-',
      currentHitRate: 0,
      baselineHitRate: 0,
      baselineNextWindowHitRate: 0,
      currentLift: 1,
      effectiveCurrentLift: 1,
      currentEvidenceWeight: 0,
      currentSlotChance: 0,
      nextWindowHitRate: 0,
      nextWindowLift: 1,
      matchedWindows: 0,
      sameWeekdayMatches: 0,
      lookbackDaysUsed: 0,
      expectedRoundIdFrom: null,
      expectedRoundIdTo: null,
      expectedRoundIdLabel: '-',
      expectedRoundIdBasis: '',
      summary: 'No rounds are stored yet, so there is no timing prediction.',
      reasons: [],
    },
    comparison: {
      band: 'wait',
      label: 'WAIT / WATCH',
      message: 'No rounds are stored yet, so timing analytics cannot be computed.',
    },
    targetCards: [],
    decision: null,
    targetReadiness: [],
    recommendationStability: null,
    bestWindowsToday: { items: [], slotMode: 'window', slotMinutes: chooseSlotMinutes(windowConfig.ms), note: '' },
    patternMatch: { available: false, examples: [], reason: 'No rounds are stored yet.' },
    cooldowns: [],
    backtest: null,
    hourlyHistory: { timeZone, rows: [], bestHours: [] },
    dayHourHeatmap: {
      focusTarget,
      focusTargetLabel: labelForTarget(focusTarget),
      days: WEEKDAYS,
      hours: Array.from({ length: 24 }, (_, hour) => ({ value: hour, label: formatHourLabel(hour) })),
      cells: [],
      strongest: [],
      weakest: [],
    },
    outlook: null,
  };
}

function buildTimingAnalyticsReport(rounds, options = {}) {
  const windowKey = normalizeTimingWindowKey(options.windowKey);
  const focusTarget = normalizeTimingTarget(options.focusTarget);
  const timeZone = normalizeTimingTimeZone(options.timeZone);
  const includeOutlook = Boolean(options.includeOutlook);
  const windowConfig = WINDOW_MAP.get(windowKey);
  const normalizedRounds = normalizeRounds(rounds);

  if (!normalizedRounds.length) {
    return buildEmptyReport(windowConfig, focusTarget, timeZone);
  }

  const latestRound = normalizedRounds[normalizedRounds.length - 1];
  const earliestRound = normalizedRounds[0];
  const currentStart = latestRound.timestamp - windowConfig.ms;
  const currentRounds = normalizedRounds.filter((round) => round.timestamp > currentStart);
  const currentSummary = summarizeRounds(currentRounds, focusTarget);

  const fixedWindows = segmentRoundsByWindow(normalizedRounds, windowConfig.ms, focusTarget);
  const completedWindows = fixedWindows.slice(0, -1);
  const baselineStats = completedWindows.length
    ? summarizeWindowCollection(completedWindows, focusTarget)
    : summarizeWindowCollection(fixedWindows, focusTarget);

  const slotMinutes = chooseSlotMinutes(windowConfig.ms);
  const slotWindows = buildSlotWindows(normalizedRounds, slotMinutes, timeZone, focusTarget);
  const slotAnalytics = buildSlotAnalytics(slotWindows, slotMinutes, windowConfig.ms, focusTarget, latestRound.timestamp, timeZone);
  const bestWindowsToday = buildBestWindowsToday(slotAnalytics, focusTarget);
  const lastHits = buildLastHitMap(normalizedRounds);
  const cooldowns = buildCooldownReport(normalizedRounds, baselineStats.perRoundHitRates, latestRound.timestamp, lastHits);
  const targetReadiness = buildTargetReadiness(currentSummary, baselineStats, slotAnalytics, cooldowns, latestRound.timestamp);
  const decision = buildDecision({
    focusTarget,
    windowLabel: windowConfig.label,
    currentSummary,
    baselineStats,
    slotAnalytics,
    cooldowns,
    readiness: targetReadiness,
  });
  const recommendationStability = buildStability(completedWindows, baselineStats, slotAnalytics, timeZone, focusTarget);
  const hourlyHistory = buildHourlyHistory(normalizedRounds, timeZone, baselineStats.perRoundHitRates);
  const dayHourHeatmap = buildHeatmap(normalizedRounds, timeZone, focusTarget, baselineStats.perRoundHitRates);
  const backtest = buildBacktest(slotWindows, slotAnalytics, focusTarget);
  const comparison = buildComparison(decision, focusTarget, currentSummary, baselineStats, bestWindowsToday);
  const targetCards = buildTargetCards(currentSummary, baselineStats);
  const patternMatch = buildPatternMatchReport(
    slotWindows,
    currentSummary,
    baselineStats,
    focusTarget,
    latestRound.timestamp,
    timeZone,
    slotAnalytics
  );
  const patternPrediction = buildPatternPrediction({
    focusTarget,
    windowLabel: windowConfig.label,
    latestRoundId: latestRound.roundId || null,
    currentSummary,
    baselineStats,
    slotAnalytics,
    patternMatch,
  });
  const outlook = includeOutlook ? buildOutlook(completedWindows, currentSummary, baselineStats, focusTarget) : null;

  return {
    ok: true,
    generatedAt: Date.now(),
    latestRoundId: latestRound.roundId || null,
    totalRounds: normalizedRounds.length,
    focusTarget,
    focusTargetLabel: labelForTarget(focusTarget),
    availableWindows: WINDOW_OPTIONS.map(({ key, label }) => ({ key, label })),
    availableTargets: TARGETS.map((value) => ({ value, label: labelForTarget(value) })),
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
      startTimestamp: latestRound.timestamp - windowConfig.ms,
      endTimestamp: latestRound.timestamp,
    },
    baseline: baselineStats,
    currentWindow: currentSummary,
    patternPrediction,
    comparison,
    targetCards,
    decision,
    targetReadiness,
    recommendationStability,
    bestWindowsToday,
    patternMatch,
    cooldowns,
    backtest,
    hourlyHistory,
    dayHourHeatmap,
    outlook,
  };
}

module.exports = {
  buildTimingAnalyticsReport,
  normalizeTimingWindowKey,
  normalizeTimingTarget,
  normalizeTimingTimeZone,
};
