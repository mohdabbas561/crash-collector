'use strict';

const TIMING_WINDOWS = {
  '5m': { key: '5m', label: '5 Minutes', ms: 5 * 60 * 1000 },
  '10m': { key: '10m', label: '10 Minutes', ms: 10 * 60 * 1000 },
  '30m': { key: '30m', label: '30 Minutes', ms: 30 * 60 * 1000 },
  '1h': { key: '1h', label: '1 Hour', ms: 60 * 60 * 1000 },
  '2h': { key: '2h', label: '2 Hours', ms: 2 * 60 * 60 * 1000 },
  '5h': { key: '5h', label: '5 Hours', ms: 5 * 60 * 60 * 1000 },
  '12h': { key: '12h', label: '12 Hours', ms: 12 * 60 * 60 * 1000 },
  '24h': { key: '24h', label: '24 Hours', ms: 24 * 60 * 60 * 1000 },
  '3d': { key: '3d', label: '3 Days', ms: 3 * 24 * 60 * 60 * 1000 },
  '7d': { key: '7d', label: '7 Days', ms: 7 * 24 * 60 * 60 * 1000 },
  '10d': { key: '10d', label: '10 Days', ms: 10 * 24 * 60 * 60 * 1000 },
  '15d': { key: '15d', label: '15 Days', ms: 15 * 24 * 60 * 60 * 1000 },
  '30d': { key: '30d', label: '30 Days', ms: 30 * 24 * 60 * 60 * 1000 },
};

const TIMING_WINDOW_KEYS = Object.keys(TIMING_WINDOWS);
const TIMING_WINDOW_LIST = TIMING_WINDOW_KEYS.map((key) => TIMING_WINDOWS[key]);
const TIMING_TARGETS = [5, 10, 20, 50, 100, 500, 1000];
const TIMING_DEFAULT_TARGET = 5;
const TIMING_HISTORY_LOOKBACK_MS = TIMING_WINDOWS['30d'].ms;
const TIMING_ANALOG_MATCHES = 36;
const TIMING_MIN_ANALOG_WINDOWS = 6;
const TIME_SLOT_MINUTES = 5;
const SLOTS_PER_DAY = (24 * 60) / TIME_SLOT_MINUTES;

const TIMING_BUCKETS = [
  { key: 'lt2', label: '<2x', min: 0, max: 2, color: '#ff4560' },
  { key: 'b2_5', label: '2-5x', min: 2, max: 5, color: '#ff8c42' },
  { key: 'b5_10', label: '5-10x', min: 5, max: 10, color: '#ffd84d' },
  { key: 'b10_20', label: '10-20x', min: 10, max: 20, color: '#aaff66' },
  { key: 'b20_50', label: '20-50x', min: 20, max: 50, color: '#00ff88' },
  { key: 'b50_100', label: '50-100x', min: 50, max: 100, color: '#00d4ff' },
  { key: 'b100_500', label: '100-500x', min: 100, max: 500, color: '#7aa2ff' },
  { key: 'b500_1000', label: '500-1000x', min: 500, max: 1000, color: '#c084fc' },
  { key: 'gt1000', label: '1000x+', min: 1000, max: Number.POSITIVE_INFINITY, color: '#ff66c4' },
];

const CHASE_WINDOW_SLOT_RANGES = {
  5: [1, 6],
  10: [1, 8],
  20: [2, 10],
  50: [3, 14],
  100: [4, 18],
  500: [6, 24],
  1000: [8, 24],
};

function roundNum(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Number(n.toFixed(digits));
}

function pct01(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Number((n * 100).toFixed(digits));
}

function quantileSorted(sorted, q) {
  if (!Array.isArray(sorted) || sorted.length === 0) return 0;
  if (q <= 0) return sorted[0];
  if (q >= 1) return sorted[sorted.length - 1];
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[Math.min(base + 1, sorted.length - 1)];
  return sorted[base] + ((next - sorted[base]) * rest);
}

function toLabel(target) {
  return `${Number(target)}x`;
}

function normalizeTimingWindowKey(rawKey) {
  const key = String(rawKey || '').trim().toLowerCase();
  return TIMING_WINDOWS[key] ? key : '5m';
}

function normalizeTimingTarget(rawTarget) {
  const numeric = Number(String(rawTarget || '').replace(/x$/i, '').trim());
  if (!Number.isFinite(numeric)) return TIMING_DEFAULT_TARGET;
  if (TIMING_TARGETS.includes(numeric)) return numeric;
  return TIMING_DEFAULT_TARGET;
}

function normalizeTimingTimeZone(rawTimeZone) {
  const timeZone = String(rawTimeZone || '').trim();
  if (!timeZone) return 'UTC';
  try {
    new Intl.DateTimeFormat('en-US', { timeZone }).format(new Date());
    return timeZone;
  } catch {
    return 'UTC';
  }
}

function cleanRounds(rounds) {
  return (rounds || [])
    .map((row) => ({
      roundId: Number(row?.roundId || 0),
      multiplier: Number(row?.multiplier || 0),
      timestamp: Number(row?.timestamp || 0),
    }))
    .filter((row) => Number.isFinite(row.roundId) && row.roundId > 0 && Number.isFinite(row.multiplier) && row.multiplier > 0 && Number.isFinite(row.timestamp) && row.timestamp > 0)
    .sort((a, b) => (a.timestamp - b.timestamp) || (a.roundId - b.roundId));
}

function summarizeRounds(rows, focusTarget, meta = {}) {
  const rounds = Array.isArray(rows) ? rows : [];
  const count = rounds.length;
  const multipliers = rounds.map((row) => Number(row.multiplier)).filter((value) => Number.isFinite(value) && value > 0);
  const sorted = [...multipliers].sort((a, b) => a - b);
  const distributionCounts = Object.fromEntries(TIMING_BUCKETS.map((bucket) => [bucket.key, 0]));
  const hitCountMap = {};

  for (const target of TIMING_TARGETS) {
    hitCountMap[target] = 0;
  }

  for (const multiplier of multipliers) {
    for (const bucket of TIMING_BUCKETS) {
      if (multiplier >= bucket.min && multiplier < bucket.max) {
        distributionCounts[bucket.key] += 1;
        break;
      }
    }
    for (const target of TIMING_TARGETS) {
      if (multiplier >= target) hitCountMap[target] += 1;
    }
  }

  const distribution = TIMING_BUCKETS.map((bucket) => ({
    key: bucket.key,
    label: bucket.label,
    color: bucket.color,
    count: distributionCounts[bucket.key],
    pct: count > 0 ? roundNum(distributionCounts[bucket.key] / count, 4) : 0,
  }));

  const targetRates = TIMING_TARGETS.map((target) => ({
    target,
    label: toLabel(target),
    hits: hitCountMap[target] || 0,
    hitRate: count > 0 ? roundNum((hitCountMap[target] || 0) / count, 4) : 0,
  }));

  const targetRateMap = Object.fromEntries(targetRates.map((item) => [item.target, item.hitRate]));
  const maxMultiplier = count > 0 ? Math.max(...multipliers) : 0;
  const minMultiplier = count > 0 ? Math.min(...multipliers) : 0;
  const mean = count > 0 ? multipliers.reduce((sum, value) => sum + value, 0) / count : 0;
  const roundsPerHour = meta.windowMs && meta.windowMs > 0
    ? roundNum((count / meta.windowMs) * 60 * 60 * 1000, 2)
    : 0;

  return {
    roundCount: count,
    avgMultiplier: roundNum(mean, 4),
    medianMultiplier: roundNum(quantileSorted(sorted, 0.5), 4),
    p75Multiplier: roundNum(quantileSorted(sorted, 0.75), 4),
    p90Multiplier: roundNum(quantileSorted(sorted, 0.9), 4),
    minMultiplier: roundNum(minMultiplier, 4),
    maxMultiplier: roundNum(maxMultiplier, 4),
    roundsPerHour,
    focusTarget,
    focusTargetLabel: toLabel(focusTarget),
    focusHits: hitCountMap[focusTarget] || 0,
    focusHitRate: targetRateMap[focusTarget] || 0,
    distribution,
    targetRates,
    targetRateMap,
    hitCountMap,
    lowRate: count > 0 ? roundNum((distributionCounts.lt2 || 0) / count, 4) : 0,
    highRate: count > 0 ? roundNum(((distributionCounts.b50_100 || 0) + (distributionCounts.b100_500 || 0) + (distributionCounts.b500_1000 || 0) + (distributionCounts.gt1000 || 0)) / count, 4) : 0,
    distributionCounts,
    featureVector: {
      focusHitRate: targetRateMap[focusTarget] || 0,
      rate5: targetRateMap[5] || 0,
      rate20: targetRateMap[20] || 0,
      rate100: targetRateMap[100] || 0,
      lowRate: count > 0 ? roundNum((distributionCounts.lt2 || 0) / count, 4) : 0,
      highRate: count > 0 ? roundNum(((distributionCounts.b50_100 || 0) + (distributionCounts.b100_500 || 0) + (distributionCounts.b500_1000 || 0) + (distributionCounts.gt1000 || 0)) / count, 4) : 0,
      avgLog: roundNum(Math.log1p(mean) / Math.log(1001), 4),
    },
  };
}

function filterByTimeRange(rounds, startTs, endTs) {
  return rounds.filter((row) => row.timestamp > startTs && row.timestamp <= endTs);
}

function buildComparison(currentSummary, baselineSummary) {
  const baselineHitRate = baselineSummary?.focusHitRate || 0;
  const currentHitRate = currentSummary?.focusHitRate || 0;
  const hitRateDelta = roundNum(currentHitRate - baselineHitRate, 4);
  const baselineAvg = Number(baselineSummary?.avgMultiplier || 0);
  const currentAvg = Number(currentSummary?.avgMultiplier || 0);
  const avgDeltaPct = baselineAvg > 0 ? roundNum((currentAvg - baselineAvg) / baselineAvg, 4) : 0;
  const lowRateDelta = roundNum((currentSummary?.lowRate || 0) - (baselineSummary?.lowRate || 0), 4);

  let band = 'steady';
  let label = 'STEADY';
  let tone = 'neutral';
  let message = `${currentSummary?.focusTargetLabel || 'Focus target'} is moving close to the last-30-day baseline.`;

  if (hitRateDelta >= 0.12 || avgDeltaPct >= 0.2) {
    band = 'hot';
    label = 'HOT';
    tone = 'good';
    message = `${currentSummary.focusTargetLabel} is landing more often than the last-30-day baseline.`;
  } else if (hitRateDelta >= 0.05 || avgDeltaPct >= 0.1) {
    band = 'good';
    label = 'GOOD';
    tone = 'good';
    message = `${currentSummary.focusTargetLabel} is slightly stronger than the recent baseline.`;
  } else if (hitRateDelta <= -0.12 || avgDeltaPct <= -0.2 || lowRateDelta >= 0.12) {
    band = 'cold';
    label = 'COLD';
    tone = 'bad';
    message = `${currentSummary.focusTargetLabel} is weaker than normal and low crashes are elevated.`;
  } else if (hitRateDelta <= -0.05 || avgDeltaPct <= -0.1 || lowRateDelta >= 0.06) {
    band = 'soft';
    label = 'SOFT';
    tone = 'bad';
    message = `${currentSummary.focusTargetLabel} is under the recent baseline right now.`;
  }

  return {
    focusTarget: currentSummary?.focusTarget || TIMING_DEFAULT_TARGET,
    focusTargetLabel: currentSummary?.focusTargetLabel || toLabel(TIMING_DEFAULT_TARGET),
    label,
    band,
    tone,
    message,
    currentHitRate,
    baselineHitRate,
    hitRateDelta,
    currentAvg,
    baselineAvg: roundNum(baselineAvg, 4),
    avgDeltaPct,
    currentLowRate: currentSummary?.lowRate || 0,
    baselineLowRate: baselineSummary?.lowRate || 0,
    lowRateDelta,
  };
}

function buildHourlyHistory(rounds, timeZone) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: '2-digit',
    hour12: false,
    hourCycle: 'h23',
  });
  const hours = Array.from({ length: 24 }, (_, hour) => ({
    hour,
    label: `${String(hour).padStart(2, '0')}:00`,
    roundCount: 0,
    avgMultiplier: 0,
    targetRates: {},
    bestTarget: null,
  }));

  const buckets = hours.map(() => ({
    multipliers: [],
    hits: Object.fromEntries(TIMING_TARGETS.map((target) => [target, 0])),
  }));

  for (const round of rounds) {
    const hour = Number(formatter.format(new Date(round.timestamp)));
    if (!Number.isFinite(hour) || hour < 0 || hour > 23) continue;
    const bucket = buckets[hour];
    bucket.multipliers.push(round.multiplier);
    for (const target of TIMING_TARGETS) {
      if (round.multiplier >= target) bucket.hits[target] += 1;
    }
  }

  for (let hour = 0; hour < 24; hour += 1) {
    const bucket = buckets[hour];
    const count = bucket.multipliers.length;
    const avg = count > 0
      ? bucket.multipliers.reduce((sum, value) => sum + value, 0) / count
      : 0;
    const targetRates = {};
    let bestTarget = null;
    let bestScore = -1;

    for (const target of TIMING_TARGETS) {
      const hitRate = count > 0 ? bucket.hits[target] / count : 0;
      targetRates[target] = roundNum(hitRate, 4);
      const score = hitRate * Math.log(target + 1);
      if (count > 0 && score > bestScore) {
        bestScore = score;
        bestTarget = target;
      }
    }

    hours[hour] = {
      hour,
      label: `${String(hour).padStart(2, '0')}:00`,
      roundCount: count,
      avgMultiplier: roundNum(avg, 4),
      targetRates,
      bestTarget: bestTarget ? toLabel(bestTarget) : '-',
    };
  }

  const bestHours = TIMING_TARGETS.map((target) => {
    const populated = hours.filter((row) => row.roundCount > 0);
    if (!populated.length) {
      return {
        target,
        label: toLabel(target),
        bestHour: null,
        bestLabel: '-',
        bestHitRate: 0,
        worstHour: null,
        worstLabel: '-',
        worstHitRate: 0,
      };
    }
    const sorted = [...populated].sort((a, b) => {
      const diff = (b.targetRates[target] || 0) - (a.targetRates[target] || 0);
      if (diff !== 0) return diff;
      return b.roundCount - a.roundCount;
    });
    const best = sorted[0];
    const worst = [...sorted].reverse()[0];
    return {
      target,
      label: toLabel(target),
      bestHour: best.hour,
      bestLabel: `${best.label} - ${String(best.hour + 1).padStart(2, '0')}:00`,
      bestHitRate: best.targetRates[target] || 0,
      bestRoundCount: best.roundCount,
      worstHour: worst.hour,
      worstLabel: `${worst.label} - ${String(worst.hour + 1).padStart(2, '0')}:00`,
      worstHitRate: worst.targetRates[target] || 0,
      worstRoundCount: worst.roundCount,
    };
  });

  return {
    timeZone,
    rows: hours,
    bestHours,
  };
}

function createTimePartsFormatter(timeZone) {
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    hourCycle: 'h23',
  });
}

function createDayPartsFormatter(timeZone) {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
}

function getSlotIndexFromTimestamp(timestamp, formatter) {
  const parts = formatter.formatToParts(new Date(timestamp));
  const hour = Number(parts.find((part) => part.type === 'hour')?.value || 0);
  const minute = Number(parts.find((part) => part.type === 'minute')?.value || 0);
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return 0;
  return Math.min(SLOTS_PER_DAY - 1, Math.max(0, (hour * (60 / TIME_SLOT_MINUTES)) + Math.floor(minute / TIME_SLOT_MINUTES)));
}

function slotIndexToLabel(slotIndex) {
  const totalMinutes = slotIndex * TIME_SLOT_MINUTES;
  const hour = Math.floor(totalMinutes / 60) % 24;
  const minute = totalMinutes % 60;
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

function formatDisplayTime(slotIndex) {
  const totalMinutes = slotIndex * TIME_SLOT_MINUTES;
  const hour24 = Math.floor(totalMinutes / 60) % 24;
  const minute = totalMinutes % 60;
  const suffix = hour24 >= 12 ? 'PM' : 'AM';
  const hour12 = hour24 % 12 || 12;
  if (minute === 0) return `${hour12}${suffix}`;
  return `${hour12}:${String(minute).padStart(2, '0')}${suffix}`;
}

function windowLabelFromSlots(startSlot, slotLength) {
  const endSlotExclusive = (startSlot + slotLength) % SLOTS_PER_DAY;
  const start = formatDisplayTime(startSlot);
  const end = formatDisplayTime(endSlotExclusive);
  return `${start} - ${end}`;
}

function listWindowSlots(startSlot, slotLength) {
  const slots = [];
  for (let offset = 0; offset < slotLength; offset += 1) {
    slots.push((startSlot + offset) % SLOTS_PER_DAY);
  }
  return slots;
}

function computeWindowStats(slotTotals, slotHits, startSlot, slotLength, baselineRate, dayCount) {
  const slots = listWindowSlots(startSlot, slotLength);
  let totalRounds = 0;
  let totalHits = 0;

  for (const slot of slots) {
    totalRounds += Number(slotTotals[slot] || 0);
    totalHits += Number(slotHits[slot] || 0);
  }

  if (totalRounds <= 0) return null;

  const priorRounds = 120;
  const rawRate = totalHits / totalRounds;
  const smoothedRate = (totalHits + (baselineRate * priorRounds)) / (totalRounds + priorRounds);
  const avgRoundsPerDay = dayCount > 0 ? totalRounds / dayCount : 0;
  const anyHitChance = avgRoundsPerDay > 0 ? 1 - Math.pow(Math.max(0, 1 - smoothedRate), avgRoundsPerDay) : 0;
  const lift = baselineRate > 0 ? smoothedRate / baselineRate : 0;
  const durationMinutes = slotLength * TIME_SLOT_MINUTES;
  const score = (smoothedRate - baselineRate) * Math.log1p(totalRounds) * Math.pow(Math.max(anyHitChance, 0.0001), 0.7) / Math.pow(slotLength, 0.28);

  return {
    startSlot,
    endSlotExclusive: (startSlot + slotLength) % SLOTS_PER_DAY,
    slotLength,
    durationMinutes,
    totalRounds,
    totalHits,
    avgRoundsPerDay: roundNum(avgRoundsPerDay, 2),
    rawRate: roundNum(rawRate, 4),
    smoothedRate: roundNum(smoothedRate, 4),
    anyHitChance: roundNum(anyHitChance, 4),
    lift: roundNum(lift, 4),
    score: roundNum(score, 6),
    slotLabel: `${slotIndexToLabel(startSlot)} - ${slotIndexToLabel((startSlot + slotLength) % SLOTS_PER_DAY)}`,
    displayLabel: windowLabelFromSlots(startSlot, slotLength),
  };
}

function buildChaseWindows(rounds, timeZone) {
  if (!rounds.length) return [];

  const timeFormatter = createTimePartsFormatter(timeZone);
  const dayFormatter = createDayPartsFormatter(timeZone);
  const dayKeys = new Set();
  const slotTotals = new Array(SLOTS_PER_DAY).fill(0);
  const perTargetSlotHits = Object.fromEntries(TIMING_TARGETS.map((target) => [target, new Array(SLOTS_PER_DAY).fill(0)]));
  const baselineHitCounts = Object.fromEntries(TIMING_TARGETS.map((target) => [target, 0]));

  for (const round of rounds) {
    const slotIndex = getSlotIndexFromTimestamp(round.timestamp, timeFormatter);
    const dayKey = dayFormatter.format(new Date(round.timestamp));
    dayKeys.add(dayKey);
    slotTotals[slotIndex] += 1;

    for (const target of TIMING_TARGETS) {
      if (round.multiplier >= target) {
        perTargetSlotHits[target][slotIndex] += 1;
        baselineHitCounts[target] += 1;
      }
    }
  }

  const dayCount = Math.max(dayKeys.size, 1);

  return TIMING_TARGETS.map((target) => {
    const baselineRate = rounds.length > 0 ? baselineHitCounts[target] / rounds.length : 0;
    const [minSlots, maxSlots] = CHASE_WINDOW_SLOT_RANGES[target] || [1, 12];
    const candidates = [];

    for (let slotLength = minSlots; slotLength <= maxSlots; slotLength += 1) {
      for (let startSlot = 0; startSlot < SLOTS_PER_DAY; startSlot += 1) {
        const stats = computeWindowStats(
          slotTotals,
          perTargetSlotHits[target],
          startSlot,
          slotLength,
          baselineRate,
          dayCount
        );
        if (!stats) continue;
        if (stats.totalRounds < Math.max(100, dayCount * 2)) continue;
        if (stats.smoothedRate <= baselineRate) continue;
        candidates.push(stats);
      }
    }

    const topWindows = candidates
      .sort((a, b) => (b.score - a.score) || (b.lift - a.lift) || (a.durationMinutes - b.durationMinutes))
      .slice(0, 3);

    const best = topWindows[0] || null;
    return {
      target,
      label: toLabel(target),
      baselineRate: roundNum(baselineRate, 4),
      bestWindow: best ? {
        startSlot: best.startSlot,
        endSlotExclusive: best.endSlotExclusive,
        slotLength: best.slotLength,
        durationMinutes: best.durationMinutes,
        label: best.displayLabel,
        rawLabel: best.slotLabel,
        anyHitChance: best.anyHitChance,
        roundHitRate: best.smoothedRate,
        lift: best.lift,
        totalRounds: best.totalRounds,
        totalHits: best.totalHits,
        avgRoundsPerDay: best.avgRoundsPerDay,
      } : null,
      topWindows: topWindows.map((item) => ({
        startSlot: item.startSlot,
        endSlotExclusive: item.endSlotExclusive,
        slotLength: item.slotLength,
        durationMinutes: item.durationMinutes,
        label: item.displayLabel,
        rawLabel: item.slotLabel,
        anyHitChance: item.anyHitChance,
        roundHitRate: item.smoothedRate,
        lift: item.lift,
        totalRounds: item.totalRounds,
        totalHits: item.totalHits,
        avgRoundsPerDay: item.avgRoundsPerDay,
      })),
    };
  });
}

function buildSegmentWindows(rounds, windowMs, focusTarget) {
  if (!rounds.length || !(windowMs > 0)) return [];
  const windows = [];
  let idx = rounds.length - 1;
  let endTs = rounds[rounds.length - 1].timestamp;

  while (idx >= 0) {
    const startTs = endTs - windowMs;
    const bucket = [];
    while (idx >= 0 && rounds[idx].timestamp > startTs && rounds[idx].timestamp <= endTs) {
      bucket.push(rounds[idx]);
      idx -= 1;
    }
    bucket.reverse();
    const summary = summarizeRounds(bucket, focusTarget, { windowMs });
    windows.unshift({
      startTs,
      endTs,
      summary,
      roundCount: summary.roundCount,
      maxMultiplier: summary.maxMultiplier,
      featureVector: summary.featureVector,
    });
    endTs = startTs;
  }

  return windows;
}

function similarityBetweenWindows(currentWindow, priorWindow) {
  const current = currentWindow?.featureVector || {};
  const prior = priorWindow?.featureVector || {};
  const weightedDiffs = [
    { key: 'focusHitRate', weight: 2.4 },
    { key: 'rate5', weight: 1.6 },
    { key: 'rate20', weight: 1.3 },
    { key: 'rate100', weight: 1.1 },
    { key: 'lowRate', weight: 1.4 },
    { key: 'highRate', weight: 1.2 },
    { key: 'avgLog', weight: 1.0 },
  ];
  let totalWeight = 0;
  let distance = 0;

  for (const item of weightedDiffs) {
    totalWeight += item.weight;
    distance += Math.abs((current[item.key] || 0) - (prior[item.key] || 0)) * item.weight;
  }

  if (!totalWeight) return 0;
  const normalizedDistance = distance / totalWeight;
  return roundNum(1 / (1 + (normalizedDistance * 8)), 6);
}

function buildOutlook(rounds, windowConfig, focusTarget) {
  const windows = buildSegmentWindows(rounds, windowConfig.ms, focusTarget);
  const currentWindow = windows[windows.length - 1];

  if (!currentWindow || currentWindow.roundCount === 0) {
    return {
      available: false,
      reason: `No completed ${windowConfig.label.toLowerCase()} window is available yet.`,
    };
  }

  const pairs = [];
  for (let index = 0; index < windows.length - 1; index += 1) {
    const source = windows[index];
    const next = windows[index + 1];
    if (!source || !next) continue;
    if (source.roundCount === 0 || next.roundCount === 0) continue;
    const similarity = similarityBetweenWindows(currentWindow, source);
    pairs.push({ source, next, similarity });
  }

  if (pairs.length < TIMING_MIN_ANALOG_WINDOWS) {
    return {
      available: false,
      reason: `Not enough completed ${windowConfig.label.toLowerCase()} windows yet for a stats-based next-window outlook.`,
      historyWindows: windows.length,
      comparableWindows: pairs.length,
    };
  }

  const matches = pairs
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, Math.min(TIMING_ANALOG_MATCHES, pairs.length));

  const totalWeight = matches.reduce((sum, item) => sum + Math.max(item.similarity, 0.02), 0);
  const candidates = TIMING_TARGETS.map((target) => {
    let anyHitWeight = 0;
    let perRoundWeight = 0;
    let avgPeakWeight = 0;
    let expectedHitsWeight = 0;

    for (const match of matches) {
      const weight = Math.max(match.similarity, 0.02);
      const nextSummary = match.next.summary;
      const targetRate = nextSummary?.targetRateMap?.[target] || 0;
      if ((match.next.maxMultiplier || 0) >= target) anyHitWeight += weight;
      perRoundWeight += targetRate * weight;
      avgPeakWeight += (nextSummary?.maxMultiplier || 0) * weight;
      expectedHitsWeight += (nextSummary?.hitCountMap?.[target] || 0) * weight;
    }

    const anyHitRate = totalWeight > 0 ? anyHitWeight / totalWeight : 0;
    const perRoundHitRate = totalWeight > 0 ? perRoundWeight / totalWeight : 0;
    const avgPeak = totalWeight > 0 ? avgPeakWeight / totalWeight : 0;
    const expectedHits = totalWeight > 0 ? expectedHitsWeight / totalWeight : 0;
    const score = anyHitRate * Math.log(target + 1);

    return {
      target,
      label: toLabel(target),
      anyHitRate: roundNum(anyHitRate, 4),
      perRoundHitRate: roundNum(perRoundHitRate, 4),
      avgPeakMultiplier: roundNum(avgPeak, 4),
      expectedHits: roundNum(expectedHits, 2),
      score: roundNum(score, 6),
      style: anyHitRate >= 0.65 ? 'safer' : (anyHitRate >= 0.35 ? 'balanced' : 'aggressive'),
    };
  }).sort((a, b) => (b.score - a.score) || (b.anyHitRate - a.anyHitRate));

  const recommendation = candidates[0] || null;
  const focusCandidate = candidates.find((item) => item.target === focusTarget) || null;
  const similarityAvg = matches.reduce((sum, item) => sum + item.similarity, 0) / matches.length;
  const confidence = Math.min(
    1,
    (Math.min(matches.length, TIMING_ANALOG_MATCHES) / TIMING_ANALOG_MATCHES) * 0.55 + (similarityAvg * 0.45)
  );

  return {
    available: true,
    method: 'analogue_windows_v1',
    windowLabel: windowConfig.label,
    basedOnMatches: matches.length,
    averageSimilarity: roundNum(similarityAvg, 4),
    confidence: roundNum(confidence, 4),
    recommendation,
    focusTarget,
    focusTargetLabel: toLabel(focusTarget),
    focusCandidate,
    candidates,
    note: `Uses the most similar completed ${windowConfig.label.toLowerCase()} windows from your history and scores what happened in the next window.`,
  };
}

function buildTargetCards(currentSummary, baselineSummary) {
  return TIMING_TARGETS.map((target) => {
    const currentHitRate = currentSummary?.targetRateMap?.[target] || 0;
    const baselineHitRate = baselineSummary?.targetRateMap?.[target] || 0;
    return {
      target,
      label: toLabel(target),
      currentHitRate,
      baselineHitRate,
      delta: roundNum(currentHitRate - baselineHitRate, 4),
      currentHits: currentSummary?.hitCountMap?.[target] || 0,
    };
  });
}

function buildTimingAnalyticsReport(rawRounds, options = {}) {
  const rounds = cleanRounds(rawRounds);
  const windowKey = normalizeTimingWindowKey(options.windowKey);
  const focusTarget = normalizeTimingTarget(options.focusTarget);
  const timeZone = normalizeTimingTimeZone(options.timeZone);
  const includeOutlook = Boolean(options.includeOutlook);
  const windowConfig = TIMING_WINDOWS[windowKey];
  const availableTargets = TIMING_TARGETS.map((target) => ({ value: target, label: toLabel(target) }));

  if (!rounds.length) {
    return {
      ok: true,
      generatedAt: new Date().toISOString(),
      availableWindows: TIMING_WINDOW_LIST,
      availableTargets,
      window: windowConfig,
      focusTarget,
      focusTargetLabel: toLabel(focusTarget),
      dataset: {
        totalRounds: 0,
        firstTimestamp: null,
        lastTimestamp: null,
      },
      currentWindow: summarizeRounds([], focusTarget, { windowMs: windowConfig.ms }),
      baseline: summarizeRounds([], focusTarget, { windowMs: TIMING_HISTORY_LOOKBACK_MS }),
      comparison: buildComparison(
        summarizeRounds([], focusTarget, { windowMs: windowConfig.ms }),
        summarizeRounds([], focusTarget, { windowMs: TIMING_HISTORY_LOOKBACK_MS })
      ),
      targetCards: [],
      hourlyHistory: {
        timeZone,
        rows: [],
        bestHours: [],
      },
      outlook: includeOutlook ? { available: false, reason: 'No rounds collected yet.' } : null,
    };
  }

  const latestRound = rounds[rounds.length - 1];
  const earliestRound = rounds[0];
  const currentWindowStart = latestRound.timestamp - windowConfig.ms;
  const historyStart = latestRound.timestamp - TIMING_HISTORY_LOOKBACK_MS;
  const currentWindowRounds = filterByTimeRange(rounds, currentWindowStart, latestRound.timestamp);
  const historyRounds = filterByTimeRange(rounds, historyStart, latestRound.timestamp);
  const comparisonPool = historyRounds.filter((row) => row.timestamp <= currentWindowStart);
  const baselineRounds = comparisonPool.length >= Math.max(25, Math.floor(currentWindowRounds.length * 0.6))
    ? comparisonPool
    : historyRounds;

  const currentSummary = summarizeRounds(currentWindowRounds, focusTarget, { windowMs: windowConfig.ms });
  const baselineSummary = summarizeRounds(baselineRounds, focusTarget, { windowMs: TIMING_HISTORY_LOOKBACK_MS });
  const comparison = buildComparison(currentSummary, baselineSummary);
  const hourlyHistory = buildHourlyHistory(historyRounds, timeZone);
  const targetCards = buildTargetCards(currentSummary, baselineSummary);
  const chaseWindows = buildChaseWindows(rounds, timeZone);

  return {
    ok: true,
    generatedAt: new Date().toISOString(),
    availableWindows: TIMING_WINDOW_LIST,
    availableTargets,
    window: {
      ...windowConfig,
      startTimestamp: currentWindowStart,
      endTimestamp: latestRound.timestamp,
    },
    focusTarget,
    focusTargetLabel: toLabel(focusTarget),
    dataset: {
      totalRounds: rounds.length,
      firstTimestamp: earliestRound.timestamp,
      lastTimestamp: latestRound.timestamp,
      firstRoundId: earliestRound.roundId,
      lastRoundId: latestRound.roundId,
    },
    currentWindow: currentSummary,
    baseline: {
      ...baselineSummary,
      label: 'Last 30 Days Baseline',
      startTimestamp: historyStart,
      endTimestamp: latestRound.timestamp,
    },
    comparison,
    targetCards,
    chaseWindows,
    hourlyHistory: {
      ...hourlyHistory,
      lookbackLabel: 'Last 30 Days',
      startTimestamp: historyStart,
      endTimestamp: latestRound.timestamp,
    },
    outlook: includeOutlook ? buildOutlook(rounds, windowConfig, focusTarget) : null,
  };
}

module.exports = {
  TIMING_WINDOWS,
  TIMING_WINDOW_KEYS,
  TIMING_WINDOW_LIST,
  TIMING_TARGETS,
  TIMING_DEFAULT_TARGET,
  normalizeTimingWindowKey,
  normalizeTimingTarget,
  normalizeTimingTimeZone,
  buildTimingAnalyticsReport,
  pct01,
};
