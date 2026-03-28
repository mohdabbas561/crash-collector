'use strict';

const BUCKETS = [
  { id: 'micro', label: 'Micro', min: 1, max: 1.99, color: '#ff4560' },
  { id: 'low', label: 'Low', min: 2, max: 4.99, color: '#ffd84d' },
  { id: 'mid', label: 'Mid', min: 5, max: 9.99, color: '#00ff88' },
  { id: 'high', label: 'High', min: 10, max: 24.99, color: '#00d4ff' },
  { id: 'moon', label: 'Moon', min: 25, max: Number.POSITIVE_INFINITY, color: '#c084fc' },
];

const THRESHOLDS = [2, 5, 10, 25, 50];
const HORIZONS = [1, 3, 5];
const CACHE_TTL_MS = 15000;

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

function betaMean(success, total, priorMean = 0.12, strength = 14) {
  const a = Math.max(0.0001, priorMean * strength);
  const b = Math.max(0.0001, (1 - priorMean) * strength);
  return (success + a) / Math.max(0.0001, total + a + b);
}

function computeB2BSignal(rounds, threshold = 5) {
  if (!rounds.length) return { score: 0, immediate: 0, near: 0, sample: 0 };
  const n = rounds.length;
  const lookback = clamp(Math.round(Math.sqrt(n) * 18), 400, 20000);
  const start = Math.max(1, n - lookback);

  let obsPairs = 0;
  let hitPairs = 0;
  let prevHit = rounds[start - 1]?.multiplier >= threshold;
  for (let i = start; i < n; i++) {
    const hit = rounds[i].multiplier >= threshold;
    if (prevHit) {
      obsPairs++;
      if (hit) hitPairs++;
    }
    prevHit = hit;
  }

  let totalGaps = 0;
  let gapLe1 = 0;
  let gapLe2 = 0;
  let lastHit = null;
  for (let i = start; i < n; i++) {
    if (rounds[i].multiplier < threshold) continue;
    if (lastHit != null) {
      const g = i - lastHit;
      totalGaps++;
      if (g <= 1) gapLe1++;
      if (g <= 2) gapLe2++;
    }
    lastHit = i;
  }

  const immediate = betaMean(hitPairs, obsPairs, 0.11, 12);
  const near = betaMean(gapLe2, totalGaps, 0.22, 14);
  const score = clamp((0.58 * immediate) + (0.42 * near), 0, 1);
  return {
    score,
    immediate,
    near,
    sample: Math.max(obsPairs, totalGaps),
  };
}

function computeHighChainSignal(rounds, threshold = 50) {
  if (!rounds.length) return { score: 0, p1: 0, p3: 0, sample: 0, active: false };
  const n = rounds.length;
  if (n < 24) return { score: 0, p1: 0, p3: 0, sample: 0, active: false };

  const lookback = clamp(Math.round(Math.sqrt(n) * 22), 600, 24000);
  const start = Math.max(2, n - lookback);
  const anchor = threshold >= 100 ? Math.max(50, threshold * 0.45) : Math.max(20, threshold * 0.5);
  const curPeak2 = Math.max(
    Number(rounds[n - 1]?.multiplier || 0),
    Number(rounds[n - 2]?.multiplier || 0)
  );
  const active = curPeak2 >= anchor;
  const curTail = mean([
    toLog(rounds[n - 1]?.multiplier || 1),
    toLog(rounds[n - 2]?.multiplier || 1),
    toLog(rounds[n - 3]?.multiplier || 1),
  ]);

  let totalW = 0;
  let hit1 = 0;
  let hit3 = 0;
  for (let i = start; i < n - 4; i++) {
    const h0 = Number(rounds[i]?.multiplier || 0);
    const h1 = Number(rounds[i - 1]?.multiplier || 0);
    const histPeak2 = Math.max(h0, h1);
    const histHot = histPeak2 >= anchor;
    const curHot = active;
    if (histHot !== curHot) continue;

    const histTail = mean([toLog(h0 || 1), toLog(h1 || 1), toLog(rounds[i - 2]?.multiplier || 1)]);
    const dist = Math.abs(curTail - histTail);
    const recency = 0.42 + (0.58 * ((i + 1) / Math.max(1, n - 1)) ** 1.2);
    const weight = Math.exp(-dist * 1.05) * recency * (histHot ? 1.1 : 1);
    if (weight < 0.001) continue;

    totalW += weight;
    const n1 = Number(rounds[i + 1]?.multiplier || 0);
    const n2 = Number(rounds[i + 2]?.multiplier || 0);
    const n3 = Number(rounds[i + 3]?.multiplier || 0);
    if (n1 >= threshold) hit1 += weight;
    if (n1 >= threshold || n2 >= threshold || n3 >= threshold) hit3 += weight;
  }

  const baseP1 = clamp(rounds.filter(r => Number(r.multiplier) >= threshold).length / Math.max(1, n), 0.0005, 0.9);
  const baseP3 = clamp(1 - ((1 - baseP1) ** 3), baseP1, 0.98);
  const alpha = totalW >= 24 ? 6 : 10;
  const p1 = clamp((hit1 + (alpha * baseP1)) / Math.max(0.0001, totalW + alpha), 0, 1);
  const p3 = clamp((hit3 + (alpha * baseP3)) / Math.max(0.0001, totalW + alpha), p1, 1);
  const score = clamp((0.4 * p1) + (0.6 * p3), 0, 1);
  return { score, p1, p3, sample: Math.round(totalW), active };
}

function computeWhiteReleaseSignal(rounds, whiteSeverity) {
  if (whiteSeverity <= 0 || rounds.length < 12) return 0;
  const logs = rounds.map(r => toLog(r.multiplier));
  const last3 = mean(logs.slice(-3));
  const prev3 = mean(logs.slice(-6, -3));
  const last8 = mean(logs.slice(-8));
  const prev8 = mean(logs.slice(-16, -8));
  const reboundShort = clamp((last3 - prev3) / 0.4, 0, 1);
  const reboundLong = clamp((last8 - prev8) / 0.22, 0, 1);
  return clamp(whiteSeverity * ((0.58 * reboundShort) + (0.42 * reboundLong)), 0, 1);
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
      model: 'cluster-pattern-hybrid-v8',
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

  const b2b5 = computeB2BSignal(cleanRounds, 5);
  const b2b10 = computeB2BSignal(cleanRounds, 10);
  const b2b50 = computeB2BSignal(cleanRounds, 50);
  const b2b100 = computeB2BSignal(cleanRounds, 100);
  const highChain50 = computeHighChainSignal(cleanRounds, 50);
  const highChain100 = computeHighChainSignal(cleanRounds, 100);
  const b2bStrength = clamp((0.62 * b2b5.score) + (0.38 * b2b10.score), 0, 1);
  if (b2bStrength > 0) {
    const adjusted = [...finalBucketDist];
    adjusted[2] *= (1 + (0.24 * b2bStrength));
    adjusted[3] *= (1 + (0.15 * b2bStrength));
    adjusted[4] *= (1 + (0.09 * b2bStrength));
    adjusted[0] *= (1 - (0.2 * b2bStrength));
    finalBucketDist = normalizeDistribution(adjusted);
  }
  const highB2BStrength = clamp(
    (0.45 * b2b50.score) +
    (0.3 * b2b100.score) +
    (0.15 * highChain50.score) +
    (0.1 * highChain100.score),
    0,
    1
  );
  if (highB2BStrength > 0) {
    const adjusted = [...finalBucketDist];
    adjusted[3] *= (1 + (0.1 * highB2BStrength));
    adjusted[4] *= (1 + (0.24 * highB2BStrength));
    adjusted[0] *= (1 - (0.08 * highB2BStrength));
    finalBucketDist = normalizeDistribution(adjusted);
  }

  const whiteRelease = computeWhiteReleaseSignal(cleanRounds, whiteSeverity);
  if (whiteRelease > 0) {
    const adjusted = [...finalBucketDist];
    adjusted[1] *= (1 + (0.18 * whiteRelease));
    adjusted[2] *= (1 + (0.22 * whiteRelease));
    adjusted[3] *= (1 + (0.14 * whiteRelease));
    adjusted[0] *= (1 - (0.22 * whiteRelease));
    finalBucketDist = normalizeDistribution(adjusted);
  }

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
    0.15 + (0.38 * maxProb) + (0.16 * sharpness) + (0.1 * evidence) + (0.08 * alignment) + (0.07 * pressureEvidence) + (0.04 * b2bStrength) + (0.03 * highB2BStrength) + (0.02 * whiteRelease),
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
    if (threshold === 25 || threshold === 50) {
      const chain = threshold === 50
        ? clamp((0.7 * highChain50.score) + (0.3 * highChain100.score), 0, 1)
        : clamp((0.65 * highChain50.score) + (0.35 * highChain100.score), 0, 1);
      byHorizon[1] = clamp(byHorizon[1] + (0.1 * chain), 0, 1);
      byHorizon[3] = clamp(byHorizon[3] + (0.18 * chain), byHorizon[1], 1);
      byHorizon[5] = clamp(byHorizon[5] + (0.22 * chain), byHorizon[3], 1);
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
    model: 'cluster-pattern-hybrid-v8',
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
        whiteReleaseSignal: roundNum(whiteRelease, 4),
      },
      b2b: {
        combined: roundNum(b2bStrength, 4),
        highCombined: roundNum(highB2BStrength, 4),
        for5x: {
          score: roundNum(b2b5.score, 4),
          immediate: roundNum(b2b5.immediate, 4),
          near: roundNum(b2b5.near, 4),
          sample: Number(b2b5.sample || 0),
        },
        for10x: {
          score: roundNum(b2b10.score, 4),
          immediate: roundNum(b2b10.immediate, 4),
          near: roundNum(b2b10.near, 4),
          sample: Number(b2b10.sample || 0),
        },
        for50x: {
          score: roundNum(b2b50.score, 4),
          immediate: roundNum(b2b50.immediate, 4),
          near: roundNum(b2b50.near, 4),
          sample: Number(b2b50.sample || 0),
        },
        for100x: {
          score: roundNum(b2b100.score, 4),
          immediate: roundNum(b2b100.immediate, 4),
          near: roundNum(b2b100.near, 4),
          sample: Number(b2b100.sample || 0),
        },
        chain50x: {
          score: roundNum(highChain50.score, 4),
          p1: roundNum(highChain50.p1, 4),
          p3: roundNum(highChain50.p3, 4),
          sample: Number(highChain50.sample || 0),
          active: Boolean(highChain50.active),
        },
        chain100x: {
          score: roundNum(highChain100.score, 4),
          p1: roundNum(highChain100.p1, 4),
          p3: roundNum(highChain100.p3, 4),
          sample: Number(highChain100.sample || 0),
          active: Boolean(highChain100.active),
        },
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

module.exports = { buildPredictionReport };
