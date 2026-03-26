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
const CACHE_TTL_MS = 12000;

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
  const matches = [];

  const maxEndIdx = Math.min(n - maxH - 1, n - patternWindow - 1);
  for (let endIdx = patternWindow - 1; endIdx <= maxEndIdx; endIdx++) {
    const start = endIdx - patternWindow + 1;
    const candidateTokens = tokens.slice(start, endIdx + 1);
    const candidateLogs = logs.slice(start, endIdx + 1);
    const similarity = patternSimilarity(currentTokens, currentLogs, candidateTokens, candidateLogs);
    if (similarity < 0.57) continue;

    const futureMax = {};
    for (const h of HORIZONS) {
      let mx = 0;
      for (let j = endIdx + 1; j <= endIdx + h; j++) {
        mx = Math.max(mx, rounds[j].multiplier);
      }
      futureMax[h] = mx;
    }

    const nextMult = rounds[endIdx + 1].multiplier;
    matches.push({
      startRoundId: rounds[start].roundId,
      endRoundId: rounds[endIdx].roundId,
      nextRoundId: rounds[endIdx + 1].roundId,
      nextMult,
      nextBucket: bucketIndex(nextMult),
      similarity,
      futureMax,
    });
  }

  matches.sort((a, b) => b.similarity - a.similarity);
  return matches.slice(0, 220);
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

function blendWeights({ clusterSupport, patternSupport, markovSupport }) {
  let cluster = clusterSupport > 0 ? clamp(clusterSupport / 1300, 0.12, 0.45) : 0;
  let pattern = patternSupport > 0 ? clamp(patternSupport / 230, 0.1, 0.5) : 0;
  let markov = markovSupport > 0 ? clamp(markovSupport / 900, 0.05, 0.24) : 0;
  let baseline = 1 - (cluster + pattern + markov);

  if (baseline < 0.1) {
    const scale = (1 - 0.1) / Math.max(0.0001, cluster + pattern + markov);
    cluster *= scale;
    pattern *= scale;
    markov *= scale;
    baseline = 0.1;
  }

  const total = cluster + pattern + markov + baseline;
  return {
    cluster: cluster / total,
    pattern: pattern / total,
    markov: markov / total,
    baseline: baseline / total,
  };
}

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

function buildSignals(context) {
  const out = [];
  const { lowStreak, highStreak, gap10, gap25, patternCount, clusterRegime, topBucket } = context;

  if (lowStreak >= 5) {
    out.push(`Low streak is ${lowStreak} rounds; rebound pressure usually increases after long micro runs.`);
  }
  if (highStreak >= 2) {
    out.push(`Back-to-back high multipliers (${highStreak}) detected; volatility regime is elevated.`);
  }
  if (gap10 >= 10) {
    out.push(`10x gap is stretched at ${gap10} rounds; this historically lifts medium/high bucket odds.`);
  }
  if (gap25 >= 30) {
    out.push(`25x has been absent for ${gap25} rounds; tail risk is suppressed but can snap sharply.`);
  }
  if (patternCount >= 60) {
    out.push(`Pattern engine found ${patternCount} close historical analogs, improving signal stability.`);
  } else if (patternCount > 0) {
    out.push(`Pattern analog count is ${patternCount}; confidence depends more on cluster and baseline structure.`);
  }

  out.push(`Current regime is ${clusterRegime}; blended model is leaning ${topBucket.label} (${topBucket.min}x+).`);
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
    return {
      model: 'cluster-pattern-hybrid-v1',
      generatedAt: new Date().toISOString(),
      asOfRound,
      sampleSize: cleanRounds.length,
      expectedMultiplier: roundNum(mean(cleanRounds.map(r => r.multiplier)), 4),
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

  const k = clamp(Math.round(Math.sqrt(samples.length / 260)), 4, 8);
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
    clusterSupport: currentClusterStats.count,
    patternSupport: patternStats.count,
    markovSupport: markov.support,
  });

  const blendedBucketDist = blendDistribution({
    weights,
    clusterDist: currentClusterStats.bucketDistribution,
    patternDist: patternStats.bucketDistribution,
    markovDist: markov.distribution,
    baselineDist: baselineStats.bucketDistribution,
  });

  const expectedFromBuckets = (dist) => {
    let out = 0;
    for (let i = 0; i < BUCKETS.length; i++) {
      out += dist[i] * bucketMidpoint(BUCKETS[i]);
    }
    return out;
  };

  const expectedMultiplier =
    (weights.cluster * currentClusterStats.meanNextMultiplier) +
    (weights.pattern * (patternStats.meanNextMultiplier || expectedFromBuckets(patternStats.bucketDistribution))) +
    (weights.markov * expectedFromBuckets(markov.distribution)) +
    (weights.baseline * baselineStats.meanNextMultiplier);

  const topBucketIdx = blendedBucketDist.reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0);
  const topBucket = BUCKETS[topBucketIdx];
  const maxProb = blendedBucketDist[topBucketIdx];
  const sharpness = 1 - entropy(blendedBucketDist);
  const evidence = clamp((currentClusterStats.count + patternStats.count + markov.support) / 1600, 0, 1);
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
    0.15 + (0.45 * maxProb) + (0.2 * sharpness) + (0.1 * evidence) + (0.1 * alignment),
    0.05,
    0.97
  );
  const confidenceBand = confidence >= 0.78 ? 'high' : confidence >= 0.6 ? 'medium' : 'low';

  const targetProbabilities = THRESHOLDS.map((threshold) => {
    const gapNow = roundsSinceHit(cleanRounds, cleanRounds.length - 1, threshold);
    const p1FromDist = probabilityFromDistribution(blendedBucketDist, threshold);
    const markovP1 = probabilityFromDistribution(markov.distribution, threshold);

    const byHorizon = {};
    for (const h of HORIZONS) {
      const clusterP = currentClusterStats.thresholdProbabilities[threshold][h];
      const patternP = patternStats.thresholdProbabilities[threshold][h];
      const baselineP = baselineStats.thresholdProbabilities[threshold][h];
      const markovPh = clamp(1 - ((1 - markovP1) ** h), 0, 1);

      byHorizon[h] = clamp(
        (weights.cluster * clusterP) +
        (weights.pattern * patternP) +
        (weights.markov * markovPh) +
        (weights.baseline * baselineP),
        0,
        1
      );
    }

    const expectedGap = p1FromDist > 0.0001 ? roundNum(1 / p1FromDist, 2) : null;
    return {
      target: threshold,
      gapNow,
      p1: roundNum(byHorizon[1], 4),
      p3: roundNum(byHorizon[3], 4),
      p5: roundNum(byHorizon[5], 4),
      expectedGap,
    };
  });

  const clusterMean = currentClusterStats.meanNextMultiplier || expectedFromBuckets(currentClusterStats.bucketDistribution);
  let clusterRegime = 'balanced';
  if (clusterMean < 2) clusterRegime = 'compression';
  else if (clusterMean < 5) clusterRegime = 'low-mid';
  else if (clusterMean < 10) clusterRegime = 'mid-volatility';
  else clusterRegime = 'expansion';

  const signals = buildSignals({
    lowStreak: streakLength(cleanRounds, cleanRounds.length - 1, m => m < 2),
    highStreak: streakLength(cleanRounds, cleanRounds.length - 1, m => m >= 10),
    gap10: roundsSinceHit(cleanRounds, cleanRounds.length - 1, 10),
    gap25: roundsSinceHit(cleanRounds, cleanRounds.length - 1, 25),
    patternCount: patternStats.count,
    clusterRegime,
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
    model: 'cluster-pattern-hybrid-v1',
    generatedAt: new Date().toISOString(),
    asOfRound: cleanRounds[cleanRounds.length - 1].roundId,
    sampleSize: cleanRounds.length,
    expectedMultiplier: roundNum(expectedMultiplier, 4),
    predictedBucket: {
      ...topBucket,
      probability: roundNum(maxProb, 4),
      confidence: roundNum(confidence, 4),
      confidenceBand,
    },
    bucketProbabilities: BUCKETS.map((bucket, i) => ({
      ...bucket,
      probability: roundNum(blendedBucketDist[i], 4),
    })),
    targetProbabilities,
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
      pattern: {
        matches: patternStats.count,
        avgSimilarity: roundNum(patternStats.avgSimilarity, 4),
      },
      markov: {
        mode: markov.mode,
        support: markov.support,
      },
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
