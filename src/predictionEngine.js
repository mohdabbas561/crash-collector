// predictionEngine.js
// ============================================================================
// Server-side prediction engine — runs after every collector poll.
// Mirrors the exact math from StatisticalEngine.jsx (no React, no browser).
// Saves prediction outcomes and locked windows to DB automatically.
// Site can be completely offline — this keeps running on the server.
// ============================================================================

'use strict';

const { getRounds, savePrediction, getPredictions, saveLockedPreds, getLockedPreds } = require('./db');

const TARGETS = [
  { label: '5x',    min: 5,    maxWidth: 3  },
  { label: '10x',   min: 10,   maxWidth: 5  },
  { label: '20x',   min: 20,   maxWidth: 8  },
  { label: '50x',   min: 50,   maxWidth: 12 },
  { label: '100x',  min: 100,  maxWidth: 18 },
  { label: '250x',  min: 250,  maxWidth: 25 },
  { label: '500x',  min: 500,  maxWidth: 30 },
  { label: '1000x', min: 1000, maxWidth: 50 },
];

const MIN_ROUNDS = 100;

// ── In-memory state (persisted across polls) ──────────────────────────────
// lockedPreds / lockedPatterns are loaded from DB on first run,
// then kept in memory and synced back to DB whenever they change.
let lockedPreds    = null; // { [label]: { low, high, confidence, targetMin, anchorRound, generation } }
let lockedPatterns = null;
let savedKeys      = new Set(); // engine outcome keys already in DB
let patSavedKeys   = new Set(); // pattern outcome keys already in DB
let lastRoundCount = 0;        // round count at last successful run
let initialised    = false;

// ============================================================================
// MATH ENGINE  v8  (exact copy from StatisticalEngine.jsx)
// ============================================================================

function hitProbability(lambda, N) {
  if (lambda <= 0 || N <= 0) return 0;
  if (lambda >= 1) return 1;
  return Math.min(1, 1 - Math.pow(1 - lambda, N));
}

function bayesLambda(hits, n) {
  return (hits + 1) / (n + 2);
}

function blendedLambda(rounds, targetMin, lambdaGlobal, precomputedRecentHits) {
  const n       = rounds.length;
  const recentN = Math.min(500, n);
  let recentHits;
  if (precomputedRecentHits !== undefined) {
    recentHits = precomputedRecentHits;
  } else {
    const start = n - recentN;
    recentHits  = 0;
    for (let i = start; i < n; i++) {
      if (rounds[i].multiplier >= targetMin) recentHits++;
    }
  }
  const lambdaRecent = bayesLambda(recentHits, recentN);
  const raw          = 0.7 * lambdaGlobal + 0.3 * lambdaRecent;
  return Math.max(1e-6, Math.min(0.5, raw));
}

function tailExpansionAdj(n, targetMin, tailCountsMap) {
  if (n < 50) return 0;
  const threshold    = targetMin * 3;
  const tailHits     = tailCountsMap.get(threshold) ?? 0;
  const observedTail = tailHits / n;
  const expectedTail = 1 / threshold;
  if (expectedTail <= 0) return 0;
  const ratio = observedTail / expectedTail;
  if (ratio > 1.4) return  3;
  if (ratio < 0.7) return -3;
  return 0;
}

function computeGlobalStats(rounds) {
  const n        = rounds.length;
  const r500     = Math.min(500, n);
  const r200     = Math.min(200, n);
  const start500 = n - r500;
  const start200 = n - r200;

  const thresholdsSorted = TARGETS.map(t => t.min * 3).sort((a, b) => b - a);

  let gLogS = 0, gLogSS = 0, rLogS = 0, rLogSS = 0, dLogS = 0;
  const gBins = [0, 0, 0, 0, 0];
  const rBins = [0, 0, 0, 0, 0];
  const tailCounts = new Map();
  thresholdsSorted.forEach(t => tailCounts.set(t, 0));

  for (let i = 0; i < n; i++) {
    const m  = rounds[i].multiplier;
    const lv = Math.log(Math.max(1.01, m));
    gLogS  += lv;
    gLogSS += lv * lv;
    if (i >= start500) { rLogS += lv; rLogSS += lv * lv; }
    if (i >= start200) dLogS += lv;
    const b = m < 2 ? 0 : m < 5 ? 1 : m < 20 ? 2 : m < 100 ? 3 : 4;
    gBins[b]++;
    if (i >= start500) rBins[b]++;
    for (let t = 0; t < thresholdsSorted.length; t++) {
      if (m < thresholdsSorted[t]) break;
      tailCounts.set(thresholdsSorted[t], tailCounts.get(thresholdsSorted[t]) + 1);
    }
  }

  const gVar = n > 0 ? gLogSS / n - (gLogS / n) ** 2 : 0;
  const rVar = r500 > 0 ? rLogSS / r500 - (rLogS / r500) ** 2 : 0;
  let regimeAdj = 0;
  if (n >= 50 && gVar > 0) {
    const ratio = rVar / gVar;
    if (ratio > 1.3) regimeAdj =  5;
    else if (ratio < 0.7) regimeAdj = -5;
  }

  function H(bins, total) {
    let h = 0;
    for (let j = 0; j < 5; j++) {
      if (bins[j] === 0) continue;
      const p = bins[j] / total;
      h -= p * Math.log2(p);
    }
    return h;
  }
  const Hg = H(gBins, n);
  const Hr = H(rBins, r500);
  const entropyAdj = (Hg > 0 && n >= 50 && (Hg - Hr) > 0.4) ? -5 : 0;

  const mlg = n > 0 ? gLogS / n : 0;
  const mlr = r200 > 0 ? dLogS / r200 : mlg;
  const dv  = mlr - mlg;
  let driftAdj = 0;
  if (n >= 50) {
    if (dv >  0.15) driftAdj =  3;
    else if (dv < -0.15) driftAdj = -3;
  }

  return { regimeAdj, entropyAdj, driftAdj, tailCounts };
}

const _mcCache = new Map();
function monteCarloAdj(lambda, maxWidth) {
  const key = lambda.toFixed(5) + '-' + maxWidth;
  if (_mcCache.has(key)) return _mcCache.get(key);
  if (_mcCache.size > 200) _mcCache.clear();
  const SIMS = 5000;
  const SEED = 0x4f3a9c1b;
  let state  = SEED;
  function rng() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  }
  let hits = 0;
  for (let s = 0; s < SIMS; s++) {
    for (let r = 0; r < maxWidth; r++) {
      if (rng() < lambda) { hits++; break; }
    }
  }
  const diff = Math.abs(hits / SIMS - hitProbability(lambda, maxWidth));
  const adj  = diff > 0.15 ? -Math.min(4, Math.round(diff * 20)) : 0;
  _mcCache.set(key, adj);
  return adj;
}

function gapVarAdj(gaps) {
  if (gaps.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < gaps.length; i++) sum += gaps[i];
  const mean = sum / gaps.length;
  let vs = 0;
  for (let i = 0; i < gaps.length; i++) vs += (gaps[i] - mean) ** 2;
  return Math.sqrt(vs / gaps.length) > mean * 1.8 ? -4 : 0;
}

function partialWeight(m, targetMin) {
  if (m >= targetMin)       return 1.00;
  if (m >= targetMin * 0.8) return 0.50;
  if (m >= targetMin * 0.6) return 0.25;
  return 0;
}

function scanRounds(rounds, targetMin) {
  const n = rounds.length;
  if (n < MIN_ROUNDS) return null;
  const usePartial = targetMin <= 50;
  const start500   = Math.max(0, n - 500);
  let hits = 0, lastIdx = -1, recent500 = 0, pwSum = 0;
  const gaps = [];
  for (let i = 0; i < n; i++) {
    const m = rounds[i].multiplier;
    if (m >= targetMin) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i;
      hits++;
      pwSum += 1.0;
      if (i >= start500) recent500++;
    } else if (usePartial) {
      pwSum += partialWeight(m, targetMin);
    }
  }
  if (hits < 2) return null;
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const lambdaGlobal = bayesLambda(hits, n);
  const lambda       = blendedLambda(rounds, targetMin, lambdaGlobal, recent500);
  let sum = 0;
  for (let i = 0; i < gaps.length; i++) sum += gaps[i];
  const meanGap = gaps.length > 0 ? sum / gaps.length : 1 / lambda;
  let vs = 0;
  for (let i = 0; i < gaps.length; i++) vs += (gaps[i] - meanGap) ** 2;
  const cv = meanGap > 0 ? (gaps.length > 1 ? Math.sqrt(vs / gaps.length) : 0) / meanGap : 1;
  return { hits, pwSum, n, lambda, lambdaGlobal, meanGap, cv, gapSinceLast, gaps };
}

function calcConfidence(hits, lambda, lambdaGlobal, cv, n, pwSum,
                        regAdj, entAdj, tailAdj, driftAdj, mcAdj, gvAdj) {
  let c = Math.min(65, 20 + Math.log2(hits + 1) * 9);
  c += Math.min(3, (pwSum / Math.max(1, hits)) * 2);
  c -= Math.min(10, Math.abs(cv - 1) * 5);
  if (lambdaGlobal > 0) {
    const d = Math.abs(lambda - lambdaGlobal) / lambdaGlobal;
    if (d < 0.2)      c += 8;
    else if (d < 0.5) c += 4;
    else              c -= 5;
  }
  c += regAdj + entAdj + tailAdj + driftAdj + mcAdj + gvAdj;
  return Math.max(25, Math.min(95, Math.round(c)));
}

let _statsCache = null;
function buildPrediction(sortedRounds, targetMin, maxWidth) {
  const gs = _statsCache ?? computeGlobalStats(sortedRounds);
  const s  = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, pwSum, n, lambda, lambdaGlobal, meanGap, cv, gapSinceLast, gaps } = s;
  const rawCenter = meanGap - gapSinceLast;
  const center    = Math.max(-maxWidth, rawCenter);
  const low       = center <= 0 ? 0 : Math.max(0, Math.round(center - maxWidth / 2));
  const high      = low + maxWidth - 1;
  const conf = calcConfidence(
    hits, lambda, lambdaGlobal, cv, n, pwSum,
    gs.regimeAdj, gs.entropyAdj,
    tailExpansionAdj(n, targetMin, gs.tailCounts),
    gs.driftAdj,
    monteCarloAdj(lambda, maxWidth),
    gapVarAdj(gaps)
  );
  return { low, high, confidence: conf, lambda, lambdaGlobal, gapSinceLast, hits, n };
}

// ── getStatus: works on roundId-based arrays ──────────────────────────────
function getStatus(sortedRounds, pred, currentRoundId) {
  const ws = pred.anchorRound + pred.low;
  const we = pred.anchorRound + pred.high;
  let lo = 0, hi = sortedRounds.length - 1, startIdx = sortedRounds.length;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedRounds[mid].roundId >= pred.anchorRound) { startIdx = mid; hi = mid - 1; }
    else lo = mid + 1;
  }
  for (let i = startIdx; i < sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > we) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < ws) return { status: 'early', hitRound: r.roundId };
    return { status: 'hit', hitRound: r.roundId };
  }
  if (currentRoundId > we)                          return { status: 'miss'    };
  if (currentRoundId >= ws && currentRoundId <= we) return { status: 'active'  };
  return { status: 'waiting' };
}

// ============================================================================
// PATTERN ENGINE  v3  (exact copy from StatisticalEngine.jsx)
// ============================================================================

function buildPatternPrediction(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;
  const W1 = 20, W2 = 60, W3 = 200;
  const s1 = Math.max(0, n - W1);
  const s2 = Math.max(0, n - W2);
  const s3 = Math.max(0, n - W3);
  let hits = 0, lastIdx = -1;
  let hW1 = 0, hW2 = 0, hW3 = 0;
  const gaps = [];
  const FAST_A = 0.15, SLOW_A = 0.03;
  let emaFast = -1, emaSlow = -1;
  for (let i = 0; i < n; i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast < 0) { emaFast = isHit; emaSlow = isHit; }
    else {
      emaFast = FAST_A * isHit + (1 - FAST_A) * emaFast;
      emaSlow = SLOW_A * isHit + (1 - SLOW_A) * emaSlow;
    }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i;
      hits++;
      if (i >= s1) hW1++;
      if (i >= s2) hW2++;
      if (i >= s3) hW3++;
    }
  }
  if (hits < 5 || gaps.length < 4) return null;
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const globalRate   = hits / n;
  let gSum = 0;
  for (let i = 0; i < gaps.length; i++) gSum += gaps[i];
  const meanGap = gSum / gaps.length;
  const sortedGaps = [...gaps].sort((a, b) => a - b);
  const mid2       = Math.floor(sortedGaps.length / 2);
  const medianGap  = sortedGaps.length % 2 === 1
    ? sortedGaps[mid2]
    : (sortedGaps[mid2 - 1] + sortedGaps[mid2]) / 2;
  let gVs = 0;
  for (let i = 0; i < gaps.length; i++) gVs += (gaps[i] - meanGap) ** 2;
  const stdGap = gaps.length > 1 ? Math.sqrt(gVs / gaps.length) : meanGap;
  const cv     = meanGap > 0 ? stdGap / meanGap : 1;
  const dW1 = hW1 / W1;
  const dW2 = hW2 / W2;
  const dW3 = hW3 / Math.min(W3, n);
  const rW1 = globalRate > 0 ? (globalRate - dW1) / globalRate : 0;
  const rW2 = globalRate > 0 ? (globalRate - dW2) / globalRate : 0;
  const rW3 = globalRate > 0 ? (globalRate - dW3) / globalRate : 0;
  const clusterScore = Math.max(-1, Math.min(1, rW1 * 0.55 + rW2 * 0.30 + rW3 * 0.15));
  const trendRaw     = emaSlow > 0 ? (emaSlow - emaFast) / emaSlow : 0;
  const trendScore   = Math.max(-1, Math.min(1, trendRaw * 3));
  let acSum = 0, varSum = 0;
  for (let i = 0; i < gaps.length; i++) varSum += (gaps[i] - meanGap) ** 2;
  for (let i = 1; i < gaps.length; i++) acSum += (gaps[i - 1] - meanGap) * (gaps[i] - meanGap);
  const autoCorr     = varSum > 0 ? acSum / varSum : 0;
  const aboveMedian  = gapSinceLast > medianGap ? 1 : -1;
  const patternScore = Math.max(-1, Math.min(1, -autoCorr * aboveMedian * 0.8));
  const composite    = clusterScore * 0.50 + trendScore * 0.30 + patternScore * 0.20;
  const scores       = [clusterScore, trendScore, patternScore];
  const sameSign     = scores.filter(s => s > 0.08).length;
  const oppSign      = scores.filter(s => s < -0.08).length;
  const agreement    = Math.max(sameSign, oppSign);
  const confidence   = Math.max(30, Math.min(92,
    Math.round(40 + Math.min(18, Math.log2(hits + 1) * 4.5) + Math.abs(composite) * 25 + (agreement - 1) * 7)
  ));
  return { confidence, hits, meanGap: Math.round(meanGap), medianGap: Math.round(medianGap),
           gapSinceLast, clusterScore, trendScore, patternScore };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const { medianGap, clusterScore, trendScore, patternScore, confidence } = patternResult;
  const scale        = Math.max(medianGap, maxWidth * 1.5);
  const clusterShift = -clusterScore * scale * 1.5;
  const trendShift   =  trendScore   * scale * 1.0;
  const patternShift = -patternScore * scale * 0.8;
  const rawCenter    = clusterShift * 0.50 + trendShift * 0.30 + patternShift * 0.20;
  const center       = Math.max(0, Math.round(rawCenter));
  const low          = Math.max(0, Math.round(center - maxWidth / 2));
  const high         = low + maxWidth - 1;
  return { low, high, confidence };
}

// ============================================================================
// KEY HELPERS (mirrors frontend dedup logic)
// ============================================================================

function histKey(r) {
  const lo = r.lo ?? 0;
  const hi = r.hi ?? 0;
  return `${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound ?? 'x'}`;
}

function patHistKey(r) {
  const lo = r.lo ?? 0;
  const hi = r.hi ?? 0;
  return `pat-${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound ?? 'x'}`;
}

// ============================================================================
// INITIALISE — load existing state from DB on first run
// ============================================================================

async function initialise() {
  if (initialised) return;
  initialised = true;

  // Load locked predictions
  try {
    const dbPreds = await getLockedPreds();
    lockedPreds    = {};
    lockedPatterns = {};
    for (const [label, pred] of Object.entries(dbPreds)) {
      const target = TARGETS.find(t => t.label === label);
      if (!target) continue;
      const eta = pred.eta || {};
      lockedPreds[label] = {
        low:         eta.low  ?? 0,
        high:        eta.high ?? (target.maxWidth - 1),
        confidence:  eta.conf ?? 50,
        targetMin:   target.min,
        anchorRound: pred.roundWhenMade ?? pred.lo,
        generation:  pred.generation ?? 1,
        stale:       true,
      };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} locked preds from DB`);
  } catch (e) {
    console.error('[engine] Failed to load locked preds:', e.message);
    lockedPreds    = {};
    lockedPatterns = {};
  }

  // Load existing outcome history to populate savedKeys (avoid re-saving)
  try {
    const rows = await getPredictions({ limit: 500 });
    for (const r of rows) {
      if (!r.source || r.source === 'engine') {
        savedKeys.add(histKey(r));
        // Also register legacy key form
        savedKeys.add(`${r.target}-${r.anchorRound ?? r.lo}-${r.outcome}-${r.hitRound ?? 'x'}`);
      } else if (r.source === 'pattern') {
        patSavedKeys.add(patHistKey(r));
        patSavedKeys.add(`pat-${r.target}-${r.anchorRound ?? r.lo}-${r.outcome}-${r.hitRound ?? 'x'}`);
      }
    }
    console.log(`[engine] Loaded ${savedKeys.size} engine keys, ${patSavedKeys.size} pattern keys`);
  } catch (e) {
    console.error('[engine] Failed to load prediction history:', e.message);
  }
}

// ============================================================================
// PROCESS ENGINE PREDICTIONS
// ============================================================================

async function processEngine(sortedRounds, lastRoundId) {
  _statsCache = computeGlobalStats(sortedRounds);
  const lockedPredsToSave = {};
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPreds[target.label];

    // No prediction yet — create one
    if (!existing) {
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth);
      if (pred) {
        lockedPreds[target.label] = {
          low:         pred.low,
          high:        pred.high,
          confidence:  pred.confidence,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  1,
        };
        anyChange = true;
        console.log(`[engine] NEW pred ${target.label}: #${lastRoundId + pred.low}–#${lastRoundId + pred.high} (${pred.confidence}%)`);
      }
      continue;
    }

    // Stale (loaded from DB) — re-anchor with fresh data
    if (existing.stale) {
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth);
      if (pred) {
        lockedPreds[target.label] = {
          low:         pred.low,
          high:        pred.high,
          confidence:  pred.confidence,
          targetMin:   target.min,
          anchorRound: existing.anchorRound,
          generation:  existing.generation,
        };
        anyChange = true;
      }
      continue;
    }

    // Check outcome
    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (status.status === 'hit' || status.status === 'miss' || status.status === 'early') {
      const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
      const record  = {
        target:      target.label,
        minMult:     target.min,
        outcome,
        lo:          existing.anchorRound + existing.low,
        hi:          existing.anchorRound + existing.high,
        anchorRound: existing.anchorRound,
        hitRound:    status.hitRound || null,
        generation:  existing.generation,
        source:      'engine',
        ts:          Date.now(),
      };
      const key = histKey(record);
      if (!savedKeys.has(key)) {
        savedKeys.add(key);
        savedKeys.add(`${record.target}-${record.anchorRound}-${record.outcome}-${record.hitRound ?? 'x'}`);
        try {
          await savePrediction(record);
          console.log(`[engine] SAVED ${target.label} ${outcome.toUpperCase()} | #${record.lo}–#${record.hi}${record.hitRound ? ` @#${record.hitRound}` : ''}`);
        } catch (e) {
          console.error(`[engine] Save failed for ${target.label}:`, e.message);
        }
      }

      // Generate next prediction
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth);
      if (pred) {
        lockedPreds[target.label] = {
          low:         pred.low,
          high:        pred.high,
          confidence:  pred.confidence,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  existing.generation + 1,
        };
        anyChange = true;
        console.log(`[engine] NEXT pred ${target.label}: #${lastRoundId + pred.low}–#${lastRoundId + pred.high} (${pred.confidence}%)`);
      }
    }
  }

  _statsCache = null;

  // Persist updated locked preds to DB
  if (anyChange) {
    for (const [label, pred] of Object.entries(lockedPreds)) {
      if (pred.stale) continue;
      lockedPredsToSave[label] = {
        lo:            pred.anchorRound + pred.low,
        hi:            pred.anchorRound + pred.high,
        roundWhenMade: pred.anchorRound,
        generation:    pred.generation,
        eta: { low: pred.low, high: pred.high, conf: pred.confidence },
      };
    }
    if (Object.keys(lockedPredsToSave).length > 0) {
      try {
        await saveLockedPreds(lockedPredsToSave);
      } catch (e) {
        console.error('[engine] Failed to save locked preds:', e.message);
      }
    }
  }
}

// ============================================================================
// PROCESS PATTERN PREDICTIONS
// ============================================================================

async function processPattern(sortedRounds, lastRoundId) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing  = lockedPatterns[target.label];
    const patPred   = buildPatternPrediction(sortedRounds, target.min);
    const win       = buildPatternWindow(patPred, target.maxWidth);

    // No prediction or stale — create/refresh
    if (!existing || existing.stale) {
      if (win) {
        lockedPatterns[target.label] = {
          low:         win.low,
          high:        win.high,
          confidence:  win.confidence,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  existing ? existing.generation : 1,
        };
        anyChange = true;
      }
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (status.status === 'hit' || status.status === 'miss' || status.status === 'early') {
      const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
      const record  = {
        target:      target.label,
        minMult:     target.min,
        outcome,
        lo:          existing.anchorRound + existing.low,
        hi:          existing.anchorRound + existing.high,
        anchorRound: existing.anchorRound,
        hitRound:    status.hitRound || null,
        generation:  existing.generation,
        source:      'pattern',
        ts:          Date.now(),
      };
      const key = patHistKey(record);
      if (!patSavedKeys.has(key)) {
        patSavedKeys.add(key);
        patSavedKeys.add(`pat-${record.target}-${record.anchorRound}-${record.outcome}-${record.hitRound ?? 'x'}`);
        try {
          await savePrediction(record);
          console.log(`[pattern] SAVED ${target.label} ${outcome.toUpperCase()} | #${record.lo}–#${record.hi}${record.hitRound ? ` @#${record.hitRound}` : ''}`);
        } catch (e) {
          console.error(`[pattern] Save failed for ${target.label}:`, e.message);
        }
      }

      if (win) {
        lockedPatterns[target.label] = {
          low:         win.low,
          high:        win.high,
          confidence:  win.confidence,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  existing.generation + 1,
        };
        anyChange = true;
      }
    }
  }
}

// ============================================================================
// MAIN ENTRY POINT — called by collector after every successful poll
// ============================================================================

async function runPredictionEngine() {
  try {
    await initialise();

    // Fetch latest rounds from DB (up to 5000 for accuracy)
    const rounds = await getRounds({ limit: 5000 });
    if (rounds.length < MIN_ROUNDS) return;

    // Sort by roundId ascending
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Skip if nothing new
    if (rounds.length === lastRoundCount) return;
    lastRoundCount = rounds.length;

    await processEngine(rounds, lastRoundId);
    await processPattern(rounds, lastRoundId);

  } catch (e) {
    console.error('[predictionEngine] Error:', e.message);
  }
}

module.exports = { runPredictionEngine };