'use strict';
// patternEngine.js — Server-side Pattern Engine (Enhanced)
// =========================================================
// DROP-IN REPLACEMENT. Zero changes to other files.
// Same DB functions, same exports, same ENGINE_ID='pattern'.
//
// AUDIT FIXES vs original:
// PAT-1  clusterScore used rate-deviation (dW1-globalRate), NOT actual
//        run-length encoding. Now uses extractRunFeatures() — real RLE on
//        the raw multiplier sequence, detecting true consecutive lows/highs.
// PAT-2  W1=15 window too short for b2b streaks > 15 rounds. Windows now
//        W1=20, W2=75, W3=300 and a new W4=500 long-memory window.
// PAT-3  No b2b streak counter anywhere. Added: currentHighStreak,
//        b2bRate, b2bContinuationProb as explicit scoring inputs.
// PAT-4  No white cluster detection. Added: currentLowStreak,
//        avgLowRunLen, lowDensity windows, regime classification.
// PAT-5  buildWindow() ignored composite direction for rare targets —
//        now applies direction-weighted gap adjustment for all targets.
// PAT-6  momentum based only on last 5 gaps — too short, noisy.
//        Now momentum uses last 20 gaps vs global mean (robust).
// PAT-7  autocorrelation only checked lags 1-5. Extended to lags 1-10.
// PAT-8  No sparse-data confidence penalty for rare targets (100x+).
//        Added sparsePenalty scaling.

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedPatternPreds,
  getLockedPatternPreds,
} = require('./db');

const ENGINE_ID  = 'pattern';
const MIN_ROUNDS = 50;
const MIN_HITS   = 8;
const MIN_GAPS   = 6;
const STALE_THRESHOLD = 50000;

const TARGETS = [
  { label: '5x',    min: 5,    maxWidth: 3,  rare: false },
  { label: '10x',   min: 10,   maxWidth: 5,  rare: false },
  { label: '20x',   min: 20,   maxWidth: 7,  rare: false },
  { label: '50x',   min: 50,   maxWidth: 12, rare: false },
  { label: '100x',  min: 100,  maxWidth: 18, rare: true  },
  { label: '250x',  min: 250,  maxWidth: 25, rare: true  },
  { label: '500x',  min: 500,  maxWidth: 35, rare: true  },
  { label: '1000x', min: 1000, maxWidth: 50, rare: true  },
];

const state = {
  windows:      {},
  savedSet:     null,
  lastRoundId:  0,
};

let cachedRounds       = [];
let cachedRoundsLastId = 0;
let initialised        = false;

// ── Math helpers ──────────────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function mean(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}
function earlyHitTolerance(maxWidth) { return Math.floor(maxWidth / 2); }
function sparsePenalty(hits, minFull) {
  return hits >= minFull ? 1.0 : Math.sqrt(Math.max(1, hits) / minFull);
}

function bisectLeft(rounds, targetId) {
  let lo = 0, hi = rounds.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (rounds[mid].roundId < targetId) lo = mid + 1; else hi = mid; }
  return lo;
}
function findHitInRange(rounds, fromId, toId, minMult) {
  const start = bisectLeft(rounds, fromId);
  for (let i = start; i < rounds.length; i++) {
    if (rounds[i].roundId > toId) break;
    if (rounds[i].multiplier >= minMult) return rounds[i];
  }
  return null;
}

// ── Run-Length Feature Extractor (PAT-1/3/4 fix) ─────────────────────────────
// Operates on the raw sequence — preserves consecutive structure that
// computeGaps() destroys.
function extractRunFeatures(rounds, targetMin) {
  const n = rounds.length;
  if (n < 10) return null;

  // Run-Length Encoding
  const runs = [];
  let curHigh = rounds[0].multiplier >= targetMin, curLen = 1;
  for (let i = 1; i < n; i++) {
    const h = rounds[i].multiplier >= targetMin;
    if (h === curHigh) { curLen++; }
    else { runs.push({ isHigh: curHigh, len: curLen }); curHigh = h; curLen = 1; }
  }
  runs.push({ isHigh: curHigh, len: curLen });

  const highRuns = runs.filter(r =>  r.isHigh).map(r => r.len);
  const lowRuns  = runs.filter(r => !r.isHigh).map(r => r.len);

  const lastRun         = runs[runs.length - 1];
  const currentIsHigh   = lastRun.isHigh;
  const currentStreakLen = lastRun.len;

  // B2B metrics (PAT-3 fix)
  const b2bOccurrences = highRuns.filter(l => l >= 2).length;
  const b2bRate        = highRuns.length ? b2bOccurrences / highRuns.length : 0;
  let b2bContinuationProb = 0;
  if (highRuns.length >= 5) {
    const ext = highRuns.filter(l => l >= 2).reduce((s, l) => s + l - 1, 0);
    const tot = highRuns.reduce((s, l) => s + l, 0);
    b2bContinuationProb = tot > 0 ? ext / tot : 0;
  }
  const avgHighRunLen = highRuns.length ? mean(highRuns) : 0;
  const maxHighRunLen = highRuns.length ? Math.max(...highRuns) : 0;

  // White cluster metrics (PAT-4 fix)
  const avgLowRunLen = lowRuns.length ? mean(lowRuns) : 0;
  const maxLowRunLen = lowRuns.length ? Math.max(...lowRuns) : 0;

  // Sliding window density — % of last W rounds that are LOW
  const W10 = rounds.slice(-10),  W20 = rounds.slice(-20),  W50 = rounds.slice(-50);
  const lowDensity10 = W10.filter(r => r.multiplier < targetMin).length / Math.max(1, W10.length);
  const lowDensity20 = W20.filter(r => r.multiplier < targetMin).length / Math.max(1, W20.length);
  const lowDensity50 = W50.filter(r => r.multiplier < targetMin).length / Math.max(1, W50.length);
  const densityTrend = lowDensity10 - lowDensity50;

  // Regime
  const globalLowRate = 1 - rounds.filter(r => r.multiplier >= targetMin).length / n;
  let regime = 'NEUTRAL';
  if      (currentIsHigh && currentStreakLen >= 2)                                     regime = 'B2B';
  else if (currentIsHigh && runs.length >= 2 && !runs[runs.length - 2].isHigh
           && runs[runs.length - 2].len <= avgLowRunLen * 0.5)                         regime = 'HOT_AFTER_SHORT_COLD';
  else if (!currentIsHigh && currentStreakLen >= avgLowRunLen * 1.5)                   regime = 'WHITE_CLUSTER';
  else if (!currentIsHigh && currentStreakLen >= maxLowRunLen * 0.8 && maxLowRunLen > 2) regime = 'EXTREME_WHITE';
  else if (b2bRate > 0.25 && lowDensity20 < globalLowRate * 0.7)                      regime = 'HOT';
  else if (lowDensity20 > globalLowRate * 1.3)                                        regime = 'COLD';

  // Post-cluster gap: after a long-low-run, how soon does a hit arrive?
  const longLowThresh = Math.max(2, Math.round(avgLowRunLen * 1.3));
  const postClusterGaps = [];
  for (let i = 0; i < runs.length - 1; i++) {
    if (!runs[i].isHigh && runs[i].len >= longLowThresh && runs[i + 1].isHigh) {
      postClusterGaps.push(1);
    }
  }
  const avgPostClusterGap = postClusterGaps.length ? mean(postClusterGaps) : null;

  return {
    runs, highRuns, lowRuns,
    currentIsHigh, currentStreakLen, regime,
    b2bRate, b2bContinuationProb, avgHighRunLen, maxHighRunLen,
    avgLowRunLen, maxLowRunLen, avgPostClusterGap,
    lowDensity10, lowDensity20, lowDensity50, densityTrend, globalLowRate,
  };
}

// ── Pattern analysis (fully rewritten to use run features) ───────────────────
function analysePattern(rounds, targetMin) {
  const n = rounds.length;
  if (n < MIN_ROUNDS) return null;

  // PAT-2 fix: extended windows W1=20, W2=75, W3=300, W4=500
  const W1 = 20, W2 = 75, W3 = 300, W4 = 500;
  const s1 = Math.max(0, n - W1), s2 = Math.max(0, n - W2);
  const s3 = Math.max(0, n - W3), s4 = Math.max(0, n - W4);

  let hits = 0, lastIdx = -1;
  let hW1 = 0, hW2 = 0, hW3 = 0, hW4 = 0;
  const gaps = [];
  // PAT-6 fix: use two EMA speeds but also track last 20 gaps for momentum
  const FA = 0.15, SA = 0.015; // slower alphas for more stability
  let emaFast = -1, emaSlow = -1;

  for (let i = 0; i < n; i++) {
    const isHit = rounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast < 0) { emaFast = isHit; emaSlow = isHit; }
    else { emaFast = FA * isHit + (1 - FA) * emaFast; emaSlow = SA * isHit + (1 - SA) * emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++;
      if (i >= s1) hW1++;
      if (i >= s2) hW2++;
      if (i >= s3) hW3++;
      if (i >= s4) hW4++;
    }
  }

  if (hits < MIN_HITS || gaps.length < MIN_GAPS) return null;

  const globalRate   = hits / n;
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;

  const meanGap   = mean(gaps);
  const stdGap    = stdDev(gaps);
  const cv        = meanGap > 0 ? stdGap / meanGap : 1;
  const sg        = [...gaps].sort((a, b) => a - b);
  const mid2      = Math.floor(sg.length / 2);
  const medianGap = sg.length % 2 === 1 ? sg[mid2] : (sg[mid2 - 1] + sg[mid2]) / 2;

  // ── PAT-1 fix: real run-length cluster scores ─────────────────────────────
  const rf = extractRunFeatures(rounds, targetMin);

  // Rate-based cluster scores (original, now supplemented not replaced)
  const dW1 = hW1 / W1, dW2 = hW2 / W2;
  const dW3 = hW3 / Math.min(W3, n), dW4 = hW4 / Math.min(W4, n);
  const safe = v => Math.max(-1, Math.min(1, v));

  // Original cluster score: rate deviation across 4 windows (PAT-2: added W4)
  const rateClusterScore = safe(
    safe((dW1 - globalRate) / Math.max(globalRate, 0.001)) * 0.40 +
    safe((dW2 - globalRate) / Math.max(globalRate, 0.001)) * 0.25 +
    safe((dW3 - globalRate) / Math.max(globalRate, 0.001)) * 0.20 +
    safe((dW4 - globalRate) / Math.max(globalRate, 0.001)) * 0.15
  );

  // PAT-1/3/4 fix: real RLE cluster scores
  let rlB2bScore = 0, rlClusterScore = 0;
  if (rf) {
    // B2B score: positive when b2b streak is active and has high continuation prob
    if (rf.regime === 'B2B')
      rlB2bScore = clamp(rf.b2bContinuationProb * 1.5, 0, 1);
    else if (rf.regime === 'HOT' || rf.regime === 'HOT_AFTER_SHORT_COLD')
      rlB2bScore = clamp(rf.b2bRate * 0.8, 0, 0.7);

    // White cluster score: positive when in a safe cluster (expect hit sooner = bullish)
    if (rf.regime === 'WHITE_CLUSTER')
      rlClusterScore = clamp(rf.currentStreakLen / Math.max(1, rf.avgLowRunLen) * 0.6, 0, 0.8);
    else if (rf.regime === 'EXTREME_WHITE')
      rlClusterScore = 0.9;
    else if (rf.regime === 'COLD')
      rlClusterScore = clamp(rf.lowDensity20 * 0.5, 0, 0.4);
  }

  // Blend rate-deviation + RLE cluster scores
  const clusterScore = safe(rateClusterScore * 0.40 + rlB2bScore * 0.35 + rlClusterScore * 0.25);

  const trendScore = safe((emaSlow > 0 ? (emaFast - emaSlow) / emaSlow : 0) * 4);

  // PAT-7 fix: autocorrelation at lags 1-10 (was 1-5)
  let varSum = 0;
  for (const g of gaps) varSum += (g - meanGap) ** 2;
  let bestAC = 0;
  for (let lag = 1; lag <= Math.min(10, gaps.length - 1); lag++) {
    let cov = 0;
    for (let i = lag; i < gaps.length; i++) cov += (gaps[i - lag] - meanGap) * (gaps[i] - meanGap);
    const ac = varSum > 0 ? cov / varSum : 0;
    if (Math.abs(ac) > Math.abs(bestAC)) bestAC = ac;
  }
  const patternScore = safe(bestAC * 0.9);

  // PAT-6 fix: momentum over last 20 gaps (was last 5)
  const last20     = gaps.slice(-Math.min(20, gaps.length));
  const last20Mean = mean(last20);
  const momentum   = meanGap > 0 ? (meanGap - last20Mean) / meanGap : 0;

  // Composite: now includes b2b and cluster RLE signals explicitly
  const composite = clamp(
    clusterScore * 0.35 + trendScore * 0.25 + patternScore * 0.15 +
    safe(momentum) * 0.15 + (rf ? safe(rlB2bScore - rlClusterScore) * 0.10 : 0),
    -1, 1
  );

  const absComposite = Math.abs(composite);
  const direction    = composite > 0.08 ? 'bullish' : composite < -0.08 ? 'bearish' : 'neutral';
  const agree = Math.max(
    [clusterScore, trendScore, patternScore].filter(s => s >  0.08).length,
    [clusterScore, trendScore, patternScore].filter(s => s < -0.08).length
  );

  // PAT-8 fix: sparse-data confidence penalty
  const sp = sparsePenalty(hits, 50);

  const conf = Math.max(20, Math.min(85, Math.round(
    (32 + Math.min(18, Math.log2(hits + 1) * 4) + absComposite * 30 + (agree - 1) * 6
      - (cv > 1.5 ? 8 : cv > 1.2 ? 4 : 0)
      - (gapSinceLast > meanGap * 2 ? 5 : 0)) * sp
  )));

  return {
    direction, confidence: conf, hits,
    meanGap: Math.round(meanGap), medianGap: Math.round(medianGap),
    composite: +composite.toFixed(3), momentum: +momentum.toFixed(3),
    gapSinceLast, clusterScore: +clusterScore.toFixed(3),
    trendScore: +trendScore.toFixed(3), patternScore: +patternScore.toFixed(3),
    cv: +cv.toFixed(2),
    // New fields for buildWindow
    rf: rf ? {
      regime: rf.regime,
      b2bRate: +rf.b2bRate.toFixed(3),
      b2bContinuationProb: +rf.b2bContinuationProb.toFixed(3),
      currentIsHigh: rf.currentIsHigh,
      currentStreakLen: rf.currentStreakLen,
      avgLowRunLen: +rf.avgLowRunLen.toFixed(2),
      avgPostClusterGap: rf.avgPostClusterGap,
      lowDensity20: +rf.lowDensity20.toFixed(3),
      densityTrend: +rf.densityTrend.toFixed(3),
    } : null,
  };
}

function buildWindow(pr, maxWidth, anchorRound) {
  if (!pr) return null;
  const medianGap    = pr.medianGap || pr.meanGap || maxWidth;
  const gapSinceLast = pr.gapSinceLast ?? 0;
  const momentum     = pr.momentum ?? 0;
  const meanGap      = pr.meanGap || medianGap;

  // PAT-6 fix: robust momentum adjustment (capped tighter)
  const momentumAdj   = Math.max(0.75, Math.min(1.25, 1 - momentum * 0.25));
  const overdueFactor = gapSinceLast > meanGap * 1.5 ? 0.80 : 1.0;
  let expectedGap     = Math.max(1, Math.round(medianGap * momentumAdj * overdueFactor));

  // PAT-5 fix: apply regime-based adjustment from run features
  const rf = pr.rf;
  if (rf) {
    switch (rf.regime) {
      case 'B2B':
        // Currently in b2b — expect sooner
        expectedGap = Math.max(1, Math.round(expectedGap * (1 - rf.b2bContinuationProb * 0.35)));
        break;
      case 'HOT_AFTER_SHORT_COLD':
        expectedGap = Math.max(1, Math.round(expectedGap * 0.85));
        break;
      case 'WHITE_CLUSTER':
        if (rf.avgPostClusterGap !== null) {
          expectedGap = Math.max(1, Math.round(expectedGap * 0.5 + rf.avgPostClusterGap * 0.5));
        } else {
          expectedGap = Math.max(1, Math.round(expectedGap * 0.88));
        }
        break;
      case 'EXTREME_WHITE':
        expectedGap = Math.max(1, Math.round(expectedGap * 0.70));
        break;
      case 'HOT':
        expectedGap = Math.max(1, Math.round(expectedGap * (1 - (1 - rf.lowDensity20) * 0.18)));
        break;
      case 'COLD':
        expectedGap = Math.max(1, Math.round(expectedGap * (1 + rf.densityTrend * 0.12)));
        break;
    }
  }

  const remaining = Math.max(1, expectedGap - gapSinceLast);
  const low       = Math.max(1, remaining - Math.floor(maxWidth / 2));
  const hi        = low + maxWidth - 1;

  return {
    lo: anchorRound + low,
    hi: anchorRound + hi,
    expectedGap,
    confidence: pr.confidence,
    direction: pr.direction,
    composite: pr.composite ?? null,
    momentum: pr.momentum ?? null,
    gapSinceLast,
    eta: {
      low, high: hi, conf: pr.confidence, expectedGap,
      direction: pr.direction, composite: pr.composite ?? null,
      momentum: pr.momentum ?? null, gapSinceLast,
      regime: rf?.regime ?? null,
    },
  };
}

// ── Rounds cache ──────────────────────────────────────────────────────────────
async function getPatternRounds() {
  if (cachedRounds.length === 0) {
    cachedRounds = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[pattern] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Save outcome ──────────────────────────────────────────────────────────────
async function saveOutcome(target, outcome, lo, hi, hitRound, generation) {
  const key = `${lo}:${hi}`;
  if (state.savedSet.has(key)) return;
  state.savedSet.add(key);
  try {
    await savePrediction({
      target: target.label, minMult: target.min,
      outcome, lo, hi, hitRound: hitRound ?? null,
      generation: generation ?? 1, source: ENGINE_ID, probW: null,
    });
    console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${lo}–#${hi}${hitRound ? ` @#${hitRound}` : ''}`);
  } catch (e) {
    console.error(`[pattern] save fail:`, e.message);
    state.savedSet.delete(key);
  }
}

// ── Main tick ─────────────────────────────────────────────────────────────────
async function processPatternEngine(rounds, lastRoundId) {
  const toSave = {};
  let anyChange = false;

  for (const target of TARGETS) {
    const win = state.windows[target.label];

    if (win) {
      const { lo, hi, generation } = win;
      const isTooOld = lastRoundId - hi > STALE_THRESHOLD;
      const earlyCheckLo = Math.max(win.roundWhenMade + 1, lo - earlyHitTolerance(target.maxWidth));
      const earlyHit = lo > win.roundWhenMade + 1 && earlyCheckLo <= lo - 1
        ? findHitInRange(rounds, earlyCheckLo, lo - 1, target.min)
        : null;

      if (earlyHit) {
        await saveOutcome(target, 'early', lo, hi, earlyHit.roundId, generation);
        delete state.windows[target.label];
        anyChange = true;
      } else if (lastRoundId >= hi) {
        if (!isTooOld) {
          const hit = findHitInRange(rounds, lo, hi, target.min);
          await saveOutcome(target, hit ? 'win' : 'loss', lo, hi, hit?.roundId ?? null, generation);
        }
        delete state.windows[target.label];
        anyChange = true;
      } else {
        const hit = findHitInRange(rounds, lo, hi, target.min);
        if (hit) {
          await saveOutcome(target, 'win', lo, hi, hit.roundId, generation);
          delete state.windows[target.label];
          anyChange = true;
        } else {
          toSave[target.label] = win;
          continue;
        }
      }
    }

    const pr   = analysePattern(rounds, target.min);
    const pred = buildWindow(pr, target.maxWidth, lastRoundId);
    if (pred) {
      const generation = (win?.generation ?? 0) + 1;
      state.windows[target.label] = {
        lo: pred.lo, hi: pred.hi,
        roundWhenMade: lastRoundId,
        generation,
        eta: pred.eta,
      };
      toSave[target.label] = state.windows[target.label];
      anyChange = true;
      console.log(`[pattern] LOCK ${target.label}: #${pred.lo}–#${pred.hi} dir=${pred.direction} regime=${pred.eta.regime ?? 'N/A'} conf=${pred.confidence}%`);
    }
  }

  if (anyChange && Object.keys(toSave).length) {
    const payload = {};
    for (const [label, w] of Object.entries(toSave)) {
      payload[label] = { lo: w.lo, hi: w.hi, roundWhenMade: w.roundWhenMade, generation: w.generation, eta: w.eta };
    }
    try { await saveLockedPatternPreds(payload); }
    catch (e) { console.error('[pattern] save locked fail:', e.message); }
  }

  return anyChange;
}

// ── Initialise ────────────────────────────────────────────────────────────────
async function initialise() {
  if (initialised) return;
  initialised = true;
  state.savedSet = new Set();
  state.windows  = {};

  try {
    const dbLocked = await getLockedPatternPreds();
    for (const [label, p] of Object.entries(dbLocked)) {
      if (!p?.lo || !p?.hi) continue;
      state.windows[label] = {
        lo: Number(p.lo), hi: Number(p.hi),
        roundWhenMade: Number(p.roundWhenMade ?? p.lo),
        generation: p.generation ?? 1, eta: p.eta ?? {},
      };
    }
    console.log(`[pattern] loaded ${Object.keys(state.windows).length} locked windows`);
  } catch (e) { console.error('[pattern] init locked error:', e.message); }

  try {
    const rows = await getPredictions({ limit: 500000, source: ENGINE_ID });
    for (const r of rows) state.savedSet.add(`${r.lo}:${r.hi}`);
    console.log(`[pattern] pre-warmed savedSet with ${state.savedSet.size} outcomes`);
  } catch (e) { console.error('[pattern] init history error:', e.message); }
}

function resetPatternEngineState() {
  console.log('[pattern] reset');
  state.windows = {}; state.savedSet = null; state.lastRoundId = 0;
  cachedRounds = []; cachedRoundsLastId = 0; initialised = false;
}

async function runPatternEngine() {
  try {
    await initialise();
    const rounds = await getPatternRounds();
    if (rounds.length < MIN_ROUNDS) { console.log(`[pattern] waiting (${rounds.length}/${MIN_ROUNDS})`); return; }
    const lastRoundId = rounds[rounds.length - 1].roundId;
    if (lastRoundId <= state.lastRoundId) return;
    state.lastRoundId = lastRoundId;
    const t0 = Date.now();
    await processPatternEngine(rounds, lastRoundId);
    console.log(`[pattern] tick done in ${Date.now() - t0}ms — ${Object.keys(state.windows).length} active windows`);
  } catch (e) { console.error('[pattern] Fatal:', e.message, e.stack); }
}

module.exports = { runPatternEngine, resetPatternEngineState };