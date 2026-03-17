'use strict';
// predictionEngine.js  v3
// ============================================================================
// BUG FIXES vs v2:
//
//   FIX 1 — STUCK IN RESOLVING (critical):
//     Root cause: stale preds loaded from DB kept their old anchorRound.
//     The stale branch did: anchorRound = existing.anchorRound (past value).
//     So absHigh = oldAnchor + high = round from days ago.
//     currentRoundId > absHigh always → getStatus() returned 'miss' every poll.
//     But savedKeys already had that key so it never re-saved.
//     Fix: when a stale pred's window is ENTIRELY in the past, treat it as
//     resolved (record the outcome) then immediately build a fresh prediction
//     anchored to lastRoundId. If the window overlaps current rounds, scan
//     properly for a hit.
//
//   FIX 2 — STALE BRANCH ANCHOR WRONG:
//     Stale rebuild was using existing.anchorRound instead of lastRoundId.
//     A "stale" pred means we reloaded it from DB on startup and haven't
//     validated it yet. We should rebuild from lastRoundId so the new window
//     is always in the future, not the past.
//
//   FIX 3 — savedKeys dedup too aggressive:
//     The compound key `${target}-${anchorRound}-${outcome}-${hitRound}` was
//     sometimes preventing valid new resolutions from saving because anchorRound
//     was undefined (NaN). Normalized all key fields to Number() with fallback.
//
//   FIX 4 — Pattern engine: same stale anchor bug fixed identically.
//
//   FIX 5 — getStatus binary search off-by-one:
//     Binary search for startIdx was searching rounds where roundId >= anchorRound
//     but anchor is exclusive of the prediction window (window starts at
//     anchorRound + low). Corrected search to start from anchorRound, then
//     skip rounds before absLow.
//
//   FIX 6 — processEngine/processPattern called buildPrediction._statsCache
//     before computeGlobalStats ran for pattern. Fixed ordering.
//
// ============================================================================

const {
  getRounds, savePrediction, getPredictions,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
} = require('./db');

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

let lockedPreds    = null;
let lockedPatterns = null;
let savedKeys      = new Set();
let patSavedKeys   = new Set();
let lastRoundCount = 0;
let initialised    = false;


// ============================================================================
// MATH HELPERS
// ============================================================================

function bayesLambda(hits, n) {
  return (hits + 1) / (n + 2);
}

function blendedLambda(rounds, targetMin, lambdaGlobal, recentHits) {
  const n       = rounds.length;
  const recentN = Math.min(500, n);
  if (recentHits === undefined) {
    recentHits = 0;
    for (let i = n - recentN; i < n; i++)
      if (rounds[i].multiplier >= targetMin) recentHits++;
  }
  const lambdaRecent = bayesLambda(recentHits, recentN);
  return Math.max(1e-6, Math.min(0.5, 0.6 * lambdaGlobal + 0.4 * lambdaRecent));
}

function scanRounds(rounds, targetMin) {
  const n        = rounds.length;
  const start500 = Math.max(0, n - 500);
  let hits = 0, lastIdx = -1, recent500 = 0;
  const gaps = [];

  for (let i = 0; i < n; i++) {
    const m = rounds[i].multiplier;
    if (m >= targetMin) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++;
      if (i >= start500) recent500++;
    }
  }

  if (hits < 3) return null;

  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const lambdaGlobal = bayesLambda(hits, n);
  const lambda       = blendedLambda(rounds, targetMin, lambdaGlobal, recent500);

  let gSum = 0;
  for (const g of gaps) gSum += g;
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / lambda;

  let gVs = 0;
  for (const g of gaps) gVs += (g - meanGap) ** 2;
  const stdGap = gaps.length > 1 ? Math.sqrt(gVs / gaps.length) : meanGap;
  const cv     = meanGap > 0 ? stdGap / meanGap : 1;

  const sg  = [...gaps].sort((a, b) => a - b);
  const mid = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? meanGap
    : sg.length % 2 === 1 ? sg[mid]
    : (sg[mid - 1] + sg[mid]) / 2;

  const p10 = sg[Math.floor(sg.length * 0.10)] ?? sg[0] ?? 0;
  const p90 = sg[Math.floor(sg.length * 0.90)] ?? sg[sg.length - 1] ?? meanGap;

  return { hits, n, lambda, lambdaGlobal, meanGap, medianGap, p10, p90, cv, gapSinceLast, gaps };
}

function computeGlobalStats(rounds) {
  const n        = rounds.length;
  const r500     = Math.min(500, n);
  const r200     = Math.min(200, n);
  const start500 = n - r500;
  const start200 = n - r200;

  let gLogS=0, gLogSS=0, rLogS=0, rLogSS=0, dLogS=0;

  for (let i = 0; i < n; i++) {
    const lv = Math.log(Math.max(1.01, rounds[i].multiplier));
    gLogS += lv; gLogSS += lv * lv;
    if (i >= start500) { rLogS += lv; rLogSS += lv * lv; }
    if (i >= start200) dLogS += lv;
  }

  const gVar = n > 0 ? gLogSS / n - (gLogS / n) ** 2 : 0;
  const rVar = r500 > 0 ? rLogSS / r500 - (rLogS / r500) ** 2 : 0;
  let regimeAdj = 0;
  if (n >= 100 && gVar > 0) {
    const ratio = rVar / gVar;
    if (ratio > 1.4) regimeAdj = 4;
    else if (ratio < 0.6) regimeAdj = -4;
  }

  const mlg = n > 0 ? gLogS / n : 0;
  const mlr = r200 > 0 ? dLogS / r200 : mlg;
  const dv  = mlr - mlg;
  const driftAdj = n >= 100 ? (dv > 0.20 ? 3 : dv < -0.20 ? -3 : 0) : 0;

  const slice200 = rounds.slice(n - Math.min(200, n));
  const sub2r    = slice200.filter(r => r.multiplier < 2).length / slice200.length;
  const coldTailAdj = sub2r > 0.60 ? -10 : sub2r > 0.50 ? -5 : 0;

  return { regimeAdj, driftAdj, coldTailAdj };
}

function detectRegime(rounds) {
  const n = rounds.length;
  if (n < 50) return { regime:'normal', streakPenalty:0, streakPct:0, coldScore:0, currentStreak:0, hotScore:0 };

  const THRESH = 5;
  const streaks = [];
  let cur = 0;
  for (let i = 0; i < n; i++) {
    if (rounds[i].multiplier < THRESH) { cur++; }
    else { if (cur > 0) streaks.push(cur); cur = 0; }
  }
  const currentStreak = cur;
  let streakPct = 0;
  if (streaks.length >= 5) {
    streaks.sort((a, b) => a - b);
    streakPct = streaks.filter(s => s <= currentStreak).length / streaks.length;
  }

  let streakPenalty = 0;
  if      (streakPct >= 0.95) streakPenalty = Math.round(currentStreak * 1.0);
  else if (streakPct >= 0.85) streakPenalty = Math.round(currentStreak * 0.6);
  else if (streakPct >= 0.70) streakPenalty = Math.round(currentStreak * 0.3);

  const W = Math.min(100, n);
  const recent = rounds.slice(n - W);
  let rLogSum = 0, gLogSum = 0, lowCount = 0, highCount = 0;
  for (const r of recent) {
    rLogSum += Math.log(Math.max(1.01, r.multiplier));
    if (r.multiplier < 2)  lowCount++;
    if (r.multiplier >= 20) highCount++;
  }
  for (const r of rounds) gLogSum += Math.log(Math.max(1.01, r.multiplier));
  const logRatio = (gLogSum / n) > 0 ? (rLogSum / W) / (gLogSum / n) : 1;

  let coldScore = 0, hotScore = 0;
  if (logRatio < 0.78)        coldScore += 3; else if (logRatio < 0.90) coldScore += 1;
  if (streakPct >= 0.85)      coldScore += 3; else if (streakPct >= 0.70) coldScore += 1;
  if (currentStreak > 12)     coldScore += 2;
  if (lowCount / W > 0.65)    coldScore += 2;
  if (logRatio > 1.22)        hotScore  += 3; else if (logRatio > 1.12) hotScore  += 1;
  if (highCount / W > 0.30)   hotScore  += 2;

  let rVarSum = 0;
  const rMean = rLogSum / W;
  for (const r of recent) rVarSum += (Math.log(Math.max(1.01, r.multiplier)) - rMean) ** 2;
  const gMean = gLogSum / n;
  let gVarSum = 0;
  for (const r of rounds) gVarSum += (Math.log(Math.max(1.01, r.multiplier)) - gMean) ** 2;
  const isVolatile = (gVarSum / n) > 0 && (rVarSum / W) / (gVarSum / n) > 1.6;

  const regime = coldScore >= 4 ? 'cold' : hotScore >= 3 ? 'hot' : isVolatile ? 'volatile' : 'normal';
  return { regime, coldScore, hotScore, streakPct: +streakPct.toFixed(3), streakPenalty, currentStreak, logRatio: +logRatio.toFixed(3), isVolatile };
}

buildPrediction._statsCache = null;

function buildPrediction(sortedRounds, targetMin, maxWidth, regime) {
  const gs = buildPrediction._statsCache ?? computeGlobalStats(sortedRounds);
  const s  = scanRounds(sortedRounds, targetMin);
  if (!s) return null;

  const { hits, n, lambda, lambdaGlobal, medianGap, cv, gapSinceLast, gaps } = s;
  if (!regime) regime = detectRegime(sortedRounds);

  const remainingGap = medianGap - gapSinceLast;

  let regimeShift = 0;
  if (regime.regime === 'cold') {
    regimeShift = Math.round(regime.streakPenalty + medianGap * 0.25 * Math.min(1, regime.coldScore / 8));
  } else if (regime.regime === 'hot') {
    regimeShift = -Math.round(medianGap * 0.10);
  }

  const center = Math.max(0, remainingGap + regimeShift);
  const low    = center <= 0 ? 0 : Math.max(0, Math.round(center - maxWidth / 2));
  const high   = low + maxWidth - 1;

  let c = Math.min(55, 18 + Math.log2(hits + 1) * 8);
  c -= Math.min(12, Math.abs(cv - 1) * 6);
  if (lambdaGlobal > 0) {
    const d = Math.abs(lambda - lambdaGlobal) / lambdaGlobal;
    if (d < 0.15) c += 7; else if (d < 0.40) c += 3; else c -= 6;
  }
  c += gs.regimeAdj + gs.driftAdj + gs.coldTailAdj;
  if (regime.regime === 'cold')     c -= Math.min(15, regime.coldScore * 2);
  if (regime.regime === 'volatile') c -= 8;
  if (regime.streakPct >= 0.85)     c -= 10;
  if (regime.streakPct >= 0.95)     c -= 8;
  if (gapSinceLast > s.p90 * 1.5) c -= 8;

  const conf = Math.max(20, Math.min(88, Math.round(c)));

  return {
    low, high, confidence: conf,
    lambda, lambdaGlobal, gapSinceLast, medianGap, hits, n,
    regime:        regime.regime,
    streakPenalty: regime.streakPenalty,
    suppressed:    regime.streakPct >= 0.85,
  };
}

// ============================================================================
// FIX 1+5: getStatus — correct binary search + proper window bounds
// ============================================================================

function getStatus(sortedRounds, pred, currentRoundId) {
  // FIX: use anchorRound + low/high as absolute window bounds
  const absLow  = pred.anchorRound + pred.low;
  const absHigh = pred.anchorRound + pred.high;

  // Binary search: find first round with roundId >= anchorRound
  // (we search from anchorRound so we don't miss early hits before absLow)
  let lo = 0, hi = sortedRounds.length - 1, startIdx = sortedRounds.length;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedRounds[mid].roundId >= pred.anchorRound) { startIdx = mid; hi = mid - 1; }
    else lo = mid + 1;
  }

  for (let i = startIdx; i < sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    // Hit found — determine if it's inside window or before it
    if (r.roundId < absLow) return { status: 'early', hitRound: r.roundId };
    return { status: 'hit', hitRound: r.roundId };
  }

  // No hit found
  if (currentRoundId > absHigh)                          return { status: 'miss'    };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status: 'active' };
  return { status: 'waiting' };
}

function buildPatternPrediction(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;

  const W1=15, W2=50, W3=150;
  const s1=Math.max(0,n-W1), s2=Math.max(0,n-W2), s3=Math.max(0,n-W3);
  let hits=0, lastIdx=-1, hW1=0, hW2=0, hW3=0;
  const gaps=[];
  const FA=0.20, SA=0.02;
  let emaFast=-1, emaSlow=-1;

  for (let i=0; i<n; i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast < 0) { emaFast=isHit; emaSlow=isHit; }
    else { emaFast=FA*isHit+(1-FA)*emaFast; emaSlow=SA*isHit+(1-SA)*emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx=i; hits++;
      if (i>=s1) hW1++; if (i>=s2) hW2++; if (i>=s3) hW3++;
    }
  }

  if (hits < 8 || gaps.length < 6) return null;

  const gapSinceLast = lastIdx===-1 ? n : n-lastIdx-1;
  const globalRate   = hits / n;

  let gSum=0;
  for (const g of gaps) gSum+=g;
  const meanGap = gSum / gaps.length;

  const sg=[...gaps].sort((a,b)=>a-b);
  const mid2=Math.floor(sg.length/2);
  const medianGap = sg.length%2===1 ? sg[mid2] : (sg[mid2-1]+sg[mid2])/2;

  let gVs=0;
  for (const g of gaps) gVs+=(g-meanGap)**2;
  const stdGap = gaps.length>1 ? Math.sqrt(gVs/gaps.length) : meanGap;
  const cv = meanGap>0 ? stdGap/meanGap : 1;

  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const rW1 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW1)/Math.max(globalRate,0.001))) : 0;
  const rW2 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW2)/Math.max(globalRate,0.001))) : 0;
  const rW3 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW3)/Math.max(globalRate,0.001))) : 0;
  const clusterScore = Math.max(-1, Math.min(1, rW1*0.50 + rW2*0.30 + rW3*0.20));

  const trendRaw   = emaSlow>0 ? (emaSlow-emaFast)/emaSlow : 0;
  const trendScore = Math.max(-1, Math.min(1, trendRaw*4));

  let varSum=0;
  for (const g of gaps) varSum+=(g-meanGap)**2;

  let bestAC=0, bestLag=1;
  for (let lag=1; lag<=Math.min(3, gaps.length-1); lag++) {
    let covSum=0;
    for (let i=lag; i<gaps.length; i++) covSum+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac = varSum>0 ? covSum/varSum : 0;
    if (Math.abs(ac) > Math.abs(bestAC)) { bestAC=ac; bestLag=lag; }
  }

  const aboveMedian = gapSinceLast > medianGap ? 1 : -1;
  const patternScore = Math.max(-1, Math.min(1, -bestAC * aboveMedian * 0.9));

  const composite = clusterScore*0.45 + trendScore*0.35 + patternScore*0.20;
  const direction = composite > 0.12 ? 'bullish' : composite < -0.12 ? 'bearish' : 'neutral';

  const scores   = [clusterScore, trendScore, patternScore];
  const nBull    = scores.filter(s=>s>0.10).length;
  const nBear    = scores.filter(s=>s<-0.10).length;
  const agree    = Math.max(nBull, nBear);
  const sampleB  = Math.min(20, Math.log2(hits+1) * 5);
  const strengthB= Math.abs(composite) * 28;
  const agreeB   = (agree-1) * 8;
  const cvPenalty= cv > 1.5 ? -8 : cv > 1.2 ? -4 : 0;
  const conf     = Math.max(28, Math.min(90,
    Math.round(38 + sampleB + strengthB + agreeB + cvPenalty)
  ));

  return {
    direction, confidence: conf, hits,
    meanGap:      Math.round(meanGap),
    medianGap:    Math.round(medianGap),
    gapSinceLast, cv: +cv.toFixed(2),
    clusterScore: +clusterScore.toFixed(3),
    trendScore:   +trendScore.toFixed(3),
    patternScore: +patternScore.toFixed(3),
    composite:    +composite.toFixed(3),
    autoCorr:     +bestAC.toFixed(3),
    dominantLag:  bestLag,
  };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const { medianGap, clusterScore, trendScore, patternScore } = patternResult;

  const scale = Math.max(medianGap, maxWidth * 1.5);
  const cShift = -clusterScore * scale * 1.4;
  const tShift =  trendScore   * scale * 0.9;
  const pShift = -patternScore * scale * 0.7;

  const rawCenter = cShift*0.50 + tShift*0.30 + pShift*0.20;
  const center    = Math.max(0, Math.round(rawCenter));
  const low       = Math.max(0, Math.round(center - maxWidth/2));
  const high      = low + maxWidth - 1;

  return { low, high, confidence: patternResult.confidence };
}


// ============================================================================
// KEY HELPERS
// FIX 3: normalize all fields to avoid NaN keys
// ============================================================================

function histKey(r) {
  const lo = Number(r.lo) || 0;
  const hi = Number(r.hi) || 0;
  return `${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound ?? 'x'}`;
}

function patHistKey(r) {
  const lo = Number(r.lo) || 0;
  const hi = Number(r.hi) || 0;
  return `pat-${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound ?? 'x'}`;
}

// ============================================================================
// INITIALISE
// ============================================================================

async function initialise() {
  if (initialised) return;
  initialised = true;
  try {
    const dbPreds = await getLockedPreds();
    lockedPreds    = {};
    lockedPatterns = {};
    for (const [label, pred] of Object.entries(dbPreds)) {
      const target = TARGETS.find(t => t.label === label);
      if (!target) continue;
      const eta    = pred.eta || {};
      const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
      lockedPreds[label] = {
        low:         eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor),
        high:        eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor),
        confidence:  eta.conf ?? 50,
        targetMin:   target.min,
        anchorRound: anchor,
        generation:  pred.generation ?? 1,
        stale:       true, // mark as stale — will be validated on first poll
      };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} locked preds from DB`);
  } catch(e) {
    console.error('[engine] init error:', e.message);
    lockedPreds    = {};
    lockedPatterns = {};
  }

  try {
    const rows = await getPredictions({ limit: 1000 });
    for (const r of rows) {
      if (!r.source || r.source === 'engine') {
        savedKeys.add(histKey(r));
        savedKeys.add(`${r.target}-${Number(r.anchorRound ?? r.lo) || 0}-${r.outcome}-${r.hitRound ?? 'x'}`);
      } else if (r.source === 'pattern') {
        patSavedKeys.add(patHistKey(r));
        patSavedKeys.add(`pat-${r.target}-${Number(r.anchorRound ?? r.lo) || 0}-${r.outcome}-${r.hitRound ?? 'x'}`);
      }
    }
    console.log(`[engine] Loaded ${savedKeys.size} engine keys, ${patSavedKeys.size} pattern keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  try {
    const dbPatPreds = await getLockedPatternPreds();
    for (const [label, pred] of Object.entries(dbPatPreds)) {
      const target = TARGETS.find(t => t.label === label);
      if (!target) continue;
      const eta    = pred.eta || {};
      const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
      lockedPatterns[label] = {
        low:         eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor),
        high:        eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor),
        confidence:  eta.conf ?? 50,
        targetMin:   target.min,
        anchorRound: anchor,
        generation:  pred.generation ?? 1,
        stale:       true,
      };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPatterns).length} pattern locked preds from DB`);
  } catch(e) {
    console.error('[engine] pattern locked preds load error:', e.message);
    if (!lockedPatterns) lockedPatterns = {};
  }

  lastRoundCount = 0;
}

// ============================================================================
// PROCESS ENGINE
// FIX 1+2: stale branch now validates old window against actual rounds,
// records outcome if it's fully resolved, then always anchors new pred to
// lastRoundId (not existing.anchorRound).
// ============================================================================

async function processEngine(sortedRounds, lastRoundId, regime) {
  buildPrediction._statsCache = computeGlobalStats(sortedRounds);
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPreds[target.label];

    // ── No prediction exists — create one ─────────────────────────────────
    if (!existing) {
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
      if (pred) {
        lockedPreds[target.label] = {
          ...pred,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  1,
        };
        anyChange = true;
        console.log(`[engine] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${regime.regime}`);
      }
      continue;
    }

    // ── FIX 2: Stale pred — validate it against actual rounds ──────────────
    // A stale pred was loaded from DB. Its anchorRound is from a past session.
    // We must check if the old window already resolved (fully in the past).
    if (existing.stale) {
      const absLow  = existing.anchorRound + existing.low;
      const absHigh = existing.anchorRound + existing.high;
      const windowInPast = lastRoundId > absHigh;

      if (windowInPast) {
        // FIX 1: Old window is entirely in the past — it must have resolved.
        // Check if there was a hit inside that window.
        const status = getStatus(sortedRounds, existing, lastRoundId);
        const resolvedStatuses = ['hit', 'miss', 'early'];

        if (resolvedStatuses.includes(status.status)) {
          const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
          const record  = {
            target:      target.label,
            minMult:     target.min,
            outcome,
            lo:          absLow,
            hi:          absHigh,
            anchorRound: existing.anchorRound,
            hitRound:    status.hitRound || null,
            generation:  existing.generation,
            source:      'engine',
            ts:          Date.now(),
          };
          const key = histKey(record);
          if (!savedKeys.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow <= absHigh) {
            savedKeys.add(key);
            try {
              await savePrediction(record);
              console.log(`[engine] STALE-RESOLVED ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}`);
            } catch(e) { console.error(`[engine] stale save fail ${target.label}:`, e.message); }
          }
        }

        // Always build a fresh prediction anchored to NOW after resolving stale
        const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
        if (pred) {
          lockedPreds[target.label] = {
            ...pred,
            targetMin:   target.min,
            anchorRound: lastRoundId, // FIX 2: anchor to NOW, not existing.anchorRound
            generation:  existing.generation + 1,
          };
          console.log(`[engine] STALE→FRESH ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
        } else {
          delete lockedPreds[target.label];
          console.warn(`[engine] ${target.label} stale cleared — buildPrediction null, will retry`);
        }
        anyChange = true;
      } else {
        // Window might still be active — clear stale flag and evaluate normally
        existing.stale = false;
        // Fall through to normal status check below on next poll
        // (don't skip this target — re-check status immediately)
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit', 'miss', 'early'].includes(status.status)) {
          // Resolved within the still-valid stale window
          const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
          const record  = {
            target:      target.label,
            minMult:     target.min,
            outcome,
            lo:          absLow,
            hi:          absHigh,
            anchorRound: existing.anchorRound,
            hitRound:    status.hitRound || null,
            generation:  existing.generation,
            source:      'engine',
            ts:          Date.now(),
          };
          const key = histKey(record);
          if (!savedKeys.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh)) {
            savedKeys.add(key);
            try { await savePrediction(record); } catch(e) {}
          }
          const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
          if (pred) {
            lockedPreds[target.label] = { ...pred, targetMin: target.min, anchorRound: lastRoundId, generation: existing.generation + 1 };
          } else {
            delete lockedPreds[target.label];
          }
          anyChange = true;
        }
        // else: status is active/waiting — leave as-is, now not stale
      }
      continue;
    }

    // ── Normal active prediction — check status ────────────────────────────
    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (['hit', 'miss', 'early'].includes(status.status)) {
      const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
      const absLow  = existing.anchorRound + existing.low;
      const absHigh = existing.anchorRound + existing.high;
      const record  = {
        target:      target.label,
        minMult:     target.min,
        outcome,
        lo:          absLow,
        hi:          absHigh,
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
        if (Number.isFinite(record.lo) && Number.isFinite(record.hi) && record.lo <= record.hi) {
          try {
            await savePrediction(record);
            console.log(`[engine] ${target.label} ${outcome.toUpperCase()} #${record.lo}–#${record.hi}${record.hitRound ? ` @#${record.hitRound}` : ''} regime=${regime.regime}`);
          } catch(e) { console.error(`[engine] save fail ${target.label}:`, e.message); }
        } else {
          console.warn(`[engine] skipped save ${target.label} — invalid lo/hi: ${record.lo}/${record.hi}`);
        }
      }
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
      if (pred) {
        lockedPreds[target.label] = {
          ...pred,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  existing.generation + 1,
        };
        console.log(`[engine] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${regime.regime}`);
      } else {
        delete lockedPreds[target.label];
        console.warn(`[engine] ${target.label} cleared after ${outcome} — buildPrediction null, will retry`);
      }
      anyChange = true;
    }
    // status === 'active' or 'waiting' → do nothing, window still open
  }

  buildPrediction._statsCache = null;

  if (anyChange) {
    const toSave = {};
    for (const [label, pred] of Object.entries(lockedPreds)) {
      if (pred.stale) continue;
      if (!Number.isFinite(pred.anchorRound) || !Number.isFinite(pred.low) || !Number.isFinite(pred.high)) continue;
      toSave[label] = {
        lo:            pred.anchorRound + pred.low,
        hi:            pred.anchorRound + pred.high,
        roundWhenMade: pred.anchorRound,
        generation:    pred.generation,
        eta:           { low: pred.low, high: pred.high, conf: pred.confidence },
      };
    }
    if (Object.keys(toSave).length > 0) {
      try { await saveLockedPreds(toSave); }
      catch(e) { console.error('[engine] saveLockedPreds fail:', e.message); }
    }
  }
}

// ============================================================================
// PROCESS PATTERN
// FIX 1+2: same stale anchor fix as processEngine
// ============================================================================

async function processPattern(sortedRounds, lastRoundId, regime) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPatterns[target.label];
    const patPred  = buildPatternPrediction(sortedRounds, target.min);
    const win      = buildPatternWindow(patPred, target.maxWidth);

    // ── No prediction — create one ─────────────────────────────────────────
    if (!existing) {
      if (win) {
        lockedPatterns[target.label] = {
          ...win,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  1,
        };
        anyChange = true;
      }
      continue;
    }

    // ── Stale pred — validate old window, build fresh ──────────────────────
    if (existing.stale) {
      const absLow  = existing.anchorRound + existing.low;
      const absHigh = existing.anchorRound + existing.high;
      const windowInPast = lastRoundId > absHigh;

      if (windowInPast) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit', 'miss', 'early'].includes(status.status)) {
          const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
          const record  = {
            target:      target.label,
            minMult:     target.min,
            outcome,
            lo:          absLow,
            hi:          absHigh,
            anchorRound: existing.anchorRound,
            hitRound:    status.hitRound || null,
            generation:  existing.generation,
            source:      'pattern',
            ts:          Date.now(),
          };
          const key = patHistKey(record);
          if (!patSavedKeys.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow <= absHigh) {
            patSavedKeys.add(key);
            try {
              await savePrediction(record);
              console.log(`[pattern] STALE-RESOLVED ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}`);
            } catch(e) { console.error(`[pattern] stale save fail:`, e.message); }
          }
        }
        // FIX 2: anchor fresh prediction to NOW
        if (win) {
          lockedPatterns[target.label] = {
            ...win,
            targetMin:   target.min,
            anchorRound: lastRoundId,
            generation:  existing.generation + 1,
          };
          console.log(`[pattern] STALE→FRESH ${target.label}: +${win.low}–+${win.high}`);
        } else {
          delete lockedPatterns[target.label];
        }
        anyChange = true;
      } else {
        // Window still valid — clear stale flag, check status
        existing.stale = false;
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit', 'miss', 'early'].includes(status.status)) {
          const absLow2  = existing.anchorRound + existing.low;
          const absHigh2 = existing.anchorRound + existing.high;
          const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
          const record  = {
            target: target.label, minMult: target.min, outcome,
            lo: absLow2, hi: absHigh2, anchorRound: existing.anchorRound,
            hitRound: status.hitRound || null, generation: existing.generation,
            source: 'pattern', ts: Date.now(),
          };
          const key = patHistKey(record);
          if (!patSavedKeys.has(key) && Number.isFinite(absLow2) && Number.isFinite(absHigh2)) {
            patSavedKeys.add(key);
            try { await savePrediction(record); } catch(e) {}
          }
          if (win) {
            lockedPatterns[target.label] = { ...win, targetMin: target.min, anchorRound: lastRoundId, generation: existing.generation + 1 };
          } else {
            delete lockedPatterns[target.label];
          }
          anyChange = true;
        }
      }
      continue;
    }

    // ── Normal active prediction ───────────────────────────────────────────
    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (['hit', 'miss', 'early'].includes(status.status)) {
      const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
      const absLow  = existing.anchorRound + existing.low;
      const absHigh = existing.anchorRound + existing.high;
      const record  = {
        target:      target.label,
        minMult:     target.min,
        outcome,
        lo:          absLow,
        hi:          absHigh,
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
        if (Number.isFinite(record.lo) && Number.isFinite(record.hi) && record.lo <= record.hi) {
          try {
            await savePrediction(record);
            console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${record.lo}–#${record.hi} regime=${regime.regime}`);
          } catch(e) { console.error(`[pattern] save fail ${target.label}:`, e.message); }
        } else {
          console.warn(`[pattern] skipped save ${target.label} — invalid lo/hi: ${record.lo}/${record.hi}`);
        }
      }
      if (win) {
        lockedPatterns[target.label] = {
          ...win,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  existing.generation + 1,
        };
        console.log(`[pattern] NEXT ${target.label}: +${win.low}–+${win.high} conf=${win.confidence}%`);
      } else {
        delete lockedPatterns[target.label];
        console.warn(`[pattern] ${target.label} cleared after ${outcome} — buildPatternPrediction null, will retry`);
      }
      anyChange = true;
    }
  }

  if (anyChange) {
    const toSavePat = {};
    for (const [label, pred] of Object.entries(lockedPatterns)) {
      if (pred.stale) continue;
      if (!Number.isFinite(pred.anchorRound) || !Number.isFinite(pred.low) || !Number.isFinite(pred.high)) continue;
      toSavePat[label] = {
        lo:            pred.anchorRound + pred.low,
        hi:            pred.anchorRound + pred.high,
        roundWhenMade: pred.anchorRound,
        generation:    pred.generation,
        eta:           { low: pred.low, high: pred.high, conf: pred.confidence },
      };
    }
    if (Object.keys(toSavePat).length > 0) {
      try { await saveLockedPatternPreds(toSavePat); }
      catch(e) { console.error('[pattern] saveLockedPatternPreds fail:', e.message); }
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function runPredictionEngine() {
  try {
    await initialise();

    const rounds = await getRounds({ limit: 5000 });
    if (rounds.length < MIN_ROUNDS) return;

    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    if (rounds.length === lastRoundCount) return;
    lastRoundCount = rounds.length;

    const regime = detectRegime(rounds);

    if (regime.regime !== 'normal' || regime.currentStreak > 5) {
      console.log(`[engine] REGIME=${regime.regime.toUpperCase()} streak=${regime.currentStreak}r(${Math.round(regime.streakPct*100)}pct) penalty=${regime.streakPenalty}r cold=${regime.coldScore} logRatio=${regime.logRatio}`);
    }

    await processEngine(rounds, lastRoundId, regime);
    await processPattern(rounds, lastRoundId, regime);

  } catch(e) {
    console.error('[predictionEngine] Fatal:', e.message, e.stack);
  }
}

module.exports = { runPredictionEngine };