'use strict';
// predictionEngine.js  v2
// ============================================================================
// Complete rebuild. Core philosophy change:
//
//   OLD: predict WHEN the next hit comes (timing only)
//   NEW: first classify WHAT REGIME we are in, then predict timing WITHIN
//        that regime. If the regime says "white streak / cold" → suppress ALL
//        bullish predictions and delay windows hard.
//
// REGIME SYSTEM (runs first, gates everything else):
//   COLD     — market suppressed, sub-5x dominating, all windows delayed
//   NORMAL   — baseline, standard timing
//   HOT      — elevated multipliers, tighten windows
//   VOLATILE — high variance, widen windows, reduce confidence
//
// WHITE STREAK SUPPRESSION:
//   When current white streak is above its historical 60th percentile →
//   add streakPenalty rounds to every window.
//   Above 85th → also reduce confidence by up to 25pts.
//   Above 95th → max suppression, confidence floor drops to 20.
//
// CROSS-TARGET COHERENCE:
//   burst5/burst10 counts how many targets fired recently.
//   Used to detect post-burst cooldowns that suppress lower targets.
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
  // Heavier weight on recent — 500-round window is responsive enough
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

  // Median — robust timing baseline
  const sg  = [...gaps].sort((a, b) => a - b);
  const mid = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? meanGap
    : sg.length % 2 === 1 ? sg[mid]
    : (sg[mid - 1] + sg[mid]) / 2;

  // P10 / P90 of gap distribution — for honest window sizing
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

  // Regime variance shift
  const gVar = n > 0 ? gLogSS / n - (gLogS / n) ** 2 : 0;
  const rVar = r500 > 0 ? rLogSS / r500 - (rLogS / r500) ** 2 : 0;
  let regimeAdj = 0;
  if (n >= 100 && gVar > 0) {
    const ratio = rVar / gVar;
    if (ratio > 1.4) regimeAdj = 4;
    else if (ratio < 0.6) regimeAdj = -4;
  }

  // Mean-log drift
  const mlg = n > 0 ? gLogS / n : 0;
  const mlr = r200 > 0 ? dLogS / r200 : mlg;
  const dv  = mlr - mlg;
  const driftAdj = n >= 100 ? (dv > 0.20 ? 3 : dv < -0.20 ? -3 : 0) : 0;

  // Cold tail: sub-2x fraction of recent 200 rounds
  const slice200 = rounds.slice(n - Math.min(200, n));
  const sub2r    = slice200.filter(r => r.multiplier < 2).length / slice200.length;
  const coldTailAdj = sub2r > 0.60 ? -10 : sub2r > 0.50 ? -5 : 0;

  return { regimeAdj, driftAdj, coldTailAdj };
}

function detectRegime(rounds) {
  const n = rounds.length;
  if (n < 50) return { regime:'normal', streakPenalty:0, streakPct:0, coldScore:0, currentStreak:0, hotScore:0 };

  // White streak
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

  // Streak penalty — only kicks in above 70th pct
  let streakPenalty = 0;
  if      (streakPct >= 0.95) streakPenalty = Math.round(currentStreak * 1.0);
  else if (streakPct >= 0.85) streakPenalty = Math.round(currentStreak * 0.6);
  else if (streakPct >= 0.70) streakPenalty = Math.round(currentStreak * 0.3);

  // Recent 100r multiplier ratio vs all-time
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

  // Cold / hot scores
  let coldScore = 0, hotScore = 0;
  if (logRatio < 0.78)        coldScore += 3; else if (logRatio < 0.90) coldScore += 1;
  if (streakPct >= 0.85)      coldScore += 3; else if (streakPct >= 0.70) coldScore += 1;
  if (currentStreak > 12)     coldScore += 2;
  if (lowCount / W > 0.65)    coldScore += 2;
  if (logRatio > 1.22)        hotScore  += 3; else if (logRatio > 1.12) hotScore  += 1;
  if (highCount / W > 0.30)   hotScore  += 2;

  // Variance
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

  // ── Window center: median-gap timing + regime shift ───────────────────
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

  // ── Confidence — honest, no inflation ─────────────────────────────────
  // Base: sample size contribution
  let c = Math.min(55, 18 + Math.log2(hits + 1) * 8);

  // Stability: low cv = regular gaps = more predictable
  c -= Math.min(12, Math.abs(cv - 1) * 6);

  // Lambda stability
  if (lambdaGlobal > 0) {
    const d = Math.abs(lambda - lambdaGlobal) / lambdaGlobal;
    if (d < 0.15) c += 7; else if (d < 0.40) c += 3; else c -= 6;
  }

  // Global stats adjustments
  c += gs.regimeAdj + gs.driftAdj + gs.coldTailAdj;

  // Regime confidence penalty
  if (regime.regime === 'cold')     c -= Math.min(15, regime.coldScore * 2);
  if (regime.regime === 'volatile') c -= 8;
  if (regime.streakPct >= 0.85)     c -= 10;
  if (regime.streakPct >= 0.95)     c -= 8;

  // Gap position penalty: if gapSinceLast is much larger than p90, prediction is unreliable
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

function getStatus(sortedRounds, pred, currentRoundId) {
  const ws = pred.anchorRound + pred.low;
  const we = pred.anchorRound + pred.high;

  let lo=0, hi=sortedRounds.length-1, startIdx=sortedRounds.length;
  while (lo <= hi) {
    const mid = (lo+hi) >>> 1;
    if (sortedRounds[mid].roundId >= pred.anchorRound) { startIdx=mid; hi=mid-1; }
    else lo=mid+1;
  }
  for (let i = startIdx; i < sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > we) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < ws) return { status:'early', hitRound:r.roundId };
    return { status:'hit', hitRound:r.roundId };
  }
  if (currentRoundId > we)                          return { status:'miss'   };
  if (currentRoundId >= ws && currentRoundId <= we) return { status:'active' };
  return { status:'waiting' };
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

  // ── CLUSTER ───────────────────────────────────────────────────────────
  // Ratio-normalized: (expected - observed) / expected
  // This makes +1 = complete drought, -1 = 2× the expected rate
  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const rW1 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW1)/Math.max(globalRate,0.001))) : 0;
  const rW2 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW2)/Math.max(globalRate,0.001))) : 0;
  const rW3 = globalRate>0 ? Math.max(-1, Math.min(1, (globalRate-dW3)/Math.max(globalRate,0.001))) : 0;
  // Short window weighted most for immediacy, long window for context
  const clusterScore = Math.max(-1, Math.min(1, rW1*0.50 + rW2*0.30 + rW3*0.20));

  // ── TREND ─────────────────────────────────────────────────────────────
  const trendRaw   = emaSlow>0 ? (emaSlow-emaFast)/emaSlow : 0;
  const trendScore = Math.max(-1, Math.min(1, trendRaw*4));

  // ── SEQUENCE PATTERN — multi-lag autocorr ────────────────────────────
  let varSum=0;
  for (const g of gaps) varSum+=(g-meanGap)**2;

  // Check lags 1, 2, 3 — take dominant (strongest absolute)
  let bestAC=0, bestLag=1;
  for (let lag=1; lag<=Math.min(3, gaps.length-1); lag++) {
    let covSum=0;
    for (let i=lag; i<gaps.length; i++) covSum+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac = varSum>0 ? covSum/varSum : 0;
    if (Math.abs(ac) > Math.abs(bestAC)) { bestAC=ac; bestLag=lag; }
  }

  // Direction: is current gap above or below median?
  const aboveMedian = gapSinceLast > medianGap ? 1 : -1;
  // Negative autocorr + above median → alternating → hit soon → positive score
  // Positive autocorr + above median → clustering  → drought extends → negative
  const patternScore = Math.max(-1, Math.min(1, -bestAC * aboveMedian * 0.9));

  // ── Composite direction ───────────────────────────────────────────────
  const composite = clusterScore*0.45 + trendScore*0.35 + patternScore*0.20;
  const direction = composite > 0.12 ? 'bullish' : composite < -0.12 ? 'bearish' : 'neutral';

  // ── Confidence — signal agreement + sample size + signal strength ─────
  const scores   = [clusterScore, trendScore, patternScore];
  const nBull    = scores.filter(s=>s>0.10).length;
  const nBear    = scores.filter(s=>s<-0.10).length;
  const agree    = Math.max(nBull, nBear); // 0,1,2,3
  const sampleB  = Math.min(20, Math.log2(hits+1) * 5);
  const strengthB= Math.abs(composite) * 28;
  const agreeB   = (agree-1) * 8; // -8, 0, +8, +16
  const cvPenalty= cv > 1.5 ? -8 : cv > 1.2 ? -4 : 0; // irregular gaps = unreliable
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
  const { medianGap, clusterScore, trendScore, patternScore, confidence } = patternResult;

  // Scale votes to medianGap, floored at maxWidth*1.5 so small medians still move
  const scale = Math.max(medianGap, maxWidth * 1.5);

  // Each signal votes independently on "rounds from NOW":
  // positive = event is later, negative = event is sooner
  const cShift = -clusterScore * scale * 1.4; // drought→sooner, burst→later
  const tShift =  trendScore   * scale * 0.9; // cooling→later, heating→sooner
  const pShift = -patternScore * scale * 0.7; // pattern says soon→sooner

  const rawCenter = cShift*0.50 + tShift*0.30 + pShift*0.20;
  const center    = Math.max(0, Math.round(rawCenter));
  const low       = Math.max(0, Math.round(center - maxWidth/2));
  const high      = low + maxWidth - 1;

  return { low, high, confidence };
}


// ============================================================================
// KEY HELPERS
// ============================================================================

function histKey(r) {
  const lo=r.lo??0, hi=r.hi??0;
  return `${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound??'x'}`;
}
function patHistKey(r) {
  const lo=r.lo??0, hi=r.hi??0;
  return `pat-${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound??'x'}`;
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
        stale:       true,
      };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} locked preds from DB`);
  } catch(e) {
    console.error('[engine] init error:', e.message);
    lockedPreds    = {};
    lockedPatterns = {};
  }
  try {
    const rows = await getPredictions({ limit: 500 });
    for (const r of rows) {
      if (!r.source || r.source === 'engine') {
        savedKeys.add(histKey(r));
        savedKeys.add(`${r.target}-${r.anchorRound??r.lo}-${r.outcome}-${r.hitRound??'x'}`);
      } else if (r.source === 'pattern') {
        patSavedKeys.add(patHistKey(r));
        patSavedKeys.add(`pat-${r.target}-${r.anchorRound??r.lo}-${r.outcome}-${r.hitRound??'x'}`);
      }
    }
    console.log(`[engine] Loaded ${savedKeys.size} engine keys, ${patSavedKeys.size} pattern keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  // Load pattern locked preds from DB
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
  }

  // Always reset round count after init so first poll always processes
  lastRoundCount = 0;
}

// ============================================================================
// PROCESS ENGINE
// ============================================================================

async function processEngine(sortedRounds, lastRoundId, regime) {
  _statsCache = computeGlobalStats(sortedRounds);
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPreds[target.label];

    if (!existing) {
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
      if (pred) {
        lockedPreds[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:1 };
        anyChange = true;
        console.log(`[engine] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${regime.regime}`);
      }
      continue;
    }

    if (existing.stale) {
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
      if (pred) {
        lockedPreds[target.label] = { ...pred, targetMin:target.min, anchorRound:existing.anchorRound, generation:existing.generation };
        anyChange = true;
      }
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
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
        savedKeys.add(`${record.target}-${record.anchorRound}-${record.outcome}-${record.hitRound??'x'}`);
        // Guard: only save valid windows (anchorRound undefined = NaN lo/hi)
        if (Number.isFinite(record.lo) && Number.isFinite(record.hi) && record.lo <= record.hi) {
          try {
            await savePrediction(record);
            console.log(`[engine] ${target.label} ${outcome.toUpperCase()} #${record.lo}–#${record.hi}${record.hitRound?` @#${record.hitRound}`:''} regime=${regime.regime}`);
          } catch(e) { console.error(`[engine] save fail ${target.label}:`, e.message); }
        } else {
          console.warn(`[engine] skipped save ${target.label} — invalid lo/hi: ${record.lo}/${record.hi} (anchorRound=${record.anchorRound})`);
        }
      }
      const pred = buildPrediction(sortedRounds, target.min, target.maxWidth, regime);
      if (pred) {
        lockedPreds[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 };
        anyChange = true;
        console.log(`[engine] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${regime.regime}`);
      }
    }
  }

  _statsCache = null;

  if (anyChange) {
    const toSave = {};
    for (const [label, pred] of Object.entries(lockedPreds)) {
      if (pred.stale) continue;
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
// ============================================================================

async function processPattern(sortedRounds, lastRoundId, regime) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPatterns[target.label];
    const patPred  = buildPatternPrediction(sortedRounds, target.min, regime);
    const win      = buildPatternWindow(patPred, target.maxWidth, regime);

    if (!existing || existing.stale) {
      if (win) {
        lockedPatterns[target.label] = { ...win, targetMin:target.min, anchorRound:lastRoundId, generation:existing?existing.generation:1 };
        anyChange = true;
      }
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);

    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
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
        patSavedKeys.add(`pat-${record.target}-${record.anchorRound}-${record.outcome}-${record.hitRound??'x'}`);
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
        lockedPatterns[target.label] = { ...win, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 };
        anyChange = true;
      }
    }
  }

  // Persist pattern locked preds to DB so they survive server restarts
  if (anyChange) {
    const toSavePat = {};
    for (const [label, pred] of Object.entries(lockedPatterns)) {
      if (pred.stale) continue;
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
    console.error('[predictionEngine] Fatal:', e.message);
  }
}

module.exports = { runPredictionEngine };