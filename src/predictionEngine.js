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

// ============================================================================
// REGIME DETECTOR  — runs once per tick, gates everything else
// ============================================================================

function detectRegime(rounds) {
  const n = rounds.length;
  if (n < 50) return { regime: 'normal', streakPenalty: 0, streakPct: 0, coldScore: 0, currentStreak: 0 };

  // ── 1. White streak analysis ──────────────────────────────────────────
  const STREAK_THRESH = 5;
  const streaks = [];
  let cur = 0;
  for (let i = 0; i < n; i++) {
    if (rounds[i].multiplier < STREAK_THRESH) {
      cur++;
    } else {
      if (cur > 0) streaks.push(cur);
      cur = 0;
    }
  }
  const currentStreak = cur;

  let streakPct = 0, streakMedian = 0;
  if (streaks.length >= 3) {
    streaks.sort((a, b) => a - b);
    const below  = streaks.filter(s => s <= currentStreak).length;
    streakPct    = below / streaks.length;
    const mid    = Math.floor(streaks.length / 2);
    streakMedian = streaks.length % 2 === 1 ? streaks[mid] : (streaks[mid-1]+streaks[mid])/2;
  }

  // Streak penalty: extra rounds added to every window
  let streakPenalty = 0;
  if      (streakPct >= 0.95) streakPenalty = Math.round(currentStreak * 1.2);
  else if (streakPct >= 0.85) streakPenalty = Math.round(currentStreak * 0.8);
  else if (streakPct >= 0.70) streakPenalty = Math.round(currentStreak * 0.4);
  else if (streakPct >= 0.60) streakPenalty = Math.round(currentStreak * 0.15);

  // ── 2. Recent multiplier distribution (last 100 rounds) ──────────────
  const W = Math.min(100, n);
  const recent = rounds.slice(n - W);
  let lowCount = 0, highCount = 0, recentLogSum = 0;
  for (let i = 0; i < recent.length; i++) {
    const m = recent[i].multiplier;
    recentLogSum += Math.log(Math.max(1.01, m));
    if (m < 2) lowCount++;
    if (m >= 20) highCount++;
  }
  const recentMeanLog = recentLogSum / W;

  let globalLogSum = 0;
  for (let i = 0; i < n; i++) globalLogSum += Math.log(Math.max(1.01, rounds[i].multiplier));
  const globalMeanLog = globalLogSum / n;
  const logRatio      = globalMeanLog > 0 ? recentMeanLog / globalMeanLog : 1;

  // ── 3. Cold score ─────────────────────────────────────────────────────
  let coldScore = 0;
  if (logRatio < 0.75)      coldScore += 3;
  else if (logRatio < 0.88) coldScore += 1;
  if (streakPct >= 0.85)    coldScore += 3;
  else if (streakPct >= 0.70) coldScore += 1;
  if (currentStreak > 10)   coldScore += 2;
  if (lowCount / W > 0.60)  coldScore += 2;

  // ── 4. Hot score ──────────────────────────────────────────────────────
  let hotScore = 0;
  if (logRatio > 1.20)         hotScore += 3;
  else if (logRatio > 1.10)    hotScore += 1;
  if (highCount / W > 0.30)    hotScore += 2;

  // ── 5. Variance (volatile) ────────────────────────────────────────────
  let rVarSum = 0;
  for (let i = 0; i < recent.length; i++)
    rVarSum += (Math.log(Math.max(1.01, recent[i].multiplier)) - recentMeanLog) ** 2;
  const recentVar = rVarSum / W;

  let gVarSum = 0;
  for (let i = 0; i < n; i++)
    gVarSum += (Math.log(Math.max(1.01, rounds[i].multiplier)) - globalMeanLog) ** 2;
  const globalVar  = gVarSum / n;
  const varRatio   = globalVar > 0 ? recentVar / globalVar : 1;
  const isVolatile = varRatio > 1.5;

  // ── 6. Classify ───────────────────────────────────────────────────────
  let regime;
  if      (coldScore >= 4)  regime = 'cold';
  else if (hotScore  >= 3)  regime = 'hot';
  else if (isVolatile)      regime = 'volatile';
  else                      regime = 'normal';

  return {
    regime,
    coldScore,
    hotScore,
    streakPct:      +streakPct.toFixed(3),
    streakPenalty,
    currentStreak,
    streakMedian,
    logRatio:       +logRatio.toFixed(3),
    varRatio:       +varRatio.toFixed(3),
    isVolatile,
  };
}

// ============================================================================
// STAT ENGINE  v9  — regime-aware
// ============================================================================

function bayesLambda(hits, n) { return (hits + 1) / (n + 2); }

function blendedLambda(rounds, targetMin, lambdaGlobal, recentHits) {
  const n       = rounds.length;
  const recentN = Math.min(500, n);
  if (recentHits === undefined) {
    recentHits = 0;
    for (let i = n - recentN; i < n; i++)
      if (rounds[i].multiplier >= targetMin) recentHits++;
  }
  const lambdaRecent = bayesLambda(recentHits, recentN);
  const raw          = 0.7 * lambdaGlobal + 0.3 * lambdaRecent;
  return Math.max(1e-6, Math.min(0.5, raw));
}

function scanRounds(rounds, targetMin) {
  const n          = rounds.length;
  const usePartial = targetMin <= 50;
  const start500   = Math.max(0, n - 500);
  let hits = 0, lastIdx = -1, recent500 = 0, pwSum = 0;
  const gaps = [];

  for (let i = 0; i < n; i++) {
    const m = rounds[i].multiplier;
    if (m >= targetMin) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++; pwSum += 1.0;
      if (i >= start500) recent500++;
    } else if (usePartial) {
      if (m >= targetMin * 0.8) pwSum += 0.5;
      else if (m >= targetMin * 0.6) pwSum += 0.25;
    }
  }

  if (hits < 2) return null;

  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const lambdaGlobal = bayesLambda(hits, n);
  const lambda       = blendedLambda(rounds, targetMin, lambdaGlobal, recent500);

  let gSum = 0;
  for (let i = 0; i < gaps.length; i++) gSum += gaps[i];
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / lambda;

  let gVs = 0;
  for (let i = 0; i < gaps.length; i++) gVs += (gaps[i] - meanGap) ** 2;
  const stdGap = gaps.length > 1 ? Math.sqrt(gVs / gaps.length) : meanGap;
  const cv     = meanGap > 0 ? stdGap / meanGap : 1;

  const sg    = [...gaps].sort((a, b) => a - b);
  const mid   = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? meanGap
    : sg.length % 2 === 1 ? sg[mid]
    : (sg[mid-1] + sg[mid]) / 2;

  return { hits, pwSum, n, lambda, lambdaGlobal, meanGap, medianGap, cv, gapSinceLast, gaps };
}

function computeGlobalStats(rounds) {
  const n = rounds.length;
  const r500 = Math.min(500, n), r200 = Math.min(200, n);
  const start500 = n - r500, start200 = n - r200;
  const thresholds = TARGETS.map(t => t.min * 3).sort((a, b) => b - a);

  let gLogS=0, gLogSS=0, rLogS=0, rLogSS=0, dLogS=0;
  const gBins=[0,0,0,0,0], rBins=[0,0,0,0,0];
  const tailCounts = new Map();
  thresholds.forEach(t => tailCounts.set(t, 0));

  for (let i = 0; i < n; i++) {
    const m  = rounds[i].multiplier;
    const lv = Math.log(Math.max(1.01, m));
    gLogS += lv; gLogSS += lv*lv;
    if (i >= start500) { rLogS += lv; rLogSS += lv*lv; }
    if (i >= start200) dLogS += lv;
    const b = m<2?0 : m<5?1 : m<20?2 : m<100?3 : 4;
    gBins[b]++;
    if (i >= start500) rBins[b]++;
    for (let t = 0; t < thresholds.length; t++) {
      if (m < thresholds[t]) break;
      tailCounts.set(thresholds[t], tailCounts.get(thresholds[t]) + 1);
    }
  }

  const gVar = n>0 ? gLogSS/n - (gLogS/n)**2 : 0;
  const rVar = r500>0 ? rLogSS/r500 - (rLogS/r500)**2 : 0;
  let regimeAdj = 0;
  if (n >= 50 && gVar > 0) {
    const ratio = rVar/gVar;
    if (ratio > 1.3) regimeAdj = 5; else if (ratio < 0.7) regimeAdj = -5;
  }

  function H(bins, total) {
    let h = 0;
    for (let j = 0; j < 5; j++) {
      if (!bins[j]) continue;
      const p = bins[j]/total;
      h -= p * Math.log2(p);
    }
    return h;
  }
  const Hg = H(gBins, n);
  const Hr = H(rBins, r500);
  const entropyAdj = (Hg > 0 && n >= 50 && (Hg - Hr) > 0.4) ? -5 : 0;

  const mlg = n > 0 ? gLogS/n : 0;
  const mlr = r200 > 0 ? dLogS/r200 : mlg;
  const dv  = mlr - mlg;
  const driftAdj = n >= 50 ? (dv > 0.15 ? 3 : dv < -0.15 ? -3 : 0) : 0;

  // Cold tail: fraction of recent 200 rounds that are sub-2x
  const sub2 = rounds.slice(n - Math.min(200,n)).filter(r => r.multiplier < 2).length;
  const sub2r = sub2 / Math.min(200, n);
  const coldTailAdj = sub2r > 0.55 ? -8 : sub2r > 0.45 ? -4 : 0;

  return { regimeAdj, entropyAdj, driftAdj, coldTailAdj, tailCounts };
}

const _mcCache = new Map();
function monteCarloAdj(lambda, maxWidth) {
  const key = `${lambda.toFixed(5)}-${maxWidth}`;
  if (_mcCache.has(key)) return _mcCache.get(key);
  if (_mcCache.size > 300) _mcCache.clear();
  const SIMS = 5000;
  let state  = 0x4f3a9c1b;
  function rng() { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state/0x100000000; }
  let hits = 0;
  for (let s = 0; s < SIMS; s++)
    for (let r = 0; r < maxWidth; r++)
      if (rng() < lambda) { hits++; break; }
  const expected = Math.min(1, 1 - Math.pow(1 - lambda, maxWidth));
  const adj = Math.abs(hits/SIMS - expected) > 0.15 ? -Math.min(4, Math.round(Math.abs(hits/SIMS - expected)*20)) : 0;
  _mcCache.set(key, adj);
  return adj;
}

let _statsCache = null;

function buildPrediction(sortedRounds, targetMin, maxWidth, regime) {
  const gs = _statsCache ?? computeGlobalStats(sortedRounds);
  const s  = scanRounds(sortedRounds, targetMin);
  if (!s) return null;

  const { hits, pwSum, n, lambda, lambdaGlobal, meanGap, medianGap, cv, gapSinceLast, gaps } = s;

  // Use median (robust) not mean as center baseline
  const baseCenter = medianGap - gapSinceLast;

  // Regime window shift
  let regimeShift = 0;
  if (regime.regime === 'cold') {
    regimeShift = Math.round(regime.streakPenalty + medianGap * 0.3 * Math.min(1, regime.coldScore / 8));
  } else if (regime.regime === 'hot') {
    regimeShift = Math.round(-medianGap * 0.15);
  }

  const center = Math.max(0, baseCenter + regimeShift);
  const low    = center <= 0 ? 0 : Math.max(0, Math.round(center - maxWidth / 2));
  const high   = low + maxWidth - 1;

  // Confidence
  let c = Math.min(65, 20 + Math.log2(hits + 1) * 9);
  c += Math.min(3, (pwSum / Math.max(1, hits)) * 2);
  c -= Math.min(10, Math.abs(cv - 1) * 5);
  if (lambdaGlobal > 0) {
    const d = Math.abs(lambda - lambdaGlobal) / lambdaGlobal;
    if (d < 0.2) c += 8; else if (d < 0.5) c += 4; else c -= 5;
  }
  c += gs.regimeAdj + gs.entropyAdj + gs.driftAdj + gs.coldTailAdj;
  c += monteCarloAdj(lambda, maxWidth);

  if (gaps.length >= 3) {
    let vsum = 0, gmean = 0;
    for (let i = 0; i < gaps.length; i++) gmean += gaps[i];
    gmean /= gaps.length;
    for (let i = 0; i < gaps.length; i++) vsum += (gaps[i]-gmean)**2;
    if (Math.sqrt(vsum/gaps.length) > gmean * 1.8) c -= 4;
  }

  // Regime confidence penalties
  if (regime.regime === 'cold')     c -= Math.min(15, regime.coldScore * 2);
  if (regime.regime === 'volatile') c -= 8;
  if (regime.streakPct >= 0.85)     c -= 10;
  if (regime.streakPct >= 0.95)     c -= 10;

  const conf = Math.max(20, Math.min(92, Math.round(c)));

  return {
    low, high, confidence: conf,
    lambda, lambdaGlobal, gapSinceLast, hits, n,
    regime:        regime.regime,
    streakPenalty: regime.streakPenalty,
    suppressed:    regime.streakPct >= 0.85,
  };
}

// ============================================================================
// PATTERN ENGINE  v4  — regime-aware
// ============================================================================

function buildPatternPrediction(sortedRounds, targetMin, regime) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;

  const W1=20, W2=60, W3=200;
  const s1=Math.max(0,n-W1), s2=Math.max(0,n-W2), s3=Math.max(0,n-W3);
  let hits=0, lastIdx=-1, hW1=0, hW2=0, hW3=0;
  const gaps=[];
  const FA=0.15, SA=0.03;
  let emaFast=-1, emaSlow=-1;

  for (let i = 0; i < n; i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast < 0) { emaFast = isHit; emaSlow = isHit; }
    else { emaFast = FA*isHit + (1-FA)*emaFast; emaSlow = SA*isHit + (1-SA)*emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++;
      if (i >= s1) hW1++; if (i >= s2) hW2++; if (i >= s3) hW3++;
    }
  }

  if (hits < 5 || gaps.length < 4) return null;

  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const globalRate   = hits / n;

  let gSum = 0;
  for (let i = 0; i < gaps.length; i++) gSum += gaps[i];
  const meanGap = gSum / gaps.length;

  const sg2   = [...gaps].sort((a,b) => a-b);
  const mid2  = Math.floor(sg2.length / 2);
  const medianGap = sg2.length%2===1 ? sg2[mid2] : (sg2[mid2-1]+sg2[mid2])/2;

  let gVs = 0;
  for (let i = 0; i < gaps.length; i++) gVs += (gaps[i]-meanGap)**2;
  const stdGap = gaps.length > 1 ? Math.sqrt(gVs/gaps.length) : meanGap;
  const cv     = meanGap > 0 ? stdGap/meanGap : 1;

  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const rW1=globalRate>0?(globalRate-dW1)/globalRate:0;
  const rW2=globalRate>0?(globalRate-dW2)/globalRate:0;
  const rW3=globalRate>0?(globalRate-dW3)/globalRate:0;
  const clusterScore = Math.max(-1, Math.min(1, rW1*0.55 + rW2*0.30 + rW3*0.15));

  const trendRaw   = emaSlow>0 ? (emaSlow-emaFast)/emaSlow : 0;
  const trendScore = Math.max(-1, Math.min(1, trendRaw*3));

  let acSum=0, varSum=0;
  for (let i=0; i<gaps.length; i++) varSum += (gaps[i]-meanGap)**2;
  for (let i=1; i<gaps.length; i++) acSum += (gaps[i-1]-meanGap)*(gaps[i]-meanGap);
  const autoCorr    = varSum>0 ? acSum/varSum : 0;
  const aboveMedian = gapSinceLast > medianGap ? 1 : -1;
  const patternScore= Math.max(-1, Math.min(1, -autoCorr*aboveMedian*0.8));

  const composite = clusterScore*0.50 + trendScore*0.30 + patternScore*0.20;
  const scores    = [clusterScore, trendScore, patternScore];
  const agreement = Math.max(scores.filter(s=>s>0.08).length, scores.filter(s=>s<-0.08).length);

  let confidence = Math.max(30, Math.min(92,
    Math.round(40 + Math.min(18, Math.log2(hits+1)*4.5) + Math.abs(composite)*25 + (agreement-1)*7)
  ));

  if (regime.regime === 'cold')     confidence = Math.max(20, confidence - Math.min(15, regime.coldScore*2));
  if (regime.streakPct >= 0.85)     confidence = Math.max(20, confidence - 12);
  if (regime.regime === 'volatile') confidence = Math.max(20, confidence - 8);

  return {
    confidence, hits,
    meanGap:      Math.round(meanGap),
    medianGap:    Math.round(medianGap),
    gapSinceLast,
    clusterScore: +clusterScore.toFixed(3),
    trendScore:   +trendScore.toFixed(3),
    patternScore: +patternScore.toFixed(3),
    composite:    +composite.toFixed(3),
    regime:       regime.regime,
    suppressed:   regime.streakPct >= 0.85,
  };
}

function buildPatternWindow(pr, maxWidth, regime) {
  if (!pr) return null;
  const { medianGap, clusterScore, trendScore, patternScore, confidence } = pr;

  const scale        = Math.max(medianGap, maxWidth * 1.5);
  const clusterShift = -clusterScore * scale * 1.5;
  const trendShift   =  trendScore   * scale * 1.0;
  const patternShift = -patternScore * scale * 0.8;
  let rawCenter      = clusterShift*0.50 + trendShift*0.30 + patternShift*0.20;

  if (regime.streakPenalty > 0) rawCenter += regime.streakPenalty;
  if (regime.regime === 'cold') rawCenter += Math.round(medianGap * 0.4 * Math.min(1, regime.coldScore/8));

  const center = Math.max(0, Math.round(rawCenter));
  const low    = Math.max(0, Math.round(center - maxWidth/2));
  const high   = low + maxWidth - 1;

  return { low, high, confidence };
}

// ============================================================================
// getStatus
// ============================================================================

function getStatus(sortedRounds, pred, currentRoundId) {
  const ws = pred.anchorRound + pred.low;
  const we = pred.anchorRound + pred.high;

  let lo=0, hi=sortedRounds.length-1, startIdx=sortedRounds.length;
  while (lo <= hi) {
    const mid = (lo+hi)>>>1;
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