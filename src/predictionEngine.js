'use strict';
// predictionEngine.js  v5
//
// RESOLVING FIX:
//   Root cause of infinite RESOLVING: the stale/expired branch did:
//     if (stale || past) { check status; if resolved → save + rebuild; else → continue; }
//   The "else continue" meant: if getStatus returned 'waiting' or 'active'
//   (which can happen when anchorRound is NaN or the rounds array doesn't
//   have data going back far enough), the loop just skipped the target
//   forever. The window stayed in RESOLVING permanently.
//
//   Fix: the stale/expired branch ALWAYS builds a fresh window after
//   attempting to resolve. We never skip without building a replacement.
//
// ENS/GEO/BAY/KM AS BACKEND ENGINES:
//   All 4 stat models now run server-side with the same lock/resolve/history
//   pattern as ENGINE and PATTERN. Each has:
//     - Its own in-memory lockedPreds map
//     - Its own savedKeys Set
//     - Its own source= identifier in the predictions table
//     - Its own /locked-<model> GET endpoint (served from api.js)
//   The frontend just fetches and displays — no frontend locking logic needed.
//
// UNIQUE HISTORY:
//   Every source (engine, pattern, ens, geo, bay, km) writes to the same
//   predictions table but with its own source= value. GET /predictions?source=X
//   returns only that source's rows. No cross-contamination.

const {
  getRounds, savePrediction, getPredictions,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedStatPreds, getLockedStatPreds,
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

// Stat models: window width offsets so each model has a distinct window
const STAT_MODELS = [
  { id: 'ens', wOffset: 1 },
  { id: 'geo', wOffset: 0 },
  { id: 'bay', wOffset: 2 },
  { id: 'km',  wOffset: 4 },
];

const MIN_ROUNDS = 100;

// ── In-memory state ───────────────────────────────────────────────────────────
let lockedPreds    = null;   // engine
let lockedPatterns = null;   // pattern
let lockedStats    = null;   // { ens:{}, geo:{}, bay:{}, km:{} }

let savedKeys      = new Set();
let patSavedKeys   = new Set();
let statSavedKeys  = {};     // { ens: Set, geo: Set, bay: Set, km: Set }

let lastRoundCount = 0;
let initialised    = false;

// ── Reset (call after any DB clear) ──────────────────────────────────────────
function resetEngineState() {
  console.log('[engine] resetEngineState() — clearing all in-memory state');
  lockedPreds    = null;
  lockedPatterns = null;
  lockedStats    = null;
  savedKeys      = new Set();
  patSavedKeys   = new Set();
  statSavedKeys  = {};
  lastRoundCount = 0;
  initialised    = false;
}

// ── Math helpers ──────────────────────────────────────────────────────────────

function bayesLambda(hits, n) { return (hits + 1) / (n + 2); }

function scanRounds(rounds, targetMin) {
  const n = rounds.length, start500 = Math.max(0, n - 500);
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
  const lambdaRecent = bayesLambda(recent500, Math.min(500, n));
  const lambda = Math.max(1e-6, Math.min(0.5, 0.6 * lambdaGlobal + 0.4 * lambdaRecent));
  let gSum = 0; for (const g of gaps) gSum += g;
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / lambda;
  let gVs = 0; for (const g of gaps) gVs += (g - meanGap) ** 2;
  const cv = meanGap > 0 ? Math.sqrt(gaps.length > 1 ? gVs / gaps.length : meanGap * meanGap) / meanGap : 1;
  const sg = [...gaps].sort((a, b) => a - b);
  const mid = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? meanGap : sg.length % 2 === 1 ? sg[mid] : (sg[mid-1]+sg[mid])/2;
  const p90 = sg[Math.floor(sg.length * 0.90)] ?? sg[sg.length-1] ?? meanGap;
  return { hits, n, lambda, lambdaGlobal, meanGap, medianGap, p90, cv, gapSinceLast };
}

function computeGlobalStats(rounds) {
  const n = rounds.length, r500 = Math.min(500,n), r200 = Math.min(200,n);
  let gLogS=0, gLogSS=0, rLogS=0, rLogSS=0, dLogS=0;
  for (let i = 0; i < n; i++) {
    const lv = Math.log(Math.max(1.01, rounds[i].multiplier));
    gLogS += lv; gLogSS += lv*lv;
    if (i >= n-r500) { rLogS += lv; rLogSS += lv*lv; }
    if (i >= n-r200) dLogS += lv;
  }
  const gVar = n > 0 ? gLogSS/n - (gLogS/n)**2 : 0;
  const rVar = r500 > 0 ? rLogSS/r500 - (rLogS/r500)**2 : 0;
  let regimeAdj = 0;
  if (n >= 100 && gVar > 0) { const ratio = rVar/gVar; regimeAdj = ratio > 1.4 ? 4 : ratio < 0.6 ? -4 : 0; }
  const dv = r200 > 0 ? (dLogS/r200) - (n > 0 ? gLogS/n : 0) : 0;
  const driftAdj = n >= 100 ? (dv > 0.20 ? 3 : dv < -0.20 ? -3 : 0) : 0;
  const slice200 = rounds.slice(n - Math.min(200,n));
  const sub2r = slice200.length > 0 ? slice200.filter(r => r.multiplier < 2).length / slice200.length : 0;
  return { regimeAdj, driftAdj, coldTailAdj: sub2r > 0.60 ? -10 : sub2r > 0.50 ? -5 : 0 };
}

function detectRegime(rounds) {
  const n = rounds.length;
  if (n < 50) return { regime:'normal', streakPenalty:0, streakPct:0, coldScore:0, currentStreak:0, hotScore:0 };
  const THRESH = 5;
  const streaks = []; let cur = 0;
  for (let i = 0; i < n; i++) {
    if (rounds[i].multiplier < THRESH) { cur++; }
    else { if (cur > 0) streaks.push(cur); cur = 0; }
  }
  const currentStreak = cur;
  let streakPct = 0;
  if (streaks.length >= 5) { streaks.sort((a,b)=>a-b); streakPct = streaks.filter(s=>s<=currentStreak).length/streaks.length; }
  const streakPenalty = streakPct >= 0.95 ? Math.round(currentStreak*1.0) : streakPct >= 0.85 ? Math.round(currentStreak*0.6) : streakPct >= 0.70 ? Math.round(currentStreak*0.3) : 0;
  const W = Math.min(100, n), recent = rounds.slice(n-W);
  let rLogSum=0, gLogSum=0, lowCount=0, highCount=0;
  for (const r of recent) { rLogSum+=Math.log(Math.max(1.01,r.multiplier)); if(r.multiplier<2)lowCount++; if(r.multiplier>=20)highCount++; }
  for (const r of rounds) gLogSum += Math.log(Math.max(1.01,r.multiplier));
  const logRatio = (gLogSum/n) > 0 ? (rLogSum/W)/(gLogSum/n) : 1;
  let coldScore=0, hotScore=0;
  if (logRatio<0.78) coldScore+=3; else if(logRatio<0.90) coldScore+=1;
  if (streakPct>=0.85) coldScore+=3; else if(streakPct>=0.70) coldScore+=1;
  if (currentStreak>12) coldScore+=2;
  if (lowCount/W>0.65) coldScore+=2;
  if (logRatio>1.22) hotScore+=3; else if(logRatio>1.12) hotScore+=1;
  if (highCount/W>0.30) hotScore+=2;
  let rVarSum=0; const rMean=rLogSum/W;
  for (const r of recent) rVarSum+=(Math.log(Math.max(1.01,r.multiplier))-rMean)**2;
  let gVarSum=0; const gMean=gLogSum/n;
  for (const r of rounds) gVarSum+=(Math.log(Math.max(1.01,r.multiplier))-gMean)**2;
  const isVolatile = (gVarSum/n)>0 && (rVarSum/W)/(gVarSum/n)>1.6;
  const regime = coldScore>=4?'cold':hotScore>=3?'hot':isVolatile?'volatile':'normal';
  return { regime, coldScore, hotScore, streakPct:+streakPct.toFixed(3), streakPenalty, currentStreak, logRatio:+logRatio.toFixed(3) };
}

// ── Prediction builders ───────────────────────────────────────────────────────

let _statsCache = null;

function buildPrediction(sortedRounds, targetMin, maxWidth, regime, gs) {
  if (!gs) gs = _statsCache ?? computeGlobalStats(sortedRounds);
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, lambda, lambdaGlobal, medianGap, cv, gapSinceLast, p90 } = s;
  if (!regime) regime = detectRegime(sortedRounds);
  let regimeShift = 0;
  if (regime.regime==='cold') regimeShift = Math.round(regime.streakPenalty + medianGap*0.25*Math.min(1,regime.coldScore/8));
  else if (regime.regime==='hot') regimeShift = -Math.round(medianGap*0.10);
  const center = Math.max(0, (medianGap - gapSinceLast) + regimeShift);
  const low    = center <= 0 ? 0 : Math.max(0, Math.round(center - maxWidth/2));
  const high   = low + maxWidth - 1;
  let c = Math.min(55, 18 + Math.log2(hits+1)*8);
  c -= Math.min(12, Math.abs(cv-1)*6);
  if (lambdaGlobal > 0) { const d=Math.abs(lambda-lambdaGlobal)/lambdaGlobal; c += d<0.15?7:d<0.40?3:-6; }
  c += gs.regimeAdj + gs.driftAdj + gs.coldTailAdj;
  if (regime.regime==='cold')     c -= Math.min(15, regime.coldScore*2);
  if (regime.regime==='volatile') c -= 8;
  if (regime.streakPct>=0.85)     c -= 10;
  if (regime.streakPct>=0.95)     c -= 8;
  if (gapSinceLast > p90*1.5)    c -= 8;
  return { low, high, confidence: Math.max(20, Math.min(88, Math.round(c))), regime: regime.regime, gapSinceLast, medianGap, hits };
}

// ENS/GEO/BAY/KM each use geometric probability but with different window widths
// and slightly different hit-rate estimators, giving genuinely different windows.
function buildStatPrediction(sortedRounds, targetMin, maxWidth, modelId, gs) {
  if (!gs) gs = _statsCache ?? computeGlobalStats(sortedRounds);
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, n, lambda, lambdaGlobal, meanGap, medianGap, cv, gapSinceLast, p90 } = s;

  // Each model uses a slightly different probability estimate
  let p, label;
  if (modelId === 'geo') {
    // Pure geometric MLE with Laplace smoothing
    p = (hits + 1) / (n + 2);
    label = 'geo';
  } else if (modelId === 'bay') {
    // Bayesian Beta-Binomial: heavier recency weight
    const r200 = sortedRounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
    const pGlobal = (hits + 1) / (n + 2);
    const pRecent = (r200 + 1) / (202);
    p = 0.80 * pGlobal + 0.20 * pRecent;
    label = 'bay';
  } else if (modelId === 'km') {
    // KM-style: use median gap instead of mean for window center
    p = lambda; // blended lambda
    label = 'km';
  } else {
    // ENS: ensemble blend
    const r200 = sortedRounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
    const pGlobal = (hits + 1) / (n + 2);
    const pRecent = (r200 + 1) / (202);
    p = 0.70 * pGlobal + 0.25 * (0.80*pGlobal + 0.20*pRecent) + 0.05 * lambda;
    label = 'ens';
  }

  p = Math.max(1e-6, Math.min(0.5, p));
  const probW = 1 - Math.pow(1 - p, maxWidth);
  const expectedGap = (1 - p) / p;

  // Window: center on expected remaining gap
  const remaining = Math.max(0, Math.round(expectedGap - gapSinceLast));
  const low  = Math.max(0, Math.round(remaining - maxWidth/2));
  const high = low + maxWidth - 1;

  // Confidence from sample size + stability + cv
  let c = Math.min(55, 18 + Math.log2(hits+1)*8);
  c -= Math.min(12, Math.abs(cv-1)*6);
  if (lambdaGlobal > 0) { const d=Math.abs(lambda-lambdaGlobal)/lambdaGlobal; c += d<0.15?7:d<0.40?3:-6; }
  c += gs.regimeAdj + gs.driftAdj + gs.coldTailAdj;
  if (gapSinceLast > p90*1.5) c -= 8;

  return {
    low, high,
    confidence: Math.max(20, Math.min(88, Math.round(c))),
    probW: +probW.toFixed(4),
    p: +p.toFixed(6),
    expectedGap: +expectedGap.toFixed(1),
    gapSinceLast,
    hits,
    model: label,
  };
}

function buildPatternPrediction(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;
  const W1=15,W2=50,W3=150, s1=Math.max(0,n-W1), s2=Math.max(0,n-W2), s3=Math.max(0,n-W3);
  let hits=0, lastIdx=-1, hW1=0, hW2=0, hW3=0;
  const gaps=[]; const FA=0.20, SA=0.02; let emaFast=-1, emaSlow=-1;
  for (let i=0;i<n;i++) {
    const isHit=sortedRounds[i].multiplier>=targetMin?1:0;
    if(emaFast<0){emaFast=isHit;emaSlow=isHit;}else{emaFast=FA*isHit+(1-FA)*emaFast;emaSlow=SA*isHit+(1-SA)*emaSlow;}
    if(isHit){if(lastIdx!==-1)gaps.push(i-lastIdx-1);lastIdx=i;hits++;if(i>=s1)hW1++;if(i>=s2)hW2++;if(i>=s3)hW3++;}
  }
  if (hits<8||gaps.length<6) return null;
  const gapSinceLast=lastIdx===-1?n:n-lastIdx-1, globalRate=hits/n;
  let gSum=0; for(const g of gaps)gSum+=g;
  const meanGap=gSum/gaps.length;
  const sg=[...gaps].sort((a,b)=>a-b); const mid2=Math.floor(sg.length/2);
  const medianGap=sg.length%2===1?sg[mid2]:(sg[mid2-1]+sg[mid2])/2;
  let gVs=0; for(const g of gaps)gVs+=(g-meanGap)**2;
  const cv=meanGap>0?Math.sqrt(gaps.length>1?gVs/gaps.length:meanGap*meanGap)/meanGap:1;
  const dW1=hW1/W1,dW2=hW2/W2,dW3=hW3/Math.min(W3,n);
  const rW1=globalRate>0?Math.max(-1,Math.min(1,(globalRate-dW1)/Math.max(globalRate,0.001))):0;
  const rW2=globalRate>0?Math.max(-1,Math.min(1,(globalRate-dW2)/Math.max(globalRate,0.001))):0;
  const rW3=globalRate>0?Math.max(-1,Math.min(1,(globalRate-dW3)/Math.max(globalRate,0.001))):0;
  const clusterScore=Math.max(-1,Math.min(1,rW1*0.50+rW2*0.30+rW3*0.20));
  const trendScore=Math.max(-1,Math.min(1,(emaSlow>0?(emaSlow-emaFast)/emaSlow:0)*4));
  let varSum=0; for(const g of gaps)varSum+=(g-meanGap)**2;
  let bestAC=0,bestLag=1;
  for(let lag=1;lag<=Math.min(3,gaps.length-1);lag++){
    let cov=0; for(let i=lag;i<gaps.length;i++)cov+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac=varSum>0?cov/varSum:0; if(Math.abs(ac)>Math.abs(bestAC)){bestAC=ac;bestLag=lag;}
  }
  const aboveMedian=gapSinceLast>medianGap?1:-1;
  const patternScore=Math.max(-1,Math.min(1,-bestAC*aboveMedian*0.9));
  const composite=clusterScore*0.45+trendScore*0.35+patternScore*0.20;
  const direction=composite>0.12?'bullish':composite<-0.12?'bearish':'neutral';
  const agree=Math.max([clusterScore,trendScore,patternScore].filter(s=>s>0.10).length,[clusterScore,trendScore,patternScore].filter(s=>s<-0.10).length);
  const conf=Math.max(28,Math.min(90,Math.round(38+Math.min(20,Math.log2(hits+1)*5)+Math.abs(composite)*28+(agree-1)*8+(cv>1.5?-8:cv>1.2?-4:0))));
  return { direction,confidence:conf,hits,meanGap:Math.round(meanGap),medianGap:Math.round(medianGap),gapSinceLast,composite:+composite.toFixed(3),clusterScore:+clusterScore.toFixed(3),trendScore:+trendScore.toFixed(3),patternScore:+patternScore.toFixed(3) };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const { medianGap, clusterScore, trendScore, patternScore, confidence } = patternResult;
  const scale = Math.max(medianGap, maxWidth*1.5);
  const rawCenter = (-clusterScore*scale*1.4)*0.50 + (trendScore*scale*0.9)*0.30 + (-patternScore*scale*0.7)*0.20;
  const center = Math.max(0, Math.round(rawCenter));
  const low    = Math.max(0, Math.round(center - maxWidth/2));
  return { low, high: low + maxWidth - 1, confidence };
}

// ── Key helpers ───────────────────────────────────────────────────────────────

function makeKey(source, r) {
  const lo = Number(r.lo)||0, hi = Number(r.hi)||0;
  return `${source}-${r.target}-${lo}-${hi}-${r.outcome}-${r.hitRound??'x'}`;
}

// ── getStatus ─────────────────────────────────────────────────────────────────
// Returns: 'hit' | 'early' | 'miss' | 'active' | 'waiting'
// NEVER returns undefined. Always returns a string.

function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);

  // Safety: if window is nonsense, treat as miss
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow) {
    return { status: 'miss', hitRound: null };
  }

  // Binary search for first round >= anchorRound
  let lo=0, hi=sortedRounds.length-1, startIdx=sortedRounds.length;
  while (lo<=hi) {
    const mid=(lo+hi)>>>1;
    if (sortedRounds[mid].roundId >= anchorRound) { startIdx=mid; hi=mid-1; }
    else lo=mid+1;
  }

  for (let i=startIdx; i<sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < absLow) return { status:'early', hitRound:r.roundId };
    return { status:'hit', hitRound:r.roundId };
  }

  if (currentRoundId > absHigh)                              return { status:'miss',    hitRound:null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status:'active',  hitRound:null };
  return { status:'waiting', hitRound:null };
}

// ── Generic window processor ──────────────────────────────────────────────────
// Handles lock/resolve/rebuild for ANY engine uniformly.
// RESOLVING FIX: always builds a new window after a stale/expired window,
// regardless of whether getStatus returned a terminal state.

async function processWindows({
  lockedMap,      // { [targetLabel]: pred }
  savedSet,       // Set of saved keys
  source,         // 'engine' | 'pattern' | 'ens' | 'geo' | 'bay' | 'km'
  sortedRounds,
  lastRoundId,
  buildFn,        // (target) => { low, high, confidence, ... } | null
  onSave,         // optional async callback after saving
}) {
  let anyChange = false;
  const toSaveDB = {};

  for (const target of TARGETS) {
    const existing = lockedMap[target.label];

    // ── No window exists — create one ─────────────────────────────────────
    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:1 };
        anyChange = true;
      }
      continue;
    }

    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow  = anchorRound + (Number(existing.low)  || 0);
    const absHigh = anchorRound + (Number(existing.high) || 0);
    const windowExpired = lastRoundId > absHigh;
    const windowNonsense = !Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow || anchorRound === 0;

    // ── RESOLVING FIX: stale OR expired OR nonsense → resolve + always rebuild ──
    if (existing.stale || windowExpired || windowNonsense) {
      // Attempt to record outcome of old window (don't fail if we can't)
      if (!windowNonsense && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow <= absHigh) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
          const record  = {
            target:target.label, minMult:target.min, outcome,
            lo:absLow, hi:absHigh, anchorRound,
            hitRound:status.hitRound||null, generation:existing.generation||1,
            source, ts:Date.now(),
          };
          const key = makeKey(source, record);
          if (!savedSet.has(key)) {
            savedSet.add(key);
            try {
              await savePrediction(record);
              console.log(`[${source}] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
              if (onSave) await onSave(record);
            } catch(e) { console.error(`[${source}] save fail ${target.label}:`, e.message); }
          }
        }
        // Note: if status is 'active' or 'waiting' on an "expired" window,
        // that means getStatus thinks it's still open — could be data lag.
        // We still rebuild to avoid permanent RESOLVING state.
      }

      // ALWAYS build a fresh window — this is the RESOLVING fix
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = {
          ...pred,
          targetMin:  target.min,
          anchorRound: lastRoundId,
          generation: (existing.generation||1) + (windowNonsense ? 0 : 1),
          stale:      false,
        };
        console.log(`[${source}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        // buildFn returned null (not enough data) — clear and retry next poll
        delete lockedMap[target.label];
        console.warn(`[${source}] ${target.label} cleared — buildFn null, will retry`);
      }
      anyChange = true;
      continue;
    }

    // ── Normal active window — check if resolved ───────────────────────────
    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
      const record  = {
        target:target.label, minMult:target.min, outcome,
        lo:absLow, hi:absHigh, anchorRound,
        hitRound:status.hitRound||null, generation:existing.generation||1,
        source, ts:Date.now(),
      };
      const key = makeKey(source, record);
      if (!savedSet.has(key)) {
        savedSet.add(key);
        try {
          await savePrediction(record);
          console.log(`[${source}] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
          if (onSave) await onSave(record);
        } catch(e) { console.error(`[${source}] save fail ${target.label}:`, e.message); }
      }
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:(existing.generation||1)+1 };
        console.log(`[${source}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete lockedMap[target.label];
        console.warn(`[${source}] ${target.label} cleared after ${outcome} — buildFn null`);
      }
      anyChange = true;
    }
    // status is 'active' or 'waiting' — window still open, nothing to do
  }

  return { anyChange, lockedMap };
}

// ── DB persistence helpers ────────────────────────────────────────────────────

function buildSavePayload(lockedMap) {
  const toSave = {};
  for (const [label, pred] of Object.entries(lockedMap)) {
    if (pred.stale) continue;
    const anchor = Number(pred.anchorRound);
    if (!Number.isFinite(anchor) || anchor === 0) continue;
    toSave[label] = {
      lo:            anchor + (Number(pred.low)||0),
      hi:            anchor + (Number(pred.high)||0),
      roundWhenMade: anchor,
      generation:    pred.generation||1,
      eta:           { low:pred.low, high:pred.high, conf:pred.confidence },
    };
  }
  return toSave;
}

function loadLockedMap(dbRows, targets) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = targets.find(t => t.label === label);
    if (!target) continue;
    const eta = pred.eta || {};
    const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
    map[label] = {
      low:         eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo)  - anchor),
      high:        eta.high != null ? eta.high : Math.max(0, Number(pred.hi)  - anchor),
      confidence:  eta.conf ?? 50,
      targetMin:   target.min,
      anchorRound: anchor,
      generation:  pred.generation ?? 1,
      stale:       true,
    };
  }
  return map;
}

// ── Initialise ────────────────────────────────────────────────────────────────

async function initialise() {
  if (initialised) return;
  initialised = true;

  // Engine locked preds
  try {
    lockedPreds = loadLockedMap(await getLockedPreds(), TARGETS);
    console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} engine locked preds`);
  } catch(e) { console.error('[engine] init error:', e.message); lockedPreds = {}; }

  // Pattern locked preds
  try {
    lockedPatterns = loadLockedMap(await getLockedPatternPreds(), TARGETS);
    console.log(`[engine] Loaded ${Object.keys(lockedPatterns).length} pattern locked preds`);
  } catch(e) { console.error('[engine] pattern init error:', e.message); lockedPatterns = {}; }

  // Stat model locked preds (ens/geo/bay/km)
  lockedStats = { ens:{}, geo:{}, bay:{}, km:{} };
  statSavedKeys = { ens:new Set(), geo:new Set(), bay:new Set(), km:new Set() };
  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      const rows = dbStats[model.id] || {};
      lockedStats[model.id] = loadLockedMap(rows, TARGETS);
      console.log(`[engine] Loaded ${Object.keys(lockedStats[model.id]).length} ${model.id} locked preds`);
    }
  } catch(e) { console.error('[engine] stat locked preds init error:', e.message); }

  // Load saved prediction keys (engine + pattern)
  try {
    const rows = await getPredictions({ limit: 3000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r);
      if (src === 'engine') { savedKeys.add(key); }
      else if (src === 'pattern') { patSavedKeys.add(key); }
      else if (statSavedKeys[src]) { statSavedKeys[src].add(key); }
    }
    console.log(`[engine] Keys: engine=${savedKeys.size} pattern=${patSavedKeys.size}`);
    for (const m of STAT_MODELS) console.log(`[engine] Keys: ${m.id}=${statSavedKeys[m.id].size}`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  lastRoundCount = 0;
}

// ── Run ───────────────────────────────────────────────────────────────────────

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
    _statsCache = computeGlobalStats(rounds);

    if (regime.regime !== 'normal' || regime.currentStreak > 5)
      console.log(`[engine] REGIME=${regime.regime.toUpperCase()} streak=${regime.currentStreak}r(${Math.round(regime.streakPct*100)}pct) penalty=${regime.streakPenalty}r cold=${regime.coldScore}`);

    // ── ENGINE ──
    { const { anyChange } = await processWindows({
        lockedMap: lockedPreds, savedSet: savedKeys, source: 'engine',
        sortedRounds: rounds, lastRoundId,
        buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth, regime, _statsCache),
      });
      if (anyChange) {
        const p = buildSavePayload(lockedPreds);
        if (Object.keys(p).length) { try { await saveLockedPreds(p); } catch(e) { console.error('[engine] saveLockedPreds:', e.message); } }
      }
    }

    // ── PATTERN ──
    { const { anyChange } = await processWindows({
        lockedMap: lockedPatterns, savedSet: patSavedKeys, source: 'pattern',
        sortedRounds: rounds, lastRoundId,
        buildFn: (t) => {
          const pp = buildPatternPrediction(rounds, t.min);
          return buildPatternWindow(pp, t.maxWidth);
        },
      });
      if (anyChange) {
        const p = buildSavePayload(lockedPatterns);
        if (Object.keys(p).length) { try { await saveLockedPatternPreds(p); } catch(e) { console.error('[pattern] saveLockedPatternPreds:', e.message); } }
      }
    }

    // ── ENS / GEO / BAY / KM ──
    for (const model of STAT_MODELS) {
      const { anyChange } = await processWindows({
        lockedMap: lockedStats[model.id],
        savedSet:  statSavedKeys[model.id],
        source:    model.id,
        sortedRounds: rounds,
        lastRoundId,
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth + model.wOffset, model.id, _statsCache),
      });
      if (anyChange) {
        const p = buildSavePayload(lockedStats[model.id]);
        if (Object.keys(p).length) {
          try { await saveLockedStatPreds(model.id, p); }
          catch(e) { console.error(`[${model.id}] saveLockedStatPreds:`, e.message); }
        }
      }
    }

    _statsCache = null;

  } catch(e) { console.error('[predictionEngine] Fatal:', e.message, e.stack); _statsCache = null; }
}

// ── Expose locked maps for API reads ─────────────────────────────────────────
function getLockedStatMap(modelId) {
  if (!lockedStats) return {};
  return lockedStats[modelId] || {};
}

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap };