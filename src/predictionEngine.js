'use strict';
// predictionEngine.js  v7
//
// ROOT CAUSE OF PERSISTENT RESOLVING:
//
//   The engine skipped processing when rounds.length === lastRoundCount.
//   But a window can expire (lastRoundId > absHigh) even when no new rounds
//   arrive — because the window is narrow (e.g. 3 rounds wide for 5x) and
//   lastRoundId already passed absHigh on the previous run.
//   The "no new rounds" guard prevented the expired window from ever being
//   rebuilt. It stayed in the DB with the old lo/hi forever → RESOLVING.
//
//   FIX: Always run processWindows. Remove the "rounds unchanged" early exit.
//   Use a separate `lastProcessedRoundId` to avoid redundant DB saves:
//   only persist to DB when lastRoundId has actually advanced.
//
// OTHER FIXES:
//   - MIN_ROUNDS = 50 (was 100, prevented engine from running on first boot)
//   - isFirstRun forces processing on startup to clear stale DB windows
//   - STALE_FORCE_REBUILD_THRESHOLD: skip scanning windows > 500r old

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

const STAT_MODELS = [
  { id: 'ens', wOffset: 1 },
  { id: 'geo', wOffset: 0 },
  { id: 'bay', wOffset: 2 },
  { id: 'km',  wOffset: 4 },
];

const MIN_ROUNDS = 50;
const STALE_FORCE_REBUILD_THRESHOLD = 500;

let lockedPreds    = null;
let lockedPatterns = null;
let lockedStats    = null;
let savedKeys      = new Set();
let patSavedKeys   = new Set();
let statSavedKeys  = {};
let lastProcessedRoundId = 0; // track last roundId we actually processed
let initialised    = false;

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetEngineState() {
  console.log('[engine] resetEngineState() — clearing all in-memory state');
  lockedPreds    = null;
  lockedPatterns = null;
  lockedStats    = null;
  savedKeys      = new Set();
  patSavedKeys   = new Set();
  statSavedKeys  = {};
  lastProcessedRoundId = 0;
  initialised    = false;
}

// ── Math ──────────────────────────────────────────────────────────────────────

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
  if (currentStreak>12) coldScore+=2; if (lowCount/W>0.65) coldScore+=2;
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

function buildStatPrediction(sortedRounds, targetMin, maxWidth, modelId, gs) {
  if (!gs) gs = _statsCache ?? computeGlobalStats(sortedRounds);
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, n, lambda, lambdaGlobal, medianGap, cv, gapSinceLast, p90 } = s;
  let p;
  if (modelId === 'geo') {
    p = (hits + 1) / (n + 2);
  } else if (modelId === 'bay') {
    const r200 = sortedRounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
    p = 0.80 * ((hits+1)/(n+2)) + 0.20 * ((r200+1)/202);
  } else if (modelId === 'km') {
    p = lambda;
  } else {
    const r200 = sortedRounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
    const pG = (hits+1)/(n+2), pR = (r200+1)/202;
    p = 0.70*pG + 0.25*(0.80*pG+0.20*pR) + 0.05*lambda;
  }
  p = Math.max(1e-6, Math.min(0.5, p));
  const probW = 1 - Math.pow(1 - p, maxWidth);
  const expectedGap = (1 - p) / p;
  const remaining = Math.max(0, Math.round(expectedGap - gapSinceLast));
  const low  = Math.max(0, Math.round(remaining - maxWidth/2));
  const high = low + maxWidth - 1;
  let c = Math.min(55, 18 + Math.log2(hits+1)*8);
  c -= Math.min(12, Math.abs(cv-1)*6);
  if (lambdaGlobal > 0) { const d=Math.abs(lambda-lambdaGlobal)/lambdaGlobal; c += d<0.15?7:d<0.40?3:-6; }
  c += gs.regimeAdj + gs.driftAdj + gs.coldTailAdj;
  if (gapSinceLast > p90*1.5) c -= 8;
  return { low, high, confidence: Math.max(20, Math.min(88, Math.round(c))), probW: +probW.toFixed(4), p: +p.toFixed(6), expectedGap: +expectedGap.toFixed(1), gapSinceLast, hits, model: modelId };
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

function makeKey(source, r) {
  return `${source}-${r.target}-${Number(r.lo)||0}-${Number(r.hi)||0}-${r.outcome}-${r.hitRound??'x'}`;
}

// ── getStatus ─────────────────────────────────────────────────────────────────

function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow) {
    return { status: 'miss', hitRound: null };
  }
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
  if (currentRoundId > absHigh)                              return { status:'miss',   hitRound:null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status:'active', hitRound:null };
  return { status:'waiting', hitRound:null };
}

// ── processWindows ────────────────────────────────────────────────────────────
// RUNS ON EVERY POLL — no early exit for "no new rounds"

async function processWindows({ lockedMap, savedSet, source, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedMap[target.label];

    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:1, stale:false };
        anyChange = true;
        console.log(`[${source}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      }
      continue;
    }

    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow || anchorRound === 0;
    const isExpired   = lastRoundId > absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      // Try to record outcome unless too old or nonsense
      if (!isNonsense && !isTooOld) {
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
            } catch(e) { console.error(`[${source}] save fail:`, e.message); }
          }
        }
      }

      // ALWAYS rebuild — no matter what getStatus returned
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = {
          ...pred,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  (existing.generation||1) + (isNonsense ? 0 : 1),
          stale:       false,
        };
        console.log(`[${source}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete lockedMap[target.label];
        console.warn(`[${source}] ${target.label} cleared — buildFn null, will retry next poll`);
      }
      anyChange = true;
      continue;
    }

    // Window is active or waiting — check if it just resolved
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
        } catch(e) { console.error(`[${source}] save fail:`, e.message); }
      }
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:(existing.generation||1)+1, stale:false };
        console.log(`[${source}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete lockedMap[target.label];
      }
      anyChange = true;
    }
    // 'active' or 'waiting' — nothing to do
  }

  return anyChange;
}

// ── DB helpers ────────────────────────────────────────────────────────────────

function buildSavePayload(lockedMap) {
  const out = {};
  for (const [label, pred] of Object.entries(lockedMap)) {
    if (pred.stale) continue;
    const anchor = Number(pred.anchorRound);
    if (!Number.isFinite(anchor) || anchor === 0) continue;
    out[label] = {
      lo:            anchor + (Number(pred.low)||0),
      hi:            anchor + (Number(pred.high)||0),
      roundWhenMade: anchor,
      generation:    pred.generation||1,
      eta:           { low: pred.low, high: pred.high, conf: pred.confidence },
    };
  }
  return out;
}

function loadLockedMap(dbRows) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = TARGETS.find(t => t.label === label);
    if (!target) continue;
    const eta    = pred.eta || {};
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

  try { lockedPreds = loadLockedMap(await getLockedPreds()); console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} engine locked preds`); }
  catch(e) { console.error('[engine] init error:', e.message); lockedPreds = {}; }

  try { lockedPatterns = loadLockedMap(await getLockedPatternPreds()); console.log(`[engine] Loaded ${Object.keys(lockedPatterns).length} pattern locked preds`); }
  catch(e) { console.error('[engine] pattern init error:', e.message); lockedPatterns = {}; }

  lockedStats   = { ens:{}, geo:{}, bay:{}, km:{} };
  statSavedKeys = { ens:new Set(), geo:new Set(), bay:new Set(), km:new Set() };
  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      lockedStats[model.id] = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] Loaded ${Object.keys(lockedStats[model.id]).length} ${model.id} locked preds`);
    }
  } catch(e) { console.error('[engine] stat locked preds init error:', e.message); }

  try {
    const rows = await getPredictions({ limit: 3000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r);
      if (src === 'engine')        savedKeys.add(key);
      else if (src === 'pattern')  patSavedKeys.add(key);
      else if (statSavedKeys[src]) statSavedKeys[src].add(key);
    }
  } catch(e) { console.error('[engine] history load error:', e.message); }

  lastProcessedRoundId = 0;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function runPredictionEngine() {
  try {
    await initialise();

    const rounds = await getRounds({ limit: 5000 });
    if (rounds.length < MIN_ROUNDS) {
      console.log(`[engine] waiting for rounds (${rounds.length}/${MIN_ROUNDS})`);
      return;
    }
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Check if ANY window is expired — if so, we must process even with same round count
    const hasExpiredWindows = () => {
      const allMaps = [lockedPreds, lockedPatterns, ...STAT_MODELS.map(m => lockedStats[m.id])].filter(Boolean);
      for (const map of allMaps) {
        for (const pred of Object.values(map)) {
          const anchor  = Number(pred.anchorRound) || 0;
          const absHigh = anchor + (Number(pred.high) || 0);
          if (pred.stale || lastRoundId > absHigh || anchor === 0) return true;
        }
      }
      return false;
    };

    // Skip only if round ID unchanged AND no expired windows
    if (lastRoundId === lastProcessedRoundId && !hasExpiredWindows()) return;
    lastProcessedRoundId = lastRoundId;

    const regime    = detectRegime(rounds);
    _statsCache     = computeGlobalStats(rounds);

    if (regime.regime !== 'normal' || regime.currentStreak > 5)
      console.log(`[engine] REGIME=${regime.regime.toUpperCase()} streak=${regime.currentStreak}r(${Math.round(regime.streakPct*100)}pct) penalty=${regime.streakPenalty}r`);

    // ENGINE
    const engChanged = await processWindows({
      lockedMap: lockedPreds, savedSet: savedKeys, source: 'engine',
      sortedRounds: rounds, lastRoundId,
      buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth, regime, _statsCache),
    });
    if (engChanged) {
      const p = buildSavePayload(lockedPreds);
      if (Object.keys(p).length) { try { await saveLockedPreds(p); } catch(e) { console.error('[engine] saveLockedPreds:', e.message); } }
    }

    // PATTERN
    const patChanged = await processWindows({
      lockedMap: lockedPatterns, savedSet: patSavedKeys, source: 'pattern',
      sortedRounds: rounds, lastRoundId,
      buildFn: (t) => { const pp = buildPatternPrediction(rounds, t.min); return buildPatternWindow(pp, t.maxWidth); },
    });
    if (patChanged) {
      const p = buildSavePayload(lockedPatterns);
      if (Object.keys(p).length) { try { await saveLockedPatternPreds(p); } catch(e) { console.error('[pattern] saveLockedPatternPreds:', e.message); } }
    }

    // ENS / GEO / BAY / KM
    for (const model of STAT_MODELS) {
      const changed = await processWindows({
        lockedMap: lockedStats[model.id], savedSet: statSavedKeys[model.id], source: model.id,
        sortedRounds: rounds, lastRoundId,
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth + model.wOffset, model.id, _statsCache),
      });
      if (changed) {
        const p = buildSavePayload(lockedStats[model.id]);
        if (Object.keys(p).length) { try { await saveLockedStatPreds(model.id, p); } catch(e) { console.error(`[${model.id}] saveLockedStatPreds:`, e.message); } }
      }
    }

    _statsCache = null;

  } catch(e) { console.error('[predictionEngine] Fatal:', e.message, e.stack); _statsCache = null; }
}

function getLockedStatMap(modelId) {
  if (!lockedStats) return {};
  return lockedStats[modelId] || {};
}

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap };