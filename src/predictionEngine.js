'use strict';
// predictionEngine.js  v4
// ============================================================================
// FIXES vs v3:
//
//   FIX A — POST-DB-CLEAR RECOVERY:
//     After DELETE on predictions/locked tables, in-memory state was stale.
//     Added resetEngineState() — call this from the API after any DELETE.
//     It nulls lockedPreds/lockedPatterns, clears savedKeys, resets
//     initialised=false so the next poll does a full re-init from (empty) DB.
//
//   FIX B — savedKeys/patSavedKeys survived DB clear:
//     These Sets were never cleared so after a DB wipe the engine would
//     silently skip re-saving outcomes it thought were already saved.
//     resetEngineState() clears both.
//
//   FIX C — lockedPreds/lockedPatterns in-memory survived DB clear:
//     Old stale windows with ancient anchorRounds stayed in memory → RESOLVING.
//     resetEngineState() nulls both maps so next poll starts clean.
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
// FIX A+B+C: Reset all in-memory state — call from API after any DB clear
// ============================================================================
function resetEngineState() {
  console.log('[engine] resetEngineState() — clearing all in-memory prediction state');
  lockedPreds    = null;
  lockedPatterns = null;
  savedKeys      = new Set();
  patSavedKeys   = new Set();
  lastRoundCount = 0;
  initialised    = false; // forces full re-init on next runPredictionEngine() call
}

// ============================================================================
// MATH HELPERS
// ============================================================================

function bayesLambda(hits, n) { return (hits + 1) / (n + 2); }

function blendedLambda(rounds, targetMin, lambdaGlobal, recentHits) {
  const n = rounds.length, recentN = Math.min(500, n);
  if (recentHits === undefined) {
    recentHits = 0;
    for (let i = n - recentN; i < n; i++)
      if (rounds[i].multiplier >= targetMin) recentHits++;
  }
  return Math.max(1e-6, Math.min(0.5, 0.6 * lambdaGlobal + 0.4 * bayesLambda(recentHits, recentN)));
}

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
  const lambda       = blendedLambda(rounds, targetMin, lambdaGlobal, recent500);
  let gSum = 0; for (const g of gaps) gSum += g;
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / lambda;
  let gVs = 0; for (const g of gaps) gVs += (g - meanGap) ** 2;
  const cv = meanGap > 0 ? Math.sqrt(gaps.length > 1 ? gVs / gaps.length : meanGap * meanGap) / meanGap : 1;
  const sg = [...gaps].sort((a, b) => a - b);
  const mid = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? meanGap : sg.length % 2 === 1 ? sg[mid] : (sg[mid-1]+sg[mid])/2;
  const p10 = sg[Math.floor(sg.length * 0.10)] ?? sg[0] ?? 0;
  const p90 = sg[Math.floor(sg.length * 0.90)] ?? sg[sg.length-1] ?? meanGap;
  return { hits, n, lambda, lambdaGlobal, meanGap, medianGap, p10, p90, cv, gapSinceLast, gaps };
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
  const sub2r = slice200.filter(r => r.multiplier < 2).length / slice200.length;
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
  if (logRatio < 0.78) coldScore+=3; else if(logRatio<0.90) coldScore+=1;
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
  return { regime, coldScore, hotScore, streakPct:+streakPct.toFixed(3), streakPenalty, currentStreak, logRatio:+logRatio.toFixed(3), isVolatile };
}

buildPrediction._statsCache = null;

function buildPrediction(sortedRounds, targetMin, maxWidth, regime) {
  const gs = buildPrediction._statsCache ?? computeGlobalStats(sortedRounds);
  const s  = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, lambda, lambdaGlobal, medianGap, cv, gapSinceLast } = s;
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
  if (gapSinceLast > s.p90*1.5)  c -= 8;
  return {
    low, high, confidence: Math.max(20, Math.min(88, Math.round(c))),
    lambda, lambdaGlobal, gapSinceLast, medianGap, hits, n: s.n,
    regime: regime.regime, streakPenalty: regime.streakPenalty, suppressed: regime.streakPct >= 0.85,
  };
}

function getStatus(sortedRounds, pred, currentRoundId) {
  const absLow = pred.anchorRound + pred.low, absHigh = pred.anchorRound + pred.high;
  let lo=0, hi=sortedRounds.length-1, startIdx=sortedRounds.length;
  while (lo<=hi) { const mid=(lo+hi)>>>1; if(sortedRounds[mid].roundId>=pred.anchorRound){startIdx=mid;hi=mid-1;}else lo=mid+1; }
  for (let i=startIdx; i<sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < absLow) return { status:'early', hitRound:r.roundId };
    return { status:'hit', hitRound:r.roundId };
  }
  if (currentRoundId > absHigh)                              return { status:'miss'    };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status:'active'  };
  return { status:'waiting' };
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
  return { direction,confidence:conf,hits,meanGap:Math.round(meanGap),medianGap:Math.round(medianGap),gapSinceLast,cv:+cv.toFixed(2),clusterScore:+clusterScore.toFixed(3),trendScore:+trendScore.toFixed(3),patternScore:+patternScore.toFixed(3),composite:+composite.toFixed(3),autoCorr:+bestAC.toFixed(3),dominantLag:bestLag };
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

function histKey(r)    { return `${r.target}-${Number(r.lo)||0}-${Number(r.hi)||0}-${r.outcome}-${r.hitRound??'x'}`; }
function patHistKey(r) { return `pat-${r.target}-${Number(r.lo)||0}-${Number(r.hi)||0}-${r.outcome}-${r.hitRound??'x'}`; }

// ============================================================================
// INITIALISE
// ============================================================================

async function initialise() {
  if (initialised) return;
  initialised = true;

  try {
    const dbPreds = await getLockedPreds();
    lockedPreds = {};
    for (const [label, pred] of Object.entries(dbPreds)) {
      const target = TARGETS.find(t => t.label === label);
      if (!target) continue;
      const eta = pred.eta || {}, anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
      lockedPreds[label] = { low: eta.low??Math.max(0,Number(pred.lo)-anchor), high: eta.high??Math.max(0,Number(pred.hi)-anchor), confidence: eta.conf??50, targetMin: target.min, anchorRound: anchor, generation: pred.generation??1, stale: true };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPreds).length} locked preds`);
  } catch(e) { console.error('[engine] init locked preds error:', e.message); lockedPreds = {}; }

  try {
    const rows = await getPredictions({ limit: 2000 });
    for (const r of rows) {
      if (!r.source || r.source === 'engine') {
        savedKeys.add(histKey(r));
        savedKeys.add(`${r.target}-${Number(r.anchorRound??r.lo)||0}-${r.outcome}-${r.hitRound??'x'}`);
      } else if (r.source === 'pattern') {
        patSavedKeys.add(patHistKey(r));
        patSavedKeys.add(`pat-${r.target}-${Number(r.anchorRound??r.lo)||0}-${r.outcome}-${r.hitRound??'x'}`);
      }
    }
    console.log(`[engine] Loaded ${savedKeys.size} engine keys, ${patSavedKeys.size} pattern keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  try {
    const dbPat = await getLockedPatternPreds();
    lockedPatterns = {};
    for (const [label, pred] of Object.entries(dbPat)) {
      const target = TARGETS.find(t => t.label === label);
      if (!target) continue;
      const eta = pred.eta || {}, anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
      lockedPatterns[label] = { low: eta.low??Math.max(0,Number(pred.lo)-anchor), high: eta.high??Math.max(0,Number(pred.hi)-anchor), confidence: eta.conf??50, targetMin: target.min, anchorRound: anchor, generation: pred.generation??1, stale: true };
    }
    console.log(`[engine] Loaded ${Object.keys(lockedPatterns).length} pattern locked preds`);
  } catch(e) { console.error('[engine] pattern locked preds error:', e.message); lockedPatterns = {}; }

  lastRoundCount = 0;
}

// ============================================================================
// SHARED RESOLVE HELPER — reduces duplication between engine + pattern
// ============================================================================

async function resolveWindow(existing, target, sortedRounds, lastRoundId, buildNext, saveKey, savedSet, source) {
  const absLow  = existing.anchorRound + existing.low;
  const absHigh = existing.anchorRound + existing.high;
  const status  = getStatus(sortedRounds, existing, lastRoundId);

  if (['hit', 'miss', 'early'].includes(status.status)) {
    const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
    const record  = { target: target.label, minMult: target.min, outcome, lo: absLow, hi: absHigh, anchorRound: existing.anchorRound, hitRound: status.hitRound||null, generation: existing.generation, source, ts: Date.now() };
    const key = saveKey(record);
    if (!savedSet.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow <= absHigh) {
      savedSet.add(key);
      try { await savePrediction(record); console.log(`[${source}] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${record.hitRound?` @#${record.hitRound}`:''}`); }
      catch(e) { console.error(`[${source}] save fail:`, e.message); }
    }
    return { resolved: true, nextPred: buildNext() };
  }
  return { resolved: false, nextPred: null };
}

// ============================================================================
// PROCESS ENGINE
// ============================================================================

async function processEngine(sortedRounds, lastRoundId, regime) {
  buildPrediction._statsCache = computeGlobalStats(sortedRounds);
  let anyChange = false;
  const buildNext = (target) => buildPrediction(sortedRounds, target.min, target.maxWidth, regime);

  for (const target of TARGETS) {
    const existing = lockedPreds[target.label];

    if (!existing) {
      const pred = buildNext(target);
      if (pred) { lockedPreds[target.label] = { ...pred, targetMin: target.min, anchorRound: lastRoundId, generation: 1 }; anyChange = true; console.log(`[engine] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`); }
      continue;
    }

    const absHigh = existing.anchorRound + existing.high;
    // Stale or past window — resolve and rebuild
    if (existing.stale || lastRoundId > absHigh) {
      if (existing.stale) existing.stale = false;
      const absLow = existing.anchorRound + existing.low;
      const status = getStatus(sortedRounds, existing, lastRoundId);
      if (['hit','miss','early'].includes(status.status)) {
        const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
        const record  = { target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound:existing.anchorRound, hitRound:status.hitRound||null, generation:existing.generation, source:'engine', ts:Date.now() };
        const key = histKey(record);
        if (!savedKeys.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow<=absHigh) {
          savedKeys.add(key);
          try { await savePrediction(record); console.log(`[engine] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}`); }
          catch(e) { console.error(`[engine] save fail:`, e.message); }
        }
        const pred = buildNext(target);
        if (pred) { lockedPreds[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 }; console.log(`[engine] NEXT ${target.label}: +${pred.low}–+${pred.high}`); }
        else delete lockedPreds[target.label];
        anyChange = true;
      }
      // If status is still active/waiting after clearing stale, leave it
      continue;
    }

    // Normal active: check status
    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
      const absLow  = existing.anchorRound + existing.low;
      const record  = { target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound:existing.anchorRound, hitRound:status.hitRound||null, generation:existing.generation, source:'engine', ts:Date.now() };
      const key = histKey(record);
      if (!savedKeys.has(key)) {
        savedKeys.add(key);
        if (Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow<=absHigh) {
          try { await savePrediction(record); console.log(`[engine] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${record.hitRound?` @#${record.hitRound}`:''}`); }
          catch(e) { console.error(`[engine] save fail:`, e.message); }
        }
      }
      const pred = buildNext(target);
      if (pred) { lockedPreds[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 }; }
      else delete lockedPreds[target.label];
      anyChange = true;
    }
  }

  buildPrediction._statsCache = null;

  if (anyChange) {
    const toSave = {};
    for (const [label, pred] of Object.entries(lockedPreds)) {
      if (pred.stale || !Number.isFinite(pred.anchorRound)) continue;
      toSave[label] = { lo: pred.anchorRound+pred.low, hi: pred.anchorRound+pred.high, roundWhenMade: pred.anchorRound, generation: pred.generation, eta: { low:pred.low, high:pred.high, conf:pred.confidence } };
    }
    if (Object.keys(toSave).length > 0) { try { await saveLockedPreds(toSave); } catch(e) { console.error('[engine] saveLockedPreds fail:', e.message); } }
  }
}

// ============================================================================
// PROCESS PATTERN
// ============================================================================

async function processPattern(sortedRounds, lastRoundId, regime) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedPatterns[target.label];
    const patPred  = buildPatternPrediction(sortedRounds, target.min);
    const win      = buildPatternWindow(patPred, target.maxWidth);

    if (!existing) {
      if (win) { lockedPatterns[target.label] = { ...win, targetMin:target.min, anchorRound:lastRoundId, generation:1 }; anyChange = true; }
      continue;
    }

    const absHigh = existing.anchorRound + existing.high;
    if (existing.stale || lastRoundId > absHigh) {
      if (existing.stale) existing.stale = false;
      const absLow = existing.anchorRound + existing.low;
      const status = getStatus(sortedRounds, existing, lastRoundId);
      if (['hit','miss','early'].includes(status.status)) {
        const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
        const record  = { target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound:existing.anchorRound, hitRound:status.hitRound||null, generation:existing.generation, source:'pattern', ts:Date.now() };
        const key = patHistKey(record);
        if (!patSavedKeys.has(key) && Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow<=absHigh) {
          patSavedKeys.add(key);
          try { await savePrediction(record); console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}`); }
          catch(e) { console.error(`[pattern] save fail:`, e.message); }
        }
        if (win) { lockedPatterns[target.label] = { ...win, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 }; console.log(`[pattern] NEXT ${target.label}: +${win.low}–+${win.high}`); }
        else delete lockedPatterns[target.label];
        anyChange = true;
      }
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
      const absLow  = existing.anchorRound + existing.low;
      const record  = { target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound:existing.anchorRound, hitRound:status.hitRound||null, generation:existing.generation, source:'pattern', ts:Date.now() };
      const key = patHistKey(record);
      if (!patSavedKeys.has(key)) {
        patSavedKeys.add(key);
        if (Number.isFinite(absLow) && Number.isFinite(absHigh) && absLow<=absHigh) {
          try { await savePrediction(record); console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${record.hitRound?` @#${record.hitRound}`:''}`); }
          catch(e) { console.error(`[pattern] save fail:`, e.message); }
        }
      }
      if (win) { lockedPatterns[target.label] = { ...win, targetMin:target.min, anchorRound:lastRoundId, generation:existing.generation+1 }; }
      else delete lockedPatterns[target.label];
      anyChange = true;
    }
  }

  if (anyChange) {
    const toSave = {};
    for (const [label, pred] of Object.entries(lockedPatterns)) {
      if (pred.stale || !Number.isFinite(pred.anchorRound)) continue;
      toSave[label] = { lo: pred.anchorRound+pred.low, hi: pred.anchorRound+pred.high, roundWhenMade: pred.anchorRound, generation: pred.generation, eta: { low:pred.low, high:pred.high, conf:pred.confidence } };
    }
    if (Object.keys(toSave).length > 0) { try { await saveLockedPatternPreds(toSave); } catch(e) { console.error('[pattern] saveLockedPatternPreds fail:', e.message); } }
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
    if (regime.regime !== 'normal' || regime.currentStreak > 5)
      console.log(`[engine] REGIME=${regime.regime.toUpperCase()} streak=${regime.currentStreak}r(${Math.round(regime.streakPct*100)}pct) penalty=${regime.streakPenalty}r cold=${regime.coldScore} logRatio=${regime.logRatio}`);
    await processEngine(rounds, lastRoundId, regime);
    await processPattern(rounds, lastRoundId, regime);
  } catch(e) { console.error('[predictionEngine] Fatal:', e.message, e.stack); }
}

module.exports = { runPredictionEngine, resetEngineState };