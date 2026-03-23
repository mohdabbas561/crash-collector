'use strict';
// advComputeEngine.js — ADV Consensus Engine v6 (FINAL PRODUCTION)
// ================================================================================
// v6 UPGRADE SUMMARY over previous version:
//
// A-1: Full v5 calibration layer injected — CALIB_DUMMY sentinel, wilsonLower
//      with NaN/Inf guards, TARGET_LOOK_AHEAD table, MIN_BIN constants,
//      module-level _calibLogCounter, computeCalibration with EPSF sampling,
//      dynamic stride (1/3/10), recency exp decay 0.999, Wilson CI gating,
//      getCalibratedAdjustment with normalized uplift, sensiHot=1.8/sensiCold=1.5,
//      maxDelta=0.26 (halved for rare), hard clamp 0.76–1.24.
//
// A-2: extractPredictiveStreakFeatures upgraded to v5 spec:
//      - hotScore strong trigger raised 55→72 (eliminates 3-soft-signal false positives)
//      - coldScore trigger raised 55→65
//      - b2bPrecursor requires garchRising vs baseline (not just garchSignal>0.15)
//      - postClusterEarlySignal raised to 1.5× threshold + densityFallingVsBaseline
//      - streakMomentum is now bidirectional (BUG-E fix)
//      - Markov uses completed runs only (BUG-C fix)
//      - predictedGapMultiplier clamped to 0.76–1.24 (H-7)
//      - calibration override applied inside EPSF
//
// A-3: applyStreakAdj upgraded to v5 spec:
//      - tc threshold raised 30→65
//      - WHITE_CLUSTER blend raised 0.55→0.65 (H-6)
//      - All regime adjustments tightened to v5 values
//
// A-4: streakConfBonus added (v5 spec):
//      - Reactive bonuses halved when !calibrated (H-5)
//      - Used in all 13 algos instead of raw tcBonus
//
// A-5: effectiveRegime tc threshold raised 40→68 (aligns with hotScore 72 trigger)
//
// A-6: All 13 algos upgraded:
//      - Pass calib to EPSF via per-tick calibration map
//      - b2bBoost amplification only when sf.calibrated===true (H-10)
//      - streakConfBonus replaces raw tcBonus
//      - sparsePenalty applied to conf
//
// A-7: computeConsensus upgraded:
//      - tcBonus only when calibrated===true && tc>75 && hotScore>=72
//
// ENGINE_IDS, TARGETS, savedSets, windows, getComputeRounds, saveOutcome,
// computeConsensus structure, locking, bustLockedCache, resolution, exports
// 100% unchanged.
// ================================================================================

const {
  getRounds,savePrediction,getPredictions,
  saveLockedAdvPreds,saveLockedConsensusPreds,
  getLockedAdvPreds,getLockedConsensusPreds,
} = require('./db');
const {bustLockedCache}=require('./advResolutionEngine');

const ENGINE_IDS=[];

const TARGETS=[
  {label:'5x',   min:5,   maxWidth:3,  rare:false},{label:'10x',  min:10,  maxWidth:5,  rare:false},
  {label:'20x',  min:20,  maxWidth:7,  rare:false},{label:'50x',  min:50,  maxWidth:12, rare:false},
  {label:'100x', min:100, maxWidth:18, rare:true },{label:'250x', min:250, maxWidth:25, rare:true },
  {label:'500x', min:500, maxWidth:35, rare:true },{label:'1000x',min:1000,maxWidth:50, rare:true },
];

const savedSets={};
for(const id of [...ENGINE_IDS,'consensus']) savedSets[id]=new Set();
let cachedRounds=[],cachedRoundsLastId=0,initialised=false;

// ── Core math helpers ─────────────────────────────────────────────────────────
function mean(arr){return arr.length?arr.reduce((s,v)=>s+v,0)/arr.length:0;}
function stdDev(arr){
  if(arr.length<2)return 0;const m=mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1));
}
function median(arr){
  if(!arr.length)return 0;
  const s=[...arr].sort((a,b)=>a-b);const m=Math.floor(s.length/2);
  return s.length%2===1?s[m]:(s[m-1]+s[m])/2;
}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}
function geoProbW(hr,w){return clamp(1-Math.pow(1-(hr||0),Math.max(1,w)),0,0.99);}
function pctile(sorted,frac){return sorted[Math.min(sorted.length-1,Math.max(0,Math.floor(frac*sorted.length)))];}
function bisectLeft(rounds,targetId){
  let lo=0,hi=rounds.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(rounds[mid].roundId<targetId)lo=mid+1;else hi=mid;}
  return lo;
}
function findHitInRange(rounds,fromId,toId,minMult){
  const start=bisectLeft(rounds,fromId);
  for(let i=start;i<rounds.length;i++){
    if(rounds[i].roundId>toId)break;if(rounds[i].multiplier>=minMult)return rounds[i];
  }
  return null;
}
function computeGaps(rounds,minMult){
  const gaps=[];let since=0;
  for(const r of rounds){since++;if(r.multiplier>=minMult){gaps.push(since);since=0;}}
  return{gaps,currentGap:since};
}
function placeWindow(expectedGap,currentGap,width){
  const remaining=Math.max(1,expectedGap-currentGap);
  const low=Math.max(1,remaining-Math.floor(width/2));
  return{low,high:low+width-1};
}
function earlyHitTolerance(width){return Math.floor(width/2);}
function olsLinear(ys){
  const n=ys.length;if(n<3)return{a:mean(ys),b:0,r2:0};
  let sx=0,sy=0,sxy=0,sxx=0;
  for(let i=0;i<n;i++){sx+=i;sy+=ys[i];sxy+=i*ys[i];sxx+=i*i;}
  const d=(n*sxx-sx*sx)||1,b=(n*sxy-sx*sy)/d,a=(sy-b*sx)/n;
  const gm=sy/n,ssTot=ys.reduce((s,v)=>s+(v-gm)**2,0),ssRes=ys.reduce((s,v,i)=>s+(v-(a+b*i))**2,0);
  return{a,b,r2:ssTot>0?clamp(1-ssRes/ssTot,0,1):0};
}
function cusumNorm(rounds,minMult){
  const n=rounds.length,p0=rounds.filter(r=>r.multiplier>=minMult).length/n;
  let cusum=0,maxC=0;
  for(const r of rounds.slice(-300)){cusum+=(r.multiplier>=minMult?1:0)-p0;if(Math.abs(cusum)>maxC)maxC=Math.abs(cusum);}
  return maxC/(Math.sqrt(Math.max(1e-9,p0*(1-p0)))*Math.sqrt(Math.min(300,n)));
}
function weibullSkew(p50,p75){return Math.max(1,Math.round(p50+0.20*(p75-p50)));}
function getDynamicBuckets(rounds){
  const mults=[...rounds].map(r=>r.multiplier).sort((a,b)=>a-b);
  if(mults.length<50)return{tL:5,tM:20,tH:100};
  return{tL:Math.max(pctile(mults,0.20),1.01),tM:Math.max(pctile(mults,0.50),1.02),tH:Math.max(pctile(mults,0.80),1.03)};
}
function sparsePenalty(hits,minFull){return hits>=minFull?1.0:Math.sqrt(Math.max(1,hits)/minFull);}

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-1: Full v5 calibration layer
// =============================================================================

// H-2: wilsonLower — Wilson score 95% CI lower bound.
// Guards: n≤0 returns 0; NaN/Inf returns 0; inner<0 clamped before sqrt.
function wilsonLower(p,n){
  if(n<=0||!isFinite(n)||isNaN(p))return 0;
  const z=1.96,z2=z*z;
  const inner=p*(1-p)/n+z2/(4*n*n);
  const sqrtTerm=Math.sqrt(Math.max(0,inner));
  const num=p+z2/(2*n)-z*sqrtTerm;
  const den=1+z2/n;
  const lower=num/den;
  if(isNaN(lower)||!isFinite(lower))return 0;
  return Math.max(0,lower);
}

// H-1: CALIB_DUMMY sentinel — prevents circularity when EPSF is called
// from within computeCalibration. hotBinCount[*]=0 → reqMinBin check fails
// → getCalibratedAdjustment returns calibMult=1.0, calibrated=false.
const CALIB_DUMMY=Object.freeze({
  hotHitRate:  new Array(10).fill(null),
  coldHitRate: new Array(10).fill(null),
  hotBinCount: new Array(10).fill(0),
  coldBinCount:new Array(10).fill(0),
  BIN_SIZE:10,minBin:20,margin:0.025,baseline:0.5,LOOK_AHEAD:20,targetRare:false,
});

// Per-target LOOK_AHEAD: long enough that P(hit) varies between regimes,
// short enough to have abundant calibration samples.
const TARGET_LOOK_AHEAD={
  '5x':20,'10x':20,'20x':20,'50x':40,
  '100x':80,'250x':150,'500x':300,'1000x':300,
};

const MIN_BIN_NON_RARE=20; // Wilson CI width ≈0.44 at p=0.5 — minimum reliable
const MIN_BIN_RARE    =30; // Sparse hit counts; 30 gives ±0.18 CI at p=0.1

// H-9: Module-level log counter — persists across ticks.
let _calibLogCounter=0;

const calibCache={};

// computeCalibration v5 — empirical P(hit in LOOK_AHEAD | hotScore bin)
// Full EPSF sampling with dynamic stride, recency decay 0.999, Wilson CI gating.
function computeCalibration(rounds,targetMin,targetLabel,targetRare){
  const cacheKey=targetLabel;
  const now=Date.now();
  const cache=calibCache[cacheKey];
  const delta=cache?rounds.length-cache.computedAt:Infinity;
  const age  =cache?now-cache.computedAtMs:Infinity;
  if(cache&&delta<50&&age<600000)return cache.result;

  const n=rounds.length;
  const LOOK_AHEAD=TARGET_LOOK_AHEAD[targetLabel]||20;
  const BIN_SIZE=10,NUM_BINS=10;
  const minBin=targetRare?MIN_BIN_RARE:MIN_BIN_NON_RARE;

  const globalHitRate=rounds.filter(r=>r.multiplier>=targetMin).length/Math.max(1,n);
  const baseline=clamp(1-Math.pow(Math.max(0,1-globalHitRate),LOOK_AHEAD),0.01,0.99);

  // H-3: Per-target minContext — skip positions that will return null EPSF.
  const expectedGapApprox=globalHitRate>0?Math.round(1/globalHitRate):n;
  const minContext=Math.min(Math.floor(n/4),Math.max(60,LOOK_AHEAD*2,expectedGapApprox*3));
  const maxPos=n-LOOK_AHEAD;

  if(maxPos<=minContext){
    const emptyResult={
      hotHitRate:new Array(NUM_BINS).fill(null),coldHitRate:new Array(NUM_BINS).fill(null),
      baseline,hotBinCount:new Array(NUM_BINS).fill(0),coldBinCount:new Array(NUM_BINS).fill(0),
      LOOK_AHEAD,BIN_SIZE,minBin,margin:targetRare?0.015:0.025,targetRare,
    };
    calibCache[cacheKey]={computedAt:n,computedAtMs:now,result:emptyResult};
    return emptyResult;
  }

  const hotBinHitsW =new Array(NUM_BINS).fill(0);
  const hotBinTotalW=new Array(NUM_BINS).fill(0);
  const hotBinCount =new Array(NUM_BINS).fill(0);
  const coldBinHitsW =new Array(NUM_BINS).fill(0);
  const coldBinTotalW=new Array(NUM_BINS).fill(0);
  const coldBinCount =new Array(NUM_BINS).fill(0);

  for(let pos=maxPos;pos>=minContext;){
    const distFromEnd=n-pos;
    const recWeight=Math.pow(0.999,distFromEnd);
    const ctx=rounds.slice(0,pos);
    // H-1: Pass CALIB_DUMMY to avoid circular calibration feedback.
    let sf=null;
    try{sf=extractPredictiveStreakFeatures(ctx,targetMin,CALIB_DUMMY);}
    catch(_){}

    if(!sf){
      const stride=distFromEnd<=4000?1:distFromEnd<=12000?3:10;
      pos-=stride;continue;
    }
    const hs=sf.hotScore,cs=sf.coldScore;
    if(!isFinite(hs)||!isFinite(cs)){
      const stride=distFromEnd<=4000?1:distFromEnd<=12000?3:10;
      pos-=stride;continue;
    }

    const hotBin =Math.min(NUM_BINS-1,Math.floor(hs/BIN_SIZE));
    const coldBin=Math.min(NUM_BINS-1,Math.floor(cs/BIN_SIZE));
    const futureHit=rounds.slice(pos,pos+LOOK_AHEAD).some(r=>r.multiplier>=targetMin);
    const noFutureHit=!futureHit;

    hotBinHitsW[hotBin]  +=futureHit  ?recWeight:0;
    hotBinTotalW[hotBin] +=recWeight;
    hotBinCount[hotBin]  +=1;
    coldBinHitsW[coldBin] +=noFutureHit?recWeight:0;
    coldBinTotalW[coldBin]+=recWeight;
    coldBinCount[coldBin] +=1;

    const stride=distFromEnd<=4000?1:distFromEnd<=12000?3:10;
    pos-=stride;
  }

  const margin=targetRare?0.015:0.025;
  const hotHitRate =new Array(NUM_BINS).fill(null);
  const coldHitRate=new Array(NUM_BINS).fill(null);

  for(let b=0;b<NUM_BINS;b++){
    if(hotBinCount[b]>=minBin&&hotBinTotalW[b]>0){
      const wr=hotBinHitsW[b]/hotBinTotalW[b];
      const wl=wilsonLower(wr,hotBinCount[b]);
      if(wl>baseline+margin)hotHitRate[b]=wr;
    }
    if(coldBinCount[b]>=minBin&&coldBinTotalW[b]>0){
      const wr=coldBinHitsW[b]/coldBinTotalW[b];
      const wl=wilsonLower(wr,coldBinCount[b]);
      if(wl>(1-baseline)+margin)coldHitRate[b]=wr;
    }
  }

  const result={hotHitRate,coldHitRate,baseline,hotBinCount,coldBinCount,
    LOOK_AHEAD,BIN_SIZE,minBin,margin,targetRare};

  _calibLogCounter++;
  if(_calibLogCounter%10===0){
    console.log(`[advCompute calib v6] ${JSON.stringify({
      target:targetLabel,baseline:baseline.toFixed(4),LOOK_AHEAD,minContext,
      positions:maxPos-minContext,binCounts:hotBinCount,
      hotHitRates:hotHitRate.map(v=>v!==null?v.toFixed(3):null),
      filledBins:hotHitRate.filter(v=>v!==null).length,
    })}`);
  }

  calibCache[cacheKey]={computedAt:n,computedAtMs:now,result};
  return result;
}

// getCalibratedAdjustment v5 — normalized uplift, Wilson CI gating, rare halved caps.
function getCalibratedAdjustment(hotScore,coldScore,calib,targetRare,sf){
  // H-1: CALIB_DUMMY or null → no adjustment.
  if(!calib||calib===CALIB_DUMMY)return{calibMult:1.0,calibConfBonus:0,calibrated:false};

  const hotBin =Math.min(9,Math.floor(hotScore /(calib.BIN_SIZE||10)));
  const coldBin=Math.min(9,Math.floor(coldScore/(calib.BIN_SIZE||10)));
  const hotCount =calib.hotBinCount?.[hotBin] ??0;
  const coldCount=calib.coldBinCount?.[coldBin]??0;
  const reqMinBin=targetRare?(calib.minBin??20)*2:(calib.minBin??20);

  // H-4: Conservative fallback in low-data zone (minBin ≤ count < 1.5×minBin).
  const densityTrend=sf?.densityTrend??0;
  const inLowDataZone=hotCount>=reqMinBin&&hotCount<reqMinBin*1.5;
  if(inLowDataZone){
    const conservMult=densityTrend>0.05?1.03:1.0;
    return{calibMult:conservMult,calibConfBonus:0,calibrated:false};
  }

  const hotRate =(hotCount >=reqMinBin)?calib.hotHitRate[hotBin] :null;
  const coldRate=(coldCount>=reqMinBin)?calib.coldHitRate[coldBin]:null;

  // Hot path: maxDelta/sensiHot unchanged.
  const maxDeltaHot=targetRare?0.13:0.26;
  const sensiHot=1.8;
  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  // maxDeltaCold raised 0.26→0.40 (non-rare), 0.13→0.20 (rare).
  // sensiCold raised 1.5→1.9. Cold trigger lowered 65→58.
  const maxDeltaCold=targetRare?0.20:0.40;
  const sensiCold=1.9;
  // === UPGRADE END ===

  let calibMult=1.0,calibConfBonus=0,calibrated=false;

  if(hotScore>=72&&hotRate!==null){
    const upliftNorm=clamp((hotRate-calib.baseline)/Math.max(0.001,1-calib.baseline),0,1);
    const reduction=clamp(upliftNorm*sensiHot,0,maxDeltaHot);
    if(reduction>0.01){
      calibMult=1.0-reduction;
      calibConfBonus=upliftNorm>0.25?12:upliftNorm>0.15?8:4;
      calibrated=true;
    }
  }

  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  if(coldScore>=58&&coldRate!==null&&!calibrated){
    const noHitBaseline=1-calib.baseline;
    const upliftNorm=clamp((coldRate-noHitBaseline)/Math.max(0.001,1-noHitBaseline),0,1);
    const extension=clamp(upliftNorm*sensiCold,0,maxDeltaCold);
    if(extension>0.01){
      calibMult=1.0+extension;
      calibConfBonus=-3;
      calibrated=true;
    }
  }
  // === UPGRADE END ===

  // Hot: floor 0.76 unchanged. Cold: ceiling raised 1.24→1.40.
  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  calibMult=clamp(calibMult,0.76,1.40);
  if(hotScore>=72&&calibMult>1.0)calibMult=1.0;
  // === UPGRADE END ===
  return{calibMult,calibConfBonus,calibrated};
}

// === UPGRADE END ===

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-2: extractPredictiveStreakFeatures v5 — full b2b/white-cluster detection
// =============================================================================
function extractPredictiveStreakFeatures(rounds,targetMin,calib){
  const n=rounds.length;if(n<10)return null;

  // 1. Run-Length Encoding
  const runs=[];
  let curHigh=rounds[0].multiplier>=targetMin,curLen=1;
  for(let i=1;i<n;i++){
    const h=rounds[i].multiplier>=targetMin;
    if(h===curHigh){curLen++;}else{runs.push({isHigh:curHigh,len:curLen});curHigh=h;curLen=1;}
  }
  runs.push({isHigh:curHigh,len:curLen});

  const highRuns=runs.filter(r=>r.isHigh).map(r=>r.len);
  const lowRuns =runs.filter(r=>!r.isHigh).map(r=>r.len);
  const lastRun=runs[runs.length-1];
  const currentIsHigh=lastRun.isHigh,currentStreakLen=lastRun.len;

  // 2. Core metrics
  const b2bOccurrences=highRuns.filter(l=>l>=2).length;
  const b2bRate       =highRuns.length?b2bOccurrences/highRuns.length:0;
  const avgHighRunLen =highRuns.length?mean(highRuns):0;
  const maxHighRunLen =highRuns.length?Math.max(...highRuns):0;
  const avgLowRunLen  =lowRuns.length ?mean(lowRuns) :0;
  const maxLowRunLen  =lowRuns.length ?Math.max(...lowRuns):0;
  const stdLowRunLen  =lowRuns.length>1?stdDev(lowRuns):0;

  let b2bContinuationProb=0;
  if(highRuns.length>=5){
    const ext=highRuns.filter(l=>l>=2).reduce((s,l)=>s+l-1,0);
    const tot=highRuns.reduce((s,l)=>s+l,0);
    b2bContinuationProb=tot>0?ext/tot:0;
  }

  // 3. Density windows
  const W5 =rounds.slice(-5), W10=rounds.slice(-10);
  const W20=rounds.slice(-20),W50=rounds.slice(-50);
  const ld5 =W5.filter(r=>r.multiplier<targetMin).length /Math.max(1,W5.length);
  const ld10=W10.filter(r=>r.multiplier<targetMin).length/Math.max(1,W10.length);
  const ld20=W20.filter(r=>r.multiplier<targetMin).length/Math.max(1,W20.length);
  const ld50=W50.filter(r=>r.multiplier<targetMin).length/Math.max(1,W50.length);
  const densityTrend=ld10-ld50;
  const globalLowRate=1-rounds.filter(r=>r.multiplier>=targetMin).length/n;

  // GARCH vs baseline (BUG-A fix from v5)
  const {gaps}=computeGaps(rounds,targetMin);
  let garchSignal=0,garchBaseline=0;
  if(gaps.length>=10){
    const gm=mean(gaps),ad=gaps.map(g=>Math.abs(g-gm));
    let cov=0,vs=0;
    for(let i=1;i<ad.length;i++)cov+=ad[i-1]*ad[i];
    for(const v of ad)vs+=v*v;
    garchSignal=vs>0?cov/vs:0;
    const oldGaps=gaps.slice(0,Math.floor(gaps.length/2));
    if(oldGaps.length>=5){
      const gm2=mean(oldGaps),ad2=oldGaps.map(g=>Math.abs(g-gm2));
      let c2=0,v2=0;
      for(let i=1;i<ad2.length;i++)c2+=ad2[i-1]*ad2[i];
      for(const v of ad2)v2+=v*v;
      garchBaseline=v2>0?c2/v2:garchSignal;
    }else{garchBaseline=garchSignal;}
  }
  const garchRising=garchSignal>garchBaseline*1.30&&garchSignal>0.25;

  // 4. Post-cluster gap
  const longLowThresh=Math.max(2,Math.round(avgLowRunLen*1.3));
  const postClusterGaps=[];
  for(let i=0;i<runs.length-1;i++){
    if(!runs[i].isHigh&&runs[i].len>=longLowThresh&&runs[i+1].isHigh)
      postClusterGaps.push(1);
  }
  const avgPostClusterGap=postClusterGaps.length?mean(postClusterGaps):null;

  // 5. Reactive regime
  let regime='NEUTRAL';
  if     (currentIsHigh&&currentStreakLen>=2)                                   regime='B2B';
  else if(currentIsHigh&&runs.length>=2&&!runs[runs.length-2].isHigh
          &&runs[runs.length-2].len<=avgLowRunLen*0.5)                          regime='HOT_AFTER_SHORT_COLD';
  else if(!currentIsHigh&&currentStreakLen>=avgLowRunLen*1.5)                   regime='WHITE_CLUSTER';
  else if(!currentIsHigh&&currentStreakLen>=maxLowRunLen*0.8&&maxLowRunLen>2)   regime='EXTREME_WHITE';
  else if(b2bRate>0.25&&ld20<globalLowRate*0.7)                                 regime='HOT';
  else if(ld20>globalLowRate*1.3)                                               regime='COLD';

  // 6. Forward-looking signals
  const lowDensityAccel=(ld5-ld10)-(ld10-ld20);

  // A-2: BUG-E fix — bidirectional streak momentum
  let streakMomentumLow=0,streakMomentumHigh=0;
  if(!currentIsHigh&&lowRuns.length>=4){
    const prev3=mean(lowRuns.slice(-4,-1));
    streakMomentumLow=(currentStreakLen-prev3)/Math.max(1,prev3);
  }
  if(currentIsHigh&&highRuns.length>=4){
    const prev3=mean(highRuns.slice(-4,-1));
    streakMomentumHigh=(currentStreakLen-prev3)/Math.max(1,prev3);
  }
  const streakMomentum=!currentIsHigh?streakMomentumLow:-streakMomentumHigh;

  // A-2: postClusterEarlySignal at 1.5× + densityFallingVsBaseline (BUG-B fix)
  let postClusterEarlySignal=false;
  if(!currentIsHigh&&lowRuns.length>=3){
    const recentLowsShortening=lowRuns.slice(-3,-1).every(l=>l<avgLowRunLen);
    const densityFallingVsBaseline=ld20<ld50*0.88;
    const inExtendedCluster=currentStreakLen>=avgLowRunLen*1.5;
    postClusterEarlySignal=inExtendedCluster&&densityFallingVsBaseline&&recentLowsShortening;
  }

  // A-2: b2bPrecursor requires garchRising vs baseline (BUG-A fix)
  let b2bPrecursor=false;
  if(lowRuns.length>=2){
    const lastCompletedLowIdx=currentIsHigh?lowRuns.length-1:lowRuns.length-2;
    const lastCompletedLow=lastCompletedLowIdx>=0?lowRuns[lastCompletedLowIdx]:null;
    if(lastCompletedLow!==null){
      const shortLowRun=lastCompletedLow<avgLowRunLen*0.55;
      b2bPrecursor=shortLowRun&&garchRising;
    }
  }

  // A-2: Markov uses completed runs only (BUG-C fix)
  let markovProbHot=(1-globalLowRate)||0.1;
  const completedRuns=runs.slice(0,-1);
  if(completedRuns.length>=4){
    const seq=completedRuns.map(r=>r.isHigh?1:0);
    const mat={};
    for(let i=2;i<seq.length;i++){
      const key=`${seq[i-2]},${seq[i-1]}`;
      if(!mat[key])mat[key]={H:0,L:0};
      if(seq[i]===1)mat[key].H++;else mat[key].L++;
    }
    const last2Key=`${seq[seq.length-2]},${seq[seq.length-1]}`;
    const cell=mat[last2Key];
    if(cell){const tot=cell.H+cell.L;if(tot>=5)markovProbHot=cell.H/tot;}
  }

  // A-2: hotScore with raised thresholds (H-8: strong trigger 72)
  const hotScore=clamp(Math.round(
    (postClusterEarlySignal  ?30:0)+
    (b2bPrecursor            ?20:0)+
    (lowDensityAccel<-0.08   ?18:lowDensityAccel<-0.04?9:0)+
    (streakMomentum<-0.35    ?15:streakMomentum<-0.15?7:0)+
    (markovProbHot>0.68      ?15:markovProbHot>0.55?7:0)+
    (b2bRate>0.28            ?10:b2bRate>0.18?5:0)+
    (ld20<ld50*0.82          ?10:0)
  ),0,100);

  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  // streakMomentum +25%: 28→35/14→17. lowDensityAccel +30%: 22→29/11→14.
  // currentStreakLen bonus +33%: 12→16.
  const coldScore=clamp(Math.round(
    (streakMomentum>0.45     ?35:streakMomentum>0.25?17:0)+
    (lowDensityAccel>0.08    ?29:lowDensityAccel>0.04?14:0)+
    (markovProbHot<0.22      ?20:markovProbHot<0.38?10:0)+
    (ld20>ld50*1.40          ?15:ld20>ld50*1.20?7:0)+
    (currentStreakLen>avgLowRunLen*1.3&&!currentIsHigh?16:0)
  ),0,100);
  // === UPGRADE END ===

  // A-2: Regime prediction — v5 thresholds and clamps (H-7, H-8)
  let predictedNextRegime='NEUTRAL',transitionConfidence=0,predictedGapMultiplier=1.0;

  if(hotScore>=72&&hotScore>coldScore+15){
    if(b2bPrecursor){
      predictedNextRegime  ='ABOUT_TO_B2B';
      transitionConfidence =clamp(hotScore,72,95);
      predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.36,0.76,0.88);
    }else{
      predictedNextRegime  ='ABOUT_TO_HOT';
      transitionConfidence =clamp(hotScore,72,90);
      predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.26,0.76,0.90);
    }
  }else if(hotScore>=62&&hotScore>coldScore+10){
    predictedNextRegime  ='ABOUT_TO_HOT';
    transitionConfidence =clamp(hotScore,62,88);
    predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.20,0.80,0.93);
  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  // Cold trigger 65→58, multiplier caps raised to 1.15–1.40 / 1.10–1.32.
  }else if(coldScore>=58&&coldScore>hotScore+12){
    if(currentStreakLen>=avgLowRunLen*1.5||regime==='EXTREME_WHITE'){
      predictedNextRegime  ='ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence =clamp(coldScore,58,88);
      predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.42,1.15,1.40);
    }else{
      predictedNextRegime  ='ABOUT_TO_COLD';
      transitionConfidence =clamp(coldScore,50,82);
      predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.30,1.10,1.32);
    }
  }
  // === UPGRADE END ===

  // Apply calibration override
  const targetRare=calib?.targetRare??false;
  const cal=getCalibratedAdjustment(hotScore,coldScore,calib,targetRare,{densityTrend});
  if(cal.calibrated){
    predictedGapMultiplier=cal.calibMult;
    if(predictedNextRegime==='NEUTRAL'&&cal.calibMult<0.95&&hotScore>=55){
      predictedNextRegime='ABOUT_TO_HOT';
      transitionConfidence=Math.min(transitionConfidence+10,85);
    }
    // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
    // Also promote NEUTRAL→ABOUT_TO_COLD when calibration confirms cold extension.
    if(predictedNextRegime==='NEUTRAL'&&cal.calibMult>1.05&&coldScore>=45){
      predictedNextRegime='ABOUT_TO_COLD';
      transitionConfidence=Math.min(transitionConfidence+8,80);
    }
    // === UPGRADE END ===
  }
  const calibConfBonus=cal.calibConfBonus;

  return{
    runs,highRuns,lowRuns,
    currentIsHigh,currentStreakLen,regime,
    b2bOccurrences:highRuns.filter(l=>l>=2).length,b2bRate,b2bContinuationProb,
    avgHighRunLen,maxHighRunLen,
    avgLowRunLen,maxLowRunLen,stdLowRunLen,avgPostClusterGap,
    lowDensity10:ld10,lowDensity20:ld20,lowDensity50:ld50,densityTrend,
    globalLowRate,garchSignal,garchBaseline,
    lowDensityAccel,streakMomentum,
    postClusterEarlySignal,b2bPrecursor,markovProbHot,
    hotScore,coldScore,
    predictedNextRegime,transitionConfidence,predictedGapMultiplier,
    calibConfBonus,calibrated:cal.calibrated,
  };
}
// === UPGRADE END ===

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-3: applyStreakAdj v5 — tc threshold 65, WHITE_CLUSTER blend 0.65
// =============================================================================
function applyStreakAdj(expectedGap,sf){
  if(!sf)return expectedGap;
  const pnr=sf.predictedNextRegime,mult=sf.predictedGapMultiplier??1.0,tc=sf.transitionConfidence??0;
  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  // Hot path: tc≥65 unchanged. Cold path: fires at tc≥58.
  const isColdPnr=pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD';
  const isHotPnr =pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT';
  if(isHotPnr&&tc>=65){
    const blend=clamp((tc-65)/45,0,1);
    return Math.max(1,Math.round(expectedGap*(1.0+(mult-1.0)*blend)));
  }
  if(isColdPnr&&tc>=58){
    const blend=clamp((tc-58)/45,0,1);
    return Math.max(1,Math.round(expectedGap*(1.0+(mult-1.0)*blend)));
  }
  if(!isHotPnr&&!isColdPnr&&pnr!=='NEUTRAL'&&tc>=65){
    const blend=clamp((tc-65)/45,0,1);
    return Math.max(1,Math.round(expectedGap*(1.0+(mult-1.0)*blend)));
  }
  // === UPGRADE END ===
  let adj=expectedGap;
  switch(sf.regime){
    case 'B2B':        adj=Math.round(adj*(1-sf.b2bContinuationProb*0.30));break;
    case 'HOT_AFTER_SHORT_COLD': adj=Math.round(adj*0.88);break;
    case 'HOT':        adj=Math.round(adj*(1-(1-sf.lowDensity20)*0.18));break;
    // A-3: H-6 WHITE_CLUSTER blend raised 0.55→0.65
    case 'WHITE_CLUSTER':
      adj=sf.avgPostClusterGap!==null
        ?Math.round(adj*0.65+sf.avgPostClusterGap*0.35)
        :Math.round(adj*0.92);break;
    case 'EXTREME_WHITE': adj=Math.round(adj*0.75);break;
    case 'COLD':       adj=Math.round(adj*(1+sf.densityTrend*0.12));break;
    default:           adj=Math.round(adj*(1-sf.densityTrend*0.06));
  }
  return Math.max(1,adj);
}
// === UPGRADE END ===

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-4: streakConfBonus v5 — reactive bonuses halved when !calibrated (H-5)
// A-5: effectiveRegime tc threshold raised 40→68
// =============================================================================
function streakConfBonus(sf,isRare){
  if(!sf)return 0;
  const pnr=sf.predictedNextRegime;
  const tc =sf.transitionConfidence??0;
  const cb =sf.calibConfBonus??0;
  let base=cb;

  if((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&sf.calibrated&&tc>75){
    base+=14+(isRare&&sf.b2bPrecursor?4:0);
  }else if((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&tc>=65){
    base+=7;
  // === v6 COLD-PATH & CALIBRATION UPGRADE START ===
  // Cold penalty strengthened -3→-6, threshold lowered 65→58.
  }else if((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD')&&tc>58){
    base-=6;
  }
  // === UPGRADE END ===

  // H-5: Reactive bonuses halved when !calibrated
  const calibMult=sf.calibrated?1.0:0.5;
  switch(sf.regime){
    case 'B2B':          base+=Math.floor((sf.b2bContinuationProb>0.3?5:2)*calibMult);break;
    case 'WHITE_CLUSTER':base+=Math.floor((sf.avgPostClusterGap!==null?4:2)*calibMult);break;
    case 'EXTREME_WHITE':base+=Math.floor(6*calibMult);break;
    case 'HOT':          base+=Math.floor(3*calibMult);break;
    case 'COLD':         base-=2;break;
  }
  return base;
}

// A-5: effectiveRegime uses tc≥68 (aligns with hotScore 72 trigger)
function effectiveRegime(sf){
  if(!sf)return'NEUTRAL';
  return(sf.transitionConfidence??0)>=68?sf.predictedNextRegime:sf.regime;
}
// === UPGRADE END ===

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-6: All 13 algos — calibrated b2bBoost, streakConfBonus, sparsePenalty
// =============================================================================
function runLSTM(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  const DECAY=0.97;let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/(wS||1),gMean=mean(gaps);
  let raw=Math.max(1,Math.round(ewaMean*0.70+gMean*0.30));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  const conf=clamp(Math.round((85-30*(stdDev(gaps)/(ewaMean||1))+streakConfBonus(sf,target.rare))*sp),30,96);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf};
}

function runXGBoost(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10)return null;
  const hrGlobal=gaps.length/rounds.length;
  const recentN=Math.min(300,Math.max(15,Math.round(5/(hrGlobal||0.01))));
  const hrRecent=(rounds.slice(-recentN).filter(r=>r.multiplier>=target.min).length+1)/(recentN+2);
  const gMean=mean(gaps);
  const{b:rawSlope}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  let raw=Math.max(1,Math.round(gMean*(1-clamp((hrRecent-hrGlobal)/(hrGlobal||0.01),-0.25,0.25))+clamp(rawSlope,-gMean*0.12,gMean*0.12)*0.5));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  const conf=clamp(Math.round((80-22*(stdDev(gaps)/(gMean||1))+streakConfBonus(sf,target.rare))*sp),30,96);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(hrGlobal,aw),conf};
}

function runRF(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10)return null;
  const N_BOOT=64,n=gaps.length;
  const primes=[3,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317];
  const bootMedians=[];
  for(let b=0;b<N_BOOT;b++){
    const offset=(b*11+5)%n,prime=primes[b%primes.length],sample=[];
    for(let i=0;i<n;i++)sample.push(gaps[(offset+i*prime)%n]);
    sample.sort((a,b)=>a-b);bootMedians.push(sample[Math.floor(n/2)]);
  }
  bootMedians.sort((a,b)=>a-b);
  let raw=Math.max(1,weibullSkew(pctile(bootMedians,0.50),pctile(bootMedians,0.75)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((52+gaps.length*0.2)*sp),40,88)};
}

function runOLS(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10)return null;
  const lin=olsLinear(gaps),gMean=mean(gaps);
  const rawGap=Math.max(1,Math.round(lin.a+lin.b*gaps.length));
  let raw=Math.max(1,Math.round(rawGap*lin.r2+gMean*(1-lin.r2)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((35+lin.r2*42+gaps.length*0.15)*sp),30,88)};
}

function runCatBoost(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8)return null;
  const hrGlobal=(gaps.length+1)/(rounds.length+2);
  const cn=cusumNorm(rounds,target.min);
  const cusumRegime=cn>1.96?'HOT':cn>1.36?'WARM':cn<-1.96?'COLD':'NEUTRAL';
  // A-6: effectiveRegime at tc≥68
  const pnr=effectiveRegime(sf);
  const rlHot=pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT'||pnr==='B2B'||pnr==='HOT'||pnr==='HOT_AFTER_SHORT_COLD';
  const rlCold=pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD'||pnr==='WHITE_CLUSTER'||pnr==='EXTREME_WHITE'||pnr==='COLD';
  const blend=cusumRegime==='HOT'?0.55:cusumRegime==='WARM'?0.35:cusumRegime==='COLD'?0.20:rlHot?0.50:rlCold?0.12:0.12;
  const hr300=(rounds.slice(-300).filter(r=>r.multiplier>=target.min).length+1)/302;
  const pBlend=hrGlobal*(1-blend)+hr300*blend;
  let raw=Math.max(1,Math.round((1-pBlend)/pBlend));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(pBlend,aw),conf:clamp(Math.round((48+gaps.length*0.25)*sp),38,91)};
}

function runHardGap(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  const sorted=[...gaps].sort((a,b)=>a-b),gMean=mean(gaps);
  let raw=Math.max(1,Math.round(gMean*0.4+weibullSkew(pctile(sorted,0.50)||gMean,pctile(sorted,0.75)||gMean)*0.6));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(30+(currentGap/(raw||1))*22),20,95)};
}

function runSoftGap(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=0.85;}
  let raw=Math.max(1,Math.round((wG/wS)*0.60+median(gaps)*0.40));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((55+gaps.length*0.2)*sp),25,88)};
}

function runMarkov(rounds,target,sf){
  if(rounds.length<30)return null;
  const{tL,tM,tH}=getDynamicBuckets(rounds);
  const bucket=m=>m>=tH?'X':m>=tM?'H':m>=tL?'M':'L';
  const states=['L','M','H','X'],mat1={};
  for(const s of states){mat1[s]={};for(const t of states)mat1[s][t]=0;}
  for(let i=1;i<rounds.length;i++)mat1[bucket(rounds[i-1].multiplier)][bucket(rounds[i].multiplier)]++;
  const cs=bucket(rounds[rounds.length-1]?.multiplier??1);
  const tot1=states.reduce((s,t)=>s+mat1[cs][t],0);
  const prob={};states.forEach(t=>{prob[t]=tot1?mat1[cs][t]/tot1:0.25;});
  let pHit=target.min>=tH?prob['X']:target.min>=tM?prob['H']+prob['X']:target.min>=tL?prob['M']+prob['H']+prob['X']:1-prob['L']*0.3;
  pHit=pHit*0.65+(rounds.filter(r=>r.multiplier>=target.min).length/rounds.length)*0.35;
  // A-6: b2bBoost only when calibrated
  if(sf&&(sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT')&&(sf.transitionConfidence??0)>=65){
    const boostMult=sf.calibrated?1+(sf.b2bContinuationProb||0)*0.5:1+(sf.b2bContinuationProb||0)*0.2;
    pHit=Math.min(0.99,pHit*boostMult);
  }
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  let raw=Math.max(1,pHit>0?Math.round(1/pHit):mean(gaps)||50);
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:clamp(pHit*(aw/Math.max(1,raw)),0,0.99),conf:clamp(Math.round(pHit*75+20),20,88)};
}

function runPercentile(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8)return null;
  const sorted=[...gaps].sort((a,b)=>a-b);
  let raw=Math.max(1,weibullSkew(pctile(sorted,0.50),pctile(sorted,0.75)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((62+gaps.length*0.15)*sp),30,90)};
}

function runBayes(rounds,target,sf){
  if(rounds.length<10)return null;
  const hits=rounds.filter(r=>r.multiplier>=target.min).length;
  const alpha=1+hits,beta=1+rounds.length-hits,mu=alpha/(alpha+beta);
  const rawGap=Math.max(1,Math.round(1/(mu||0.001)));
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  const sorted=[...gaps].sort((a,b)=>a-b),p50=pctile(sorted,0.50)||rawGap,p75=pctile(sorted,0.75)||rawGap;
  const muVar=alpha*beta/((alpha+beta)**2*(alpha+beta+1)),certainty=clamp(1-Math.sqrt(muVar)/mu,0,1);
  let raw=Math.max(1,Math.round(rawGap*certainty+weibullSkew(p50,p75)*(1-certainty)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(mu,aw),conf:clamp(Math.round((40+rounds.length*0.06+certainty*20)*sp),35,90)};
}

function runSHA256(rounds,target,sf){
  if(rounds.length<20)return null;
  const obsP=(rounds.filter(r=>r.multiplier>=target.min).length+1)/(rounds.length+2);
  const trust=clamp((rounds.length-20)/180,0,1);
  let raw=Math.max(1,Math.round((1/obsP)*trust+(target.min/(1-0.01))*(1-trust)));
  // A-6: SHA entropy bias only when sf.calibrated
  if(sf&&sf.highRuns&&sf.highRuns.length>=5){
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    // H-10: only amplify when calibrated
    const hotW=sf.calibrated?(sf.hotScore??0)/100:(sf.hotScore??0)/200;
    const streakBias=(1-normHrEnt)*hotW*0.12;
    raw=Math.max(1,Math.round(raw*(1-streakBias)));
  }
  raw=applyStreakAdj(raw,sf);
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(obsP,aw),conf:clamp(Math.round((55+trust*15+streakConfBonus(sf,target.rare))*sp),40,90)};
}

function runMersenne(rounds,target,sf){
  if(rounds.length<50)return null;
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  let raw=Math.max(1,Math.round(median(gaps)*0.6+mean(gaps)*0.4));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((42+gaps.length*0.15)*sp),30,85)};
}

function runLCG(rounds,target,sf){
  if(rounds.length<30)return null;
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  let raw=Math.max(1,Math.round(median(gaps)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth,sp=sparsePenalty(gaps.length,50);
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round((25+gaps.length*0.15)*sp),20,80)};
}
// === UPGRADE END ===

const ALGOS={
  lstm:runLSTM,xgb:runXGBoost,rf:runRF,ols:runOLS,cat:runCatBoost,
  hardgap:runHardGap,softgap:runSoftGap,markov:runMarkov,
  percentile:runPercentile,bayes:runBayes,sha256:runSHA256,mt:runMersenne,lcg:runLCG,
};

// ── Infrastructure (UNCHANGED) ────────────────────────────────────────────────
async function getComputeRounds(){
  if(cachedRounds.length===0){
    cachedRounds=await getRounds({limit:100000,order:'ASC'});
    cachedRoundsLastId=cachedRounds.length?cachedRounds[cachedRounds.length-1].roundId:0;
    console.log(`[advCompute] loaded ${cachedRounds.length} rounds`);
  }else{
    const nr=await getRounds({limit:5000,minRoundId:cachedRoundsLastId+1});
    if(nr.length){cachedRounds=[...cachedRounds,...nr];cachedRoundsLastId=cachedRounds[cachedRounds.length-1].roundId;}
  }
  return cachedRounds;
}

async function saveOutcome(engineId,target,outcome,lo,hi,hitRound,generation){
  const key=`${lo}:${hi}`;
  if(savedSets[engineId].has(key))return;
  savedSets[engineId].add(key);
  try{
    await savePrediction({target:target.label,minMult:target.min,outcome,lo,hi,hitRound:hitRound??null,generation:generation??1,source:engineId,probW:null});
    console.log(`[advCompute] ${engineId} ${target.label} ${outcome.toUpperCase()} #${lo}-#${hi}${hitRound?` @#${hitRound}`:''}`);
  }catch(e){console.error(`[advCompute] save fail ${engineId}:`,e.message);savedSets[engineId].delete(key);}
}

// =============================================================================
// === ADV v6 PRODUCTION UPGRADE START ===
// A-7: computeConsensus — tcBonus only when calibrated && tc>75 && hotScore>=72
// =============================================================================
function computeConsensus(allResults,lastRoundId,streakFeatures){
  const consensus={};
  for(const target of TARGETS){
    const windows=[];
    for(const[eid,res]of Object.entries(allResults)){
      const r=res[target.label];if(!r)continue;
      windows.push({engineId:eid,lo:lastRoundId+r.low,hi:lastRoundId+r.high});
    }
    if(windows.length<3){consensus[target.label]=null;continue;}
    let bestGroup=[],bestLo=0,bestHi=0;
    for(let i=0;i<windows.length;i++){
      const grp=[windows[i]];let rLo=windows[i].lo,rHi=windows[i].hi;
      for(let j=0;j<windows.length;j++){
        if(j===i)continue;
        const nl=Math.max(rLo,windows[j].lo),nh=Math.min(rHi,windows[j].hi);
        if(nl<=nh){grp.push(windows[j]);rLo=nl;rHi=nh;}
      }
      if(grp.length>bestGroup.length){bestGroup=grp;bestLo=rLo;bestHi=rHi;}
    }
    if(bestGroup.length<2){consensus[target.label]=null;continue;}
    const baseW=target.maxWidth;
    if(bestHi-bestLo+1<baseW){const c=Math.round((bestLo+bestHi)/2);bestLo=c-Math.floor(baseW/2);bestHi=bestLo+baseW-1;}
    if(bestLo<=lastRoundId){bestLo=lastRoundId+1;bestHi=bestLo+baseW-1;}
    // A-7: tcBonus only when calibrated && tc>75 && hotScore>=72
    const sf=streakFeatures?.[target.label];
    const tcBonus=(sf?.calibrated===true&&(sf?.transitionConfidence??0)>75&&(sf?.hotScore??0)>=72&&
      (sf?.predictedNextRegime==='ABOUT_TO_B2B'||sf?.predictedNextRegime==='ABOUT_TO_HOT'))?8:0;
    consensus[target.label]={lo:bestLo,hi:bestHi,engineCount:bestGroup.length,engines:bestGroup.map(w=>w.engineId),tcBonus};
  }
  return consensus;
}
// === UPGRADE END ===

const windows={};
for(const id of [...ENGINE_IDS,'consensus']){windows[id]={};}

async function runAdvComputeEngine(){
  try{
    const rounds=await getComputeRounds();
    if(rounds.length<50)return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // === ADV v6 PRODUCTION UPGRADE START ===
    // A-1+A-2: Compute calibrations first, then pass to EPSF
    const calibrations={};
    for(const target of TARGETS){
      try{calibrations[target.label]=computeCalibration(rounds,target.min,target.label,target.rare);}
      catch(e){calibrations[target.label]=null;console.error(`[advCompute] calib/${target.label}:`,e.message);}
    }

    const streakFeatures={};
    for(const target of TARGETS){
      try{
        streakFeatures[target.label]=extractPredictiveStreakFeatures(
          rounds,target.min,calibrations[target.label]??CALIB_DUMMY
        );
      }catch(e){streakFeatures[target.label]=null;}
    }
    // === UPGRADE END ===

    const ALL_ALGO_IDS=Object.keys(ALGOS);
    const allResults={};
    for(const engineId of ALL_ALGO_IDS){
      const algo=ALGOS[engineId];if(!algo)continue;
      allResults[engineId]={};
      for(const target of TARGETS){
        try{
          const sf=streakFeatures[target.label];
          const r=algo(rounds,target,sf);
          if(r)allResults[engineId][target.label]=r;
        }catch(e){console.error(`[advCompute] ${engineId}/${target.label}:`,e.message);}
      }
    }

    const advPayload={};
    for(const engineId of ENGINE_IDS){
      advPayload[engineId]={};
      for(const target of TARGETS){
        const win=windows[engineId][target.label];
        const fresh=allResults[engineId][target.label];
        if(win){
          const{lo,hi,generation,roundWhenMade}=win;
          const earlyCheckLo=Math.max(roundWhenMade+1,lo-earlyHitTolerance(target.maxWidth));
          const earlyHit=lo>roundWhenMade+1&&earlyCheckLo<=lo-1?findHitInRange(rounds,earlyCheckLo,lo-1,target.min):null;
          if(earlyHit){await saveOutcome(engineId,target,'early',lo,hi,earlyHit.roundId,generation);delete windows[engineId][target.label];}
          else if(lastRoundId>=hi){const hit=findHitInRange(rounds,lo,hi,target.min);await saveOutcome(engineId,target,hit?'win':'loss',lo,hi,hit?.roundId??null,generation);delete windows[engineId][target.label];}
          else{const hit=findHitInRange(rounds,lo,hi,target.min);if(hit){await saveOutcome(engineId,target,'win',lo,hi,hit.roundId,generation);delete windows[engineId][target.label];}else{advPayload[engineId][target.label]={lo,hi,roundWhenMade,generation,eta:win.eta};continue;}}
        }
        if(fresh){
          const newLo=lastRoundId+fresh.low,newHi=lastRoundId+fresh.high;
          const gen=(windows[engineId][target.label]?.generation??0)+1;
          windows[engineId][target.label]={lo:newLo,hi:newHi,roundWhenMade:lastRoundId,generation:gen,eta:{probW:fresh.probW,conf:fresh.conf,expectedGap:fresh.expectedGap}};
          advPayload[engineId][target.label]=windows[engineId][target.label];
        }
      }
      if(Object.keys(advPayload[engineId]).length)await saveLockedAdvPreds(engineId,advPayload[engineId]);
    }

    // A-7: pass streakFeatures to computeConsensus for tcBonus
    const consensus=computeConsensus(allResults,lastRoundId,streakFeatures);
    const consPayload={};
    for(const target of TARGETS){
      const c=consensus[target.label];
      const win=windows['consensus'][target.label];
      if(win){
        const consEarlyLo=Math.max(win.roundWhenMade+1,win.lo-earlyHitTolerance(target.maxWidth));
        const earlyHit=win.lo>win.roundWhenMade+1&&consEarlyLo<=win.lo-1?findHitInRange(rounds,consEarlyLo,win.lo-1,target.min):null;
        if(earlyHit){await saveOutcome('consensus',target,'early',win.lo,win.hi,earlyHit.roundId,win.generation);delete windows['consensus'][target.label];}
        else if(lastRoundId>=win.hi){const hit=findHitInRange(rounds,win.lo,win.hi,target.min);await saveOutcome('consensus',target,hit?'win':'loss',win.lo,win.hi,hit?.roundId??null,win.generation);delete windows['consensus'][target.label];}
        else{const hit=findHitInRange(rounds,win.lo,win.hi,target.min);if(hit){await saveOutcome('consensus',target,'win',win.lo,win.hi,hit.roundId,win.generation);delete windows['consensus'][target.label];}else{consPayload[target.label]={lo:win.lo,hi:win.hi,roundWhenMade:win.roundWhenMade,generation:win.generation,eta:win.eta};continue;}}
      }
      if(c){
        const gen=(windows['consensus'][target.label]?.generation??0)+1;
        windows['consensus'][target.label]={lo:c.lo,hi:c.hi,roundWhenMade:lastRoundId,generation:gen,eta:{engineCount:c.engineCount,engines:c.engines}};
        consPayload[target.label]=windows['consensus'][target.label];
      }
    }
    if(Object.keys(consPayload).length){
      await saveLockedConsensusPreds(consPayload);
      const newWins=Object.keys(consPayload).filter(t=>consPayload[t].roundWhenMade===lastRoundId);
      if(newWins.length){
        const targets=newWins.map(t=>`${t}:#${consPayload[t].lo}-#${consPayload[t].hi}`).join(' ');
        console.log(`[advCompute] consensus NEW windows: ${targets}`);
      }
    }
    bustLockedCache();
  }catch(e){console.error('[advCompute] Fatal:',e.message,e.stack);}
}

async function initAdvCompute(){
  if(initialised)return;initialised=true;
  try{
    const existing=await getLockedAdvPreds();
    for(const engineId of ENGINE_IDS){
      for(const target of TARGETS){
        const w=existing[engineId]?.[target.label];
        if(w?.lo&&w?.hi)windows[engineId][target.label]={lo:Number(w.lo),hi:Number(w.hi),roundWhenMade:Number(w.roundWhenMade??w.lo),generation:w.generation??1,eta:w.eta??{}};
      }
    }
    const cons=await getLockedConsensusPreds();
    for(const target of TARGETS){
      const w=cons[target.label];
      if(w?.lo&&w?.hi)windows['consensus'][target.label]={lo:Number(w.lo),hi:Number(w.hi),roundWhenMade:Number(w.roundWhenMade??w.lo),generation:w.generation??1,eta:w.eta??{}};
    }
    console.log(`[advCompute] loaded existing locked windows`);
  }catch(e){console.error('[advCompute] init locked error:',e.message);}
  try{
    const allIds=[...ENGINE_IDS,'consensus'];
    for(const engineId of allIds){
      const rows=await getPredictions({limit:500000,source:engineId});
      for(const r of rows)savedSets[engineId].add(`${r.lo}:${r.hi}`);
    }
    const total=[...ENGINE_IDS,'consensus'].reduce((s,id)=>s+savedSets[id].size,0);
    console.log(`[advCompute] pre-warmed savedSets with ${total} outcomes`);
  }catch(e){console.error('[advCompute] init history error:',e.message);}
}

const _origRun=runAdvComputeEngine;
async function runAdvComputeEngineWithInit(){await initAdvCompute();await _origRun();}
function resetAdvComputeState(){
  for(const id of [...ENGINE_IDS,'consensus']){windows[id]={};savedSets[id]=new Set();}
  cachedRounds=[];cachedRoundsLastId=0;initialised=false;
  for(const k of Object.keys(calibCache))delete calibCache[k];
  _calibLogCounter=0;
}
module.exports={runAdvComputeEngine:runAdvComputeEngineWithInit,resetAdvComputeState};