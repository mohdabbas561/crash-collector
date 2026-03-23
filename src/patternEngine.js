'use strict';
// patternEngine.js — Pattern Engine with Predictive Early-Warning Layer
// ======================================================================
// DROP-IN REPLACEMENT. Same exports, same DB, same ENGINE_ID, same locks.
// PREDICTIVE UPGRADE: detects regime shifts BEFORE they start.
// Every reactive rf.regime check replaced with rf.predictedNextRegime.

const {
  getRounds, savePrediction, getPredictions,
  saveLockedPatternPreds, getLockedPatternPreds,
} = require('./db');

const ENGINE_ID  = 'pattern';
const MIN_ROUNDS = 50, MIN_HITS = 8, MIN_GAPS = 6, STALE_THRESHOLD = 50000;

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

const state = { windows:{}, savedSet:null, lastRoundId:0 };
let cachedRounds=[], cachedRoundsLastId=0, initialised=false;

function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}
function mean(arr){return arr.length?arr.reduce((s,v)=>s+v,0)/arr.length:0;}
function stdDev(arr){
  if(arr.length<2)return 0; const m=mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1));
}
function earlyHitTolerance(maxWidth){return Math.floor(maxWidth/2);}
function sparsePenalty(hits,minFull){return hits>=minFull?1.0:Math.sqrt(Math.max(1,hits)/minFull);}
function bisectLeft(rounds,targetId){
  let lo=0,hi=rounds.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(rounds[mid].roundId<targetId)lo=mid+1;else hi=mid;}
  return lo;
}
function findHitInRange(rounds,fromId,toId,minMult){
  const start=bisectLeft(rounds,fromId);
  for(let i=start;i<rounds.length;i++){
    if(rounds[i].roundId>toId)break;
    if(rounds[i].multiplier>=minMult)return rounds[i];
  }
  return null;
}

// =============================================================================
// === PREDICTIVE CHANGE START ===
// extractPredictiveRunFeatures() — replaces extractRunFeatures()
// Adds forward-looking precursor signals on top of all reactive RLE metrics.
// =============================================================================
function extractPredictiveRunFeatures(rounds, targetMin) {
  const n = rounds.length;
  if (n < 10) return null;

  const runs=[];
  let curHigh=rounds[0].multiplier>=targetMin,curLen=1;
  for(let i=1;i<n;i++){
    const h=rounds[i].multiplier>=targetMin;
    if(h===curHigh){curLen++;}
    else{runs.push({isHigh:curHigh,len:curLen});curHigh=h;curLen=1;}
  }
  runs.push({isHigh:curHigh,len:curLen});

  const highRuns=runs.filter(r=>r.isHigh).map(r=>r.len);
  const lowRuns =runs.filter(r=>!r.isHigh).map(r=>r.len);
  const lastRun=runs[runs.length-1];
  const currentIsHigh=lastRun.isHigh, currentStreakLen=lastRun.len;

  const b2bOccurrences=highRuns.filter(l=>l>=2).length;
  const b2bRate=highRuns.length?b2bOccurrences/highRuns.length:0;
  let b2bContinuationProb=0;
  if(highRuns.length>=5){
    const ext=highRuns.filter(l=>l>=2).reduce((s,l)=>s+l-1,0);
    const tot=highRuns.reduce((s,l)=>s+l,0);
    b2bContinuationProb=tot>0?ext/tot:0;
  }
  const avgHighRunLen=highRuns.length?mean(highRuns):0;
  const maxHighRunLen=highRuns.length?Math.max(...highRuns):0;
  const avgLowRunLen=lowRuns.length?mean(lowRuns):0;
  const maxLowRunLen=lowRuns.length?Math.max(...lowRuns):0;

  // Density windows
  const W5=rounds.slice(-5),W10=rounds.slice(-10),W20=rounds.slice(-20),W50=rounds.slice(-50);
  const ld5 =W5.filter(r=>r.multiplier<targetMin).length/Math.max(1,W5.length);
  const ld10=W10.filter(r=>r.multiplier<targetMin).length/Math.max(1,W10.length);
  const ld20=W20.filter(r=>r.multiplier<targetMin).length/Math.max(1,W20.length);
  const ld50=W50.filter(r=>r.multiplier<targetMin).length/Math.max(1,W50.length);
  const densityTrend=ld10-ld50;
  const globalLowRate=1-rounds.filter(r=>r.multiplier>=targetMin).length/n;

  // Post-cluster gap
  const longLowThresh=Math.max(2,Math.round(avgLowRunLen*1.3));
  const postClusterGaps=[];
  for(let i=0;i<runs.length-1;i++){
    if(!runs[i].isHigh&&runs[i].len>=longLowThresh&&runs[i+1].isHigh)postClusterGaps.push(1);
  }
  const avgPostClusterGap=postClusterGaps.length?mean(postClusterGaps):null;

  // Reactive regime (kept for fallback)
  let regime='NEUTRAL';
  if     (currentIsHigh&&currentStreakLen>=2) regime='B2B';
  else if(currentIsHigh&&runs.length>=2&&!runs[runs.length-2].isHigh
          &&runs[runs.length-2].len<=avgLowRunLen*0.5) regime='HOT_AFTER_SHORT_COLD';
  else if(!currentIsHigh&&currentStreakLen>=avgLowRunLen*1.5) regime='WHITE_CLUSTER';
  else if(!currentIsHigh&&currentStreakLen>=maxLowRunLen*0.8&&maxLowRunLen>2) regime='EXTREME_WHITE';
  else if(b2bRate>0.25&&ld20<globalLowRate*0.7) regime='HOT';
  else if(ld20>globalLowRate*1.3) regime='COLD';

  // === PREDICTIVE SIGNALS ===
  // A: 2nd derivative of low-density (acceleration)
  const lowDensityAccel=(ld5-ld10)-(ld10-ld20);

  // B: streak momentum — rate of change in current low-run vs previous 3
  let streakMomentum=0;
  if(!currentIsHigh&&lowRuns.length>=4){
    const prev3=mean(lowRuns.slice(-4,-1));
    streakMomentum=(currentStreakLen-prev3)/Math.max(1,prev3);
  }

  // C: post-cluster early signal
  let postClusterEarlySignal=false;
  if(!currentIsHigh&&lowRuns.length>=3){
    const recentShortening=lowRuns.slice(-3,-1).every(l=>l<avgLowRunLen);
    postClusterEarlySignal=currentStreakLen>=avgLowRunLen*1.0&&lowDensityAccel<-0.05&&recentShortening;
  }

  // D: b2b precursor
  let b2bPrecursor=false;
  if(lowRuns.length>=2){
    const lastCompletedLow=lowRuns[lowRuns.length-(currentIsHigh?1:2)];
    if(lastCompletedLow!==undefined){
      // For pattern engine, use autocorrelation of gaps as proxy for garchSignal
      let garchProxy=0;
      if(highRuns.length>=5){
        const gm=mean(highRuns),ad=highRuns.map(g=>Math.abs(g-gm));
        let cov=0,vs=0;
        for(let i=1;i<ad.length;i++) cov+=ad[i-1]*ad[i];
        for(const v of ad) vs+=v*v;
        garchProxy=vs>0?cov/vs:0;
      }
      b2bPrecursor=lastCompletedLow<avgLowRunLen*0.60&&garchProxy>0.10;
    }
  }

  // E: Markov transition probability (bigram on runs)
  let markovProbHot=(1-globalLowRate)||0.1;
  if(runs.length>=4){
    const seq=runs.map(r=>r.isHigh?1:0);
    const mat={};
    for(let i=2;i<seq.length;i++){
      const k=`${seq[i-2]},${seq[i-1]}`;
      if(!mat[k]) mat[k]={H:0,L:0};
      if(seq[i]===1) mat[k].H++; else mat[k].L++;
    }
    const last2Key=`${seq[seq.length-2]},${seq[seq.length-1]}`;
    const cell=mat[last2Key];
    if(cell){const tot=cell.H+cell.L;if(tot>=3)markovProbHot=cell.H/tot;}
  }

  // F: composite scores
  const hotScore=clamp(Math.round(
    (postClusterEarlySignal?35:0)+(b2bPrecursor?25:0)+
    (lowDensityAccel<-0.08?20:lowDensityAccel<-0.03?10:0)+
    (streakMomentum<-0.3?15:streakMomentum<-0.1?7:0)+
    (markovProbHot>0.65?15:markovProbHot>0.50?7:0)+
    (b2bRate>0.25?10:b2bRate>0.15?5:0)
  ),0,100);

  const coldScore=clamp(Math.round(
    (streakMomentum>0.4?30:streakMomentum>0.2?15:0)+
    (lowDensityAccel>0.08?25:lowDensityAccel>0.03?12:0)+
    (markovProbHot<0.25?20:markovProbHot<0.40?10:0)+
    (ld20>globalLowRate*1.4?15:ld20>globalLowRate*1.2?7:0)+
    (currentStreakLen>avgLowRunLen*1.2?10:0)
  ),0,100);

  // G: predictedNextRegime + confidence + gap multiplier
  let predictedNextRegime='NEUTRAL',transitionConfidence=0,predictedGapMultiplier=1.0;
  if(hotScore>=55&&hotScore>coldScore+15){
    if(b2bPrecursor||b2bRate>0.20){
      predictedNextRegime='ABOUT_TO_B2B';
      transitionConfidence=clamp(hotScore,55,95);
      predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.55,0.35,0.75);
    } else {
      predictedNextRegime='ABOUT_TO_HOT';
      transitionConfidence=clamp(hotScore,50,88);
      predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.40,0.50,0.80);
    }
  } else if(coldScore>=55&&coldScore>hotScore+15){
    if(currentStreakLen>=avgLowRunLen*1.5||regime==='EXTREME_WHITE'){
      predictedNextRegime='ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence=clamp(coldScore,50,85);
      predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.45,1.20,1.60);
    } else {
      predictedNextRegime='ABOUT_TO_COLD';
      transitionConfidence=clamp(coldScore,45,80);
      predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.30,1.10,1.40);
    }
  } else if(hotScore>=35&&hotScore>coldScore){
    predictedNextRegime='ABOUT_TO_HOT';
    transitionConfidence=clamp(hotScore,30,60);
    predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.25,0.70,0.90);
  } else if(coldScore>=35&&coldScore>hotScore){
    predictedNextRegime='ABOUT_TO_COLD';
    transitionConfidence=clamp(coldScore,30,60);
    predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.20,1.05,1.25);
  }
  // === PREDICTIVE CHANGE END ===

  return {
    runs,highRuns,lowRuns,currentIsHigh,currentStreakLen,regime,
    b2bRate,b2bContinuationProb,avgHighRunLen,maxHighRunLen,
    avgLowRunLen,maxLowRunLen,avgPostClusterGap,
    lowDensity10:ld10,lowDensity20:ld20,lowDensity50:ld50,densityTrend,globalLowRate,
    // Predictive fields
    lowDensityAccel,streakMomentum,postClusterEarlySignal,b2bPrecursor,markovProbHot,
    hotScore,coldScore,predictedNextRegime,transitionConfidence,predictedGapMultiplier,
  };
}
// === PREDICTIVE CHANGE END ===

// Helper: effective regime from predicted or reactive
function pEffective(rf) {
  if(!rf) return 'NEUTRAL';
  return (rf.transitionConfidence??0)>=40 ? rf.predictedNextRegime : rf.regime;
}

function analysePattern(rounds, targetMin) {
  const n=rounds.length;
  if(n<MIN_ROUNDS) return null;

  const W1=20,W2=75,W3=300,W4=500;
  const s1=Math.max(0,n-W1),s2=Math.max(0,n-W2),s3=Math.max(0,n-W3),s4=Math.max(0,n-W4);
  let hits=0,lastIdx=-1,hW1=0,hW2=0,hW3=0,hW4=0;
  const gaps=[];
  const FA=0.15,SA=0.015;
  let emaFast=-1,emaSlow=-1;
  for(let i=0;i<n;i++){
    const isHit=rounds[i].multiplier>=targetMin?1:0;
    if(emaFast<0){emaFast=isHit;emaSlow=isHit;}
    else{emaFast=FA*isHit+(1-FA)*emaFast;emaSlow=SA*isHit+(1-SA)*emaSlow;}
    if(isHit){
      if(lastIdx!==-1)gaps.push(i-lastIdx-1);
      lastIdx=i;hits++;
      if(i>=s1)hW1++;if(i>=s2)hW2++;if(i>=s3)hW3++;if(i>=s4)hW4++;
    }
  }
  if(hits<MIN_HITS||gaps.length<MIN_GAPS) return null;

  const globalRate=hits/n;
  const gapSinceLast=lastIdx===-1?n:n-lastIdx-1;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1,cv=gMean>0?gStd/gMean:1;
  const sg=[...gaps].sort((a,b)=>a-b);
  const mid2=Math.floor(sg.length/2);
  const medianGap=sg.length%2===1?sg[mid2]:(sg[mid2-1]+sg[mid2])/2;

  const dW1=hW1/W1,dW2=hW2/W2,dW3=hW3/Math.min(W3,n),dW4=hW4/Math.min(W4,n);
  const safe=v=>Math.max(-1,Math.min(1,v));

  // === PREDICTIVE CHANGE START ===
  // Extract predictive run features for pattern scoring
  const rf=extractPredictiveRunFeatures(rounds,targetMin);

  const rateClusterScore=safe(
    safe((dW1-globalRate)/Math.max(globalRate,0.001))*0.40+
    safe((dW2-globalRate)/Math.max(globalRate,0.001))*0.25+
    safe((dW3-globalRate)/Math.max(globalRate,0.001))*0.20+
    safe((dW4-globalRate)/Math.max(globalRate,0.001))*0.15
  );

  // RLE cluster scores
  let rlB2bScore=0,rlClusterScore=0;
  if(rf){
    // Use PREDICTIVE signals for scoring, not just reactive regime
    const pnr=pEffective(rf);
    if(pnr==='ABOUT_TO_B2B'||pnr==='B2B')
      rlB2bScore=clamp((rf.b2bContinuationProb||0)*1.8+(rf.hotScore||0)/100*0.5,0,1);
    else if(pnr==='ABOUT_TO_HOT'||pnr==='HOT'||pnr==='HOT_AFTER_SHORT_COLD')
      rlB2bScore=clamp((rf.b2bRate||0)*1.2+(rf.hotScore||0)/100*0.4,0,0.8);
    if(pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER')
      rlClusterScore=clamp((rf.currentStreakLen||0)/Math.max(1,rf.avgLowRunLen||1)*0.7,0,0.9);
    else if(pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE'||pnr==='COLD')
      rlClusterScore=clamp((rf.coldScore||0)/100*0.8,0,0.7);

    // postClusterEarlySignal boosts the composite BEFORE regime is detected
    if(rf.postClusterEarlySignal) rlB2bScore=Math.min(rlB2bScore+0.25,1.0);
    if(rf.b2bPrecursor)           rlB2bScore=Math.min(rlB2bScore+0.20,1.0);
  }
  // === PREDICTIVE CHANGE END ===

  const clusterScore=safe(rateClusterScore*0.35+rlB2bScore*0.40+rlClusterScore*0.25);
  const trendScore=safe((emaSlow>0?(emaFast-emaSlow)/emaSlow:0)*4);

  let varSum=0;
  for(const g of gaps) varSum+=(g-gMean)**2;
  let bestAC=0;
  for(let lag=1;lag<=Math.min(10,gaps.length-1);lag++){
    let cov=0;
    for(let i=lag;i<gaps.length;i++) cov+=(gaps[i-lag]-gMean)*(gaps[i]-gMean);
    const ac=varSum>0?cov/varSum:0;
    if(Math.abs(ac)>Math.abs(bestAC)) bestAC=ac;
  }
  const patternScore=safe(bestAC*0.9);

  const last20=gaps.slice(-Math.min(20,gaps.length));
  const momentum=gMean>0?(gMean-mean(last20))/gMean:0;

  const composite=clamp(
    clusterScore*0.35+trendScore*0.25+patternScore*0.15+
    safe(momentum)*0.15+(rf?safe(rlB2bScore-rlClusterScore)*0.10:0),
    -1,1
  );
  const absComposite=Math.abs(composite);
  const direction=composite>0.08?'bullish':composite<-0.08?'bearish':'neutral';
  const agree=Math.max(
    [clusterScore,trendScore,patternScore].filter(s=>s>0.08).length,
    [clusterScore,trendScore,patternScore].filter(s=>s<-0.08).length
  );
  const sp=sparsePenalty(hits,50);
  const conf=Math.max(20,Math.min(88,Math.round(
    (32+Math.min(18,Math.log2(hits+1)*4)+absComposite*30+(agree-1)*6
      -(cv>1.5?8:cv>1.2?4:0)
      -(gapSinceLast>gMean*2?5:0)
      // === PREDICTIVE CHANGE START ===
      // +15 confidence when high-confidence hot transition detected early
      +((rf&&(rf.transitionConfidence??0)>70&&
         (rf.predictedNextRegime==='ABOUT_TO_B2B'||rf.predictedNextRegime==='ABOUT_TO_HOT'))?15:0)
      // === PREDICTIVE CHANGE END ===
    )*sp
  )));

  return {
    direction,confidence:conf,hits,
    meanGap:Math.round(gMean),medianGap:Math.round(medianGap),
    composite:+composite.toFixed(3),momentum:+momentum.toFixed(3),
    gapSinceLast,clusterScore:+clusterScore.toFixed(3),
    trendScore:+trendScore.toFixed(3),patternScore:+patternScore.toFixed(3),
    cv:+cv.toFixed(2),
    rf: rf ? {
      regime:rf.regime,
      // === PREDICTIVE CHANGE START ===
      predictedNextRegime:rf.predictedNextRegime,
      transitionConfidence:rf.transitionConfidence,
      predictedGapMultiplier:rf.predictedGapMultiplier,
      hotScore:rf.hotScore, coldScore:rf.coldScore,
      postClusterEarlySignal:rf.postClusterEarlySignal,
      b2bPrecursor:rf.b2bPrecursor,
      // === PREDICTIVE CHANGE END ===
      b2bRate:+rf.b2bRate.toFixed(3),
      b2bContinuationProb:+rf.b2bContinuationProb.toFixed(3),
      currentIsHigh:rf.currentIsHigh,currentStreakLen:rf.currentStreakLen,
      avgLowRunLen:+rf.avgLowRunLen.toFixed(2),avgPostClusterGap:rf.avgPostClusterGap,
      lowDensity20:+rf.lowDensity20.toFixed(3),densityTrend:+rf.densityTrend.toFixed(3),
    } : null,
  };
}

function buildWindow(pr, maxWidth, anchorRound) {
  if(!pr) return null;
  const medianGap=pr.medianGap||pr.meanGap||maxWidth;
  const gapSinceLast=pr.gapSinceLast??0;
  const momentum=pr.momentum??0;
  const meanGap=pr.meanGap||medianGap;
  const momentumAdj=Math.max(0.75,Math.min(1.25,1-momentum*0.25));
  const overdueFactor=gapSinceLast>meanGap*1.5?0.80:1.0;
  let expectedGap=Math.max(1,Math.round(medianGap*momentumAdj*overdueFactor));

  const rf=pr.rf;
  if(rf){
    // === PREDICTIVE CHANGE START ===
    // Use predictedNextRegime + predictedGapMultiplier for window placement
    const pnr=pEffective(rf);
    const tc=rf.transitionConfidence??0;
    const mult=rf.predictedGapMultiplier??1.0;

    if(pnr!=='NEUTRAL'&&tc>=30){
      // Blend based on confidence — high confidence = full multiplier
      const blend=clamp((tc-30)/70,0,1);
      const blendedMult=1.0+(mult-1.0)*blend;
      expectedGap=Math.max(1,Math.round(expectedGap*blendedMult));
    } else {
      // Reactive fallback
      switch(rf.regime){
        case 'B2B':            expectedGap=Math.max(1,Math.round(expectedGap*(1-rf.b2bContinuationProb*0.35))); break;
        case 'HOT_AFTER_SHORT_COLD': expectedGap=Math.max(1,Math.round(expectedGap*0.85)); break;
        case 'WHITE_CLUSTER':  expectedGap=rf.avgPostClusterGap!==null
          ?Math.max(1,Math.round(expectedGap*0.5+rf.avgPostClusterGap*0.5))
          :Math.max(1,Math.round(expectedGap*0.88)); break;
        case 'EXTREME_WHITE':  expectedGap=Math.max(1,Math.round(expectedGap*0.70)); break;
        case 'HOT':            expectedGap=Math.max(1,Math.round(expectedGap*(1-(1-rf.lowDensity20)*0.18))); break;
        case 'COLD':           expectedGap=Math.max(1,Math.round(expectedGap*(1+rf.densityTrend*0.12))); break;
      }
    }

    // postClusterEarlySignal fires BEFORE regime reaches WHITE_CLUSTER
    if(rf.postClusterEarlySignal&&rf.avgPostClusterGap!==null&&pnr==='ABOUT_TO_B2B'){
      expectedGap=Math.max(1,Math.round(expectedGap*0.65+rf.avgPostClusterGap*0.35));
    }
    // === PREDICTIVE CHANGE END ===
  }

  const remaining=Math.max(1,expectedGap-gapSinceLast);
  const low=Math.max(1,remaining-Math.floor(maxWidth/2));
  const hi=low+maxWidth-1;

  return {
    lo:anchorRound+low,hi:anchorRound+hi,expectedGap,
    confidence:pr.confidence,direction:pr.direction,
    composite:pr.composite??null,momentum:pr.momentum??null,gapSinceLast,
    eta:{
      low,high:hi,conf:pr.confidence,expectedGap,
      direction:pr.direction,composite:pr.composite??null,
      momentum:pr.momentum??null,gapSinceLast,
      regime:rf?.regime??null,
      // === PREDICTIVE CHANGE START ===
      predictedNextRegime:rf?.predictedNextRegime??null,
      transitionConfidence:rf?.transitionConfidence??0,
      hotScore:rf?.hotScore??0,
      coldScore:rf?.coldScore??0,
      // === PREDICTIVE CHANGE END ===
    },
  };
}

async function getPatternRounds() {
  if(cachedRounds.length===0){
    cachedRounds=await getRounds({limit:100000,order:'ASC'});
    cachedRoundsLastId=cachedRounds.length?cachedRounds[cachedRounds.length-1].roundId:0;
    console.log(`[pattern] loaded ${cachedRounds.length} rounds`);
  } else {
    const nr=await getRounds({limit:5000,minRoundId:cachedRoundsLastId+1});
    if(nr.length){cachedRounds=[...cachedRounds,...nr];cachedRoundsLastId=cachedRounds[cachedRounds.length-1].roundId;}
  }
  return cachedRounds;
}

async function saveOutcome(target,outcome,lo,hi,hitRound,generation){
  const key=`${lo}:${hi}`;
  if(state.savedSet.has(key)) return;
  state.savedSet.add(key);
  try{
    await savePrediction({target:target.label,minMult:target.min,outcome,lo,hi,
      hitRound:hitRound??null,generation:generation??1,source:ENGINE_ID,probW:null});
    console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${lo}–#${hi}${hitRound?` @#${hitRound}`:''}`);
  }catch(e){console.error(`[pattern] save fail:`,e.message);state.savedSet.delete(key);}
}

async function processPatternEngine(rounds,lastRoundId){
  const toSave={};let anyChange=false;
  for(const target of TARGETS){
    const win=state.windows[target.label];
    if(win){
      const{lo,hi,generation}=win;
      const isTooOld=lastRoundId-hi>STALE_THRESHOLD;
      const earlyCheckLo=Math.max(win.roundWhenMade+1,lo-earlyHitTolerance(target.maxWidth));
      const earlyHit=lo>win.roundWhenMade+1&&earlyCheckLo<=lo-1?findHitInRange(rounds,earlyCheckLo,lo-1,target.min):null;
      if(earlyHit){
        await saveOutcome(target,'early',lo,hi,earlyHit.roundId,generation);
        delete state.windows[target.label];anyChange=true;
      } else if(lastRoundId>=hi){
        if(!isTooOld){const hit=findHitInRange(rounds,lo,hi,target.min);await saveOutcome(target,hit?'win':'loss',lo,hi,hit?.roundId??null,generation);}
        delete state.windows[target.label];anyChange=true;
      } else {
        const hit=findHitInRange(rounds,lo,hi,target.min);
        if(hit){await saveOutcome(target,'win',lo,hi,hit.roundId,generation);delete state.windows[target.label];anyChange=true;}
        else{toSave[target.label]=win;continue;}
      }
    }
    const pr=analysePattern(rounds,target.min);
    const pred=buildWindow(pr,target.maxWidth,lastRoundId);
    if(pred){
      const generation=(win?.generation??0)+1;
      state.windows[target.label]={lo:pred.lo,hi:pred.hi,roundWhenMade:lastRoundId,generation,eta:pred.eta};
      toSave[target.label]=state.windows[target.label];anyChange=true;
      // === PREDICTIVE CHANGE START ===
      const pnr=pred.eta.predictedNextRegime??'N/A';
      const tc=pred.eta.transitionConfidence??0;
      console.log(`[pattern] LOCK ${target.label}: #${pred.lo}–#${pred.hi} pnr=${pnr}(${tc}%) dir=${pred.direction} conf=${pred.confidence}%`);
      // === PREDICTIVE CHANGE END ===
    }
  }
  if(anyChange&&Object.keys(toSave).length){
    const payload={};
    for(const[label,w]of Object.entries(toSave)){
      payload[label]={lo:w.lo,hi:w.hi,roundWhenMade:w.roundWhenMade,generation:w.generation,eta:w.eta};
    }
    try{await saveLockedPatternPreds(payload);}
    catch(e){console.error('[pattern] save locked fail:',e.message);}
  }
  return anyChange;
}

async function initialise(){
  if(initialised) return;initialised=true;
  state.savedSet=new Set();state.windows={};
  try{
    const dbLocked=await getLockedPatternPreds();
    for(const[label,p]of Object.entries(dbLocked)){
      if(!p?.lo||!p?.hi) continue;
      state.windows[label]={lo:Number(p.lo),hi:Number(p.hi),roundWhenMade:Number(p.roundWhenMade??p.lo),generation:p.generation??1,eta:p.eta??{}};
    }
    console.log(`[pattern] loaded ${Object.keys(state.windows).length} locked windows`);
  }catch(e){console.error('[pattern] init locked error:',e.message);}
  try{
    const rows=await getPredictions({limit:500000,source:ENGINE_ID});
    for(const r of rows) state.savedSet.add(`${r.lo}:${r.hi}`);
    console.log(`[pattern] pre-warmed savedSet with ${state.savedSet.size} outcomes`);
  }catch(e){console.error('[pattern] init history error:',e.message);}
}

function resetPatternEngineState(){
  console.log('[pattern] reset');
  state.windows={};state.savedSet=null;state.lastRoundId=0;
  cachedRounds=[];cachedRoundsLastId=0;initialised=false;
}

async function runPatternEngine(){
  try{
    await initialise();
    const rounds=await getPatternRounds();
    if(rounds.length<MIN_ROUNDS){console.log(`[pattern] waiting (${rounds.length}/${MIN_ROUNDS})`);return;}
    const lastRoundId=rounds[rounds.length-1].roundId;
    if(lastRoundId<=state.lastRoundId) return;
    state.lastRoundId=lastRoundId;
    const t0=Date.now();
    await processPatternEngine(rounds,lastRoundId);
    console.log(`[pattern] tick done in ${Date.now()-t0}ms — ${Object.keys(state.windows).length} active windows`);
  }catch(e){console.error('[pattern] Fatal:',e.message,e.stack);}
}

module.exports={runPatternEngine,resetPatternEngineState};