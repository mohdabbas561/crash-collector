'use strict';
// advComputeEngine.js — ADV Consensus Engine with Predictive Early-Warning Layer
// ================================================================================
// DROP-IN REPLACEMENT. Same exports, DB, engine IDs, consensus algorithm.
// PREDICTIVE UPGRADE: all 13 algos now use predictedNextRegime BEFORE it starts.

const {
  getRounds,savePrediction,getPredictions,
  saveLockedAdvPreds,saveLockedConsensusPreds,
  getLockedAdvPreds,getLockedConsensusPreds,
} = require('./db');
const {bustLockedCache}=require('./advResolutionEngine');

const ENGINE_IDS=[];

const TARGETS=[
  {label:'5x',   min:5,   maxWidth:3 },{label:'10x',  min:10,  maxWidth:5 },
  {label:'20x',  min:20,  maxWidth:7 },{label:'50x',  min:50,  maxWidth:12},
  {label:'100x', min:100, maxWidth:18},{label:'250x', min:250, maxWidth:25},
  {label:'500x', min:500, maxWidth:35},{label:'1000x',min:1000,maxWidth:50},
];

const savedSets={};
for(const id of [...ENGINE_IDS,'consensus']) savedSets[id]=new Set();
let cachedRounds=[],cachedRoundsLastId=0,initialised=false;

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

// =============================================================================
// === PREDICTIVE CHANGE START ===
// extractPredictiveStreakFeatures — same predictive layer as ngComputeEngine
// Fully self-contained copy — no shared state with ng engine.
// =============================================================================
function extractPredictiveStreakFeatures(rounds,targetMin){
  const n=rounds.length;if(n<10)return null;
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
  const b2bOcc=highRuns.filter(l=>l>=2).length;
  const b2bRate=highRuns.length?b2bOcc/highRuns.length:0;
  let b2bCont=0;
  if(highRuns.length>=5){const ext=highRuns.filter(l=>l>=2).reduce((s,l)=>s+l-1,0);const tot=highRuns.reduce((s,l)=>s+l,0);b2bCont=tot>0?ext/tot:0;}
  const avgHighRunLen=highRuns.length?mean(highRuns):0;
  const avgLowRunLen=lowRuns.length?mean(lowRuns):0;
  const maxLowRunLen=lowRuns.length?Math.max(...lowRuns):0;
  const W5=rounds.slice(-5),W10=rounds.slice(-10),W20=rounds.slice(-20),W50=rounds.slice(-50);
  const ld5=W5.filter(r=>r.multiplier<targetMin).length/Math.max(1,W5.length);
  const ld10=W10.filter(r=>r.multiplier<targetMin).length/Math.max(1,W10.length);
  const ld20=W20.filter(r=>r.multiplier<targetMin).length/Math.max(1,W20.length);
  const ld50=W50.filter(r=>r.multiplier<targetMin).length/Math.max(1,W50.length);
  const densityTrend=ld10-ld50;
  const globalLowRate=1-rounds.filter(r=>r.multiplier>=targetMin).length/n;
  let garchSignal=0;
  const{gaps}=computeGaps(rounds,targetMin);
  if(gaps.length>=10){const gm=mean(gaps),ad=gaps.map(g=>Math.abs(g-gm));let cov=0,vs=0;for(let i=1;i<ad.length;i++)cov+=ad[i-1]*ad[i];for(const v of ad)vs+=v*v;garchSignal=vs>0?cov/vs:0;}
  const longLowThresh=Math.max(2,Math.round(avgLowRunLen*1.3));
  const pcGaps=[];
  for(let i=0;i<runs.length-1;i++){if(!runs[i].isHigh&&runs[i].len>=longLowThresh&&runs[i+1].isHigh)pcGaps.push(1);}
  const avgPostClusterGap=pcGaps.length?mean(pcGaps):null;
  let regime='NEUTRAL';
  if(currentIsHigh&&currentStreakLen>=2) regime='B2B';
  else if(currentIsHigh&&runs.length>=2&&!runs[runs.length-2].isHigh&&runs[runs.length-2].len<=avgLowRunLen*0.5) regime='HOT_AFTER_SHORT_COLD';
  else if(!currentIsHigh&&currentStreakLen>=avgLowRunLen*1.5) regime='WHITE_CLUSTER';
  else if(!currentIsHigh&&currentStreakLen>=maxLowRunLen*0.8&&maxLowRunLen>2) regime='EXTREME_WHITE';
  else if(b2bRate>0.25&&ld20<globalLowRate*0.7) regime='HOT';
  else if(ld20>globalLowRate*1.3) regime='COLD';
  // Predictive signals
  const lowDensityAccel=(ld5-ld10)-(ld10-ld20);
  let streakMomentum=0;
  if(!currentIsHigh&&lowRuns.length>=4){const p3=mean(lowRuns.slice(-4,-1));streakMomentum=(currentStreakLen-p3)/Math.max(1,p3);}
  let postClusterEarlySignal=false;
  if(!currentIsHigh&&lowRuns.length>=3){
    const rs=lowRuns.slice(-3,-1).every(l=>l<avgLowRunLen);
    postClusterEarlySignal=currentStreakLen>=avgLowRunLen*1.0&&lowDensityAccel<-0.05&&rs;
  }
  let b2bPrecursor=false;
  if(lowRuns.length>=2){const lcl=lowRuns[lowRuns.length-(currentIsHigh?1:2)];if(lcl!==undefined) b2bPrecursor=lcl<avgLowRunLen*0.60&&garchSignal>0.15;}
  let markovProbHot=(1-globalLowRate)||0.1;
  if(runs.length>=4){
    const seq=runs.map(r=>r.isHigh?1:0);const mat={};
    for(let i=2;i<seq.length;i++){const k=`${seq[i-2]},${seq[i-1]}`;if(!mat[k])mat[k]={H:0,L:0};if(seq[i]===1)mat[k].H++;else mat[k].L++;}
    const lk=`${seq[seq.length-2]},${seq[seq.length-1]}`;const cell=mat[lk];
    if(cell){const tot=cell.H+cell.L;if(tot>=3)markovProbHot=cell.H/tot;}
  }
  const hotScore=clamp(Math.round((postClusterEarlySignal?35:0)+(b2bPrecursor?25:0)+(lowDensityAccel<-0.08?20:lowDensityAccel<-0.03?10:0)+(streakMomentum<-0.3?15:streakMomentum<-0.1?7:0)+(markovProbHot>0.65?15:markovProbHot>0.50?7:0)+(b2bRate>0.25?10:b2bRate>0.15?5:0)),0,100);
  const coldScore=clamp(Math.round((streakMomentum>0.4?30:streakMomentum>0.2?15:0)+(lowDensityAccel>0.08?25:lowDensityAccel>0.03?12:0)+(markovProbHot<0.25?20:markovProbHot<0.40?10:0)+(ld20>globalLowRate*1.4?15:ld20>globalLowRate*1.2?7:0)+(currentStreakLen>avgLowRunLen*1.2?10:0)),0,100);
  let predictedNextRegime='NEUTRAL',transitionConfidence=0,predictedGapMultiplier=1.0;
  if(hotScore>=55&&hotScore>coldScore+15){
    if(b2bPrecursor||b2bRate>0.20){predictedNextRegime='ABOUT_TO_B2B';transitionConfidence=clamp(hotScore,55,95);predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.55,0.35,0.75);}
    else{predictedNextRegime='ABOUT_TO_HOT';transitionConfidence=clamp(hotScore,50,88);predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.40,0.50,0.80);}
  } else if(coldScore>=55&&coldScore>hotScore+15){
    if(currentStreakLen>=avgLowRunLen*1.5||regime==='EXTREME_WHITE'){predictedNextRegime='ABOUT_TO_WHITE_CLUSTER';transitionConfidence=clamp(coldScore,50,85);predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.45,1.20,1.60);}
    else{predictedNextRegime='ABOUT_TO_COLD';transitionConfidence=clamp(coldScore,45,80);predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.30,1.10,1.40);}
  } else if(hotScore>=35&&hotScore>coldScore){
    predictedNextRegime='ABOUT_TO_HOT';transitionConfidence=clamp(hotScore,30,60);predictedGapMultiplier=clamp(1.0-(hotScore/100)*0.25,0.70,0.90);
  } else if(coldScore>=35&&coldScore>hotScore){
    predictedNextRegime='ABOUT_TO_COLD';transitionConfidence=clamp(coldScore,30,60);predictedGapMultiplier=clamp(1.0+(coldScore/100)*0.20,1.05,1.25);
  }
  return{
    currentIsHigh,currentStreakLen,regime,b2bRate,b2bContinuationProb:b2bCont,
    avgHighRunLen,avgLowRunLen,maxLowRunLen,avgPostClusterGap,
    lowDensity10:ld10,lowDensity20:ld20,lowDensity50:ld50,densityTrend,globalLowRate,
    garchSignal,highRuns,lowRuns,
    lowDensityAccel,streakMomentum,postClusterEarlySignal,b2bPrecursor,markovProbHot,
    hotScore,coldScore,predictedNextRegime,transitionConfidence,predictedGapMultiplier,
  };
}
// === PREDICTIVE CHANGE END ===

// === PREDICTIVE CHANGE START ===
// applyStreakAdj — uses predictedNextRegime + predictedGapMultiplier
// === PREDICTIVE CHANGE END ===
function applyStreakAdj(expectedGap,sf){
  if(!sf) return expectedGap;
  const pnr=sf.predictedNextRegime,mult=sf.predictedGapMultiplier??1.0,tc=sf.transitionConfidence??0;
  if(pnr!=='NEUTRAL'&&tc>=30){
    const blend=clamp((tc-30)/70,0,1);
    return Math.max(1,Math.round(expectedGap*(1.0+(mult-1.0)*blend)));
  }
  let adj=expectedGap;
  switch(sf.regime){
    case 'B2B':adj=Math.round(adj*(1-sf.b2bContinuationProb*0.35));break;
    case 'HOT_AFTER_SHORT_COLD':adj=Math.round(adj*0.85);break;
    case 'WHITE_CLUSTER':adj=sf.avgPostClusterGap!==null?Math.round(adj*0.5+sf.avgPostClusterGap*0.5):Math.round(adj*0.90);break;
    case 'EXTREME_WHITE':adj=Math.round(adj*0.70);break;
    case 'HOT':adj=Math.round(adj*(1-(1-sf.lowDensity20)*0.18));break;
    case 'COLD':adj=Math.round(adj*(1+sf.densityTrend*0.12));break;
    default:adj=Math.round(adj*(1-sf.densityTrend*0.07));
  }
  return Math.max(1,adj);
}

function effectiveRegime(sf){
  if(!sf)return'NEUTRAL';
  return(sf.transitionConfidence??0)>=40?sf.predictedNextRegime:sf.regime;
}

// All 13 engine algorithms — each now accepts sf and uses predictive regime
function runLSTM(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  const DECAY=0.97;let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/wS,gMean=mean(gaps);
  let raw=Math.max(1,Math.round(ewaMean*0.70+gMean*0.30));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  // === PREDICTIVE CHANGE START ===
  // +15 conf bonus when high-confidence hot predicted
  const tcBonus=(sf&&(sf.transitionConfidence??0)>70&&(sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT'))?15:0;
  // === PREDICTIVE CHANGE END ===
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(85-30*(stdDev(gaps)/(ewaMean||1)))+tcBonus,30,96)};
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
  const aw=target.maxWidth;
  const tcBonus=(sf&&(sf.transitionConfidence??0)>70&&(sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT'))?15:0;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(hrGlobal,aw),conf:clamp(Math.round(80-22*(stdDev(gaps)/(gMean||1)))+tcBonus,30,96)};
}

function runRF(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10)return null;
  const N_BOOT=64,n=gaps.length;
  const primes=[3,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317];
  const bootMedians=[];
  for(let b=0;b<N_BOOT;b++){
    const offset=(b*11+5)%n,prime=primes[b%primes.length],sample=[];
    for(let i=0;i<n;i++) sample.push(gaps[(offset+i*prime)%n]);
    sample.sort((a,b)=>a-b);bootMedians.push(sample[Math.floor(n/2)]);
  }
  bootMedians.sort((a,b)=>a-b);
  let raw=Math.max(1,weibullSkew(pctile(bootMedians,0.50),pctile(bootMedians,0.75)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(52+gaps.length*0.2),40,88)};
}

function runOLS(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10)return null;
  const lin=olsLinear(gaps),gMean=mean(gaps);
  const rawGap=Math.max(1,Math.round(lin.a+lin.b*gaps.length));
  let raw=Math.max(1,Math.round(rawGap*lin.r2+gMean*(1-lin.r2)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(35+lin.r2*42+gaps.length*0.15),30,88)};
}

function runCatBoost(rounds,target,sf){
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8)return null;
  const hrGlobal=(gaps.length+1)/(rounds.length+2);
  const cn=cusumNorm(rounds,target.min);
  const cusumRegime=cn>1.96?'HOT':cn>1.36?'WARM':cn<-1.96?'COLD':'NEUTRAL';
  const hr300=(rounds.slice(-300).filter(r=>r.multiplier>=target.min).length+1)/302;
  // === PREDICTIVE CHANGE START ===
  const pnr=effectiveRegime(sf);
  const rlHot=pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT'||pnr==='B2B'||pnr==='HOT'||pnr==='HOT_AFTER_SHORT_COLD';
  const rlCold=pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD'||pnr==='WHITE_CLUSTER'||pnr==='EXTREME_WHITE'||pnr==='COLD';
  // === PREDICTIVE CHANGE END ===
  const blend=cusumRegime==='HOT'?0.55:cusumRegime==='WARM'?0.35:cusumRegime==='COLD'?0.20:rlHot?0.50:rlCold?0.12:0.12;
  const pBlend=hrGlobal*(1-blend)+hr300*blend;
  let raw=Math.max(1,Math.round((1-pBlend)/pBlend));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(pBlend,aw),conf:clamp(Math.round(48+gaps.length*0.25),38,91)};
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
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(55+gaps.length*0.2),25,88)};
}

function runMarkov(rounds,target,sf){
  if(rounds.length<30)return null;
  const{tL,tM,tH}=getDynamicBuckets(rounds);
  const bucket=m=>m>=tH?'X':m>=tM?'H':m>=tL?'M':'L';
  const states=['L','M','H','X'],mat1={};
  for(const s of states){mat1[s]={};for(const t of states)mat1[s][t]=0;}
  for(let i=1;i<rounds.length;i++) mat1[bucket(rounds[i-1].multiplier)][bucket(rounds[i].multiplier)]++;
  const cs=bucket(rounds[rounds.length-1]?.multiplier??1);
  const tot1=states.reduce((s,t)=>s+mat1[cs][t],0);
  const prob={};states.forEach(t=>{prob[t]=tot1?mat1[cs][t]/tot1:0.25;});
  let pHit=target.min>=tH?prob['X']:target.min>=tM?prob['H']+prob['X']:target.min>=tL?prob['M']+prob['H']+prob['X']:1-prob['L']*0.3;
  pHit=pHit*0.65+(rounds.filter(r=>r.multiplier>=target.min).length/rounds.length)*0.35;
  // === PREDICTIVE CHANGE START ===
  if(sf&&(sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT')&&(sf.transitionConfidence??0)>40){
    pHit=Math.min(0.99,pHit*(1+(sf.b2bContinuationProb||0)*0.5));
  }
  // === PREDICTIVE CHANGE END ===
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
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(62+gaps.length*0.15),30,90)};
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
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(mu,aw),conf:clamp(Math.round(40+rounds.length*0.06+certainty*20),35,90)};
}

function runSHA256(rounds,target,sf){
  if(rounds.length<20)return null;
  const obsP=(rounds.filter(r=>r.multiplier>=target.min).length+1)/(rounds.length+2);
  const trust=clamp((rounds.length-20)/180,0,1);
  let raw=Math.max(1,Math.round((1/obsP)*trust+(target.min/(1-0.01))*(1-trust)));
  // === PREDICTIVE CHANGE START ===
  if(sf&&sf.highRuns&&sf.highRuns.length>=5){
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    const hotW=(sf.hotScore??0)/100; // continuous, not binary
    const streakBias=(1-normHrEnt)*hotW*0.12;
    raw=Math.max(1,Math.round(raw*(1-streakBias)));
  }
  // === PREDICTIVE CHANGE END ===
  raw=applyStreakAdj(raw,sf);
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(obsP,aw),conf:clamp(Math.round(55+trust*15),40,90)};
}

function runMersenne(rounds,target,sf){
  if(rounds.length<50)return null;
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  let raw=Math.max(1,Math.round(median(gaps)*0.6+mean(gaps)*0.4));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(42+gaps.length*0.15),30,85)};
}

function runLCG(rounds,target,sf){
  if(rounds.length<30)return null;
  const{gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5)return null;
  let raw=Math.max(1,Math.round(median(gaps)));
  raw=applyStreakAdj(raw,sf);
  const aw=target.maxWidth;
  return{...placeWindow(raw,currentGap,aw),expectedGap:raw,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(25+gaps.length*0.15),20,80)};
}

const ALGOS={
  lstm:runLSTM,xgb:runXGBoost,rf:runRF,ols:runOLS,cat:runCatBoost,
  hardgap:runHardGap,softgap:runSoftGap,markov:runMarkov,
  percentile:runPercentile,bayes:runBayes,sha256:runSHA256,mt:runMersenne,lcg:runLCG,
};

async function getComputeRounds(){
  if(cachedRounds.length===0){
    cachedRounds=await getRounds({limit:100000,order:'ASC'});
    cachedRoundsLastId=cachedRounds.length?cachedRounds[cachedRounds.length-1].roundId:0;
    console.log(`[advCompute] loaded ${cachedRounds.length} rounds`);
  } else {
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

function computeConsensus(allResults,lastRoundId){
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
    consensus[target.label]={lo:bestLo,hi:bestHi,engineCount:bestGroup.length,engines:bestGroup.map(w=>w.engineId)};
  }
  return consensus;
}

const windows={};
for(const id of [...ENGINE_IDS,'consensus']){windows[id]={};}

async function runAdvComputeEngine(){
  try{
    const rounds=await getComputeRounds();
    if(rounds.length<50)return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // === PREDICTIVE CHANGE START ===
    const streakFeatures={};
    for(const target of TARGETS){
      try{streakFeatures[target.label]=extractPredictiveStreakFeatures(rounds,target.min);}
      catch(e){streakFeatures[target.label]=null;}
    }
    // === PREDICTIVE CHANGE END ===

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

    const consensus=computeConsensus(allResults,lastRoundId);
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
    if(Object.keys(consPayload).length) {
      await saveLockedConsensusPreds(consPayload);
      // Only log targets where window was freshly created this tick (lo === lastRoundId + something new)
      const newWins = Object.keys(consPayload).filter(t => consPayload[t].roundWhenMade === lastRoundId);
      if(newWins.length) {
        const targets = newWins.map(t => `${t}:#${consPayload[t].lo}-#${consPayload[t].hi}`).join(' ');
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
    const total=allIds.reduce((s,id)=>s+savedSets[id].size,0);
    console.log(`[advCompute] pre-warmed savedSets with ${total} outcomes`);
  }catch(e){console.error('[advCompute] init history error:',e.message);}
}

const _origRun=runAdvComputeEngine;
async function runAdvComputeEngineWithInit(){await initAdvCompute();await _origRun();}
function resetAdvComputeState(){
  for(const id of [...ENGINE_IDS,'consensus']){windows[id]={};savedSets[id]=new Set();}
  cachedRounds=[];cachedRoundsLastId=0;initialised=false;
}
module.exports={runAdvComputeEngine:runAdvComputeEngineWithInit,resetAdvComputeState};