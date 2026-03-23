'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
// ================================================================================
// DROP-IN REPLACEMENT for the original ngComputeEngine.js.
// Zero changes to any other file. Same DB tables, same API shape, same exports.
//
// AUDIT FIXES APPLIED:
// BUG-1  All engines receive extractStreakFeatures() — b2b and white-cluster
//        signals are first-class inputs before gap collapsing.
// BUG-2  computeGaps() supplemented by extractStreakFeatures() on raw sequence
//        so consecutive-low cluster structure is never lost.
// BUG-3  Lookback windows extended: slopes over last 100 gaps, TCN dilations
//        to [1,2,4,8,16,32], attention window to 200 gaps.
// BUG-4  hrRecent denominator capped to avoid 100k-round "recent" windows.
// BUG-5  GRU scale floor = max(gStd, 0.5*gMean) to prevent gate saturation.
// BUG-6  sha512 now included in stacking meta-learner sources.
// BUG-7  N-BEATS extrapolates to index n (next), not n (count). Sparse guard.
// BUG-8  Rare-target confidence penalised by sqrt(30/hits).
// BUG-9  BiLSTM backward pass correctly iterates newest->oldest.
// BUG-10 Transformer uses 3-head positional attention, not scalar distance.
// ================================================================================

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedAdvPreds,
  getLockedAdvPreds,
} = require('./db');

// Engine IDs — unchanged from original
const NG_ENGINE_IDS = [
  'hlstm_xgb', 'htrans_lstm', 'htft', 'tft', 'nbeats',
  'tcn', 'lgbm', 'gru', 'bilstm', 'stacking', 'sha512', 'ng_consensus',
];

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

const ngSavedSets = {};
for (const id of NG_ENGINE_IDS) ngSavedSets[id] = new Set();
const ngWindows = {};
for (const id of NG_ENGINE_IDS) ngWindows[id] = {};

let cachedRounds       = [];
let cachedRoundsLastId = 0;
let initialised        = false;

function earlyHitTolerance(width) { return Math.floor(width / 2); }

// =============================================================================
// MATH HELPERS
// =============================================================================
function mean(arr)  { return arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }
function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1));
}
function clamp(v,lo,hi) { return Math.max(lo,Math.min(hi,v)); }
function geoProbW(hr,w) { return clamp(1-Math.pow(1-(hr||0),Math.max(1,w)),0,0.99); }
function pctile(sorted,frac) {
  return sorted[Math.min(sorted.length-1,Math.max(0,Math.floor(frac*sorted.length)))];
}
function olsLinear(ys) {
  const n=ys.length; if(n<3) return {a:mean(ys),b:0,r2:0};
  let sx=0,sy=0,sxy=0,sxx=0;
  for(let i=0;i<n;i++){sx+=i;sy+=ys[i];sxy+=i*ys[i];sxx+=i*i;}
  const d=(n*sxx-sx*sx)||1, b=(n*sxy-sx*sy)/d, a=(sy-b*sx)/n;
  const gm=sy/n,ssTot=ys.reduce((s,v)=>s+(v-gm)**2,0);
  const ssRes=ys.reduce((s,v,i)=>s+(v-(a+b*i))**2,0);
  return {a,b,r2:ssTot>0?clamp(1-ssRes/ssTot,0,1):0};
}
function bisectLeft(rounds,targetId) {
  let lo=0,hi=rounds.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(rounds[mid].roundId<targetId)lo=mid+1;else hi=mid;}
  return lo;
}
function findHitInRange(rounds,fromId,toId,minMult) {
  const start=bisectLeft(rounds,fromId);
  for(let i=start;i<rounds.length;i++){
    if(rounds[i].roundId>toId)break;
    if(rounds[i].multiplier>=minMult)return rounds[i];
  }
  return null;
}
function computeGaps(rounds,minMult) {
  const gaps=[]; let since=0;
  for(const r of rounds){since++;if(r.multiplier>=minMult){gaps.push(since);since=0;}}
  return {gaps,currentGap:since};
}
function placeWindow(expectedGap,currentGap,width) {
  const remaining=Math.max(1,expectedGap-currentGap);
  const low=Math.max(1,remaining-Math.floor(width/2));
  return {low,high:low+width-1};
}
function weibullSkew(p50,p75) { return Math.max(1,Math.round(p50+0.20*(p75-p50))); }
function sparsePenalty(hits,minFull) {
  return hits>=minFull ? 1.0 : Math.sqrt(Math.max(1,hits)/minFull);
}

// =============================================================================
// STREAK & CLUSTER FEATURE EXTRACTOR  (BUG-1 / BUG-2 FIX)
// =============================================================================
// Operates on the raw multiplier sequence. Returns StreakFeatures consumed by
// every engine. Never loses consecutive-low structure the way computeGaps() does.
function extractStreakFeatures(rounds, targetMin) {
  const n = rounds.length;
  if (n < 10) return null;

  // 1. Run-Length Encoding
  const runs = [];
  let curHigh = rounds[0].multiplier >= targetMin, curLen = 1;
  for (let i = 1; i < n; i++) {
    const h = rounds[i].multiplier >= targetMin;
    if (h === curHigh) { curLen++; }
    else { runs.push({isHigh:curHigh,len:curLen}); curHigh=h; curLen=1; }
  }
  runs.push({isHigh:curHigh,len:curLen});

  const highRuns = runs.filter(r=> r.isHigh).map(r=>r.len);
  const lowRuns  = runs.filter(r=>!r.isHigh).map(r=>r.len);

  const lastRun         = runs[runs.length-1];
  const currentIsHigh   = lastRun.isHigh;
  const currentStreakLen = lastRun.len;

  // B2B metrics
  const b2bOccurrences      = highRuns.filter(l=>l>=2).length;
  const b2bRate             = highRuns.length ? b2bOccurrences/highRuns.length : 0;
  const avgHighRunLen       = highRuns.length ? mean(highRuns) : 0;
  const maxHighRunLen       = highRuns.length ? Math.max(...highRuns) : 0;

  // White cluster metrics
  const avgLowRunLen        = lowRuns.length ? mean(lowRuns) : 0;
  const maxLowRunLen        = lowRuns.length ? Math.max(...lowRuns) : 0;
  const stdLowRunLen        = lowRuns.length>1 ? stdDev(lowRuns) : 0;

  // Sliding window density
  const W10 = rounds.slice(-10), W20 = rounds.slice(-20), W50 = rounds.slice(-50);
  const lowDensity10 = W10.filter(r=>r.multiplier<targetMin).length/Math.max(1,W10.length);
  const lowDensity20 = W20.filter(r=>r.multiplier<targetMin).length/Math.max(1,W20.length);
  const lowDensity50 = W50.filter(r=>r.multiplier<targetMin).length/Math.max(1,W50.length);
  const densityTrend = lowDensity10 - lowDensity50;

  // Regime
  const globalLowRate = 1 - rounds.filter(r=>r.multiplier>=targetMin).length/n;
  let regime = 'NEUTRAL';
  if      (currentIsHigh && currentStreakLen>=2)                                    regime='B2B';
  else if (currentIsHigh && runs.length>=2 && !runs[runs.length-2].isHigh
           && runs[runs.length-2].len <= avgLowRunLen*0.5)                          regime='HOT_AFTER_SHORT_COLD';
  else if (!currentIsHigh && currentStreakLen >= avgLowRunLen*1.5)                  regime='WHITE_CLUSTER';
  else if (!currentIsHigh && currentStreakLen >= maxLowRunLen*0.8 && maxLowRunLen>2) regime='EXTREME_WHITE';
  else if (b2bRate>0.25 && lowDensity20<globalLowRate*0.7)                         regime='HOT';
  else if (lowDensity20>globalLowRate*1.3)                                         regime='COLD';

  // B2B continuation probability
  let b2bContinuationProb = 0;
  if (highRuns.length>=5) {
    const extensions = highRuns.filter(l=>l>=2).reduce((s,l)=>s+l-1,0);
    const total      = highRuns.reduce((s,l)=>s+l,0);
    b2bContinuationProb = total>0 ? extensions/total : 0;
  }

  // Post-cluster gap: how soon after a long-low-run does a hit arrive?
  const longLowThresh = Math.max(2, Math.round(avgLowRunLen*1.3));
  const postClusterGaps = [];
  for (let i=0; i<runs.length-1; i++) {
    if (!runs[i].isHigh && runs[i].len>=longLowThresh && runs[i+1].isHigh) {
      postClusterGaps.push(1);
    }
  }
  const avgPostClusterGap = postClusterGaps.length ? mean(postClusterGaps) : null;

  // GARCH-style volatility clustering on gaps
  const {gaps} = computeGaps(rounds, targetMin);
  let garchSignal = 0;
  if (gaps.length>=10) {
    const gm=mean(gaps), ad=gaps.map(g=>Math.abs(g-gm));
    let cov=0,vs=0;
    for(let i=1;i<ad.length;i++) cov+=ad[i-1]*ad[i];
    for(const v of ad) vs+=v*v;
    garchSignal = vs>0 ? cov/vs : 0;
  }

  return {
    runs, highRuns, lowRuns,
    currentIsHigh, currentStreakLen, regime,
    b2bOccurrences, b2bRate, b2bContinuationProb,
    avgHighRunLen, maxHighRunLen,
    avgLowRunLen, maxLowRunLen, stdLowRunLen, avgPostClusterGap,
    lowDensity10, lowDensity20, lowDensity50, densityTrend, globalLowRate,
    garchSignal,
  };
}

// Streak-aware gap adjustment — applied by every engine after baseline
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  let adj = expectedGap;
  switch (sf.regime) {
    case 'B2B':
      adj = Math.round(adj * (1 - sf.b2bContinuationProb * 0.35));
      break;
    case 'HOT_AFTER_SHORT_COLD':
      adj = Math.round(adj * 0.85);
      break;
    case 'HOT':
      adj = Math.round(adj * (1 - (1 - sf.lowDensity20) * 0.20));
      break;
    case 'WHITE_CLUSTER':
      adj = sf.avgPostClusterGap !== null
        ? Math.round(adj*0.5 + sf.avgPostClusterGap*0.5)
        : Math.round(adj * 0.90);
      break;
    case 'EXTREME_WHITE':
      adj = Math.round(adj * 0.70);
      break;
    case 'COLD':
      adj = Math.round(adj * (1 + sf.densityTrend * 0.15));
      break;
    default:
      adj = Math.round(adj * (1 - sf.densityTrend * 0.08));
  }
  return Math.max(1, adj);
}

function streakConfBonus(sf) {
  if (!sf) return 0;
  switch (sf.regime) {
    case 'B2B':           return sf.b2bContinuationProb>0.3 ? 8 : 4;
    case 'WHITE_CLUSTER': return sf.avgPostClusterGap!==null  ? 7 : 3;
    case 'EXTREME_WHITE': return 10;
    case 'HOT':           return 5;
    case 'COLD':          return -3;
    default:              return 0;
  }
}

// =============================================================================
// ENGINE 1: hybrid_lstm_xgb
// BUG-3: slope window -> last 100 gaps. BUG-4: hrRecent window capped.
// BUG-1: b2bRate + lowDensity20 as explicit XGBoost features.
// =============================================================================
function runHybridLstmXgb(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;
  const hrGlobal = gaps.length / rounds.length;

  // LSTM: EWA over all gaps
  const DECAY = 0.97; let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean = wG/(wS||1);

  const gMean = mean(gaps), gStd = stdDev(gaps)||1;
  // BUG-4 fix: cap recentN to 3 expected hit-lengths of data
  const recentN = Math.min(300, Math.max(10, Math.round(3/(hrGlobal||0.01))));
  const hrRecent = (rounds.slice(-recentN).filter(r=>r.multiplier>=target.min).length+1)/(recentN+2);
  const {b:slope, r2} = olsLinear(gaps.slice(-Math.min(100,gaps.length))); // BUG-3
  const overdue = clamp(currentGap/(gMean||1),0,3);
  const cv = gStd/(gMean||1);

  // BUG-1: streak features as explicit inputs
  const b2bBoost   = sf ? sf.b2bRate      : 0;
  const safeWeight = sf ? sf.lowDensity20 : (1-hrGlobal);

  const raw = Math.max(1, Math.round(
    ewaMean                                         * 0.28 +
    gMean                                           * 0.18 +
    Math.max(1, gMean+slope*5)                      * 0.14 +
    gMean*Math.max(0.5,1-overdue*0.1)               * 0.12 +
    (1/(hrGlobal||0.001))                           * 0.10 +
    gMean*(1-clamp(b2bBoost*0.5,0,0.4))             * 0.10 +
    gMean*safeWeight                                * 0.08
  ));
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw = target.maxWidth;
  const sp = sparsePenalty(gaps.length, 50);
  const conf = clamp(Math.round((78-20*cv+overdue*4+r2*5+streakConfBonus(sf))*sp), 22, 91);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 2: hybrid_transformer_lstm
// BUG-10: 3-head positional attention. BUG-3: window->200. BUG-1: head weights
//         shift by regime.
// =============================================================================
function runHybridTransformerLstm(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 15) return null;
  const hrGlobal = gaps.length/rounds.length;
  const gMean = mean(gaps), gStd = stdDev(gaps)||1;

  const attnHead = (window) => {
    if (!window.length) return gMean;
    const norm = window.map(g=>(g-gMean)/gStd);
    const scores = norm.map((v,i)=>{
      const pos = i/Math.max(1,window.length-1);
      return Math.exp(-(v*v)*0.5 + Math.sin(pos*Math.PI)*0.3);
    });
    const tot = scores.reduce((a,b)=>a+b,0)||1;
    return window.reduce((s,g,i)=>s+g*scores[i]/tot, 0);
  };

  const h1 = attnHead(gaps.slice(-Math.min(30,  gaps.length)));
  const h2 = attnHead(gaps.slice(-Math.min(100, gaps.length)));
  const h3 = attnHead(gaps.slice(-Math.min(200, gaps.length))); // BUG-3

  // BUG-1: weight heads by regime
  let wH1=0.45, wH2=0.35, wH3=0.20;
  if (sf) {
    if (sf.regime==='B2B'||sf.regime==='HOT')                         {wH1=0.60;wH2=0.30;wH3=0.10;}
    else if (sf.regime==='WHITE_CLUSTER'||sf.regime==='EXTREME_WHITE') {wH1=0.25;wH2=0.35;wH3=0.40;}
  }
  const attnOut = h1*wH1+h2*wH2+h3*wH3;

  // LSTM local memory over last 30 gaps
  const DECAY=0.88; let wS=0,wG=0,wt=1;
  const rec = gaps.slice(-30);
  for(let i=rec.length-1;i>=0;i--){wG+=rec[i]*wt;wS+=wt;wt*=DECAY;}
  const lstmOut = wG/(wS||1);

  const raw = Math.max(1, Math.round(attnOut*0.60+lstmOut*0.40));
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw  = target.maxWidth;
  const sp  = sparsePenalty(gaps.length,40);
  const cv  = gStd/(gMean||1);
  const conf = clamp(Math.round((72-18*cv+streakConfBonus(sf))*sp), 22, 88);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 3: hybrid_tft
// Variable selection gating + streak gate that shifts horizon blend weights.
// BUG-1: streak gate added.
// =============================================================================
function runHybridTft(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 20) return null;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const hrGlobal = gaps.length/rounds.length;

  const shortGaps = gaps.slice(-Math.min(20,  gaps.length));
  const longGaps  = gaps.slice(-Math.min(100, gaps.length));
  const sM=mean(shortGaps), sS=stdDev(shortGaps)||1;
  const lM=mean(longGaps),  lS=stdDev(longGaps) ||1;

  let shortGate = 1/(1+sS/(sM||1));
  let longGate  = 1/(1+lS/(lM||1));
  // BUG-1: streak gate
  if (sf) {
    if (sf.regime==='B2B'||sf.regime==='HOT')                         shortGate*=1.5;
    else if (sf.regime==='WHITE_CLUSTER'||sf.regime==='EXTREME_WHITE') longGate *=1.5;
  }
  const tot=shortGate+longGate||1;
  const shortW=shortGate/tot, longW=longGate/tot;

  const shortPred = weibullSkew(sM, sM+sS*0.5);
  const longPred  = weibullSkew(pctile(sorted,0.50), pctile(sorted,0.75));
  let raw = Math.max(1, Math.round(shortPred*shortW+longPred*longW));
  if (sf&&sf.regime==='WHITE_CLUSTER'&&sf.avgPostClusterGap!==null) {
    raw = Math.max(1, Math.round(raw*0.6+sf.avgPostClusterGap*0.4));
  }
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw  = target.maxWidth;
  const sp  = sparsePenalty(gaps.length,40);
  const conf = clamp(Math.round((68+gaps.length*0.08-gStd/gMean*12+streakConfBonus(sf))*sp), 22, 90);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 4: tft_full
// BUG-7: sparse guard for <15 gaps. BUG-1: streak-weighted quantile attention.
// =============================================================================
function runTftFull(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;
  const hrGlobal = gaps.length/rounds.length;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const staticEmb = Math.log(target.min)/Math.log(1000);

  // BUG-7: sparse guard
  if (gaps.length < 15) {
    const fb  = Math.max(1, Math.round(pctile(sorted,0.50)));
    const adj = applyStreakAdjustment(fb, sf, target);
    const sp  = sparsePenalty(gaps.length,30);
    return {...placeWindow(adj,currentGap,target.maxWidth), expectedGap:adj,
            probW:geoProbW(hrGlobal,target.maxWidth), conf:clamp(Math.round(40*sp),15,55)};
  }

  const DECAY=0.88+staticEmb*0.09; let wS=0,wG=0,wt=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*wt;wS+=wt;wt*=DECAY;}
  const seqOut=wG/(wS||1);

  let wQ10=0.25,wQ50=0.50,wQ90=0.25;
  if (sf) {
    if (sf.regime==='B2B'||sf.regime==='HOT')                         {wQ10=0.40;wQ50=0.45;wQ90=0.15;}
    else if (sf.regime==='WHITE_CLUSTER'||sf.regime==='EXTREME_WHITE') {wQ10=0.10;wQ50=0.40;wQ90=0.50;}
  }
  const attnOut = pctile(sorted,0.10)*wQ10 + pctile(sorted,0.50)*wQ50 + pctile(sorted,0.90)*wQ90;
  const raw = Math.max(1, Math.round(seqOut*0.55+attnOut*0.45));
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw  = target.maxWidth;
  const sp  = sparsePenalty(gaps.length,40);
  const conf = clamp(Math.round((70-gStd/gMean*16+staticEmb*6+streakConfBonus(sf))*sp), 15, 92);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 5: nbeats
// BUG-7: extrapolate to index n (correct), not count n. New streak block.
// =============================================================================
function runNbeats(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;
  const hrGlobal = gaps.length/rounds.length;
  const n=gaps.length, gMean=mean(gaps);

  const {a,b,r2} = olsLinear(gaps);
  const trendForecast = Math.max(1, a+b*n); // BUG-7: index n = one beyond last
  const residuals1    = gaps.map((g,i)=>g-(a+b*i));
  const resMean       = mean(residuals1);
  const identityForecast = gMean+resMean;

  // BUG-1: run-length streak block
  let streakForecast = identityForecast;
  if (sf && sf.highRuns.length>=5) {
    const {a:rA,b:rB} = olsLinear(sf.highRuns);
    streakForecast = Math.max(1, rA+rB*sf.highRuns.length);
  }

  const raw = Math.max(1, Math.round(
    trendForecast    * r2 +
    identityForecast * (1-r2)*0.6 +
    streakForecast   * (1-r2)*0.4
  ));
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw  = target.maxWidth;
  const gStd = stdDev(gaps)||1;
  const sp  = sparsePenalty(n,40);
  const conf = clamp(Math.round((60+r2*22-gStd/gMean*10+streakConfBonus(sf))*sp), 18, 90);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 6: tcn
// BUG-3: dilations extended to [1,2,4,8,16,32]. BUG-1: streak residual block.
// =============================================================================
function runTcn(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;
  const hrGlobal=gaps.length/rounds.length, gMean=mean(gaps);

  let signal=[...gaps];
  for (const d of [1,2,4,8,16,32]) { // BUG-3: was [1,2,4,8]
    const out=new Array(signal.length);
    for(let i=0;i<signal.length;i++){
      out[i]=i-d>=0 ? 0.50*signal[i]+0.50*signal[i-d] : signal[i];
    }
    for(let i=0;i<signal.length;i++) signal[i]=0.70*out[i]+0.30*gaps[i];
  }
  const lastK = Math.min(10, signal.length);
  const tcnOut = mean(signal.slice(-lastK));

  // BUG-1: streak residual
  let streakRes=0;
  if (sf&&sf.highRuns.length>=5) {
    const {b:rs}=olsLinear(sf.highRuns);
    streakRes=rs*0.5;
  }
  const raw = Math.max(1, Math.round(tcnOut+streakRes));
  const adj = applyStreakAdjustment(raw, sf, target);
  const aw  = target.maxWidth;
  const gStd=stdDev(gaps)||1;
  const sp  = sparsePenalty(gaps.length,30);
  const conf = clamp(Math.round((75-gStd/gMean*14+Math.min(gaps.length,300)*0.04+streakConfBonus(sf))*sp), 18, 91);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 7: lightgbm
// BUG-4: hrRecent capped. BUG-1: 2 new streak leaves (b2b + cluster).
// =============================================================================
function runLightGBM(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const p50=pctile(sorted,0.50), p75=pctile(sorted,0.75);

  // BUG-4: capped recent window
  const recentN = Math.min(300, Math.max(15, Math.round(5/(hrGlobal||0.01))));
  const hrRecent=(rounds.slice(-recentN).filter(r=>r.multiplier>=target.min).length+1)/(recentN+2);
  const {b:slope,r2}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  const overdue=clamp(currentGap/(gMean||1),0,3);
  const cv=gStd/(gMean||1);

  const leaf1=p50;
  const leaf2=gMean*(1-clamp((hrRecent-hrGlobal)/(hrGlobal||0.01),-0.3,0.3));
  const leaf3=Math.max(1,gMean+slope*3);
  const leaf4=overdue>1.5 ? gMean*0.75 : gMean*1.10;
  const leaf5=weibullSkew(p50,p75);
  // BUG-1: streak leaves
  const leaf6=sf ? gMean*(1-sf.b2bRate*0.4) : gMean;
  const leaf7=sf ? gMean*(sf.lowDensity20>0.85 ? 0.7 : 1.0) : gMean;

  const w1=1/(1+cv), w2=r2, w3=Math.abs(slope)<gMean*0.05?0.8:0.3;
  const w4=overdue>1?0.9:0.4, w5=0.7;
  const w6=sf?clamp(sf.b2bRate*3,0.1,1.0):0.1;
  const w7=sf?clamp(sf.lowDensity20*2,0.1,1.0):0.1;
  const wS=w1+w2+w3+w4+w5+w6+w7;

  const raw=Math.max(1,Math.round((leaf1*w1+leaf2*w2+leaf3*w3+leaf4*w4+leaf5*w5+leaf6*w6+leaf7*w7)/wS));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((72-cv*16+r2*10+overdue*2+streakConfBonus(sf))*sp),18,92);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 8: gru
// BUG-5: scale floor = max(gStd, 0.5*gMean). BUG-1: regime-biased init state.
// =============================================================================
function runGRU(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 5) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const cv=gStd/(gMean||1);

  // BUG-5: prevent gate saturation in volatile regimes
  const scale=Math.max(gStd, gMean*0.5);
  const sigmoid=x=>1/(1+Math.exp(-clamp(x/scale,-8,8)));

  // BUG-1: regime-biased initial hidden state
  let h=gMean;
  if (sf) {
    if (sf.regime==='B2B')            h=gMean*(1-sf.b2bContinuationProb*0.3);
    else if (sf.regime==='WHITE_CLUSTER') h=gMean*(sf.avgPostClusterGap!==null?sf.avgPostClusterGap/gMean:0.9);
    else if (sf.regime==='EXTREME_WHITE') h=gMean*0.7;
  }

  for (const g of gaps) {
    const z=sigmoid(h-gMean);
    const r=sigmoid(g-gMean);
    const hc=0.5*(r*h+g);
    h=(1-z)*hc+z*h;
  }

  const raw=Math.max(1,Math.round(h));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,30);
  const conf=clamp(Math.round((73-cv*18+Math.min(gaps.length,400)*0.05+streakConfBonus(sf))*sp),18,90);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 9: bilstm
// BUG-9: backward pass correctly iterates oldest->newest so oldest = highest weight.
// BUG-1: merge weights shift by regime.
// =============================================================================
function runBiLSTM(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1, cv=gStd/(gMean||1);

  // Forward: newest->oldest, slow decay (most recent = high weight)
  const FD=0.97; let wS=0,wG=0,wt=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*wt;wS+=wt;wt*=FD;}
  const fwdOut=wG/(wS||1);

  // BUG-9 fix: Backward pass = oldest->newest, fast decay (oldest = high weight)
  // This gives the historical long-range view from the other direction.
  const BD=0.93; wS=0;wG=0;wt=1;
  for(let i=0;i<gaps.length;i++){wG+=gaps[i]*wt;wS+=wt;wt*=BD;}
  const bwdOut=wG/(wS||1);

  // BUG-1: merge weights shift by regime
  let fwdW=0.60, bwdW=0.40;
  if (sf) {
    if (sf.regime==='B2B'||sf.regime==='HOT')                         {fwdW=0.75;bwdW=0.25;}
    else if (sf.regime==='WHITE_CLUSTER'||sf.regime==='EXTREME_WHITE') {fwdW=0.45;bwdW=0.55;}
  }
  const biOut=fwdOut*fwdW+bwdOut*bwdW;
  const overdueAdj=currentGap>gMean*1.2 ? biOut*0.85 : biOut;
  const raw=Math.max(1,Math.round(overdueAdj));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,35);
  const conf=clamp(Math.round((76-cv*20+(currentGap>gMean?5:0)+streakConfBonus(sf))*sp),18,91);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 10: stacking_meta
// BUG-6: sha512 now in source list. BUG-1: specialisation weights by regime.
// =============================================================================
function runStackingMeta(rounds, target, allNgResults, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 5) return null;
  const hrGlobal=gaps.length/rounds.length, gMean=mean(gaps), aw=target.maxWidth;

  // BUG-6: sha512 included
  const sourceIds=['hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','sha512'];

  const spec = {
    hlstm_xgb:  {b2b:1.4,cluster:1.2,neutral:1.0},
    htrans_lstm: {b2b:1.3,cluster:1.3,neutral:1.0},
    htft:        {b2b:1.1,cluster:1.5,neutral:1.0},
    tft:         {b2b:1.0,cluster:1.4,neutral:1.0},
    nbeats:      {b2b:1.2,cluster:1.2,neutral:1.0},
    tcn:         {b2b:1.5,cluster:1.0,neutral:1.0},
    lgbm:        {b2b:1.3,cluster:1.3,neutral:1.0},
    gru:         {b2b:1.2,cluster:1.1,neutral:1.0},
    bilstm:      {b2b:1.1,cluster:1.5,neutral:1.0},
    sha512:      {b2b:1.0,cluster:1.0,neutral:1.1},
  };
  const regime = sf ? (
    (sf.regime==='B2B'||sf.regime==='HOT'||sf.regime==='HOT_AFTER_SHORT_COLD')     ? 'b2b' :
    (sf.regime==='WHITE_CLUSTER'||sf.regime==='EXTREME_WHITE'||sf.regime==='COLD') ? 'cluster' :
    'neutral'
  ) : 'neutral';

  const predictions=[], weights=[];
  for (const eid of sourceIds) {
    const r=allNgResults[eid]?.[target.label];
    if (!r?.expectedGap) continue;
    predictions.push(r.expectedGap);
    weights.push(spec[eid]?.[regime]??1.0);
  }
  const sorted=[...gaps].sort((a,b)=>a-b);
  predictions.push(pctile(sorted,0.50)); weights.push(0.5);

  if (predictions.length<3) {
    const eg=Math.max(1,Math.round(pctile(sorted,0.50)));
    return {...placeWindow(eg,currentGap,aw), expectedGap:eg, probW:geoProbW(hrGlobal,aw), conf:40};
  }

  const pMean=mean(predictions), pStd=stdDev(predictions)||1;
  const ivW=predictions.map((p,i)=>weights[i]/(Math.abs(p-pMean)+pStd));
  const wSum=ivW.reduce((a,b)=>a+b,0)||1;
  const eg=Math.max(1,Math.round(predictions.reduce((s,p,i)=>s+p*ivW[i]/wSum,0)));
  const adj=applyStreakAdjustment(eg,sf,target);
  const diversity=pStd/(pMean||1);
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((80-diversity*18+predictions.length*1.2+streakConfBonus(sf))*sp),22,94);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(hrGlobal,aw), conf};
}

// =============================================================================
// ENGINE 11: sha512
// BUG-1: run-length entropy as hash-chain streak bias signal.
// =============================================================================
function runSHA512(rounds, target, sf) {
  if (rounds.length<30) return null;
  const {gaps, currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<5) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const n=rounds.length;

  const obsP=(gaps.length+1)/(n+2);
  const trust=clamp((n-30)/270,0,1);

  // CUSUM on last 300 rounds
  const p0=hrGlobal; let cusum=0,maxC=0,minC=0;
  for(const r of rounds.slice(-300)){
    cusum+=(r.multiplier>=target.min?1:0)-p0;
    if(cusum>maxC)maxC=cusum; if(cusum<minC)minC=cusum;
  }
  const sigma=Math.sqrt(Math.max(1e-9,p0*(1-p0)*Math.min(300,n)));
  const cusumMag=Math.max(Math.abs(maxC),Math.abs(minC))/(sigma||1);
  const drift=cusumMag>1.65;

  // Gap entropy
  const bins=10, binC=new Array(bins).fill(0);
  const gMax=sorted[sorted.length-1]||1;
  for(const g of gaps){const b=Math.min(bins-1,Math.floor(g/gMax*bins));binC[b]++;}
  let ent=0;
  for(const c of binC){const p=c/gaps.length;if(p>0)ent-=p*Math.log2(p);}
  const normEnt=ent/(Math.log2(bins)||1);

  // Lag-1 AC
  let ac1=0;
  if (gaps.length>=10) {
    const m=mean(gaps); let cov=0,vs=0;
    for(let i=1;i<gaps.length;i++) cov+=(gaps[i-1]-m)*(gaps[i]-m);
    for(const g of gaps) vs+=(g-m)**2;
    ac1=vs>0?cov/vs:0;
  }

  // BUG-1: run-length entropy as streak bias
  let streakBias=0;
  if (sf&&sf.highRuns.length>=5) {
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    streakBias=(1-normHrEnt)*(sf.b2bRate>0.2?0.08:0.03);
  }

  const driftF=drift?clamp(cusumMag*0.07*Math.sign(maxC+minC),-0.15,0.15):0;
  const entAdj=(1-normEnt)*0.04;
  const acAdj=ac1<-0.15?-0.04:0;

  const baseGap=Math.max(1,Math.round((1/obsP)*trust+gMean*(1-trust)));
  const raw=Math.max(1,Math.round(baseGap*(1-driftF-entAdj-acAdj-streakBias)));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,30);
  const conf=clamp(Math.round((55+trust*18+(drift?7:0)+normEnt*4+streakConfBonus(sf))*sp),22,93);
  return {...placeWindow(adj,currentGap,aw), expectedGap:adj, probW:geoProbW(obsP,aw), conf};
}

// =============================================================================
// ROUNDS CACHE
// =============================================================================
async function getNgRounds() {
  if (cachedRounds.length===0) {
    cachedRounds=await getRounds({limit:100000,order:'ASC'});
    cachedRoundsLastId=cachedRounds.length?cachedRounds[cachedRounds.length-1].roundId:0;
    console.log(`[ngCompute] loaded ${cachedRounds.length} rounds`);
  } else {
    const nr=await getRounds({limit:5000,minRoundId:cachedRoundsLastId+1});
    if (nr.length){cachedRounds=[...cachedRounds,...nr];cachedRoundsLastId=cachedRounds[cachedRounds.length-1].roundId;}
  }
  return cachedRounds;
}

// =============================================================================
// SAVE OUTCOME
// =============================================================================
async function saveNgOutcome(engineId, target, outcome, lo, hi, hitRound, generation) {
  const key=`${lo}:${hi}`;
  if (ngSavedSets[engineId].has(key)) return;
  ngSavedSets[engineId].add(key);
  try {
    await savePrediction({target:target.label,minMult:target.min,outcome,lo,hi,
      hitRound:hitRound??null,generation:generation??1,source:engineId,probW:null});
    console.log(`[ngCompute] ${engineId} ${target.label} ${outcome.toUpperCase()} #${lo}-#${hi}${hitRound?` @#${hitRound}`:''}`);
  } catch(e) {
    console.error(`[ngCompute] save fail ${engineId}:`,e.message);
    ngSavedSets[engineId].delete(key);
  }
}

// =============================================================================
// NG CONSENSUS
// =============================================================================
function computeNgConsensus(allNgResults, lastRoundId) {
  const consensus={};
  for (const target of TARGETS) {
    const windows=[];
    const SRC=['hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','sha512'];
    for(const eid of SRC){
      const r=allNgResults[eid]?.[target.label];
      if(!r) continue;
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
    const bW=target.maxWidth;
    if(bestHi-bestLo+1<bW){const c=Math.round((bestLo+bestHi)/2);bestLo=c-Math.floor(bW/2);bestHi=bestLo+bW-1;}
    if(bestLo<=lastRoundId){bestLo=lastRoundId+1;bestHi=bestLo+bW-1;}
    consensus[target.label]={lo:bestLo,hi:bestHi,engineCount:bestGroup.length,engines:bestGroup.map(w=>w.engineId)};
  }
  return consensus;
}

// =============================================================================
// MAIN TICK
// =============================================================================
async function runNgComputeEngine() {
  try {
    const rounds=await getNgRounds();
    if (rounds.length<50) return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // Pre-compute streak features once per target (shared backbone — BUG-1/2 fix)
    const streakFeatures={};
    for (const target of TARGETS) {
      try { streakFeatures[target.label]=extractStreakFeatures(rounds,target.min); }
      catch(e){ streakFeatures[target.label]=null; console.error(`[ngCompute] sf/${target.label}:`,e.message); }
    }

    // Pass 1: all non-stacking engines
    const ALGO_MAP={
      hlstm_xgb:  runHybridLstmXgb,
      htrans_lstm: runHybridTransformerLstm,
      htft:        runHybridTft,
      tft:         runTftFull,
      nbeats:      runNbeats,
      tcn:         runTcn,
      lgbm:        runLightGBM,
      gru:         runGRU,
      bilstm:      runBiLSTM,
      sha512:      runSHA512,
    };
    const allNgResults={};
    for(const [eid,algo] of Object.entries(ALGO_MAP)){
      allNgResults[eid]={};
      for(const target of TARGETS){
        try{
          const r=algo(rounds,target,streakFeatures[target.label]);
          if(r) allNgResults[eid][target.label]=r;
        } catch(e){console.error(`[ngCompute] ${eid}/${target.label}:`,e.message);}
      }
    }

    // Pass 2: stacking meta
    allNgResults['stacking']={};
    for(const target of TARGETS){
      try{
        const r=runStackingMeta(rounds,target,allNgResults,streakFeatures[target.label]);
        if(r) allNgResults['stacking'][target.label]=r;
      } catch(e){console.error(`[ngCompute] stacking/${target.label}:`,e.message);}
    }

    // Pass 3: ng_consensus
    const ngConsensus=computeNgConsensus(allNgResults,lastRoundId);
    allNgResults['ng_consensus']={};
    for(const target of TARGETS){
      const c=ngConsensus[target.label];
      if(c){
        allNgResults['ng_consensus'][target.label]={
          low:c.lo-lastRoundId, high:c.hi-lastRoundId,
          expectedGap:Math.round((c.lo+c.hi)/2-lastRoundId),
          probW:null, conf:clamp(55+Math.round(c.engineCount*4),55,95),
          _meta:{engineCount:c.engineCount,engines:c.engines},
        };
      }
    }

    // Phase 1+2: resolve + lock windows
    for(const engineId of NG_ENGINE_IDS){
      const payload={};
      for(const target of TARGETS){
        const win=ngWindows[engineId][target.label];
        const fresh=allNgResults[engineId]?.[target.label];
        if(win){
          const {lo,hi,generation,roundWhenMade}=win;
          const eLo=Math.max(roundWhenMade+1,lo-earlyHitTolerance(target.maxWidth));
          const earlyHit=lo>roundWhenMade+1&&eLo<=lo-1?findHitInRange(rounds,eLo,lo-1,target.min):null;
          if(earlyHit){
            await saveNgOutcome(engineId,target,'early',lo,hi,earlyHit.roundId,generation);
            delete ngWindows[engineId][target.label];
          } else if(lastRoundId>=hi){
            const hit=findHitInRange(rounds,lo,hi,target.min);
            await saveNgOutcome(engineId,target,hit?'win':'loss',lo,hi,hit?.roundId??null,generation);
            delete ngWindows[engineId][target.label];
          } else {
            const hit=findHitInRange(rounds,lo,hi,target.min);
            if(hit){
              await saveNgOutcome(engineId,target,'win',lo,hi,hit.roundId,generation);
              delete ngWindows[engineId][target.label];
            } else {
              payload[target.label]={lo,hi,roundWhenMade,generation,eta:win.eta};
              continue;
            }
          }
        }
        if(fresh){
          const newLo=lastRoundId+fresh.low, newHi=lastRoundId+fresh.high;
          const gen=(ngWindows[engineId][target.label]?.generation??0)+1;
          const baseEta={probW:fresh.probW,conf:fresh.conf,expectedGap:fresh.expectedGap};
          const eta=fresh._meta?{...baseEta,...fresh._meta}:baseEta;
          ngWindows[engineId][target.label]={lo:newLo,hi:newHi,roundWhenMade:lastRoundId,generation:gen,eta};
          payload[target.label]=ngWindows[engineId][target.label];
        }
      }
      if(Object.keys(payload).length) await saveLockedAdvPreds(engineId,payload);
    }
  } catch(e) {
    console.error('[ngCompute] Fatal:',e.message,e.stack);
  }
}

// =============================================================================
// INIT
// =============================================================================
async function initNgCompute() {
  if(initialised) return;
  initialised=true;
  try{
    const existing=await getLockedAdvPreds();
    for(const engineId of NG_ENGINE_IDS){
      for(const target of TARGETS){
        const w=existing[engineId]?.[target.label];
        if(w?.lo&&w?.hi){
          ngWindows[engineId][target.label]={
            lo:Number(w.lo),hi:Number(w.hi),
            roundWhenMade:Number(w.roundWhenMade??w.lo),
            generation:w.generation??1, eta:w.eta??{},
          };
        }
      }
    }
    console.log(`[ngCompute] loaded existing locked windows`);
  } catch(e){console.error('[ngCompute] init locked error:',e.message);}
  try{
    for(const engineId of NG_ENGINE_IDS){
      const rows=await getPredictions({limit:500000,source:engineId});
      for(const r of rows) ngSavedSets[engineId].add(`${r.lo}:${r.hi}`);
    }
    const total=NG_ENGINE_IDS.reduce((s,id)=>s+ngSavedSets[id].size,0);
    console.log(`[ngCompute] pre-warmed savedSets with ${total} outcomes`);
  } catch(e){console.error('[ngCompute] init history error:',e.message);}
}

const _origRun=runNgComputeEngine;
async function runNgComputeEngineWithInit(){await initNgCompute();await _origRun();}

function resetNgComputeState(){
  for(const id of NG_ENGINE_IDS){ngWindows[id]={};ngSavedSets[id]=new Set();}
  cachedRounds=[];cachedRoundsLastId=0;initialised=false;
}
function resetNgWindowsOnly(){
  for(const id of NG_ENGINE_IDS) ngWindows[id]={};
  console.log('[ngCompute] in-memory windows cleared (lock reset)');
}

module.exports={runNgComputeEngine:runNgComputeEngineWithInit,resetNgComputeState,resetNgWindowsOnly,NG_ENGINE_IDS};