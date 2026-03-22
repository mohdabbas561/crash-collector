'use strict';
// advComputeEngine.js — Server-side window computation for all 13 ADV engines
// ═══════════════════════════════════════════════════════════════════════════════
// PURPOSE: Makes ADV engines fully independent of the browser.
// Previously: frontend computed windows → posted to /locked-adv → server resolved.
// Now:        THIS FILE computes windows → saves to locked_preds_adv → server resolves.
// Frontend still computes its own windows (for display), but the server is the
// authoritative source. If browser is offline, this keeps running every 8s tick.
// ═══════════════════════════════════════════════════════════════════════════════

const {
  getRounds,
  saveLockedAdvPreds,
  saveLockedConsensusPreds,
  getLockedAdvPreds,
  getLockedConsensusPreds,
} = require('./db');
const { bustLockedCache } = require('./advResolutionEngine');

// ── Constants ─────────────────────────────────────────────────────────────────
const ENGINE_IDS = [
  'lstm','xgb','rf','ols','cat',
  'hardgap','softgap','markov','percentile','bayes',
  'sha256','mt','lcg',
];

const TARGETS = [
  { label: '5x',    min: 5,    maxWidth: 3  },
  { label: '10x',   min: 10,   maxWidth: 5  },
  { label: '20x',   min: 20,   maxWidth: 7  },
  { label: '50x',   min: 50,   maxWidth: 12 },
  { label: '100x',  min: 100,  maxWidth: 18 },
  { label: '250x',  min: 250,  maxWidth: 25 },
  { label: '500x',  min: 500,  maxWidth: 35 },
  { label: '1000x', min: 1000, maxWidth: 50 },
];

let cachedRounds       = [];
let cachedRoundsLastId = 0;
let initialised        = false;

// ── Math utilities (ported from EngineWorker.js — pure JS, no DOM) ─────────────
function mean(arr){ return arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }
function median(arr){
  if(!arr.length) return 0;
  const s=[...arr].sort((a,b)=>a-b);
  const m=Math.floor(s.length/2);
  return s.length%2===1 ? s[m] : (s[m-1]+s[m])/2;
}
function stdDev(arr){
  if(arr.length<2) return 0;
  const m=mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1));
}
function clamp(v,lo,hi){ return Math.max(lo,Math.min(hi,v)); }
function geoProbW(hr,w){ return clamp(1-Math.pow(1-(hr||0),Math.max(1,w)),0,0.99); }
function pctile(sorted,frac){ return sorted[Math.min(sorted.length-1,Math.max(0,Math.floor(frac*sorted.length)))]; }

function bisectLeft(rounds,targetId){
  let lo=0,hi=rounds.length;
  while(lo<hi){const mid=(lo+hi)>>>1;if(rounds[mid].roundId<targetId)lo=mid+1;else hi=mid;}
  return lo;
}

function computeGaps(rounds,minMult){
  const gaps=[]; let since=0;
  for(const r of rounds){since++;if(r.multiplier>=minMult){gaps.push(since);since=0;}}
  return {gaps,currentGap:since};
}

function placeWindow(expectedGap,currentGap,width){
  const remaining=Math.max(1,expectedGap-currentGap);
  const low=Math.max(1,remaining-Math.floor(width/2));
  return {low,high:low+width-1};
}

function olsLinear(ys){
  const n=ys.length;
  if(n<3) return {a:mean(ys),b:0,r2:0};
  let sx=0,sy=0,sxy=0,sxx=0;
  for(let i=0;i<n;i++){sx+=i;sy+=ys[i];sxy+=i*ys[i];sxx+=i*i;}
  const b=(n*sxy-sx*sy)/(n*sxx-sx*sx||1);
  const a=(sy-b*sx)/n;
  const gm=sy/n;
  const ssTot=ys.reduce((s,v)=>s+(v-gm)**2,0);
  const ssRes=ys.reduce((s,v,i)=>s+(v-(a+b*i))**2,0);
  return {a,b,r2:ssTot>0?clamp(1-ssRes/ssTot,0,1):0};
}

function cusumNorm(rounds,minMult){
  const n=rounds.length;
  const p0=rounds.filter(r=>r.multiplier>=minMult).length/n;
  let cusum=0,maxC=0;
  const win=rounds.slice(-300);
  for(const r of win){cusum+=(r.multiplier>=minMult?1:0)-p0;if(Math.abs(cusum)>maxC)maxC=Math.abs(cusum);}
  const sigma=Math.sqrt(Math.max(1e-9,p0*(1-p0)));
  return maxC/(sigma*Math.sqrt(win.length));
}

function weibullSkew(p50,p75){ return Math.max(1,Math.round(p50+0.20*(p75-p50))); }

function getDynamicBuckets(rounds){
  const mults=[...rounds].map(r=>r.multiplier).sort((a,b)=>a-b);
  if(mults.length<50) return {tL:5,tM:20,tH:100};
  return {
    tL:Math.max(pctile(mults,0.20),1.01),
    tM:Math.max(pctile(mults,0.50),1.02),
    tH:Math.max(pctile(mults,0.80),1.03),
  };
}

// ── Engine implementations (ported from EngineWorker.js v3.0) ─────────────────
function runLSTM(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const DECAY=0.97;
  let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/wS;
  const gMean=mean(gaps);
  const expectedGap=Math.max(1,Math.round(ewaMean*0.70+gMean*0.30));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(85-30*stdDev(gaps)/(ewaMean||1)),30,88)};
}

function runXGBoost(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const hrGlobal=gaps.length/rounds.length;
  const recentN=Math.min(100,gaps.length);
  const recentRounds=Math.round(recentN/(hrGlobal||0.01));
  const hrRecent=(rounds.slice(-recentRounds).filter(r=>r.multiplier>=target.min).length+1)/(recentRounds+2);
  const rateDrift=hrRecent-hrGlobal;
  const gMean=mean(gaps);
  const {b:rawSlope}=olsLinear(gaps.slice(-50));
  const slope=clamp(rawSlope,-gMean*0.12,gMean*0.12);
  const expectedGap=Math.max(1,Math.round(gMean*(1-clamp(rateDrift/(hrGlobal||0.01),-0.25,0.25))+slope*0.5));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(hrGlobal,aw),conf:clamp(Math.round(80-22*stdDev(gaps)/(gMean||1)),30,88)};
}

function runRF(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const N_BOOT=64,n=gaps.length;
  const primes=[3,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317];
  const bootMedians=[];
  for(let b=0;b<N_BOOT;b++){
    const offset=(b*11+5)%n,prime=primes[b%primes.length];
    const sample=[];
    for(let i=0;i<n;i++) sample.push(gaps[(offset+i*prime)%n]);
    sample.sort((a,b)=>a-b);
    bootMedians.push(sample[Math.floor(n/2)]);
  }
  bootMedians.sort((a,b)=>a-b);
  const p50=pctile(bootMedians,0.50),p75=pctile(bootMedians,0.75);
  const expectedGap=Math.max(1,weibullSkew(p50,p75));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(52+gaps.length*0.2),40,88)};
}

function runOLS(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const lin=olsLinear(gaps);
  const gMean=mean(gaps);
  const rawGap=Math.max(1,Math.round(lin.a+lin.b*gaps.length));
  const expectedGap=Math.max(1,Math.round(rawGap*lin.r2+gMean*(1-lin.r2)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(35+lin.r2*42+gaps.length*0.15),30,88)};
}

function runCatBoost(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8) return null;
  const hrGlobal=(gaps.length+1)/(rounds.length+2);
  const cn=cusumNorm(rounds,target.min);
  const localRegime=cn>1.96?'HOT':cn>1.36?'WARM':cn<-1.96?'COLD':'NEUTRAL';
  const hr300=(rounds.slice(-300).filter(r=>r.multiplier>=target.min).length+1)/302;
  const blend=localRegime==='HOT'?0.55:localRegime==='WARM'?0.35:localRegime==='COLD'?0.20:0.12;
  const pBlend=hrGlobal*(1-blend)+hr300*blend;
  const expectedGap=Math.max(1,Math.round((1-pBlend)/pBlend));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(pBlend,aw),conf:clamp(Math.round(48+gaps.length*0.25),38,91)};
}

function runHardGap(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const p50=pctile(sorted,0.50),p75=pctile(sorted,0.75);
  const gMean=mean(gaps);
  const skewGap=weibullSkew(p50||gMean,p75||gMean);
  const expectedGap=Math.max(1,Math.round(gMean*0.4+skewGap*0.6));
  const overdue=currentGap/(expectedGap||1);
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(30+overdue*22),20,95)};
}

function runSoftGap(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=0.85;}
  const ewmaFast=wG/wS;
  const gMedian=median(gaps);
  const expectedGap=Math.max(1,Math.round(ewmaFast*0.60+gMedian*0.40));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(55+gaps.length*0.2),25,88)};
}

function runMarkov(rounds,target){
  if(rounds.length<30) return null;
  const {tL,tM,tH}=getDynamicBuckets(rounds);
  const bucket=m=>m>=tH?'X':m>=tM?'H':m>=tL?'M':'L';
  const states=['L','M','H','X'];
  const mat1={};
  for(const s of states){mat1[s]={};for(const t of states)mat1[s][t]=0;}
  for(let i=1;i<rounds.length;i++) mat1[bucket(rounds[i-1].multiplier)][bucket(rounds[i].multiplier)]++;
  const cs=bucket(rounds[rounds.length-1]?.multiplier??1);
  const tot1=states.reduce((s,t)=>s+mat1[cs][t],0);
  const prob={};states.forEach(t=>{prob[t]=tot1?mat1[cs][t]/tot1:0.25;});
  let pHit=target.min>=tH?prob['X']:target.min>=tM?prob['H']+prob['X']:target.min>=tL?prob['M']+prob['H']+prob['X']:1-prob['L']*0.3;
  const globalHR=rounds.filter(r=>r.multiplier>=target.min).length/rounds.length;
  pHit=pHit*0.65+globalHR*0.35;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const expectedGap=Math.max(1,pHit>0?Math.round(1/pHit):mean(gaps)||50);
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:clamp(pHit*(aw/Math.max(1,expectedGap)),0,0.99),conf:clamp(Math.round(pHit*75+20),20,88)};
}

function runPercentile(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8) return null;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const p50=pctile(sorted,0.50),p75=pctile(sorted,0.75);
  const expectedGap=Math.max(1,weibullSkew(p50,p75));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(62+gaps.length*0.15),30,90)};
}

function runBayes(rounds,target){
  if(rounds.length<10) return null;
  const hits=rounds.filter(r=>r.multiplier>=target.min).length;
  const alpha=1+hits,beta=1+rounds.length-hits;
  const mu=alpha/(alpha+beta);
  const rawGap=Math.max(1,Math.round(1/(mu||0.001)));
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const sorted=[...gaps].sort((a,b)=>a-b);
  const p50=pctile(sorted,0.50)||rawGap,p75=pctile(sorted,0.75)||rawGap;
  const muVar=alpha*beta/((alpha+beta)**2*(alpha+beta+1));
  const certainty=clamp(1-Math.sqrt(muVar)/mu,0,1);
  const expectedGap=Math.max(1,Math.round(rawGap*certainty+weibullSkew(p50,p75)*(1-certainty)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(mu,aw),conf:clamp(Math.round(40+rounds.length*0.06+certainty*20),35,90)};
}

function runSHA256(rounds,target){
  if(rounds.length<20) return null;
  const obsHits=rounds.filter(r=>r.multiplier>=target.min).length;
  const obsP=(obsHits+1)/(rounds.length+2);
  const obsGap=Math.max(1,Math.round(1/obsP));
  const theorGap=Math.max(1,Math.round(target.min/(1-0.01)));
  const trust=clamp((rounds.length-20)/180,0,1);
  const expectedGap=Math.max(1,Math.round(obsGap*trust+theorGap*(1-trust)));
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(obsP,aw),conf:clamp(Math.round(55+trust*15),40,90)};
}

function runMersenne(rounds,target){
  if(rounds.length<50) return null;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const gMedian=median(gaps),gMean=mean(gaps);
  const expectedGap=Math.max(1,Math.round(gMedian*0.6+gMean*0.4));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(42+gaps.length*0.15),30,85)};
}

function runLCG(rounds,target){
  if(rounds.length<30) return null;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const gMean=mean(gaps),gMedian=median(gaps);
  const expectedGap=Math.max(1,Math.round(gMedian));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(25+gaps.length*0.15),20,80)};
}

const ALGOS = {
  lstm:runLSTM, xgb:runXGBoost, rf:runRF, ols:runOLS, cat:runCatBoost,
  hardgap:runHardGap, softgap:runSoftGap, markov:runMarkov,
  percentile:runPercentile, bayes:runBayes,
  sha256:runSHA256, mt:runMersenne, lcg:runLCG,
};

// ── Rounds cache ───────────────────────────────────────────────────────────────
async function getComputeRounds() {
  if (cachedRounds.length === 0) {
    const all = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRounds       = all;
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[advCompute] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds       = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Compute consensus: find overlapping window across all engines for each target ──
function computeConsensus(allResults, lastRoundId) {
  const consensus = {};
  for (const target of TARGETS) {
    const windows = [];
    for (const [engineId, result] of Object.entries(allResults)) {
      const r = result[target.label];
      if (!r) continue;
      const lo = lastRoundId + r.low;
      const hi = lastRoundId + r.high;
      windows.push({ engineId, lo, hi });
    }
    if (windows.length < 3) { consensus[target.label] = null; continue; }

    // Find the group of engines with maximum overlap
    let bestGroup = [], bestLo = 0, bestHi = 0;
    for (let i = 0; i < windows.length; i++) {
      const grp = [windows[i]];
      let runLo = windows[i].lo, runHi = windows[i].hi;
      for (let j = 0; j < windows.length; j++) {
        if (j === i) continue;
        const nl = Math.max(runLo, windows[j].lo), nh = Math.min(runHi, windows[j].hi);
        if (nl <= nh) { grp.push(windows[j]); runLo = nl; runHi = nh; }
      }
      if (grp.length > bestGroup.length) { bestGroup = grp; bestLo = runLo; bestHi = runHi; }
    }

    if (bestGroup.length < 2) { consensus[target.label] = null; continue; }

    // Ensure minimum width
    const baseW = target.maxWidth;
    if (bestHi - bestLo + 1 < baseW) {
      const center = Math.round((bestLo + bestHi) / 2);
      bestLo = center - Math.floor(baseW / 2);
      bestHi = bestLo + baseW - 1;
    }
    // Must start after current round
    if (bestLo <= lastRoundId) { bestLo = lastRoundId + 1; bestHi = bestLo + baseW - 1; }

    consensus[target.label] = {
      lo: bestLo, hi: bestHi,
      engineCount: bestGroup.length,
      engines: bestGroup.map(w => w.engineId),
    };
  }
  return consensus;
}

// ── Main: compute windows and save to DB ──────────────────────────────────────
async function runAdvComputeEngine() {
  try {
    const rounds = await getComputeRounds();
    if (rounds.length < 50) return;

    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Get existing locked windows to compare
    const existingAdv  = await getLockedAdvPreds();
    const existingCons = await getLockedConsensusPreds();

    const allResults = {}; // { engineId: { targetLabel: {low, high, expectedGap, ...} } }

    // Compute windows for all 13 engines
    for (const engineId of ENGINE_IDS) {
      const algo = ALGOS[engineId];
      if (!algo) continue;
      allResults[engineId] = {};
      for (const target of TARGETS) {
        try {
          const r = algo(rounds, target);
          if (r) allResults[engineId][target.label] = r;
        } catch(e) {
          console.error(`[advCompute] ${engineId}/${target.label}:`, e.message);
        }
      }
    }

    // Save windows for each engine if changed
    let savedEngines = 0;
    for (const engineId of ENGINE_IDS) {
      const results = allResults[engineId];
      if (!Object.keys(results).length) continue;

      const payload = {};
      let hasChanges = false;

      for (const target of TARGETS) {
        const r = results[target.label];
        if (!r) continue;

        const newLo = lastRoundId + r.low;
        const newHi = lastRoundId + r.high;

        const existing = existingAdv[engineId]?.[target.label];
        // Only update if window changed meaningfully (different lo/hi)
        // or if existing window has expired
        const existingExpired = existing && existing.hi <= lastRoundId;
        const windowChanged   = !existing || existing.lo !== newLo || existing.hi !== newHi;

        if (existingExpired || windowChanged) {
          payload[target.label] = {
            lo: newLo,
            hi: newHi,
            roundWhenMade: lastRoundId,
            generation: (existing?.generation ?? 0) + (existingExpired ? 1 : 0),
            eta: { probW: r.probW, conf: r.conf, expectedGap: r.expectedGap },
          };
          hasChanges = true;
        }
      }

      if (hasChanges) {
        await saveLockedAdvPreds(engineId, payload);
        savedEngines++;
      }
    }

    // Compute and save consensus
    const consensus = computeConsensus(allResults, lastRoundId);
    const consPayload = {};
    let consHasChanges = false;

    for (const target of TARGETS) {
      const c = consensus[target.label];
      if (!c) continue;

      const existing = existingCons[target.label];
      const existingExpired = existing && existing.hi <= lastRoundId;
      const windowChanged   = !existing || existing.lo !== c.lo || existing.hi !== c.hi;

      if (existingExpired || windowChanged) {
        consPayload[target.label] = {
          lo: c.lo,
          hi: c.hi,
          roundWhenMade: lastRoundId,
          generation: (existing?.generation ?? 0) + (existingExpired ? 1 : 0),
          eta: { engineCount: c.engineCount, engines: c.engines },
        };
        consHasChanges = true;
      }
    }

    if (consHasChanges) {
      await saveLockedConsensusPreds(consPayload);
    }

    if (savedEngines > 0 || consHasChanges) {
      bustLockedCache(); // tell advResolutionEngine to re-read locked windows
      console.log(`[advCompute] updated ${savedEngines} engines + consensus=${consHasChanges} @ #${lastRoundId}`);
    }

  } catch(e) {
    console.error('[advCompute] Fatal:', e.message, e.stack);
  }
}

function resetAdvComputeState() {
  cachedRounds       = [];
  cachedRoundsLastId = 0;
  initialised        = false;
}

module.exports = { runAdvComputeEngine, resetAdvComputeState };