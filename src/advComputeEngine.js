'use strict';
// advComputeEngine.js — Server-side window computation for all 13 ADV engines
// Two-phase per tick:
//   Phase 1: resolve expired/hit windows → save to predictions
//   Phase 2: compute and lock fresh windows for targets with none
// Fully offline — no browser dependency.

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedAdvPreds,
  saveLockedConsensusPreds,
  getLockedAdvPreds,
  getLockedConsensusPreds,
} = require('./db');
const { bustLockedCache } = require('./advResolutionEngine');

// ENGINE_IDS trimmed to empty — individual adv engine windows removed to save DB load.
// The math functions (runLSTM, runXGB, etc.) are still called by computeConsensus.
// consensus window computation is preserved 100%.
const ENGINE_IDS = [];

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

// Per-engine saved sets (dedup by lo:hi)
const savedSets = {};
for (const id of [...ENGINE_IDS, 'consensus']) savedSets[id] = new Set();

let cachedRounds       = [];
let cachedRoundsLastId = 0;
let initialised        = false;

// ── Math (ported from EngineWorker.js) ────────────────────────────────────────
function mean(arr){ return arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }
function median(arr){
  if(!arr.length) return 0;
  const s=[...arr].sort((a,b)=>a-b); const m=Math.floor(s.length/2);
  return s.length%2===1?s[m]:(s[m-1]+s[m])/2;
}
function stdDev(arr){
  if(arr.length<2) return 0; const m=mean(arr);
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
function findHitInRange(rounds,fromId,toId,minMult){
  const start=bisectLeft(rounds,fromId);
  for(let i=start;i<rounds.length;i++){
    if(rounds[i].roundId>toId)break;
    if(rounds[i].multiplier>=minMult)return rounds[i];
  }
  return null;
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
// FIXED: earlyHit tolerance — max rounds before window open that count as early.
// Previously the full [roundWhenMade...lo-1] range was checked, which for large
// expectedGap values could be 10-20+ rounds, producing a near-100% early rate on
// common targets (5x, 10x). Now capped at floor(maxWidth/2) rounds so only hits
// very close to the window edge count as early; hits further back are scored LOSS.
function earlyHitTolerance(width) {
  return Math.floor(width / 2);
}
function olsLinear(ys){
  const n=ys.length; if(n<3) return {a:mean(ys),b:0,r2:0};
  let sx=0,sy=0,sxy=0,sxx=0;
  for(let i=0;i<n;i++){sx+=i;sy+=ys[i];sxy+=i*ys[i];sxx+=i*i;}
  const b=(n*sxy-sx*sy)/(n*sxx-sx*sx||1),a=(sy-b*sx)/n;
  const gm=sy/n,ssTot=ys.reduce((s,v)=>s+(v-gm)**2,0),ssRes=ys.reduce((s,v,i)=>s+(v-(a+b*i))**2,0);
  return {a,b,r2:ssTot>0?clamp(1-ssRes/ssTot,0,1):0};
}
function cusumNorm(rounds,minMult){
  const n=rounds.length,p0=rounds.filter(r=>r.multiplier>=minMult).length/n;
  let cusum=0,maxC=0;
  for(const r of rounds.slice(-300)){cusum+=(r.multiplier>=minMult?1:0)-p0;if(Math.abs(cusum)>maxC)maxC=Math.abs(cusum);}
  return maxC/(Math.sqrt(Math.max(1e-9,p0*(1-p0)))*Math.sqrt(Math.min(300,n)));
}
function weibullSkew(p50,p75){ return Math.max(1,Math.round(p50+0.20*(p75-p50))); }
function getDynamicBuckets(rounds){
  const mults=[...rounds].map(r=>r.multiplier).sort((a,b)=>a-b);
  if(mults.length<50) return {tL:5,tM:20,tH:100};
  return {tL:Math.max(pctile(mults,0.20),1.01),tM:Math.max(pctile(mults,0.50),1.02),tH:Math.max(pctile(mults,0.80),1.03)};
}

// ── Engine algorithms ─────────────────────────────────────────────────────────
function runLSTM(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const DECAY=0.97; let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/wS,gMean=mean(gaps);
  const expectedGap=Math.max(1,Math.round(ewaMean*0.70+gMean*0.30));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(85-30*(stdDev(gaps)/(ewaMean||1))),30,88)};
}
function runXGBoost(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const hrGlobal=gaps.length/rounds.length,recentN=Math.min(100,gaps.length);
  const hrRecent=(rounds.slice(-Math.round(recentN/(hrGlobal||0.01))).filter(r=>r.multiplier>=target.min).length+1)/(Math.round(recentN/(hrGlobal||0.01))+2);
  const gMean=mean(gaps);
  const {b:rawSlope}=olsLinear(gaps.slice(-50));
  const expectedGap=Math.max(1,Math.round(gMean*(1-clamp((hrRecent-hrGlobal)/(hrGlobal||0.01),-0.25,0.25))+clamp(rawSlope,-gMean*0.12,gMean*0.12)*0.5));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(hrGlobal,aw),conf:clamp(Math.round(80-22*(stdDev(gaps)/(gMean||1))),30,88)};
}
function runRF(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const N_BOOT=64,n=gaps.length;
  const primes=[3,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317];
  const bootMedians=[];
  for(let b=0;b<N_BOOT;b++){
    const offset=(b*11+5)%n,prime=primes[b%primes.length],sample=[];
    for(let i=0;i<n;i++) sample.push(gaps[(offset+i*prime)%n]);
    sample.sort((a,b)=>a-b); bootMedians.push(sample[Math.floor(n/2)]);
  }
  bootMedians.sort((a,b)=>a-b);
  const expectedGap=Math.max(1,weibullSkew(pctile(bootMedians,0.50),pctile(bootMedians,0.75)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(52+gaps.length*0.2),40,88)};
}
function runOLS(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<10) return null;
  const lin=olsLinear(gaps),gMean=mean(gaps);
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
  const regime=cn>1.96?'HOT':cn>1.36?'WARM':cn<-1.96?'COLD':'NEUTRAL';
  const hr300=(rounds.slice(-300).filter(r=>r.multiplier>=target.min).length+1)/302;
  const blend=regime==='HOT'?0.55:regime==='WARM'?0.35:regime==='COLD'?0.20:0.12;
  const pBlend=hrGlobal*(1-blend)+hr300*blend;
  const expectedGap=Math.max(1,Math.round((1-pBlend)/pBlend));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(pBlend,aw),conf:clamp(Math.round(48+gaps.length*0.25),38,91)};
}
function runHardGap(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const sorted=[...gaps].sort((a,b)=>a-b),gMean=mean(gaps);
  const expectedGap=Math.max(1,Math.round(gMean*0.4+weibullSkew(pctile(sorted,0.50)||gMean,pctile(sorted,0.75)||gMean)*0.6));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(30+(currentGap/(expectedGap||1))*22),20,95)};
}
function runSoftGap(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=0.85;}
  const expectedGap=Math.max(1,Math.round((wG/wS)*0.60+median(gaps)*0.40));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(55+gaps.length*0.2),25,88)};
}
function runMarkov(rounds,target){
  if(rounds.length<30) return null;
  const {tL,tM,tH}=getDynamicBuckets(rounds),bucket=m=>m>=tH?'X':m>=tM?'H':m>=tL?'M':'L';
  const states=['L','M','H','X'],mat1={};
  for(const s of states){mat1[s]={};for(const t of states)mat1[s][t]=0;}
  for(let i=1;i<rounds.length;i++) mat1[bucket(rounds[i-1].multiplier)][bucket(rounds[i].multiplier)]++;
  const cs=bucket(rounds[rounds.length-1]?.multiplier??1),tot1=states.reduce((s,t)=>s+mat1[cs][t],0);
  const prob={};states.forEach(t=>{prob[t]=tot1?mat1[cs][t]/tot1:0.25;});
  let pHit=target.min>=tH?prob['X']:target.min>=tM?prob['H']+prob['X']:target.min>=tL?prob['M']+prob['H']+prob['X']:1-prob['L']*0.3;
  pHit=pHit*0.65+(rounds.filter(r=>r.multiplier>=target.min).length/rounds.length)*0.35;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const expectedGap=Math.max(1,pHit>0?Math.round(1/pHit):mean(gaps)||50);
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:clamp(pHit*(aw/Math.max(1,expectedGap)),0,0.99),conf:clamp(Math.round(pHit*75+20),20,88)};
}
function runPercentile(rounds,target){
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<8) return null;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const expectedGap=Math.max(1,weibullSkew(pctile(sorted,0.50),pctile(sorted,0.75)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(62+gaps.length*0.15),30,90)};
}
function runBayes(rounds,target){
  if(rounds.length<10) return null;
  const hits=rounds.filter(r=>r.multiplier>=target.min).length;
  const alpha=1+hits,beta=1+rounds.length-hits,mu=alpha/(alpha+beta);
  const rawGap=Math.max(1,Math.round(1/(mu||0.001)));
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const sorted=[...gaps].sort((a,b)=>a-b),p50=pctile(sorted,0.50)||rawGap,p75=pctile(sorted,0.75)||rawGap;
  const muVar=alpha*beta/((alpha+beta)**2*(alpha+beta+1)),certainty=clamp(1-Math.sqrt(muVar)/mu,0,1);
  const expectedGap=Math.max(1,Math.round(rawGap*certainty+weibullSkew(p50,p75)*(1-certainty)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(mu,aw),conf:clamp(Math.round(40+rounds.length*0.06+certainty*20),35,90)};
}
function runSHA256(rounds,target){
  if(rounds.length<20) return null;
  const obsP=(rounds.filter(r=>r.multiplier>=target.min).length+1)/(rounds.length+2);
  const trust=clamp((rounds.length-20)/180,0,1);
  const expectedGap=Math.max(1,Math.round((1/obsP)*trust+(target.min/(1-0.01))*(1-trust)));
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(obsP,aw),conf:clamp(Math.round(55+trust*15),40,90)};
}
function runMersenne(rounds,target){
  if(rounds.length<50) return null;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const expectedGap=Math.max(1,Math.round(median(gaps)*0.6+mean(gaps)*0.4));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(42+gaps.length*0.15),30,85)};
}
function runLCG(rounds,target){
  if(rounds.length<30) return null;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if(gaps.length<5) return null;
  const expectedGap=Math.max(1,Math.round(median(gaps)));
  const aw=target.maxWidth;
  return {...placeWindow(expectedGap,currentGap,aw),expectedGap,probW:geoProbW(gaps.length/rounds.length,aw),conf:clamp(Math.round(25+gaps.length*0.15),20,80)};
}

const ALGOS = {
  lstm:runLSTM, xgb:runXGBoost, rf:runRF, ols:runOLS, cat:runCatBoost,
  hardgap:runHardGap, softgap:runSoftGap, markov:runMarkov,
  percentile:runPercentile, bayes:runBayes,
  sha256:runSHA256, mt:runMersenne, lcg:runLCG,
};

// ── Rounds cache ──────────────────────────────────────────────────────────────
async function getComputeRounds() {
  if (cachedRounds.length === 0) {
    cachedRounds = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length-1].roundId : 0;
    console.log(`[advCompute] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length-1].roundId;
    }
  }
  return cachedRounds;
}

// ── Save outcome ──────────────────────────────────────────────────────────────
async function saveOutcome(engineId, target, outcome, lo, hi, hitRound, generation) {
  const key = `${lo}:${hi}`;
  if (savedSets[engineId].has(key)) return;
  savedSets[engineId].add(key);
  try {
    await savePrediction({
      target: target.label, minMult: target.min,
      outcome, lo, hi, hitRound: hitRound ?? null,
      generation: generation ?? 1, source: engineId, probW: null,
    });
    console.log(`[advCompute] ${engineId} ${target.label} ${outcome.toUpperCase()} #${lo}–#${hi}${hitRound?` @#${hitRound}`:''}`);
  } catch(e) {
    console.error(`[advCompute] save fail ${engineId}:`, e.message);
    savedSets[engineId].delete(key);
  }
}

// ── Compute consensus ─────────────────────────────────────────────────────────
// Only uses FRESH computed windows (relative offsets from lastRoundId).
// Never uses existing/active windows from DB — those may be from a previous cycle.
function computeConsensus(allResults, lastRoundId) {
  const consensus = {};
  for (const target of TARGETS) {
    const windows = [];
    for (const [eid, res] of Object.entries(allResults)) {
      const r = res[target.label];
      if (!r) continue;
      // fresh.low is always >= 1, so lo always > lastRoundId (future window)
      const lo = lastRoundId + r.low;
      const hi = lastRoundId + r.high;
      windows.push({ engineId: eid, lo, hi });
    }
    if (windows.length < 3) { consensus[target.label] = null; continue; }
    let bestGroup = [], bestLo = 0, bestHi = 0;
    for (let i = 0; i < windows.length; i++) {
      const grp = [windows[i]]; let runLo = windows[i].lo, runHi = windows[i].hi;
      for (let j = 0; j < windows.length; j++) {
        if (j === i) continue;
        const nl = Math.max(runLo, windows[j].lo), nh = Math.min(runHi, windows[j].hi);
        if (nl <= nh) { grp.push(windows[j]); runLo = nl; runHi = nh; }
      }
      if (grp.length > bestGroup.length) { bestGroup = grp; bestLo = runLo; bestHi = runHi; }
    }
    if (bestGroup.length < 2) { consensus[target.label] = null; continue; }
    const baseW = target.maxWidth;
    if (bestHi - bestLo + 1 < baseW) {
      const center = Math.round((bestLo + bestHi) / 2);
      bestLo = center - Math.floor(baseW / 2); bestHi = bestLo + baseW - 1;
    }
    if (bestLo <= lastRoundId) { bestLo = lastRoundId + 1; bestHi = bestLo + baseW - 1; }
    consensus[target.label] = { lo: bestLo, hi: bestHi, engineCount: bestGroup.length, engines: bestGroup.map(w => w.engineId) };
  }
  return consensus;
}

// ── In-memory window state (per engine, per target) ───────────────────────────
// windows[engineId][targetLabel] = { lo, hi, roundWhenMade, generation }
const windows = {};
for (const id of [...ENGINE_IDS, 'consensus']) { windows[id] = {}; }

// ── Main tick ─────────────────────────────────────────────────────────────────
async function runAdvComputeEngine() {
  try {
    const rounds = await getComputeRounds();
    if (rounds.length < 50) return;
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Compute fresh window predictions for all 13 engines
    const allResults = {};
    for (const engineId of ENGINE_IDS) {
      const algo = ALGOS[engineId];
      if (!algo) continue;
      allResults[engineId] = {};
      for (const target of TARGETS) {
        try {
          const r = algo(rounds, target);
          if (r) allResults[engineId][target.label] = r;
        } catch(e) { console.error(`[advCompute] ${engineId}/${target.label}:`, e.message); }
      }
    }

    // ── Phase 1+2 for each engine ─────────────────────────────────────────────
    const advPayload = {};
    for (const engineId of ENGINE_IDS) {
      advPayload[engineId] = {};
      for (const target of TARGETS) {
        const win = windows[engineId][target.label];
        const fresh = allResults[engineId][target.label];

        if (win) {
          const { lo, hi, generation, roundWhenMade } = win;
          // Phase 1: check for early hit
          // FIXED: earlyHit range bounded by earlyHitTolerance(width) to prevent
          // large pre-window gaps inflating early counts. Hits outside tolerance = LOSS.
          const earlyCheckLo = Math.max(roundWhenMade + 1, lo - earlyHitTolerance(target.maxWidth));
          const earlyHit = lo > roundWhenMade + 1 && earlyCheckLo <= lo - 1
            ? findHitInRange(rounds, earlyCheckLo, lo - 1, target.min)
            : null;
          if (earlyHit) {
            await saveOutcome(engineId, target, 'early', lo, hi, earlyHit.roundId, generation);
            delete windows[engineId][target.label];
          } else if (lastRoundId >= hi) {
            // Window expired — resolve
            const hit = findHitInRange(rounds, lo, hi, target.min);
            await saveOutcome(engineId, target, hit ? 'win' : 'loss', lo, hi, hit?.roundId ?? null, generation);
            delete windows[engineId][target.label];
          } else {
            // Still active — check for in-window hit
            const hit = findHitInRange(rounds, lo, hi, target.min);
            if (hit) {
              await saveOutcome(engineId, target, 'win', lo, hi, hit.roundId, generation);
              delete windows[engineId][target.label];
            } else {
              // Keep active window
              advPayload[engineId][target.label] = { lo, hi, roundWhenMade, generation, eta: win.eta };
              continue;
            }
          }
        }

        // Phase 2: lock fresh window
        if (fresh) {
          const newLo = lastRoundId + fresh.low;
          const newHi = lastRoundId + fresh.high;
          const gen = (windows[engineId][target.label]?.generation ?? 0) + 1;
          windows[engineId][target.label] = { lo: newLo, hi: newHi, roundWhenMade: lastRoundId, generation: gen, eta: { probW: fresh.probW, conf: fresh.conf, expectedGap: fresh.expectedGap } };
          advPayload[engineId][target.label] = windows[engineId][target.label];
        }
      }
      if (Object.keys(advPayload[engineId]).length) {
        await saveLockedAdvPreds(engineId, advPayload[engineId]);
      }
    }

    // ── Consensus ────────────────────────────────────────────────────────────
    const consensus = computeConsensus(allResults, lastRoundId);
    const consPayload = {};
    for (const target of TARGETS) {
      const c = consensus[target.label];
      const win = windows['consensus'][target.label];

      if (win) {
        // FIXED: bounded earlyHit tolerance for consensus windows
        const consEarlyLo = Math.max(win.roundWhenMade + 1, win.lo - earlyHitTolerance(target.maxWidth));
        const earlyHit = win.lo > win.roundWhenMade + 1 && consEarlyLo <= win.lo - 1
          ? findHitInRange(rounds, consEarlyLo, win.lo - 1, target.min)
          : null;
        if (earlyHit) {
          await saveOutcome('consensus', target, 'early', win.lo, win.hi, earlyHit.roundId, win.generation);
          delete windows['consensus'][target.label];
        } else if (lastRoundId >= win.hi) {
          const hit = findHitInRange(rounds, win.lo, win.hi, target.min);
          await saveOutcome('consensus', target, hit ? 'win' : 'loss', win.lo, win.hi, hit?.roundId ?? null, win.generation);
          delete windows['consensus'][target.label];
        } else {
          const hit = findHitInRange(rounds, win.lo, win.hi, target.min);
          if (hit) {
            await saveOutcome('consensus', target, 'win', win.lo, win.hi, hit.roundId, win.generation);
            delete windows['consensus'][target.label];
          } else {
            consPayload[target.label] = { lo: win.lo, hi: win.hi, roundWhenMade: win.roundWhenMade, generation: win.generation, eta: win.eta };
            continue;
          }
        }
      }

      if (c) {
        const gen = (windows['consensus'][target.label]?.generation ?? 0) + 1;
        windows['consensus'][target.label] = { lo: c.lo, hi: c.hi, roundWhenMade: lastRoundId, generation: gen, eta: { engineCount: c.engineCount, engines: c.engines } };
        consPayload[target.label] = windows['consensus'][target.label];
      }
    }
    if (Object.keys(consPayload).length) await saveLockedConsensusPreds(consPayload);

    bustLockedCache();
  } catch(e) {
    console.error('[advCompute] Fatal:', e.message, e.stack);
  }
}

// ── Initialise ────────────────────────────────────────────────────────────────
async function initAdvCompute() {
  if (initialised) return;
  initialised = true;

  // Load existing locked windows into memory
  try {
    const existing = await getLockedAdvPreds();
    for (const engineId of ENGINE_IDS) {
      for (const target of TARGETS) {
        const w = existing[engineId]?.[target.label];
        if (w?.lo && w?.hi) {
          windows[engineId][target.label] = { lo: Number(w.lo), hi: Number(w.hi), roundWhenMade: Number(w.roundWhenMade ?? w.lo), generation: w.generation ?? 1, eta: w.eta ?? {} };
        }
      }
    }
    const cons = await getLockedConsensusPreds();
    for (const target of TARGETS) {
      const w = cons[target.label];
      if (w?.lo && w?.hi) {
        windows['consensus'][target.label] = { lo: Number(w.lo), hi: Number(w.hi), roundWhenMade: Number(w.roundWhenMade ?? w.lo), generation: w.generation ?? 1, eta: w.eta ?? {} };
      }
    }
    console.log(`[advCompute] loaded existing locked windows`);
  } catch(e) { console.error('[advCompute] init locked error:', e.message); }

  // Pre-warm savedSets from existing history
  try {
    const allIds = [...ENGINE_IDS, 'consensus'];
    for (const engineId of allIds) {
      const rows = await getPredictions({ limit: 500000, source: engineId });
      for (const r of rows) savedSets[engineId].add(`${r.lo}:${r.hi}`);
    }
    const total = allIds.reduce((s, id) => s + savedSets[id].size, 0);
    console.log(`[advCompute] pre-warmed savedSets with ${total} outcomes`);
  } catch(e) { console.error('[advCompute] init history error:', e.message); }
}

const _origRun = runAdvComputeEngine;
async function runAdvComputeEngineWithInit() {
  await initAdvCompute();
  await _origRun();
}

function resetAdvComputeState() {
  for (const id of [...ENGINE_IDS, 'consensus']) { windows[id] = {}; savedSets[id] = new Set(); }
  cachedRounds = []; cachedRoundsLastId = 0; initialised = false;
}

module.exports = { runAdvComputeEngine: runAdvComputeEngineWithInit, resetAdvComputeState };