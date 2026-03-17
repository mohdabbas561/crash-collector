'use strict';
// predictionEngine.js  v10
//
// FIXES:
//
// 1. IMMEDIATE REBUILD — per-engine dirty flags replace global lastProcessedRoundId.
//    Each engine (engine/pattern/ens/geo/bay/km) has its own `needsRebuild` flag.
//    When any window resolves, only THAT engine's flag is set dirty.
//    runPredictionEngine() always runs all 6 engines — each checks its own flag.
//    The collector now always calls runPredictionEngine() on every poll tick,
//    regardless of whether new rounds arrived.
//
// 2. NO DUPLICATE HISTORY — makeKey uses only (source, target, lo, hi).
//    Outcome and hitRound excluded from key — window identity is lo+hi only.
//    DB constraint is (source, target, window_lo, window_hi) — same logic.
//    This means one window can only ever produce ONE history record.
//
// 3. INDEPENDENT ENGINES — each engine has its own:
//    - lockedMap (in-memory windows)
//    - savedSet (history keys seen)
//    - needsRebuild flag
//    No shared state between engines except the rounds array.
//
// 4. POLL ALWAYS RUNS ENGINE — collector calls runPredictionEngine() on every
//    tick, not just when new rounds arrive. Engine skips internally if nothing
//    changed and no dirty flags set.

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
const STALE_FORCE_REBUILD_THRESHOLD = 200;

// ── Per-engine state (fully independent) ──────────────────────────────────────

const STATE = {
  engine:  { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  pattern: { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ens:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  geo:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  bay:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  km:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
};

let initialised = false;

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetEngineState() {
  console.log('[engine] resetEngineState()');
  for (const id of Object.keys(STATE)) {
    STATE[id].lockedMap    = null;
    STATE[id].savedSet     = null;
    STATE[id].needsRebuild = true;
    STATE[id].lastRoundId  = 0;
  }
  initialised = false;
}

// ── Core math ─────────────────────────────────────────────────────────────────

function scanRounds(rounds, targetMin) {
  const n = rounds.length;
  let hits = 0, lastIdx = -1;
  const gaps = [];
  for (let i = 0; i < n; i++) {
    if (rounds[i].multiplier >= targetMin) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i;
      hits++;
    }
  }
  if (hits < 3) return null;
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
  const pGlobal = (hits + 1) / (n + 2);
  const r200hits = rounds.slice(-200).filter(r => r.multiplier >= targetMin).length;
  const pRecent  = (r200hits + 1) / 202;

  // CUSUM: detect genuine rate shift
  const p0 = hits / n;
  let cusum = 0, maxCusum = 0;
  const cusumWindow = rounds.slice(-150);
  for (const r of cusumWindow) {
    cusum += (r.multiplier >= targetMin ? 1 : 0) - p0;
    if (Math.abs(cusum) > maxCusum) maxCusum = Math.abs(cusum);
  }
  const sigma = Math.sqrt(Math.max(1e-9, p0 * (1 - p0)));
  const cusumNorm = maxCusum / (sigma * Math.sqrt(cusumWindow.length));
  const rateShifted = cusumNorm > 1.36;

  // Blend: if rate shift detected, weight recent more
  const p = rateShifted
    ? Math.max(1e-6, Math.min(0.5, 0.70 * pGlobal + 0.30 * pRecent))
    : Math.max(1e-6, Math.min(0.5, 0.95 * pGlobal + 0.05 * pRecent));

  const sg = [...gaps].sort((a, b) => a - b);
  const mid = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1/p) :
    sg.length % 2 === 1 ? sg[mid] : (sg[mid-1] + sg[mid]) / 2;
  let gSum = 0; for (const g of gaps) gSum += g;
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / p;
  let gVs = 0; for (const g of gaps) gVs += (g - meanGap) ** 2;
  const cv = meanGap > 0 ? Math.sqrt(gaps.length > 1 ? gVs / gaps.length : meanGap * meanGap) / meanGap : 1;
  const p90 = sg[Math.floor(sg.length * 0.90)] ?? sg[sg.length - 1] ?? meanGap;
  const p95 = sg[Math.floor(sg.length * 0.95)] ?? sg[sg.length - 1] ?? meanGap;

  return { hits, n, p, pGlobal, pRecent, rateShifted, cusumNorm: +cusumNorm.toFixed(3), gapSinceLast, meanGap, medianGap, cv, p90, p95, gaps };
}

function computeConf(probW, hits) {
  let c = probW * 100;
  if (hits < 10)      c -= 15;
  else if (hits < 20) c -= 8;
  else if (hits < 40) c -= 3;
  return Math.max(20, Math.min(88, Math.round(c)));
}

function detectRegime(rounds) {
  const n = rounds.length;
  if (n < 50) return { regime: 'normal', currentStreak: 0 };
  const W = Math.min(100, n);
  const recent = rounds.slice(n - W);
  let rLogSum = 0, gLogSum = 0, lowCount = 0;
  for (const r of recent) { rLogSum += Math.log(Math.max(1.01, r.multiplier)); if (r.multiplier < 2) lowCount++; }
  for (const r of rounds) gLogSum += Math.log(Math.max(1.01, r.multiplier));
  const logRatio = (gLogSum / n) > 0 ? (rLogSum / W) / (gLogSum / n) : 1;
  let cur = 0;
  for (let i = n - 1; i >= 0; i--) { if (rounds[i].multiplier < 5) cur++; else break; }
  const regime = logRatio < 0.82 ? 'cold' : logRatio > 1.18 ? 'hot' : lowCount / W > 0.62 ? 'cold' : 'normal';
  return { regime, currentStreak: cur, logRatio: +logRatio.toFixed(3) };
}

// ── Build functions (each engine is fully independent) ────────────────────────

function buildPrediction(sortedRounds, targetMin, maxWidth) {
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, p, pGlobal, gapSinceLast, medianGap, p90, rateShifted, cusumNorm } = s;
  const probW = 1 - Math.pow(1 - p, maxWidth);
  const regime = detectRegime(sortedRounds);

  // Expected gap = median of historical gaps (more robust than mean for skewed distributions)
  // Weight toward p90 slightly when rate-shifted (conservative)
  const expectedGap = rateShifted
    ? Math.round(medianGap * 0.7 + p90 * 0.3)
    : Math.round(medianGap);

  // Window opens in the last W rounds of the expected gap
  // low = max(0, expectedGap - W), high = low + W - 1
  // This means: "we expect the hit around round expectedGap, and we open a W-round window for it"
  const low  = Math.max(0, expectedGap - maxWidth);
  const high = low + maxWidth - 1;

  return {
    low, high,
    expectedGap,
    opensIn: low, // rounds from anchor until window opens
    confidence: computeConf(probW, hits),
    probW: +probW.toFixed(4),
    p: +p.toFixed(6),
    rateShifted, cusumNorm,
    regime: regime.regime,
    gapSinceLast, hits,
  };
}

function buildStatPrediction(sortedRounds, targetMin, maxWidth, modelId) {
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, n, p, pGlobal, pRecent, gapSinceLast, rateShifted, gaps } = s;

  let pModel;
  if (modelId === 'geo') {
    // Pure Laplace MLE — no recency at all
    pModel = (hits + 1) / (n + 2);

  } else if (modelId === 'bay') {
    // Bayesian: global + recency blend, heavier recency if CUSUM shift
    const recencyW = rateShifted ? 0.35 : 0.10;
    pModel = (1 - recencyW) * pGlobal + recencyW * pRecent;

  } else if (modelId === 'km') {
    // KM: empirical P(gap ≤ maxWidth) from historical gaps
    if (gaps.length < 5) return null;
    const hitsInW = gaps.filter(g => g <= maxWidth).length;
    pModel = (hitsInW + 1) / (gaps.length + 2);

  } else {
    // ENS: independent weighted combination — geo 0.60, bay 0.30, km 0.10
    const pGeo = (hits + 1) / (n + 2);
    const recencyW = rateShifted ? 0.35 : 0.10;
    const pBay = (1 - recencyW) * pGlobal + recencyW * pRecent;
    const pKm = gaps.length >= 5
      ? (gaps.filter(g => g <= maxWidth).length + 1) / (gaps.length + 2)
      : pGeo;
    pModel = 0.60 * pGeo + 0.30 * pBay + 0.10 * pKm;
  }

  pModel = Math.max(1e-6, Math.min(0.999, pModel));
  // KM pModel is already P(gap ≤ W), others need geometric transform
  const probW = modelId === 'km' ? pModel : 1 - Math.pow(1 - pModel, maxWidth);

  // Expected gap from pGlobal (unblended, pure base rate)
  const rawExpectedGap = (1 - pGlobal) / pGlobal;
  // Use median gap for window placement — more robust
  const { medianGap, p90 } = s;
  const expectedGap = rateShifted
    ? Math.round(medianGap * 0.7 + p90 * 0.3)
    : Math.round(medianGap);

  // Window: last W rounds of the expected gap
  const low  = Math.max(0, expectedGap - maxWidth);
  const high = low + maxWidth - 1;

  return {
    low, high,
    expectedGap,
    opensIn: low,
    confidence: computeConf(probW, hits),
    probW: +probW.toFixed(4),
    p: +pModel.toFixed(6),
    rawExpectedGap: +rawExpectedGap.toFixed(1),
    gapSinceLast, hits, rateShifted, model: modelId,
  };
}

function buildPatternPrediction(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;
  const W1=15, W2=50, W3=150;
  const s1=Math.max(0,n-W1), s2=Math.max(0,n-W2), s3=Math.max(0,n-W3);
  let hits=0, lastIdx=-1, hW1=0, hW2=0, hW3=0;
  const gaps=[];
  const FA=0.20, SA=0.02; let emaFast=-1, emaSlow=-1;
  for (let i=0;i<n;i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast<0) { emaFast=isHit; emaSlow=isHit; }
    else { emaFast=FA*isHit+(1-FA)*emaFast; emaSlow=SA*isHit+(1-SA)*emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx=i; hits++;
      if (i>=s1) hW1++;
      if (i>=s2) hW2++;
      if (i>=s3) hW3++;
    }
  }
  if (hits < 8 || gaps.length < 6) return null;
  const globalRate = hits / n;
  let gSum=0; for (const g of gaps) gSum+=g;
  const meanGap = gSum / gaps.length;
  const sg=[...gaps].sort((a,b)=>a-b);
  const mid2=Math.floor(sg.length/2);
  const medianGap=sg.length%2===1?sg[mid2]:(sg[mid2-1]+sg[mid2])/2;
  let gVs=0; for (const g of gaps) gVs+=(g-meanGap)**2;
  const cv=meanGap>0?Math.sqrt(gaps.length>1?gVs/gaps.length:meanGap*meanGap)/meanGap:1;

  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const safe = (v) => Math.max(-1, Math.min(1, v));
  const rW1=globalRate>0?safe((dW1-globalRate)/Math.max(globalRate,0.001)):0;
  const rW2=globalRate>0?safe((dW2-globalRate)/Math.max(globalRate,0.001)):0;
  const rW3=globalRate>0?safe((dW3-globalRate)/Math.max(globalRate,0.001)):0;
  const clusterScore=safe(rW1*0.50+rW2*0.30+rW3*0.20);
  const trendScore=safe((emaSlow>0?(emaFast-emaSlow)/emaSlow:0)*4);

  let varSum=0; for (const g of gaps) varSum+=(g-meanGap)**2;
  let bestAC=0;
  for (let lag=1;lag<=Math.min(3,gaps.length-1);lag++) {
    let cov=0;
    for (let i=lag;i<gaps.length;i++) cov+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac=varSum>0?cov/varSum:0;
    if (Math.abs(ac)>Math.abs(bestAC)) bestAC=ac;
  }
  const patternScore=safe(bestAC*0.9);
  const composite=clusterScore*0.50+trendScore*0.35+patternScore*0.15;
  const absComposite=Math.abs(composite);
  const direction=composite>0.10?'bullish':composite<-0.10?'bearish':'neutral';
  const agree=Math.max(
    [clusterScore,trendScore,patternScore].filter(s=>s>0.10).length,
    [clusterScore,trendScore,patternScore].filter(s=>s<-0.10).length
  );
  const conf=Math.max(25,Math.min(82,Math.round(
    32 + Math.min(18,Math.log2(hits+1)*4) + absComposite*30 + (agree-1)*6 - (cv>1.5?8:cv>1.2?4:0)
  )));
  return { direction, confidence: conf, hits, meanGap: Math.round(meanGap), medianGap: Math.round(medianGap), composite:+composite.toFixed(3) };
}

function buildPatternWindow(patternResult, maxWidth, sortedRounds, targetMin) {
  if (!patternResult) return null;
  // Pattern uses its own meanGap as expected gap
  const expectedGap = patternResult.medianGap || patternResult.meanGap || maxWidth;
  const low  = Math.max(0, expectedGap - maxWidth);
  const high = low + maxWidth - 1;
  return {
    low, high,
    expectedGap,
    opensIn: low,
    confidence: patternResult.confidence,
    direction: patternResult.direction,
  };
}

// ── makeKey — window identity only (no outcome/hitRound to prevent dupes) ─────

function makeKey(source, target, lo, hi) {
  return `${source}-${target}-${Number(lo)||0}-${Number(hi)||0}`;
}

// ── getStatus ─────────────────────────────────────────────────────────────────

function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow)
    return { status: 'miss', hitRound: null };

  // Binary search to round >= anchorRound
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

// ── processEngine — runs ONE engine, returns whether DB needs saving ──────────

async function processEngine({ engineId, state, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = state.lockedMap[target.label];

    // No window yet — build one
    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId + 1, generation:1, stale:false };
        anyChange = true;
        console.log(`[${engineId}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'}`);
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
      // Record outcome if resolvable
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
          // Key: window identity only (no outcome/hitRound) — prevents dupes
          const key = makeKey(engineId, target.label, absLow, absHigh);
          if (!state.savedSet.has(key)) {
            state.savedSet.add(key);
            try {
              await savePrediction({
                target: target.label, minMult: target.min, outcome,
                lo: absLow, hi: absHigh, anchorRound,
                hitRound: status.hitRound || null,
                generation: existing.generation || 1,
                source: engineId,
              });
              console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
            } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
          }
        }
      }

      // Rebuild window — anchor to lastRoundId+1 so new window never overlaps the just-resolved hit
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min, anchorRound: lastRoundId + 1,
          generation: (existing.generation||1) + (isNonsense ? 0 : 1),
          stale: false,
        };
        console.log(`[${engineId}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete state.lockedMap[target.label];
        console.warn(`[${engineId}] ${target.label} cleared — insufficient data`);
      }
      anyChange = true;
      state.needsRebuild = false; // handled
      continue;
    }

    // Window still active/waiting — check if it just resolved NOW
    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit' ? 'win' : status.status==='early' ? 'early' : 'loss';
      const key = makeKey(engineId, target.label, absLow, absHigh);
      if (!state.savedSet.has(key)) {
        state.savedSet.add(key);
        try {
          await savePrediction({
            target: target.label, minMult: target.min, outcome,
            lo: absLow, hi: absHigh, anchorRound,
            hitRound: status.hitRound || null,
            generation: existing.generation || 1,
            source: engineId,
          });
          console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
        } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
      }
      // Immediately build next window — anchor to lastRoundId+1 so it starts AFTER the hit
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min, anchorRound: lastRoundId + 1,
          generation: (existing.generation||1) + 1, stale: false,
        };
        console.log(`[${engineId}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}%`);
      } else {
        delete state.lockedMap[target.label];
      }
      anyChange = true;
    }
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
      eta:           { low: pred.low, high: pred.high, conf: pred.confidence, probW: pred.probW, expectedGap: pred.expectedGap, opensIn: pred.opensIn },
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
      low:         eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor),
      high:        eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor),
      confidence:  eta.conf ?? 50,
      probW:       eta.probW ?? null,
      expectedGap: eta.expectedGap ?? null,
      opensIn:     eta.opensIn ?? null,
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

  // Init all saved sets
  for (const id of Object.keys(STATE)) {
    STATE[id].savedSet = new Set();
  }

  // Load locked preds from DB
  try {
    STATE.engine.lockedMap = loadLockedMap(await getLockedPreds());
    console.log(`[engine] loaded ${Object.keys(STATE.engine.lockedMap).length} engine preds`);
  } catch(e) { console.error('[engine] init error:', e.message); STATE.engine.lockedMap = {}; }

  try {
    STATE.pattern.lockedMap = loadLockedMap(await getLockedPatternPreds());
    console.log(`[engine] loaded ${Object.keys(STATE.pattern.lockedMap).length} pattern preds`);
  } catch(e) { console.error('[engine] pattern init error:', e.message); STATE.pattern.lockedMap = {}; }

  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      STATE[model.id].lockedMap = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] loaded ${Object.keys(STATE[model.id].lockedMap).length} ${model.id} preds`);
    }
  } catch(e) {
    console.error('[engine] stat init error:', e.message);
    for (const model of STAT_MODELS) STATE[model.id].lockedMap = {};
  }

  // Load history keys — key format: source-target-lo-hi (no outcome/hitRound)
  try {
    const rows = await getPredictions({ limit: 10000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r.target, r.lo, r.hi);
      if (STATE[src]?.savedSet) STATE[src].savedSet.add(key);
    }
    console.log(`[engine] loaded ${rows.length} history keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  // All engines start dirty on first run
  for (const id of Object.keys(STATE)) STATE[id].needsRebuild = true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function runPredictionEngine() {
  try {
    await initialise();

    const rounds = await getRounds({ limit: 5000, order: 'DESC' });
    if (rounds.length < MIN_ROUNDS) {
      console.log(`[engine] waiting for rounds (${rounds.length}/${MIN_ROUNDS})`);
      return;
    }
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // Each engine runs independently — checks its own lastRoundId + needsRebuild
    const allEngines = [
      {
        id: 'engine',
        state: STATE.engine,
        buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedPreds(p); },
      },
      {
        id: 'pattern',
        state: STATE.pattern,
        buildFn: (t) => { const pp = buildPatternPrediction(rounds, t.min); return buildPatternWindow(pp, t.maxWidth, rounds, t.min); },
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedPatternPreds(p); },
      },
      ...STAT_MODELS.map(model => ({
        id: model.id,
        state: STATE[model.id],
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth + model.wOffset, model.id),
        saveFn:  async (p) => { if (Object.keys(p).length) await saveLockedStatPreds(model.id, p); },
      })),
    ];

    for (const eng of allEngines) {
      const roundAdvanced = lastRoundId > eng.state.lastRoundId;
      const shouldRun     = roundAdvanced || eng.state.needsRebuild;
      if (!shouldRun) continue;

      eng.state.needsRebuild = false;

      const changed = await processEngine({
        engineId:     eng.id,
        state:        eng.state,
        sortedRounds: rounds,
        lastRoundId,
        buildFn:      eng.buildFn,
      });

      eng.state.lastRoundId = lastRoundId;

      if (changed) {
        const p = buildSavePayload(eng.state.lockedMap);
        try { await eng.saveFn(p); }
        catch(e) { console.error(`[${eng.id}] save locked error:`, e.message); }
      }
    }

  } catch(e) {
    console.error('[predictionEngine] Fatal:', e.message, e.stack);
  }
}

function getLockedStatMap(modelId) {
  return STATE[modelId]?.lockedMap || {};
}

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap };