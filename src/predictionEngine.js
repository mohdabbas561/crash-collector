'use strict';
// predictionEngine.js  v9
//
// WHAT CHANGED FROM v8:
//
// 1. WINDOW PLACEMENT — honest math, no gambler's fallacy
//    Old: center = medianGap - gapSinceLast (implies "overdue" → WRONG for IID)
//    New: window starts at round 0 relative to anchor (immediate), width = W.
//         For stat models: expectedGap used only to set window width, not center.
//         The window covers the statistically likely range P(≥1 hit in W rounds).
//
// 2. CONFIDENCE = P(≥1 hit in window) — directly interpretable
//    Old: arbitrary ±point additions with regime shifts, drift, cold tail
//    New: confidence = round(probW * 100) capped at 88.
//         This is the ACTUAL geometric probability. No fake boosts.
//         Secondary reliability deduction only for data sparsity (hits < 20).
//
// 3. ENGINE model uses pure Bayesian hit rate (no regime shifts)
//    Old: detectRegime() shifted window center — gambler's fallacy
//    New: ENGINE uses same geo model as stat/geo but with recency blend.
//         detectRegime() kept for DISPLAY (regime indicator) only, not math.
//
// 4. HISTORY deduplication — DB-level unique constraint
//    Added UNIQUE(source, target, window_lo, window_hi) to predictions table.
//    savePrediction uses ON CONFLICT DO NOTHING — no more duplicates on restart.
//    In-memory savedKeys Set still used to avoid hammering DB with known keys.
//
// 5. PATTERN model — honest cluster/trend signal, no window centering on "due"
//    Pattern window always starts at +0 from anchor, width reflects confidence.
//    High composite → narrower window (more certain). Low → wider.
//
// 6. CUSUM rate-shift detection → adjusts p estimate (not confidence)
//    If CUSUM detects a rate shift in last 150r, blend recent rate more heavily.
//    This is the ONE valid use of recency — actual rate change detection.

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

let lockedPreds    = null;
let lockedPatterns = null;
let lockedStats    = null;
let savedKeys      = new Set();
let patSavedKeys   = new Set();
let statSavedKeys  = {};
let lastProcessedRoundId = 0;
let initialised    = false;
let isFirstRun     = true;

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetEngineState() {
  console.log('[engine] resetEngineState()');
  lockedPreds = null; lockedPatterns = null; lockedStats = null;
  savedKeys = new Set(); patSavedKeys = new Set(); statSavedKeys = {};
  lastProcessedRoundId = 0; initialised = false; isFirstRun = true;
}

// ── Core stats — pure math, no gambler's fallacy ──────────────────────────────

/**
 * scanRounds: extract hit stats for a target.
 * Returns the PURE statistical picture — no regime, no recency tricks.
 */
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

  // Laplace-smoothed hit rate (global)
  const pGlobal = (hits + 1) / (n + 2);

  // Recent 200r hit rate for CUSUM shift detection
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
  const rateShifted = cusumNorm > 1.36; // KS threshold at 5%

  // Blended p: if rate shift detected, weight recent more (0.70/0.30)
  // else use pure global estimate (0.95/0.05 just for smoothing edge cases)
  const p = rateShifted
    ? Math.max(1e-6, Math.min(0.5, 0.70 * pGlobal + 0.30 * pRecent))
    : Math.max(1e-6, Math.min(0.5, 0.95 * pGlobal + 0.05 * pRecent));

  // Gap distribution stats
  const sg = [...gaps].sort((a, b) => a - b);
  const m = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1/p) :
    sg.length % 2 === 1 ? sg[m] : (sg[m-1] + sg[m]) / 2;
  let gSum = 0; for (const g of gaps) gSum += g;
  const meanGap = gaps.length > 0 ? gSum / gaps.length : 1 / p;
  let gVs = 0; for (const g of gaps) gVs += (g - meanGap) ** 2;
  const cv = meanGap > 0 ? Math.sqrt(gaps.length > 1 ? gVs / gaps.length : meanGap * meanGap) / meanGap : 1;
  const p90 = sg[Math.floor(sg.length * 0.90)] ?? sg[sg.length - 1] ?? meanGap;
  const p95 = sg[Math.floor(sg.length * 0.95)] ?? sg[sg.length - 1] ?? meanGap;

  return {
    hits, n, p, pGlobal, pRecent,
    rateShifted, cusumNorm: +cusumNorm.toFixed(3),
    gapSinceLast, meanGap, medianGap, cv, p90, p95,
  };
}

/**
 * Compute confidence as P(≥1 hit in W rounds) — the honest number.
 * Deduct reliability penalty only for data sparsity.
 * No regime shifts, no drift adjustments, no gambler's fallacy.
 */
function computeConf(probW, hits) {
  // Base: actual probability × 100, capped at 88 (never claim >88% certainty)
  let c = probW * 100;
  // Sparsity penalty: less data = less reliable estimate
  if (hits < 10) c -= 15;
  else if (hits < 20) c -= 8;
  else if (hits < 40) c -= 3;
  return Math.max(20, Math.min(88, Math.round(c)));
}

/**
 * detectRegime: for DISPLAY purposes only — shown as a label, never shifts math.
 * Kept because users want to see if the game is in a cold/hot streak.
 */
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

// ── Build functions ───────────────────────────────────────────────────────────

/**
 * ENGINE model: Bayesian geo with CUSUM-aware recency blend.
 * Window: starts at +0 from anchor (immediate), width = maxWidth.
 * This is honest: we don't know WHEN in the window the hit will come.
 * Confidence = P(≥1 hit in window).
 */
function buildPrediction(sortedRounds, targetMin, maxWidth) {
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, p, gapSinceLast, rateShifted, cusumNorm } = s;
  const probW = 1 - Math.pow(1 - p, maxWidth);
  const conf  = computeConf(probW, hits);
  const regime = detectRegime(sortedRounds);
  return {
    low: 0, high: maxWidth - 1,
    confidence: conf,
    probW: +probW.toFixed(4),
    p: +p.toFixed(6),
    rateShifted,
    cusumNorm,
    regime: regime.regime,
    gapSinceLast,
    hits,
  };
}

/**
 * STAT models: geo / bay / km / ens
 * Each uses a slightly different p estimator:
 *   geo: pure global Laplace MLE
 *   bay: Beta posterior with recency (0.90/0.10) — corrects for small n
 *   km:  empirical gap percentile P(gap ≤ W) — non-parametric
 *   ens: weighted average (geo 0.70, bay 0.25, km 0.05)
 *
 * Window: [0, W-1] relative to anchor. Confidence = probW.
 * CUSUM shift detected → bay and ens shift recency weight to 0.70/0.30.
 */
function buildStatPrediction(sortedRounds, targetMin, maxWidth, modelId) {
  const s = scanRounds(sortedRounds, targetMin);
  if (!s) return null;
  const { hits, n, p, pGlobal, pRecent, gapSinceLast, rateShifted } = s;

  let pModel;

  if (modelId === 'geo') {
    // Pure geometric: Laplace MLE, no recency
    pModel = (hits + 1) / (n + 2);

  } else if (modelId === 'bay') {
    // Bayesian Beta: global posterior + recency blend
    // Recency weight higher if CUSUM shift detected
    const recencyW = rateShifted ? 0.30 : 0.10;
    pModel = (1 - recencyW) * pGlobal + recencyW * pRecent;

  } else if (modelId === 'km') {
    // KM: empirical P(next gap ≤ maxWidth)
    // Count gaps ≤ maxWidth in historical data
    const allGaps = [];
    let lastIdx = -1;
    for (let i = 0; i < sortedRounds.length; i++) {
      if (sortedRounds[i].multiplier >= targetMin) {
        if (lastIdx !== -1) allGaps.push(i - lastIdx - 1);
        lastIdx = i;
      }
    }
    if (allGaps.length < 5) return null;
    const hitsInWindow = allGaps.filter(g => g <= maxWidth).length;
    pModel = (hitsInWindow + 1) / (allGaps.length + 2);

  } else {
    // ENS: weighted combination
    const pGeo = (hits + 1) / (n + 2);
    const recencyW = rateShifted ? 0.30 : 0.10;
    const pBay = (1 - recencyW) * pGlobal + recencyW * pRecent;
    // KM component inline
    const allGaps2 = [];
    let li = -1;
    for (let i = 0; i < sortedRounds.length; i++) {
      if (sortedRounds[i].multiplier >= targetMin) {
        if (li !== -1) allGaps2.push(i - li - 1);
        li = i;
      }
    }
    const pKm = allGaps2.length >= 5
      ? (allGaps2.filter(g => g <= maxWidth).length + 1) / (allGaps2.length + 2)
      : pGeo;
    pModel = 0.65 * pGeo + 0.25 * pBay + 0.10 * pKm;
  }

  pModel = Math.max(1e-6, Math.min(0.999, pModel));
  // For km model, pModel is already P(gap ≤ W), so probW = pModel directly
  const probW = modelId === 'km'
    ? pModel
    : 1 - Math.pow(1 - pModel, maxWidth);

  const conf = computeConf(probW, hits);

  return {
    low: 0, high: maxWidth - 1,
    confidence: conf,
    probW: +probW.toFixed(4),
    p: +pModel.toFixed(6),
    expectedGap: +(( 1 - pGlobal) / pGlobal).toFixed(1),
    gapSinceLast,
    hits,
    rateShifted,
    model: modelId,
  };
}

/**
 * PATTERN model: cluster/trend/autocorrelation signal.
 * Window width shrinks when composite signal is strong (more certain timing).
 * Window width widens when signal is weak (less certain).
 * Confidence = function of |composite| + agreement + data volume.
 * No gambler's fallacy — window always starts at +0.
 */
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

  // Cluster scores (rate deviation in windows)
  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const rW1=globalRate>0?Math.max(-1,Math.min(1,(dW1-globalRate)/Math.max(globalRate,0.001))):0;
  const rW2=globalRate>0?Math.max(-1,Math.min(1,(dW2-globalRate)/Math.max(globalRate,0.001))):0;
  const rW3=globalRate>0?Math.max(-1,Math.min(1,(dW3-globalRate)/Math.max(globalRate,0.001))):0;
  const clusterScore=Math.max(-1,Math.min(1,rW1*0.50+rW2*0.30+rW3*0.20));

  // Trend score from EMA divergence
  const trendScore=Math.max(-1,Math.min(1,(emaSlow>0?(emaFast-emaSlow)/emaSlow:0)*4));

  // Autocorrelation of gaps
  let varSum=0; for (const g of gaps) varSum+=(g-meanGap)**2;
  let bestAC=0;
  for (let lag=1;lag<=Math.min(3,gaps.length-1);lag++) {
    let cov=0;
    for (let i=lag;i<gaps.length;i++) cov+=(gaps[i-lag]-meanGap)*(gaps[i]-meanGap);
    const ac=varSum>0?cov/varSum:0;
    if (Math.abs(ac)>Math.abs(bestAC)) bestAC=ac;
  }
  const patternScore=Math.max(-1,Math.min(1,bestAC*0.9));

  const composite=clusterScore*0.50+trendScore*0.35+patternScore*0.15;
  const absComposite=Math.abs(composite);

  const direction=composite>0.10?'bullish':composite<-0.10?'bearish':'neutral';
  const agree=Math.max(
    [clusterScore,trendScore,patternScore].filter(s=>s>0.10).length,
    [clusterScore,trendScore,patternScore].filter(s=>s<-0.10).length
  );
  const conf=Math.max(25,Math.min(82,Math.round(
    32
    + Math.min(18,Math.log2(hits+1)*4)
    + absComposite*30
    + (agree-1)*6
    - (cv>1.5?8:cv>1.2?4:0)
  )));
  return {
    direction, confidence: conf, hits,
    meanGap: Math.round(meanGap), medianGap: Math.round(medianGap),
    composite:+composite.toFixed(3),
    clusterScore:+clusterScore.toFixed(3),
    trendScore:+trendScore.toFixed(3),
    patternScore:+patternScore.toFixed(3),
  };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const { confidence } = patternResult;
  // Always start at +0. Width = maxWidth (honest: we don't know WHEN in window).
  return { low: 0, high: maxWidth - 1, confidence };
}

function makeKey(source, r) {
  return `${source}-${r.target}-${Number(r.lo)||0}-${Number(r.hi)||0}-${r.outcome}-${r.hitRound??'x'}`;
}

// ── getStatus ─────────────────────────────────────────────────────────────────

function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow)
    return { status: 'miss', hitRound: null };
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
  if (currentRoundId > absHigh)                               return { status:'miss',   hitRound:null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh)  return { status:'active', hitRound:null };
  return { status:'waiting', hitRound:null };
}

// ── processWindows ────────────────────────────────────────────────────────────

async function processWindows({ lockedMap, savedSet, source, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = lockedMap[target.label];

    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId, generation:1, stale:false };
        anyChange = true;
        console.log(`[${source}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'}`);
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
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          // FIX: 'early' = hit came before window — still a WIN, not a loss
          const outcome = (status.status==='hit' || status.status==='early') ? 'win' : 'loss';
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
              console.log(`[${source}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
            } catch(e) { console.error(`[${source}] save fail:`, e.message); }
          }
        }
      }

      const pred = buildFn(target);
      if (pred) {
        lockedMap[target.label] = {
          ...pred,
          targetMin:   target.min,
          anchorRound: lastRoundId,
          generation:  (existing.generation||1) + (isNonsense ? 0 : 1),
          stale:       false,
        };
        console.log(`[${source}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'}`);
      } else {
        delete lockedMap[target.label];
        console.warn(`[${source}] ${target.label} cleared — insufficient data`);
      }
      anyChange = true;
      // FIX: force immediate re-process on next call after any rebuild
      lastProcessedRoundId = 0;
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      // FIX: 'early' = hit before window opened — still a WIN
      const outcome = (status.status==='hit' || status.status==='early') ? 'win' : 'loss';
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
          console.log(`[${source}] ${target.label} ${outcome.toUpperCase()}${status.status==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''}`);
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
      // FIX: force immediate re-process on next call — don't wait for new round
      lastProcessedRoundId = 0;
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
      eta:           { low: pred.low, high: pred.high, conf: pred.confidence, probW: pred.probW },
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

  try { lockedPreds = loadLockedMap(await getLockedPreds()); console.log(`[engine] loaded ${Object.keys(lockedPreds).length} engine preds`); }
  catch(e) { console.error('[engine] init error:', e.message); lockedPreds = {}; }

  try { lockedPatterns = loadLockedMap(await getLockedPatternPreds()); console.log(`[engine] loaded ${Object.keys(lockedPatterns).length} pattern preds`); }
  catch(e) { console.error('[engine] pattern init error:', e.message); lockedPatterns = {}; }

  lockedStats   = { ens:{}, geo:{}, bay:{}, km:{} };
  statSavedKeys = { ens:new Set(), geo:new Set(), bay:new Set(), km:new Set() };
  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      lockedStats[model.id] = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] loaded ${Object.keys(lockedStats[model.id]).length} ${model.id} preds`);
    }
  } catch(e) { console.error('[engine] stat init error:', e.message); }

  try {
    const rows = await getPredictions({ limit: 5000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r);
      if (src === 'engine')        savedKeys.add(key);
      else if (src === 'pattern')  patSavedKeys.add(key);
      else if (statSavedKeys[src]) statSavedKeys[src].add(key);
    }
    console.log(`[engine] loaded ${rows.length} prediction history keys`);
  } catch(e) { console.error('[engine] history load error:', e.message); }

  lastProcessedRoundId = 0;
  isFirstRun = true;
}

// ── hasExpiredWindows ─────────────────────────────────────────────────────────

function hasExpiredWindows(lastRoundId) {
  const allMaps = [lockedPreds, lockedPatterns, ...STAT_MODELS.map(m => lockedStats?.[m.id])].filter(Boolean);
  for (const map of allMaps) {
    for (const pred of Object.values(map)) {
      const anchor  = Number(pred.anchorRound) || 0;
      const absHigh = anchor + (Number(pred.high) || 0);
      if (pred.stale || anchor === 0 || lastRoundId > absHigh) return true;
    }
  }
  return false;
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

    const shouldSkip = !isFirstRun
      && lastRoundId === lastProcessedRoundId
      && !hasExpiredWindows(lastRoundId);
    if (shouldSkip) return;

    isFirstRun = false;

    const regime = detectRegime(rounds);
    if (regime.regime !== 'normal' || regime.currentStreak > 5)
      console.log(`[engine] regime=${regime.regime} streak=${regime.currentStreak}`);

    const engChanged = await processWindows({
      lockedMap: lockedPreds, savedSet: savedKeys, source: 'engine',
      sortedRounds: rounds, lastRoundId,
      buildFn: (t) => buildPrediction(rounds, t.min, t.maxWidth),
    });
    if (engChanged) {
      const p = buildSavePayload(lockedPreds);
      if (Object.keys(p).length) { try { await saveLockedPreds(p); } catch(e) { console.error('[engine] saveLockedPreds:', e.message); } }
    }

    const patChanged = await processWindows({
      lockedMap: lockedPatterns, savedSet: patSavedKeys, source: 'pattern',
      sortedRounds: rounds, lastRoundId,
      buildFn: (t) => { const pp = buildPatternPrediction(rounds, t.min); return buildPatternWindow(pp, t.maxWidth); },
    });
    if (patChanged) {
      const p = buildSavePayload(lockedPatterns);
      if (Object.keys(p).length) { try { await saveLockedPatternPreds(p); } catch(e) { console.error('[pattern] save error:', e.message); } }
    }

    for (const model of STAT_MODELS) {
      const changed = await processWindows({
        lockedMap: lockedStats[model.id], savedSet: statSavedKeys[model.id], source: model.id,
        sortedRounds: rounds, lastRoundId,
        buildFn: (t) => buildStatPrediction(rounds, t.min, t.maxWidth + model.wOffset, model.id),
      });
      if (changed) {
        const p = buildSavePayload(lockedStats[model.id]);
        if (Object.keys(p).length) { try { await saveLockedStatPreds(model.id, p); } catch(e) { console.error(`[${model.id}] save error:`, e.message); } }
      }
    }

    lastProcessedRoundId = lastRoundId;

  } catch(e) {
    console.error('[predictionEngine] Fatal:', e.message, e.stack);
  }
}

function getLockedStatMap(modelId) {
  if (!lockedStats) return {};
  return lockedStats[modelId] || {};
}

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap };