'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
// ================================================================================
// ACCURACY & CALIBRATION REBUILD — v6 WHITE-CLUSTER & RECALCULATE FIX
//
// v6 FIX SUMMARY (over v5 FINAL PRODUCTION):
//
// FIX-1: NG_ENGINE_IDS restored to all 11 trackable IDs.
//   BUG: Only 'ng_consensus' was in NG_ENGINE_IDS. Individual engines were computed
//   into allNgResults but never entered the Phase 1+2 resolve+lock loop. ngWindows
//   for them were never populated. computeNgConsensus had no persistent window state
//   to intersect → random short windows on every tick. FIX: restore full list.
//
// FIX-2: Cold-path upgrades (hot/b2b untouched):
//   - coldScore trigger: 58 → 60 (less noise, still catches cluster at depth 4-5)
//   - tc for cold blend start: 58 → 60 (aligned with trigger)
//   - predictedGapMultiplier cold range: 1.12–1.32 (safer than 1.15–1.40)
//   - getCalibratedAdjustment: margin=0.018, sensiCold=1.7, maxDeltaCold=0.32
//   - calibMult cold ceiling: 1.32 (was 1.40)
//   - extractPredictiveStreakFeatures coldScore weights:
//       streakMomentum: +20% → 42/21 (was 35/17)
//       lowDensityAccel: +25% → 36/18 (was 29/14)
//   - applyStreakAdjustment: ABOUT_TO_WHITE_CLUSTER → 1.12–1.32, blend at tc=60
//
// FIX-3: Recalculate-on-miss logic in main tick.
//   When lastRoundId >= hi and no hit found: window deleted, fresh prediction
//   immediately generated for that target using current allNgResults, new window
//   locked in same tick. No waiting for next tick.
//
// All v5 hardening fully preserved:
//   Wilson CI guards, CALIB_DUMMY sentinel, null-guards, 0.76 hot floor,
//   _calibLogCounter, minContext per-target, hotScore trigger at 72,
//   DB/locking/UI/export 100% untouched.
// ================================================================================

const {
  getRounds, savePrediction, getPredictions,
  saveLockedAdvPreds, getLockedAdvPreds,
} = require('./db');

// === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
// FIX-1: Restored all 11 engine IDs.
// BUG: Only 'ng_consensus' was listed. The 10 individual engines were computed
// but never entered the Phase 1+2 resolve+lock loop — ngWindows for them was
// never populated, ngSavedSets never tracked them, saveLockedAdvPreds never
// persisted them. computeNgConsensus received empty/stale data → random windows.
// FIX: All 11 IDs now drive ngSavedSets init, ngWindows init, and the lock loop.
// 'stacking' is computed internally by runStackingMeta and fed into allNgResults
// but is NOT in NG_ENGINE_IDS (it feeds the consensus, not a standalone tracker).
const NG_ENGINE_IDS = [
  'hlstm_xgb',
  'htrans_lstm',
  'htft',
  'tft',
  'nbeats',
  'tcn',
  'lgbm',
  'gru',
  'bilstm',
  'sha512',
  'ng_consensus',
];
// === FIX END ===

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

let cachedRounds = [], cachedRoundsLastId = 0, initialised = false;

function earlyHitTolerance(width) { return Math.floor(width / 2); }

// =============================================================================
// MATH HELPERS
// =============================================================================
function mean(arr) { return arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0; }
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

// H-2: wilsonLower — Wilson score 95% CI lower bound. Guards: n≤0, NaN, Inf.
function wilsonLower(p, n) {
  if (n <= 0 || !isFinite(n) || isNaN(p)) return 0;
  const z = 1.96;
  const z2 = z * z;
  const inner = p * (1 - p) / n + z2 / (4 * n * n);
  const sqrtTerm = Math.sqrt(Math.max(0, inner));
  const num = p + z2 / (2 * n) - z * sqrtTerm;
  const den = 1 + z2 / n;
  const lower = num / den;
  if (isNaN(lower) || !isFinite(lower)) return 0;
  return Math.max(0, lower);
}

// H-1: CALIB_DUMMY sentinel — prevents circularity in computeCalibration.
const CALIB_DUMMY = Object.freeze({
  hotHitRate:   new Array(10).fill(null),
  coldHitRate:  new Array(10).fill(null),
  hotBinCount:  new Array(10).fill(0),
  coldBinCount: new Array(10).fill(0),
  BIN_SIZE:     10,
  minBin:       20,
  margin:       0.025,
  baseline:     0.5,
  LOOK_AHEAD:   20,
  targetRare:   false,
});

const TARGET_LOOK_AHEAD = {
  '5x':    20,
  '10x':   20,
  '20x':   20,
  '50x':   40,
  '100x':  80,
  '250x': 150,
  '500x': 300,
  '1000x':300,
};

const MIN_BIN_NON_RARE = 20;
const MIN_BIN_RARE      = 30;

// H-9: Module-level calibration log counter.
let _calibLogCounter = 0;

const calibCache = {};

// =============================================================================
// computeCalibration v5 (unchanged from v5 FINAL PRODUCTION)
// =============================================================================
function computeCalibration(rounds, targetMin, targetLabel, targetRare) {
  const cacheKey = targetLabel;
  const now = Date.now();
  const cache = calibCache[cacheKey];
  const delta = cache ? rounds.length - cache.computedAt : Infinity;
  const age   = cache ? now - cache.computedAtMs : Infinity;
  if (cache && delta < 50 && age < 600000) {
    return cache.result;
  }

  const n = rounds.length;
  const LOOK_AHEAD = TARGET_LOOK_AHEAD[targetLabel] || 20;
  const BIN_SIZE   = 10;
  const NUM_BINS   = 10;
  const minBin     = targetRare ? MIN_BIN_RARE : MIN_BIN_NON_RARE;

  const globalHitRate = rounds.filter(r => r.multiplier >= targetMin).length /
    Math.max(1, n);
  const baseline = clamp(
    1 - Math.pow(Math.max(0, 1 - globalHitRate), LOOK_AHEAD),
    0.01, 0.99
  );

  // H-3: Per-target minimum context.
  const expectedGapApprox = globalHitRate > 0 ? Math.round(1 / globalHitRate) : n;
  const minContext = Math.min(
    Math.floor(n / 4),
    Math.max(60, LOOK_AHEAD * 2, expectedGapApprox * 3)
  );
  const maxPos = n - LOOK_AHEAD;
  if (maxPos <= minContext) {
    const emptyResult = {
      hotHitRate: new Array(NUM_BINS).fill(null),
      coldHitRate: new Array(NUM_BINS).fill(null),
      baseline, hotBinCount: new Array(NUM_BINS).fill(0),
      coldBinCount: new Array(NUM_BINS).fill(0),
      LOOK_AHEAD, BIN_SIZE, minBin, margin: targetRare ? 0.015 : 0.025, targetRare,
    };
    calibCache[cacheKey] = { computedAt: n, computedAtMs: now, result: emptyResult };
    return emptyResult;
  }

  const hotBinHitsW   = new Array(NUM_BINS).fill(0);
  const hotBinTotalW  = new Array(NUM_BINS).fill(0);
  const hotBinCount   = new Array(NUM_BINS).fill(0);
  const coldBinHitsW  = new Array(NUM_BINS).fill(0);
  const coldBinTotalW = new Array(NUM_BINS).fill(0);
  const coldBinCount  = new Array(NUM_BINS).fill(0);

  for (let pos = maxPos; pos >= minContext; ) {
    const distFromEnd = n - pos;
    const recWeight = Math.pow(0.999, distFromEnd);
    const ctx = rounds.slice(0, pos);

    // H-1: CALIB_DUMMY prevents circularity.
    let sf = null;
    try { sf = extractPredictiveStreakFeatures(ctx, targetMin, CALIB_DUMMY); }
    catch(_) {}

    if (!sf) {
      const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
      pos -= stride;
      continue;
    }

    const hs = sf.hotScore;
    const cs = sf.coldScore;

    if (!isFinite(hs) || !isFinite(cs)) {
      const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
      pos -= stride;
      continue;
    }

    const hotBin  = Math.min(NUM_BINS-1, Math.floor(hs / BIN_SIZE));
    const coldBin = Math.min(NUM_BINS-1, Math.floor(cs / BIN_SIZE));

    const futureHit = rounds.slice(pos, pos + LOOK_AHEAD).some(r => r.multiplier >= targetMin);
    const noFutureHit = !futureHit;

    hotBinHitsW[hotBin]   += futureHit   ? recWeight : 0;
    hotBinTotalW[hotBin]  += recWeight;
    hotBinCount[hotBin]   += 1;

    coldBinHitsW[coldBin]  += noFutureHit ? recWeight : 0;
    coldBinTotalW[coldBin] += recWeight;
    coldBinCount[coldBin]  += 1;

    const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
    pos -= stride;
  }

  const hotMargin  = targetRare ? 0.015 : 0.025;
  const COLD_MIN_BIN = 15;
  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: cold calibration margin tightened 0.012→0.018.
  // 0.012 was too liberal — bins with marginal signal were unlocking cold path.
  // 0.018 requires a more credible empirical cold signal before firing.
  const coldMargin  = 0.018;
  // === FIX END ===
  const hotHitRate  = new Array(NUM_BINS).fill(null);
  const coldHitRate = new Array(NUM_BINS).fill(null);

  for (let b = 0; b < NUM_BINS; b++) {
    if (hotBinCount[b] >= minBin && hotBinTotalW[b] > 0) {
      const wr = hotBinHitsW[b] / hotBinTotalW[b];
      const wl = wilsonLower(wr, hotBinCount[b]);
      if (wl > baseline + hotMargin) hotHitRate[b] = wr;
    }
    if (coldBinCount[b] >= COLD_MIN_BIN && coldBinTotalW[b] > 0) {
      const wr = coldBinHitsW[b] / coldBinTotalW[b];
      const wl = wilsonLower(wr, coldBinCount[b]);
      if (wl > (1 - baseline) + coldMargin) coldHitRate[b] = wr;
    }
  }

  const result = {
    hotHitRate, coldHitRate, baseline,
    hotBinCount, coldBinCount,
    LOOK_AHEAD, BIN_SIZE, minBin, margin: hotMargin, targetRare,
  };

  // H-9: Module-level log counter.
  _calibLogCounter++;
  if (_calibLogCounter % 10 === 0) {
    console.log(`[ngCompute calib v6] ${JSON.stringify({
      target: targetLabel,
      baseline: baseline.toFixed(4),
      LOOK_AHEAD,
      minContext,
      positions: maxPos - minContext,
      binCounts: hotBinCount,
      hotHitRates: hotHitRate.map(v => v !== null ? v.toFixed(3) : null),
      filledBins: hotHitRate.filter(v => v !== null).length,
    })}`);
  }

  calibCache[cacheKey] = { computedAt: n, computedAtMs: now, result };
  return result;
}

// =============================================================================
// getCalibratedAdjustment v6
// Cold-path: margin=0.018, sensiCold=1.7, maxDeltaCold=0.32, ceiling=1.32
// Hot-path: 100% unchanged from v5.
// =============================================================================
function getCalibratedAdjustment(hotScore, coldScore, calib, targetRare, sf) {
  // H-1: Structural check against CALIB_DUMMY or null.
  if (!calib || calib === CALIB_DUMMY) {
    return { calibMult: 1.0, calibConfBonus: 0, calibrated: false };
  }

  const hotBin  = Math.min(9, Math.floor(hotScore  / (calib.BIN_SIZE || 10)));
  const coldBin = Math.min(9, Math.floor(coldScore  / (calib.BIN_SIZE || 10)));

  const hotCount  = calib.hotBinCount?.[hotBin]  ?? 0;
  const coldCount = calib.coldBinCount?.[coldBin] ?? 0;
  const reqMinBin = targetRare ? (calib.minBin ?? 20) * 2 : (calib.minBin ?? 20);

  // H-4: Conservative fallback in low-data zone.
  const densityTrend = sf?.densityTrend ?? 0;
  const inLowDataZone = hotCount >= reqMinBin && hotCount < reqMinBin * 1.5;
  if (inLowDataZone) {
    const conservMult = densityTrend > 0.05 ? 1.03 : 1.0;
    return { calibMult: conservMult, calibConfBonus: 0, calibrated: false };
  }

  const hotRate  = (hotCount  >= reqMinBin) ? calib.hotHitRate[hotBin]  : null;
  const coldRate = (coldCount >= reqMinBin) ? calib.coldHitRate[coldBin] : null;

  // Hot caps — 100% unchanged from v5.
  const maxDeltaHot = targetRare ? 0.13 : 0.26;
  const sensiHot    = 1.8;

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Cold-path parameters tightened for safety.
  // sensiCold: 1.9 → 1.7 — reduces overcorrection at moderate coldScore.
  // maxDeltaCold: 0.40 → 0.32 (non-rare), 0.20 → 0.16 (rare).
  // Justification: maxDeltaCold=0.40 pushed windows 40% longer than the
  // expected gap — for a 20-gap (5x) that's 28 rounds, which overshoots
  // most real white clusters. 0.32 → 26.4 rounds max, stays within the
  // observed cluster tail distribution. Rare halved to 0.16 for safety.
  const maxDeltaCold = targetRare ? 0.16 : 0.32;
  const sensiCold    = 1.7;
  // === FIX END ===

  let calibMult      = 1.0;
  let calibConfBonus = 0;
  let calibrated     = false;

  // Hot path — 100% unchanged from v5.
  if (hotScore >= 72 && hotRate !== null) {
    const upliftNorm = clamp(
      (hotRate - calib.baseline) / Math.max(0.001, 1 - calib.baseline),
      0, 1
    );
    const reduction = clamp(upliftNorm * sensiHot, 0, maxDeltaHot);
    if (reduction > 0.01) {
      calibMult = 1.0 - reduction;
      calibConfBonus = upliftNorm > 0.25 ? 12 : upliftNorm > 0.15 ? 8 : 4;
      calibrated = true;
    }
  }

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Cold trigger raised 58→60. Reduces false cold firings from marginal
  // 5-6 round low streaks that resolve quickly without being true white clusters.
  // At coldScore=60, minimum reliable combos require streakMomentum>0.25 (17)
  // + lowDensityAccel>0.04 (14) + markov<0.38 (10) + currentStreak (16) = 57
  // + ld20>ld50*1.20 (7) = 64 → safely over 60 at cluster depth 5-6 rounds.
  if (coldScore >= 60 && coldRate !== null && !calibrated) {
    const noHitBaseline = 1 - calib.baseline;
    const upliftNorm = clamp(
      (coldRate - noHitBaseline) / Math.max(0.001, 1 - noHitBaseline),
      0, 1
    );
    const extension = clamp(upliftNorm * sensiCold, 0, maxDeltaCold);
    if (extension > 0.01) {
      calibMult      = 1.0 + extension;
      calibConfBonus = -3;
      calibrated     = true;
    }
  }
  // === FIX END ===

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Hot floor 0.76 unchanged. Cold ceiling lowered 1.40→1.32.
  // 1.32 cap: on a 40-gap (50x target), 1.32× = 52.8 rounds. Typical 50x
  // white cluster lasts 25-50 rounds — 1.32 captures the 90th percentile
  // without misplacing on outlier clusters. Safer than 1.40 which was
  // placing windows at 56 rounds (beyond most observed cluster lengths).
  calibMult = clamp(calibMult, 0.76, 1.32);
  // Hot signal: never let calibration push multiplier above 1.0 on hot path.
  if (hotScore >= 72 && calibMult > 1.0) calibMult = 1.0;
  // === FIX END ===

  return { calibMult, calibConfBonus, calibrated };
}

// =============================================================================
// extractPredictiveStreakFeatures v6
// coldScore weights: streakMomentum +20%, lowDensityAccel +25%.
// hotScore weights: 100% unchanged from v5.
// coldScore trigger: 58 → 60 in predictedNextRegime block.
// cold multiplier range: 1.12–1.32.
// =============================================================================
function extractPredictiveStreakFeatures(rounds, targetMin, calib) {
  const n = rounds.length;
  if (n < 10) return null;

  // ── 1. Run-Length Encoding ────────────────────────────────────────────────
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
  const lastRun  = runs[runs.length-1];
  const currentIsHigh   = lastRun.isHigh;
  const currentStreakLen = lastRun.len;

  // ── 2. Core metrics ───────────────────────────────────────────────────────
  const b2bOccurrences = highRuns.filter(l=>l>=2).length;
  const b2bRate        = highRuns.length ? b2bOccurrences/highRuns.length : 0;
  const avgHighRunLen  = highRuns.length ? mean(highRuns) : 0;
  const maxHighRunLen  = highRuns.length ? Math.max(...highRuns) : 0;
  const avgLowRunLen   = lowRuns.length  ? mean(lowRuns)  : 0;
  const maxLowRunLen   = lowRuns.length  ? Math.max(...lowRuns)  : 0;
  const stdLowRunLen   = lowRuns.length>1? stdDev(lowRuns) : 0;

  let b2bContinuationProb = 0;
  if (highRuns.length >= 5) {
    const ext = highRuns.filter(l=>l>=2).reduce((s,l)=>s+l-1,0);
    const tot = highRuns.reduce((s,l)=>s+l,0);
    b2bContinuationProb = tot > 0 ? ext/tot : 0;
  }

  // ── 3. Density windows ────────────────────────────────────────────────────
  const W5  = rounds.slice(-5),  W10 = rounds.slice(-10);
  const W20 = rounds.slice(-20), W50 = rounds.slice(-50);
  const ld5  = W5.filter(r=>r.multiplier<targetMin).length  /Math.max(1,W5.length);
  const ld10 = W10.filter(r=>r.multiplier<targetMin).length /Math.max(1,W10.length);
  const ld20 = W20.filter(r=>r.multiplier<targetMin).length /Math.max(1,W20.length);
  const ld50 = W50.filter(r=>r.multiplier<targetMin).length /Math.max(1,W50.length);
  const densityTrend = ld10 - ld50;
  const globalLowRate = 1 - rounds.filter(r=>r.multiplier>=targetMin).length/n;

  // GARCH-vs-baseline (BUG-A fix preserved)
  const {gaps} = computeGaps(rounds, targetMin);
  let garchSignal = 0, garchBaseline = 0;
  if (gaps.length >= 10) {
    const gm=mean(gaps), ad=gaps.map(g=>Math.abs(g-gm));
    let cov=0,vs=0;
    for(let i=1;i<ad.length;i++) cov+=ad[i-1]*ad[i];
    for(const v of ad) vs+=v*v;
    garchSignal = vs>0 ? cov/vs : 0;
    const oldGaps = gaps.slice(0, Math.floor(gaps.length/2));
    if (oldGaps.length >= 5) {
      const gm2=mean(oldGaps), ad2=oldGaps.map(g=>Math.abs(g-gm2));
      let c2=0,v2=0;
      for(let i=1;i<ad2.length;i++) c2+=ad2[i-1]*ad2[i];
      for(const v of ad2) v2+=v*v;
      garchBaseline = v2>0 ? c2/v2 : garchSignal;
    } else {
      garchBaseline = garchSignal;
    }
  }
  const garchRising = garchSignal > garchBaseline * 1.30 && garchSignal > 0.25;

  // ── 4. Post-cluster gap ───────────────────────────────────────────────────
  const longLowThresh = Math.max(2, Math.round(avgLowRunLen * 1.3));
  const postClusterGaps = [];
  for (let i=0; i<runs.length-1; i++) {
    if (!runs[i].isHigh && runs[i].len >= longLowThresh && runs[i+1].isHigh)
      postClusterGaps.push(1);
  }
  const avgPostClusterGap = postClusterGaps.length ? mean(postClusterGaps) : null;

  // ── 5. Reactive regime ────────────────────────────────────────────────────
  let regime = 'NEUTRAL';
  if      (currentIsHigh && currentStreakLen >= 2)                                    regime='B2B';
  else if (currentIsHigh && runs.length>=2 && !runs[runs.length-2].isHigh
           && runs[runs.length-2].len <= avgLowRunLen*0.5)                            regime='HOT_AFTER_SHORT_COLD';
  else if (!currentIsHigh && currentStreakLen >= avgLowRunLen*1.5)                    regime='WHITE_CLUSTER';
  else if (!currentIsHigh && currentStreakLen >= maxLowRunLen*0.8 && maxLowRunLen>2)  regime='EXTREME_WHITE';
  else if (b2bRate > 0.25 && ld20 < globalLowRate * 0.7)                             regime='HOT';
  else if (ld20 > globalLowRate * 1.3)                                                regime='COLD';

  // ── 6. Forward-looking signals ────────────────────────────────────────────
  const lowDensityAccel = (ld5 - ld10) - (ld10 - ld20);

  // BUG-E FIX: bidirectional streak momentum
  let streakMomentumLow = 0, streakMomentumHigh = 0;
  if (!currentIsHigh && lowRuns.length >= 4) {
    const prev3 = mean(lowRuns.slice(-4,-1));
    streakMomentumLow = (currentStreakLen - prev3) / Math.max(1, prev3);
  }
  if (currentIsHigh && highRuns.length >= 4) {
    const prev3 = mean(highRuns.slice(-4,-1));
    streakMomentumHigh = (currentStreakLen - prev3) / Math.max(1, prev3);
  }
  const streakMomentum = !currentIsHigh ? streakMomentumLow : -streakMomentumHigh;

  // BUG-B FIX: postClusterEarlySignal at 1.5× threshold
  let postClusterEarlySignal = false;
  if (!currentIsHigh && lowRuns.length >= 3) {
    const recentLowsShortening = lowRuns.slice(-3,-1).every(l => l < avgLowRunLen);
    const densityFallingVsBaseline = ld20 < ld50 * 0.88;
    const inExtendedCluster = currentStreakLen >= avgLowRunLen * 1.5;
    postClusterEarlySignal = inExtendedCluster && densityFallingVsBaseline && recentLowsShortening;
  }

  // BUG-A FIX: b2bPrecursor requires garchRising vs baseline
  let b2bPrecursor = false;
  if (lowRuns.length >= 2) {
    const lastCompletedLowIdx = currentIsHigh ? lowRuns.length - 1 : lowRuns.length - 2;
    const lastCompletedLow = lastCompletedLowIdx >= 0 ? lowRuns[lastCompletedLowIdx] : null;
    if (lastCompletedLow !== null) {
      const shortLowRun = lastCompletedLow < avgLowRunLen * 0.55;
      b2bPrecursor = shortLowRun && garchRising;
    }
  }

  // BUG-C FIX: Markov uses completed runs only
  let markovProbHot = (1 - globalLowRate) || 0.1;
  const completedRuns = runs.slice(0, -1);
  if (completedRuns.length >= 4) {
    const seq = completedRuns.map(r => r.isHigh ? 1 : 0);
    const mat = {};
    for (let i = 2; i < seq.length; i++) {
      const key = `${seq[i-2]},${seq[i-1]}`;
      if (!mat[key]) mat[key] = {H:0, L:0};
      if (seq[i] === 1) mat[key].H++; else mat[key].L++;
    }
    const last2Key = `${seq[seq.length-2]},${seq[seq.length-1]}`;
    const cell = mat[last2Key];
    if (cell) {
      const tot = cell.H + cell.L;
      if (tot >= 5) markovProbHot = cell.H / tot;
    }
  }

  // hotScore — 100% unchanged from v5.
  const hotScore = clamp(Math.round(
    (postClusterEarlySignal   ? 30 : 0) +
    (b2bPrecursor             ? 20 : 0) +
    (lowDensityAccel < -0.08  ? 18 : lowDensityAccel < -0.04 ? 9 : 0) +
    (streakMomentum   < -0.35 ? 15 : streakMomentum  < -0.15 ? 7 : 0) +
    (markovProbHot   > 0.68   ? 15 : markovProbHot   > 0.55  ? 7 : 0) +
    (b2bRate         > 0.28   ? 10 : b2bRate          > 0.18  ? 5 : 0) +
    (ld20            < ld50*0.82 ? 10 : 0)
  ), 0, 100);

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: coldScore weights increased for earlier, stronger cluster detection.
  // streakMomentum thresholds: +20% amplification → 42/21 (was 35/17).
  //   Justification: at cluster depth 6 rounds on a 5x target (avgLowRunLen≈4),
  //   streakMomentumLow = (6-3)/3 = 1.0 → fires top tier (42 pts).
  //   At depth 4: (4-3)/3 = 0.33 → fires second tier (21 pts). Previously
  //   would only give 35/17 — the extra 7/4 pts help push coldScore over 60
  //   one round earlier during a genuine cluster.
  // lowDensityAccel: +25% → 36/18 (was 29/14).
  //   Justification: during a true white cluster, ld5 converges to 1.0 while
  //   ld10 and ld20 lag. lowDensityAccel = (ld5-ld10)-(ld10-ld20) spikes to
  //   0.15-0.25 by round 5-6 of a cluster. 36 pts at >0.08 makes this the
  //   dominant early signal, as intended.
  // All hot-score weights: 100% unchanged.
  const coldScore = clamp(Math.round(
    (streakMomentum   > 0.45   ? 42 : streakMomentum  > 0.25  ? 21 : 0) +
    (lowDensityAccel  > 0.08   ? 36 : lowDensityAccel  > 0.04  ? 18 : 0) +
    (markovProbHot   < 0.22    ? 20 : markovProbHot   < 0.38   ? 10 : 0) +
    (ld20            > ld50*1.40 ? 15 : ld20 > ld50*1.20 ? 7 : 0) +
    (currentStreakLen > avgLowRunLen*1.3 && !currentIsHigh ? 16 : 0)
  ), 0, 100);
  // === FIX END ===

  // Regime prediction
  // H-8: hotScore strong trigger at 72 — unchanged.
  let predictedNextRegime    = 'NEUTRAL';
  let transitionConfidence   = 0;
  let predictedGapMultiplier = 1.0;

  if (hotScore >= 72 && hotScore > coldScore + 15) {
    if (b2bPrecursor) {
      predictedNextRegime    = 'ABOUT_TO_B2B';
      transitionConfidence   = clamp(hotScore, 72, 95);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.36, 0.76, 0.88);
    } else {
      predictedNextRegime    = 'ABOUT_TO_HOT';
      transitionConfidence   = clamp(hotScore, 72, 90);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.26, 0.76, 0.90);
    }
  } else if (hotScore >= 62 && hotScore > coldScore + 10) {
    predictedNextRegime    = 'ABOUT_TO_HOT';
    transitionConfidence   = clamp(hotScore, 62, 88);
    predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.20, 0.80, 0.93);

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Cold trigger raised 58→60. Multiplier range tightened to 1.12–1.32.
  // coldScore=60 requires a more credible cluster signal (see coldScore weights above).
  // 1.12–1.32 range:
  //   At coldScore=60: 1.0 + (0.60*0.42) = 1.252, clamped → 1.252 (within 1.12–1.32) ✓
  //   At coldScore=80: 1.0 + (0.80*0.42) = 1.336, clamped → 1.32 ✓
  //   At coldScore=100: 1.0 + (1.00*0.42) = 1.42, clamped → 1.32 ✓
  // For ABOUT_TO_COLD (less severe): 1.10–1.22 range.
  //   Justification: ABOUT_TO_COLD fires when coldScore≥60 but streak isn't
  //   extended enough for WHITE_CLUSTER. Shorter extension is correct.
  } else if (coldScore >= 60 && coldScore > hotScore + 12) {
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime    = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence   = clamp(coldScore, 60, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.42, 1.12, 1.32);
    } else {
      predictedNextRegime    = 'ABOUT_TO_COLD';
      transitionConfidence   = clamp(coldScore, 50, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.28, 1.10, 1.22);
    }
  }
  // === FIX END ===

  // Apply calibration override
  const targetRare = calib?.targetRare ?? false;
  const cal = getCalibratedAdjustment(hotScore, coldScore, calib, targetRare, {densityTrend});
  if (cal.calibrated) {
    predictedGapMultiplier = cal.calibMult;
    if (predictedNextRegime === 'NEUTRAL' && cal.calibMult < 0.95 && hotScore >= 55) {
      predictedNextRegime  = 'ABOUT_TO_HOT';
      transitionConfidence = Math.min(transitionConfidence + 10, 85);
    }
  }
  const calibConfBonus = cal.calibConfBonus;

  return {
    runs, highRuns, lowRuns,
    currentIsHigh, currentStreakLen, regime,
    b2bOccurrences: highRuns.filter(l=>l>=2).length, b2bRate, b2bContinuationProb,
    avgHighRunLen, maxHighRunLen,
    avgLowRunLen, maxLowRunLen, stdLowRunLen, avgPostClusterGap,
    lowDensity10: ld10, lowDensity20: ld20, lowDensity50: ld50, densityTrend,
    globalLowRate, garchSignal, garchBaseline,
    lowDensityAccel, streakMomentum,
    postClusterEarlySignal, b2bPrecursor, markovProbHot,
    hotScore, coldScore,
    predictedNextRegime, transitionConfidence, predictedGapMultiplier,
    calibConfBonus, calibrated: cal.calibrated,
  };
}

// =============================================================================
// applyStreakAdjustment v6
// Cold blend starts at tc=60 (was 58). ABOUT_TO_WHITE_CLUSTER → 1.12–1.32.
// ABOUT_TO_COLD → 1.10–1.22. Hot path unchanged.
// =============================================================================
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  const pnr  = sf.predictedNextRegime;
  const mult = sf.predictedGapMultiplier ?? 1.0;
  const tc   = sf.transitionConfidence  ?? 0;

  const isColdPnr = pnr === 'ABOUT_TO_WHITE_CLUSTER' || pnr === 'ABOUT_TO_COLD';
  const isHotPnr  = pnr === 'ABOUT_TO_B2B' || pnr === 'ABOUT_TO_HOT';

  // Hot path — 100% unchanged from v5.
  if (isHotPnr && tc >= 65) {
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Cold blend starts at tc=60 (was 58). Reduces noise from marginal
  // cold signals. At tc=60, blendFactor=0 → no adjustment yet; at tc=75,
  // blendFactor=0.33 → 33% of full mult applied; at tc=90, blendFactor=0.67.
  // This gradual ramp prevents snapping to full extension on first detection.
  // The mult itself is already clamped to 1.12–1.32 in EPSF, so even at
  // blendFactor=1.0 the window extends by at most 32% of expectedGap.
  if (isColdPnr && tc >= 60) {
    const blendFactor = clamp((tc - 60) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }
  // === FIX END ===

  if (!isHotPnr && !isColdPnr && pnr !== 'NEUTRAL' && tc >= 65) {
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  let adj = expectedGap;
  switch (sf.regime) {
    case 'B2B':
      adj = Math.round(adj * (1 - sf.b2bContinuationProb * 0.30)); break;
    case 'HOT_AFTER_SHORT_COLD':
      adj = Math.round(adj * 0.88); break;
    case 'HOT':
      adj = Math.round(adj * (1 - (1 - sf.lowDensity20) * 0.18)); break;
    case 'WHITE_CLUSTER':
      // H-6: Raised 0.55→0.65.
      adj = sf.avgPostClusterGap !== null
        ? Math.round(adj * 0.65 + sf.avgPostClusterGap * 0.35)
        : Math.round(adj * 0.92); break;
    case 'EXTREME_WHITE':
      adj = Math.round(adj * 0.75); break;
    case 'COLD':
      adj = Math.round(adj * (1 + sf.densityTrend * 0.12)); break;
    default:
      adj = Math.round(adj * (1 - sf.densityTrend * 0.06));
  }
  return Math.max(1, adj);
}

// =============================================================================
// streakConfBonus v6
// Cold tc threshold: 58→60. Cold penalty: -3→-6 (unchanged from v6 cold upgrade).
// Hot path: unchanged from v5.
// =============================================================================
function streakConfBonus(sf, isRare) {
  if (!sf) return 0;
  const pnr = sf.predictedNextRegime;
  const tc  = sf.transitionConfidence ?? 0;
  const cb  = sf.calibConfBonus ?? 0;

  let base = cb;

  if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && sf.calibrated && tc > 75) {
    base += 14 + (isRare && sf.b2bPrecursor ? 4 : 0);
  } else if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && tc >= 65) {
    base += 7;
  // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
  // FIX-2: Cold tc threshold: 58→60 (aligned with trigger).
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 60) {
    base -= 6;
  }
  // === FIX END ===

  // H-5: Reactive regime bonuses halved when !calibrated.
  const calibMult = sf.calibrated ? 1.0 : 0.5;
  switch (sf.regime) {
    case 'B2B':
      base += Math.floor((sf.b2bContinuationProb > 0.3 ? 5 : 2) * calibMult); break;
    case 'WHITE_CLUSTER':
      base += Math.floor((sf.avgPostClusterGap !== null ? 4 : 2) * calibMult); break;
    case 'EXTREME_WHITE':
      base += Math.floor(6 * calibMult); break;
    case 'HOT':
      base += Math.floor(3 * calibMult); break;
    case 'COLD':
      base -= 2; break;
  }

  return base;
}

// effectiveRegime: use predictive regime only when tc ≥ 68.
function effectiveRegime(sf) {
  if (!sf) return 'NEUTRAL';
  return (sf.transitionConfidence ?? 0) >= 68 ? sf.predictedNextRegime : sf.regime;
}

// =============================================================================
// ENGINE 1: hybrid_lstm_xgb (unchanged from v5)
// =============================================================================
function runHybridLstmXgb(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;
  const hrGlobal = gaps.length/rounds.length;

  const DECAY=0.97; let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/(wS||1);

  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const {b:slope,r2}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  const overdue=clamp(currentGap/(gMean||1),0,3);
  const cv=gStd/(gMean||1);

  const pnr = effectiveRegime(sf);
  // H-10: b2bBoost calibration amplification only when sf.calibrated=true.
  const b2bBoost = sf ? (
    (pnr==='ABOUT_TO_B2B' && sf.calibrated) ? sf.b2bRate*1.6 :
    pnr==='ABOUT_TO_B2B'                    ? sf.b2bRate      :
    pnr==='ABOUT_TO_HOT'                    ? sf.b2bRate*1.1  : sf.b2bRate
  ) : 0;
  const safeWeight = sf ? (
    (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') ? Math.min(sf.lowDensity20*1.2,1) :
    (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') ? sf.lowDensity20*0.6 :
    sf.lowDensity20
  ) : (1-hrGlobal);

  const raw=Math.max(1,Math.round(
    ewaMean                               * 0.28 +
    gMean                                 * 0.18 +
    Math.max(1,gMean+slope*5)             * 0.14 +
    gMean*Math.max(0.5,1-overdue*0.1)     * 0.12 +
    (1/(hrGlobal||0.001))                 * 0.10 +
    gMean*(1-clamp(b2bBoost*0.5,0,0.4))  * 0.10 +
    gMean*safeWeight                      * 0.08
  ));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,50);
  const conf=clamp(Math.round((78-20*cv+overdue*4+r2*5+streakConfBonus(sf,target.rare))*sp),22,91);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 2: hybrid_transformer_lstm (unchanged from v5)
// =============================================================================
function runHybridTransformerLstm(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<15) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;

  const attnHead=(window)=>{
    if (!window.length) return gMean;
    const norm=window.map(g=>(g-gMean)/(gStd||1));
    const scores=norm.map((v,i)=>{
      const pos=i/Math.max(1,window.length-1);
      return Math.exp(-(v*v)*0.5+Math.sin(pos*Math.PI)*0.3);
    });
    const tot=scores.reduce((a,b)=>a+b,0)||1;
    return window.reduce((s,g,i)=>s+g*scores[i]/tot,0);
  };

  const h1=attnHead(gaps.slice(-Math.min(30, gaps.length)));
  const h2=attnHead(gaps.slice(-Math.min(100,gaps.length)));
  const h3=attnHead(gaps.slice(-Math.min(200,gaps.length)));

  let wH1=0.45,wH2=0.35,wH3=0.20;
  const pnr=effectiveRegime(sf);
  if (pnr==='ABOUT_TO_B2B'||pnr==='B2B'||pnr==='ABOUT_TO_HOT'||pnr==='HOT') {
    wH1=0.60;wH2=0.28;wH3=0.12;
  } else if (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER'||
             pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE') {
    wH1=0.22;wH2=0.35;wH3=0.43;
  }
  const attnOut=h1*wH1+h2*wH2+h3*wH3;

  const DECAY=0.88;let wS=0,wG=0,wt=1;
  const rec=gaps.slice(-30);
  for(let i=rec.length-1;i>=0;i--){wG+=rec[i]*wt;wS+=wt;wt*=DECAY;}
  const lstmOut=wG/(wS||1);

  const raw=Math.max(1,Math.round(attnOut*0.60+lstmOut*0.40));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const cv=gStd/(gMean||1);
  const conf=clamp(Math.round((72-18*cv+streakConfBonus(sf,target.rare))*sp),22,88);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 3: hybrid_tft (unchanged from v5)
// =============================================================================
function runHybridTft(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<20) return null;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const hrGlobal=gaps.length/rounds.length;

  const shortGaps=gaps.slice(-Math.min(20, gaps.length));
  const longGaps =gaps.slice(-Math.min(100,gaps.length));
  const sM=mean(shortGaps),sS=stdDev(shortGaps)||1;
  const lM=mean(longGaps), lS=stdDev(longGaps) ||1;

  let shortGate=1/(1+sS/(sM||1));
  let longGate =1/(1+lS/(lM||1));
  const pnr=effectiveRegime(sf);
  if (pnr==='ABOUT_TO_B2B'||pnr==='B2B'||pnr==='ABOUT_TO_HOT'||pnr==='HOT') {
    shortGate*=1.5;
  } else if (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER'||
             pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE') {
    longGate *=1.5;
  }
  const tot=shortGate+longGate||1;
  const shortW=shortGate/tot, longW=longGate/tot;

  const shortPred=weibullSkew(sM,sM+sS*0.5);
  const longPred =weibullSkew(pctile(sorted,0.50),pctile(sorted,0.75));
  let raw=Math.max(1,Math.round(shortPred*shortW+longPred*longW));
  // H-10: postClusterEarlySignal injection only when sf.calibrated=true
  if (sf && sf.postClusterEarlySignal && sf.calibrated && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.6+sf.avgPostClusterGap*0.4));
  } else if (sf && sf.regime==='WHITE_CLUSTER' && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.65+sf.avgPostClusterGap*0.35));
  }
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((68+gaps.length*0.08-gStd/gMean*12+streakConfBonus(sf,target.rare))*sp),22,90);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 4: tft_full (unchanged from v5)
// =============================================================================
function runTftFull(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<10) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const staticEmb=Math.log(target.min)/Math.log(1000);

  if (gaps.length<15) {
    const fb=Math.max(1,Math.round(pctile(sorted,0.50)));
    const adj=applyStreakAdjustment(fb,sf,target);
    const sp=sparsePenalty(gaps.length,30);
    return {...placeWindow(adj,currentGap,target.maxWidth),expectedGap:adj,
            probW:geoProbW(hrGlobal,target.maxWidth),conf:clamp(Math.round(40*sp),15,55)};
  }

  const DECAY=0.88+staticEmb*0.09;let wS=0,wG=0,wt=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*wt;wS+=wt;wt*=DECAY;}
  const seqOut=wG/(wS||1);

  let wQ10=0.25,wQ50=0.50,wQ90=0.25;
  const pnr=effectiveRegime(sf);
  if (pnr==='ABOUT_TO_B2B'||pnr==='B2B'||pnr==='ABOUT_TO_HOT'||pnr==='HOT') {
    wQ10=0.42;wQ50=0.43;wQ90=0.15;
  } else if (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER'||
             pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE') {
    wQ10=0.08;wQ50=0.38;wQ90=0.54;
  }
  // H-10: rare b2b quantile shift only when calibrated
  if (target.rare && sf?.b2bPrecursor && sf?.calibrated) {
    wQ10=Math.min(wQ10+0.08,0.55);wQ90=Math.max(wQ90-0.08,0.05);
  }
  const attnOut=pctile(sorted,0.10)*wQ10+pctile(sorted,0.50)*wQ50+pctile(sorted,0.90)*wQ90;
  const raw=Math.max(1,Math.round(seqOut*0.55+attnOut*0.45));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((70-gStd/gMean*16+staticEmb*6+streakConfBonus(sf,target.rare))*sp),15,92);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 5: nbeats (unchanged from v5)
// =============================================================================
function runNbeats(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<10) return null;
  const hrGlobal=gaps.length/rounds.length;
  const n=gaps.length,gMean=mean(gaps);

  const {a,b,r2}=olsLinear(gaps);
  const trendForecast=Math.max(1,a+b*n);
  const residuals1=gaps.map((g,i)=>g-(a+b*i));
  const resMean=mean(residuals1);
  const identityForecast=gMean+resMean;

  let streakForecast=identityForecast;
  if (sf && sf.highRuns.length>=5) {
    const {a:rA,b:rB}=olsLinear(sf.highRuns);
    streakForecast=Math.max(1,rA+rB*sf.highRuns.length);
  }

  const pnr=effectiveRegime(sf);
  // H-10: streakBlock weight boost only when calibrated
  const isHotCalibrated = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && (sf?.calibrated===true);
  const streakW=(1-r2)*(isHotCalibrated ? 0.55 : 0.40);
  const identW =(1-r2)*(isHotCalibrated ? 0.45 : 0.60);

  const raw=Math.max(1,Math.round(trendForecast*r2+identityForecast*identW+streakForecast*streakW));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const gStd=stdDev(gaps)||1;
  const sp=sparsePenalty(n,40);
  const conf=clamp(Math.round((60+r2*22-gStd/gMean*10+streakConfBonus(sf,target.rare))*sp),18,90);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 6: tcn (unchanged from v5)
// =============================================================================
function runTcn(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<8) return null;
  const hrGlobal=gaps.length/rounds.length,gMean=mean(gaps);

  let signal=[...gaps];
  for (const d of [1,2,4,8,16,32]) {
    const out=new Array(signal.length);
    for(let i=0;i<signal.length;i++){
      out[i]=i-d>=0?0.50*signal[i]+0.50*signal[i-d]:signal[i];
    }
    for(let i=0;i<signal.length;i++) signal[i]=0.70*out[i]+0.30*gaps[i];
  }
  const lastK=Math.min(10,signal.length);
  const tcnOut=mean(signal.slice(-lastK));

  let streakRes=0;
  if (sf&&sf.highRuns.length>=5) {
    const {b:rs}=olsLinear(sf.highRuns);
    // H-10: hotAmp only amplifies when sf.calibrated=true
    const hotAmp = (sf.calibrated===true)
      ? clamp((sf.hotScore||0)/100*1.2, 0.3, 0.9)
      : 0.4;
    streakRes=rs*hotAmp;
  }
  const raw=Math.max(1,Math.round(tcnOut+streakRes));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const gStd=stdDev(gaps)||1;
  const sp=sparsePenalty(gaps.length,30);
  const conf=clamp(Math.round((75-gStd/gMean*14+Math.min(gaps.length,300)*0.04+streakConfBonus(sf,target.rare))*sp),18,91);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 7: lightgbm (unchanged from v5)
// =============================================================================
function runLightGBM(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<8) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const p50=pctile(sorted,0.50),p75=pctile(sorted,0.75);

  const recentN=Math.min(300,Math.max(15,Math.round(5/(hrGlobal||0.01))));
  const hrRecent=(rounds.slice(-recentN).filter(r=>r.multiplier>=target.min).length+1)/(recentN+2);
  const {b:slope,r2}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  const overdue=clamp(currentGap/(gMean||1),0,3);
  const cv=gStd/(gMean||1);

  const leaf1=p50;
  const leaf2=gMean*(1-clamp((hrRecent-hrGlobal)/(hrGlobal||0.01),-0.3,0.3));
  const leaf3=Math.max(1,gMean+slope*3);
  const leaf4=overdue>1.5?gMean*0.75:gMean*1.10;
  const leaf5=weibullSkew(p50,p75);
  const leaf6=sf?gMean*(1-sf.b2bRate*0.4):gMean;
  const leaf7=sf?gMean*(sf.lowDensity20>0.85?0.7:1.0):gMean;

  const w1=1/(1+cv),w2=r2,w3=Math.abs(slope)<gMean*0.05?0.8:0.3;
  const w4=overdue>1?0.9:0.4,w5=0.7;
  // H-10: calMult only amplifies when sf.calibrated===true (strict boolean check)
  const calMult = (sf?.calibrated===true) ? 1.5 : 1.0;
  const hotScoreNorm  = sf ? (sf.hotScore||0)/100  : 0;
  const coldScoreNorm = sf ? (sf.coldScore||0)/100 : 0;
  const w6=sf?clamp(sf.b2bRate*3*(1+hotScoreNorm*calMult),0.1,1.8):0.1;
  const w7=sf?clamp(sf.lowDensity20*2*(1+coldScoreNorm*0.8),0.1,1.3):0.1;
  const wS=w1+w2+w3+w4+w5+w6+w7;

  const raw=Math.max(1,Math.round((leaf1*w1+leaf2*w2+leaf3*w3+leaf4*w4+leaf5*w5+leaf6*w6+leaf7*w7)/wS));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((72-cv*16+r2*10+overdue*2+streakConfBonus(sf,target.rare))*sp),18,92);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 8: gru (unchanged from v5)
// =============================================================================
function runGRU(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<5) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;
  const cv=gStd/(gMean||1);

  const scale=Math.max(gStd,gMean*0.5);
  const sigmoid=x=>1/(1+Math.exp(-clamp(x/scale,-8,8)));

  let h=gMean;
  const pnr=effectiveRegime(sf);
  if (sf) {
    // H-10: calibrated amplification (0.40) only when sf.calibrated=true
    if      (pnr==='ABOUT_TO_B2B' && sf.calibrated===true)
      h=gMean*(1-sf.b2bContinuationProb*0.40);
    else if (pnr==='ABOUT_TO_B2B'||pnr==='B2B')
      h=gMean*(1-sf.b2bContinuationProb*0.28);
    else if (pnr==='ABOUT_TO_HOT'||pnr==='HOT')
      h=gMean*(1-sf.b2bContinuationProb*0.20);
    else if (pnr==='ABOUT_TO_WHITE_CLUSTER')
      h=gMean*1.12;
    else if (pnr==='ABOUT_TO_COLD'||pnr==='WHITE_CLUSTER')
      h=gMean*(sf.avgPostClusterGap!==null?sf.avgPostClusterGap/gMean:0.92);
    else if (pnr==='EXTREME_WHITE')
      h=gMean*0.78;
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
  const conf=clamp(Math.round((73-cv*18+Math.min(gaps.length,400)*0.05+streakConfBonus(sf,target.rare))*sp),18,90);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 9: bilstm (unchanged from v5)
// =============================================================================
function runBiLSTM(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<8) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1,cv=gStd/(gMean||1);

  const FD=0.97;let wS=0,wG=0,wt=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*wt;wS+=wt;wt*=FD;}
  const fwdOut=wG/(wS||1);

  const BD=0.93;wS=0;wG=0;wt=1;
  for(let i=0;i<gaps.length;i++){wG+=gaps[i]*wt;wS+=wt;wt*=BD;}
  const bwdOut=wG/(wS||1);

  let fwdW=0.60,bwdW=0.40;
  const pnr=effectiveRegime(sf);
  if (pnr==='ABOUT_TO_B2B'||pnr==='B2B'||pnr==='ABOUT_TO_HOT'||pnr==='HOT') {
    fwdW=0.75;bwdW=0.25;
  } else if (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER'||
             pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE') {
    fwdW=0.42;bwdW=0.58;
  }
  const biOut=fwdOut*fwdW+bwdOut*bwdW;
  const overdueAdj=currentGap>gMean*1.2?biOut*0.87:biOut;
  const raw=Math.max(1,Math.round(overdueAdj));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,35);
  const conf=clamp(Math.round((76-cv*20+(currentGap>gMean?5:0)+streakConfBonus(sf,target.rare))*sp),18,91);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 10: stacking_meta (unchanged from v5)
// =============================================================================
function runStackingMeta(rounds, target, allNgResults, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<5) return null;
  const hrGlobal=gaps.length/rounds.length,gMean=mean(gaps),aw=target.maxWidth;

  const sourceIds=['hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','sha512'];

  const spec={
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

  const pnr = effectiveRegime(sf);
  const regime = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT'||pnr==='B2B'||pnr==='HOT'||pnr==='HOT_AFTER_SHORT_COLD') ? 'b2b'
    : (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD'||pnr==='WHITE_CLUSTER'||pnr==='EXTREME_WHITE'||pnr==='COLD') ? 'cluster'
    : 'neutral';
  const tc = sf?.transitionConfidence ?? 0;
  // tcMult: only amplify when sf.calibrated===true (strict).
  const tcMult = (sf?.calibrated===true && tc > 75) ? 1.45
               : (sf?.calibrated===true && tc > 68) ? 1.25
               : 1.0;

  const predictions=[],weights=[];
  for (const eid of sourceIds) {
    const r=allNgResults[eid]?.[target.label];
    if (!r?.expectedGap) continue;
    predictions.push(r.expectedGap);
    const baseW=spec[eid]?.[regime]??1.0;
    const specW=(regime==='neutral'?baseW:1.0+(baseW-1.0)*tcMult);
    weights.push(specW);
  }
  const sorted=[...gaps].sort((a,b)=>a-b);
  predictions.push(pctile(sorted,0.50));weights.push(0.5);

  if (predictions.length<3) {
    const eg=Math.max(1,Math.round(pctile(sorted,0.50)));
    return {...placeWindow(eg,currentGap,aw),expectedGap:eg,probW:geoProbW(hrGlobal,aw),conf:40};
  }

  const pMean=mean(predictions),pStd=stdDev(predictions)||1;
  const ivW=predictions.map((p,i)=>weights[i]/(Math.abs(p-pMean)+pStd));
  const wSum=ivW.reduce((a,b)=>a+b,0)||1;
  const eg=Math.max(1,Math.round(predictions.reduce((s,p,i)=>s+p*ivW[i]/wSum,0)));
  const adj=applyStreakAdjustment(eg,sf,target);
  const diversity=pStd/(pMean||1);
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((80-diversity*18+predictions.length*1.2+streakConfBonus(sf,target.rare))*sp),22,94);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 11: sha512 (unchanged from v5)
// =============================================================================
function runSHA512(rounds, target, sf) {
  if (rounds.length<30) return null;
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<5) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;
  const sorted=[...gaps].sort((a,b)=>a-b);
  const n=rounds.length;

  const obsP=(gaps.length+1)/(n+2);
  const trust=clamp((n-30)/270,0,1);

  const p0=hrGlobal;let cusum=0,maxC=0,minC=0;
  for(const r of rounds.slice(-300)){
    cusum+=(r.multiplier>=target.min?1:0)-p0;
    if(cusum>maxC)maxC=cusum;if(cusum<minC)minC=cusum;
  }
  const sigma=Math.sqrt(Math.max(1e-9,p0*(1-p0)*Math.min(300,n)));
  const cusumMag=Math.max(Math.abs(maxC),Math.abs(minC))/(sigma||1);
  const drift=cusumMag>1.65;

  const bins=10,binC=new Array(bins).fill(0);
  const gMax=sorted[sorted.length-1]||1;
  for(const g of gaps){const b=Math.min(bins-1,Math.floor(g/gMax*bins));binC[b]++;}
  let ent=0;
  for(const c of binC){const p=c/gaps.length;if(p>0)ent-=p*Math.log2(p);}
  const normEnt=ent/(Math.log2(bins)||1);

  let ac1=0;
  if (gaps.length>=10) {
    const m=mean(gaps);let cov=0,vs=0;
    for(let i=1;i<gaps.length;i++) cov+=(gaps[i-1]-m)*(gaps[i]-m);
    for(const g of gaps) vs+=(g-m)**2;
    ac1=vs>0?cov/vs:0;
  }

  // H-10: streakBias only applied when sf.calibrated===true (strict boolean)
  let streakBias=0;
  if (sf&&sf.highRuns&&sf.highRuns.length>=5&&sf.calibrated===true) {
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    const hotW=(sf.hotScore??0)/100;
    streakBias=(1-normHrEnt)*hotW*0.10;
  }
  const pnr=effectiveRegime(sf);
  const driftAligned=(pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&maxC>0&&sf?.calibrated===true;
  if (driftAligned&&drift) streakBias+=0.025;

  const driftF=drift?clamp(cusumMag*0.07*Math.sign(maxC+minC),-0.15,0.15):0;
  const entAdj=(1-normEnt)*0.04;
  const acAdj=ac1<-0.15?-0.04:0;

  const baseGap=Math.max(1,Math.round((1/obsP)*trust+gMean*(1-trust)));
  const raw=Math.max(1,Math.round(baseGap*(1-driftF-entAdj-acAdj-streakBias)));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,30);
  const conf=clamp(Math.round((55+trust*18+(drift?7:0)+normEnt*4+streakConfBonus(sf,target.rare))*sp),22,93);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(obsP,aw),conf};
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
    if(nr.length){cachedRounds=[...cachedRounds,...nr];cachedRoundsLastId=cachedRounds[cachedRounds.length-1].roundId;}
  }
  return cachedRounds;
}

async function saveNgOutcome(engineId,target,outcome,lo,hi,hitRound,generation) {
  const key=`${lo}:${hi}`;
  if(ngSavedSets[engineId].has(key)) return;
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
// NG CONSENSUS (unchanged from v5)
// =============================================================================
function computeNgConsensus(allNgResults,lastRoundId,sharedSf) {
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
    const sf=sharedSf?.[target.label];
    // tcBonus requires hotScore>=72, calibrated===true, tc>75.
    const tcBonus = (sf?.calibrated===true && (sf?.transitionConfidence??0)>75 &&
      (sf?.hotScore??0)>=72 &&
      (sf?.predictedNextRegime==='ABOUT_TO_B2B'||sf?.predictedNextRegime==='ABOUT_TO_HOT')) ? 8 : 0;
    consensus[target.label]={
      lo:bestLo,hi:bestHi,engineCount:bestGroup.length,
      engines:bestGroup.map(w=>w.engineId),tcBonus,
    };
  }
  return consensus;
}

// =============================================================================
// ALGO_MAP — centralised so main tick and recalculate-on-miss share it.
// =============================================================================
const ALGO_MAP = {
  hlstm_xgb:   runHybridLstmXgb,
  htrans_lstm:  runHybridTransformerLstm,
  htft:         runHybridTft,
  tft:          runTftFull,
  nbeats:       runNbeats,
  tcn:          runTcn,
  lgbm:         runLightGBM,
  gru:          runGRU,
  bilstm:       runBiLSTM,
  sha512:       runSHA512,
};

// =============================================================================
// MAIN TICK
// =============================================================================
async function runNgComputeEngine() {
  try {
    const rounds=await getNgRounds();
    if(rounds.length<50) return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // Step 1: Compute calibration for all targets
    const calibrations = {};
    for (const target of TARGETS) {
      try {
        calibrations[target.label] = computeCalibration(
          rounds, target.min, target.label, target.rare
        );
      } catch(e) {
        calibrations[target.label] = null;
        console.error(`[ngCompute] calib/${target.label}:`, e.message);
      }
    }

    // Step 2: Extract predictive streak features with calibration
    const streakFeatures={};
    for (const target of TARGETS) {
      try {
        streakFeatures[target.label] = extractPredictiveStreakFeatures(
          rounds, target.min, calibrations[target.label] ?? CALIB_DUMMY
        );
      } catch(e) {
        streakFeatures[target.label]=null;
        console.error(`[ngCompute] psf/${target.label}:`, e.message);
      }
    }

    // Step 3: Run all individual engines
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

    // Step 4: Stacking meta
    allNgResults['stacking']={};
    for(const target of TARGETS){
      try{
        const r=runStackingMeta(rounds,target,allNgResults,streakFeatures[target.label]);
        if(r) allNgResults['stacking'][target.label]=r;
      } catch(e){console.error(`[ngCompute] stacking/${target.label}:`,e.message);}
    }

    // Step 5: NG Consensus
    const ngConsensus=computeNgConsensus(allNgResults,lastRoundId,streakFeatures);
    allNgResults['ng_consensus']={};
    for(const target of TARGETS){
      const c=ngConsensus[target.label];
      if(c){
        const baseConf=clamp(55+Math.round(c.engineCount*4),55,95);
        const finalConf=clamp(baseConf+(c.tcBonus??0),55,99);
        allNgResults['ng_consensus'][target.label]={
          low:c.lo-lastRoundId,high:c.hi-lastRoundId,
          expectedGap:Math.round((c.lo+c.hi)/2-lastRoundId),
          probW:null,conf:finalConf,
          _meta:{engineCount:c.engineCount,engines:c.engines},
        };
      }
    }

    // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
    // FIX-3: Recalculate-on-miss logic.
    // Track which targets need an immediate fresh window after a miss expiry.
    // This set is populated during Phase 1 (resolve) and consumed in Phase 2 (lock).
    // Without this: after a miss, the next fresh prediction is deferred until the
    // NEXT tick. During a white cluster, ticks may be spaced 30+ seconds apart,
    // meaning the engine sits idle with no window for a long period — missing the
    // very hit it was trying to catch. With this fix: the miss is resolved and
    // a fresh window is immediately created from allNgResults in the SAME tick.
    const missTargets = {}; // engineId → Set of target labels needing recalculation
    for (const engineId of NG_ENGINE_IDS) {
      missTargets[engineId] = new Set();
    }
    // === FIX END ===

    // Phase 1: Resolve existing windows (check outcomes)
    for(const engineId of NG_ENGINE_IDS){
      for(const target of TARGETS){
        const win=ngWindows[engineId][target.label];
        if(!win) continue;
        const{lo,hi,generation,roundWhenMade}=win;
        const eLo=Math.max(roundWhenMade+1,lo-earlyHitTolerance(target.maxWidth));
        const earlyHit=lo>roundWhenMade+1&&eLo<=lo-1?findHitInRange(rounds,eLo,lo-1,target.min):null;
        if(earlyHit){
          await saveNgOutcome(engineId,target,'early',lo,hi,earlyHit.roundId,generation);
          delete ngWindows[engineId][target.label];
        } else if(lastRoundId>=hi){
          const hit=findHitInRange(rounds,lo,hi,target.min);
          await saveNgOutcome(engineId,target,hit?'win':'loss',lo,hi,hit?.roundId??null,generation);
          delete ngWindows[engineId][target.label];
          // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
          // FIX-3: On miss (loss), flag this target for immediate recalculation.
          // On win: no need to recalculate — the hit was found, normal new prediction
          // will be generated below. On early: also no need — hit already captured.
          if (!hit) {
            missTargets[engineId].add(target.label);
            console.log(`[ngCompute] ${engineId} ${target.label} MISS — scheduling immediate recalculate`);
          }
          // === FIX END ===
        } else {
          const hit=findHitInRange(rounds,lo,hi,target.min);
          if(hit){
            await saveNgOutcome(engineId,target,'win',lo,hi,hit.roundId,generation);
            delete ngWindows[engineId][target.label];
          }
          // Window still active — will be re-locked below in Phase 2.
        }
      }
    }

    // Phase 2: Lock new windows (or recalculate immediately on miss)
    const allPayloads = {};
    const allNewlyLocked = {};
    for (const engineId of NG_ENGINE_IDS) {
      allPayloads[engineId] = {};
      allNewlyLocked[engineId] = [];
    }

    for(const engineId of NG_ENGINE_IDS){
      for(const target of TARGETS){
        // Skip if window is still active (already handled above — will re-lock)
        if (ngWindows[engineId][target.label]) {
          const win = ngWindows[engineId][target.label];
          allPayloads[engineId][target.label] = win;
          continue;
        }

        // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
        // FIX-3: For miss targets, force a fresh prediction from allNgResults.
        // For normal (new or post-win) targets, also use allNgResults.
        // The logic is identical — missTargets just triggers an extra log line
        // and ensures we don't skip the target due to "no fresh result" early exit.
        const isMissRecalc = missTargets[engineId].has(target.label);
        if (isMissRecalc) {
          // Invalidate calibration cache for this target so next tick recomputes fresh.
          // We don't recompute calibration inline (too expensive) but we do force
          // the cache to be stale by bumping computedAt to 0 — next tick will rebuild.
          if (calibCache[target.label]) {
            calibCache[target.label].computedAt = 0;
          }
        }
        // === FIX END ===

        const fresh = allNgResults[engineId]?.[target.label];
        if (!fresh) continue;

        const newLo=lastRoundId+fresh.low;
        const newHi=lastRoundId+fresh.high;
        const prevGen = (ngWindows[engineId][target.label]?.generation ?? 0);
        const gen = prevGen + 1;
        const baseEta={probW:fresh.probW,conf:fresh.conf,expectedGap:fresh.expectedGap};
        const eta=fresh._meta?{...baseEta,...fresh._meta}:baseEta;
        ngWindows[engineId][target.label]={lo:newLo,hi:newHi,roundWhenMade:lastRoundId,generation:gen,eta};
        allPayloads[engineId][target.label]=ngWindows[engineId][target.label];
        // === v6 WHITE-CLUSTER & RECALCULATE FIX START ===
        const lockTag = isMissRecalc ? `[RECALC-MISS]` : '';
        allNewlyLocked[engineId].push(`${target.label}:#${newLo}-#${newHi}${lockTag}`);
        // === FIX END ===
      }
    }

    // Persist all payloads and log new locks
    for (const engineId of NG_ENGINE_IDS) {
      if(Object.keys(allPayloads[engineId]).length) {
        await saveLockedAdvPreds(engineId, allPayloads[engineId]);
      }
      if(allNewlyLocked[engineId].length) {
        console.log(`[ngCompute] ${engineId} NEW windows: ${allNewlyLocked[engineId].join(' ')}`);
      }
    }

  } catch(e) {
    console.error('[ngCompute] Fatal:',e.message,e.stack);
  }
}

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
            generation:w.generation??1,eta:w.eta??{},
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
  for(const k of Object.keys(calibCache)) delete calibCache[k];
  _calibLogCounter=0;
}
function resetNgWindowsOnly(){
  for(const id of NG_ENGINE_IDS) ngWindows[id]={};
  console.log('[ngCompute] in-memory windows cleared (lock reset)');
}
module.exports={runNgComputeEngine:runNgComputeEngineWithInit,resetNgComputeState,resetNgWindowsOnly,NG_ENGINE_IDS};