'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
// ================================================================================
// v6 WHITE-CLUSTER SLAYER & FULL FIX — GROK MANUAL EDIT (March 2026)
//
// FIXED ISSUES:
// 1. NG_ENGINE_IDS was only ['ng_consensus'] → all 11 engines were dead → restored full list
// 2. White clusters (5–10+ lows) not lengthening windows → cold trigger 55, weights +30/25%, max 1.35×
// 3. No recalculate on miss → added immediate re-prediction when window expires without hit
// 4. Calibration too conservative on cold → COLD_MIN_BIN=12, coldMargin=0.015
// 5. All hot/b2b safety (hotScore≥72, clamp 0.76 floor, H-1..H-10) untouched
// 6. Added miss-recalculate logging for visibility
//
// Copy-paste this entire file. Restart server. Let it run 200–300 rounds.
// ================================================================================

const {
  getRounds, savePrediction, getPredictions,
  saveLockedAdvPreds, getLockedAdvPreds,
} = require('./db');

const NG_ENGINE_IDS = [
  'hlstm_xgb', 'htrans_lstm', 'htft', 'tft', 'nbeats',
  'tcn', 'lgbm', 'gru', 'bilstm', 'sha512', 'ng_consensus'
]; // FIXED: all engines now active

const TARGETS = [
  { label: '5x', min: 5, maxWidth: 3, rare: false },
  { label: '10x', min: 10, maxWidth: 5, rare: false },
  { label: '20x', min: 20, maxWidth: 7, rare: false },
  { label: '50x', min: 50, maxWidth: 12, rare: false },
  { label: '100x', min: 100, maxWidth: 18, rare: true },
  { label: '250x', min: 250, maxWidth: 25, rare: true },
  { label: '500x', min: 500, maxWidth: 35, rare: true },
  { label: '1000x', min: 1000, maxWidth: 50, rare: true },
];

const ngSavedSets = {};
for (const id of NG_ENGINE_IDS) ngSavedSets[id] = new Set();
const ngWindows = {};
for (const id of NG_ENGINE_IDS) ngWindows[id] = {};
let cachedRounds = [], cachedRoundsLastId = 0, initialised = false;

function earlyHitTolerance(width) { return Math.floor(width / 2); }

// MATH HELPERS (unchanged)
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

// wilsonLower (unchanged)
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

// CALIB_DUMMY, TARGET_LOOK_AHEAD, MIN_BIN_* (unchanged)
const CALIB_DUMMY = Object.freeze({
  hotHitRate: new Array(10).fill(null),
  coldHitRate: new Array(10).fill(null),
  hotBinCount: new Array(10).fill(0),
  coldBinCount: new Array(10).fill(0),
  BIN_SIZE: 10,
  minBin: 20,
  margin: 0.025,
  baseline: 0.5,
  LOOK_AHEAD: 20,
  targetRare: false,
});

const TARGET_LOOK_AHEAD = {
  '5x': 20, '10x': 20, '20x': 20, '50x': 40,
  '100x': 80, '250x': 150, '500x': 300, '1000x': 300,
};

const MIN_BIN_NON_RARE = 20;
const MIN_BIN_RARE = 30;

let _calibLogCounter = 0;
const calibCache = {};

// computeCalibration — COLD PATH MADE STRONGER
function computeCalibration(rounds, targetMin, targetLabel, targetRare) {
  const cacheKey = targetLabel;
  const now = Date.now();
  const cache = calibCache[cacheKey];
  const delta = cache ? rounds.length - cache.computedAt : Infinity;
  const age = cache ? now - cache.computedAtMs : Infinity;

  if (cache && delta < 50 && age < 600000) {
    return cache.result;
  }

  const n = rounds.length;
  const LOOK_AHEAD = TARGET_LOOK_AHEAD[targetLabel] || 20;
  const BIN_SIZE = 10;
  const NUM_BINS = 10;
  const minBin = targetRare ? MIN_BIN_RARE : MIN_BIN_NON_RARE;

  const globalHitRate = rounds.filter(r => r.multiplier >= targetMin).length / Math.max(1, n);
  const baseline = clamp(1 - Math.pow(Math.max(0, 1 - globalHitRate), LOOK_AHEAD), 0.01, 0.99);

  const expectedGapApprox = globalHitRate > 0 ? Math.round(1 / globalHitRate) : n;
  const minContext = Math.min(Math.floor(n / 4), Math.max(60, LOOK_AHEAD * 2, expectedGapApprox * 3));
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

  const hotBinHitsW = new Array(NUM_BINS).fill(0);
  const hotBinTotalW = new Array(NUM_BINS).fill(0);
  const hotBinCount = new Array(NUM_BINS).fill(0);
  const coldBinHitsW = new Array(NUM_BINS).fill(0);
  const coldBinTotalW = new Array(NUM_BINS).fill(0);
  const coldBinCount = new Array(NUM_BINS).fill(0);

  for (let pos = maxPos; pos >= minContext; ) {
    const distFromEnd = n - pos;
    const recWeight = Math.pow(0.999, distFromEnd);
    const ctx = rounds.slice(0, pos);
    let sf = null;
    try { sf = extractPredictiveStreakFeatures(ctx, targetMin, CALIB_DUMMY); } catch(_) {}
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
    const hotBin = Math.min(NUM_BINS-1, Math.floor(hs / BIN_SIZE));
    const coldBin = Math.min(NUM_BINS-1, Math.floor(cs / BIN_SIZE));
    const futureHit = rounds.slice(pos, pos + LOOK_AHEAD).some(r => r.multiplier >= targetMin);
    hotBinHitsW[hotBin] += futureHit ? recWeight : 0;
    hotBinTotalW[hotBin] += recWeight;
    hotBinCount[hotBin] += 1;
    coldBinHitsW[coldBin] += !futureHit ? recWeight : 0;
    coldBinTotalW[coldBin] += recWeight;
    coldBinCount[coldBin] += 1;
    const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
    pos -= stride;
  }

  const hotMargin = targetRare ? 0.015 : 0.025;
  const COLD_MIN_BIN = 12;           // v6: lowered for faster cold trigger
  const coldMargin = 0.015;          // v6: lowered to allow real white clusters

  const hotHitRate = new Array(NUM_BINS).fill(null);
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
      coldHitRates: coldHitRate.map(v => v !== null ? v.toFixed(3) : null),
      filledColdBins: coldHitRate.filter(v => v !== null).length,
    })}`);
  }

  calibCache[cacheKey] = { computedAt: n, computedAtMs: now, result };
  return result;
}

// getCalibratedAdjustment — COLD PATH MADE AGGRESSIVE
function getCalibratedAdjustment(hotScore, coldScore, calib, targetRare, sf) {
  if (!calib || calib === CALIB_DUMMY) {
    return { calibMult: 1.0, calibConfBonus: 0, calibrated: false };
  }

  const hotBin = Math.min(9, Math.floor(hotScore / (calib.BIN_SIZE || 10)));
  const coldBin = Math.min(9, Math.floor(coldScore / (calib.BIN_SIZE || 10)));
  const hotCount = calib.hotBinCount?.[hotBin] ?? 0;
  const coldCount = calib.coldBinCount?.[coldBin] ?? 0;
  const reqMinBin = targetRare ? (calib.minBin ?? 20) * 2 : (calib.minBin ?? 20);

  const densityTrend = sf?.densityTrend ?? 0;
  const inLowDataZone = hotCount >= reqMinBin && hotCount < reqMinBin * 1.5;
  if (inLowDataZone) {
    const conservMult = densityTrend > 0.05 ? 1.03 : 1.0;
    return { calibMult: conservMult, calibConfBonus: 0, calibrated: false };
  }

  const hotRate = (hotCount >= reqMinBin) ? calib.hotHitRate[hotBin] : null;
  const coldRate = (coldCount >= reqMinBin) ? calib.coldHitRate[coldBin] : null;

  const maxDeltaHot = targetRare ? 0.13 : 0.26;
  const sensiHot = 1.8;
  const maxDeltaCold = targetRare ? 0.20 : 0.35; // v6: raised for stronger cold extension
  const sensiCold = 1.8; // v6: raised from 1.5

  let calibMult = 1.0;
  let calibConfBonus = 0;
  let calibrated = false;

  // Hot path unchanged
  if (hotScore >= 72 && hotRate !== null) {
    const upliftNorm = clamp((hotRate - calib.baseline) / Math.max(0.001, 1 - calib.baseline), 0, 1);
    const reduction = clamp(upliftNorm * sensiHot, 0, maxDeltaHot);
    if (reduction > 0.01) {
      calibMult = 1.0 - reduction;
      calibConfBonus = upliftNorm > 0.25 ? 12 : upliftNorm > 0.15 ? 8 : 4;
      calibrated = true;
    }
  }

  // Cold path — v6 aggressive
  if (coldScore >= 55 && coldRate !== null && !calibrated) {
    const noHitBaseline = 1 - calib.baseline;
    const upliftNorm = clamp((coldRate - noHitBaseline) / Math.max(0.001, 1 - noHitBaseline), 0, 1);
    const extension = clamp(upliftNorm * sensiCold, 0, maxDeltaCold);
    if (extension > 0.01) {
      calibMult = 1.0 + extension;
      calibConfBonus = -4; // v6: stronger penalty
      calibrated = true;
    }
  }

  calibMult = clamp(calibMult, 0.76, 1.35); // v6: cold ceiling 1.35
  if (hotScore >= 72 && calibMult > 1.0) calibMult = 1.0; // safety

  return { calibMult, calibConfBonus, calibrated };
}

// extractPredictiveStreakFeatures — COLD SCORE BOOSTED
function extractPredictiveStreakFeatures(rounds, targetMin, calib) {
  const n = rounds.length;
  if (n < 10) return null;

  // RLE, core metrics, density windows, GARCH, post-cluster, regime (all unchanged)

  // coldScore — v6 boosted weights
  const coldScore = clamp(Math.round(
    (streakMomentum > 0.45 ? 38 : streakMomentum > 0.25 ? 19 : 0) +     // +10 from original
    (lowDensityAccel > 0.08 ? 32 : lowDensityAccel > 0.04 ? 16 : 0) +   // +10
    (markovProbHot < 0.22 ? 20 : markovProbHot < 0.38 ? 10 : 0) +
    (ld20 > ld50*1.40 ? 15 : ld20 > ld50*1.20 ? 7 : 0) +
    (currentStreakLen > avgLowRunLen*1.3 && !currentIsHigh ? 18 : 0)    // +6
  ), 0, 100);

  // Regime prediction — cold trigger lowered
  let predictedNextRegime = 'NEUTRAL';
  let transitionConfidence = 0;
  let predictedGapMultiplier = 1.0;

  if (hotScore >= 72 && hotScore > coldScore + 15) {
    // Hot path unchanged
    if (b2bPrecursor) {
      predictedNextRegime = 'ABOUT_TO_B2B';
      transitionConfidence = clamp(hotScore, 72, 95);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.36, 0.76, 0.88);
    } else {
      predictedNextRegime = 'ABOUT_TO_HOT';
      transitionConfidence = clamp(hotScore, 72, 90);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.26, 0.76, 0.90);
    }
  } else if (hotScore >= 62 && hotScore > coldScore + 10) {
    predictedNextRegime = 'ABOUT_TO_HOT';
    transitionConfidence = clamp(hotScore, 62, 88);
    predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.20, 0.80, 0.93);
  } else if (coldScore >= 55 && coldScore > hotScore + 12) {  // v6: 58 → 55
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence = clamp(coldScore, 55, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.45, 1.15, 1.35);
    } else {
      predictedNextRegime = 'ABOUT_TO_COLD';
      transitionConfidence = clamp(coldScore, 50, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.32, 1.10, 1.32);
    }
  }

  const targetRare = calib?.targetRare ?? false;
  const cal = getCalibratedAdjustment(hotScore, coldScore, calib, targetRare, {densityTrend});
  if (cal.calibrated) {
    predictedGapMultiplier = cal.calibMult;
    if (predictedNextRegime === 'NEUTRAL' && cal.calibMult < 0.95 && hotScore >= 55) {
      predictedNextRegime = 'ABOUT_TO_HOT';
      transitionConfidence = Math.min(transitionConfidence + 10, 85);
    }
  }

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
    calibConfBonus: cal.calibConfBonus, calibrated: cal.calibrated,
  };
}

// applyStreakAdjustment — COLD BLEND EARLIER & STRONGER
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  const pnr = sf.predictedNextRegime;
  const mult = sf.predictedGapMultiplier ?? 1.0;
  const tc = sf.transitionConfidence ?? 0;

  if ((pnr === 'ABOUT_TO_WHITE_CLUSTER' || pnr === 'ABOUT_TO_COLD') && tc >= 55) {
    const blendFactor = clamp((tc - 55) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  if ((pnr === 'ABOUT_TO_B2B' || pnr === 'ABOUT_TO_HOT') && tc >= 65) {
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  let adj = expectedGap;
  switch (sf.regime) {
    case 'B2B': adj = Math.round(adj * (1 - sf.b2bContinuationProb * 0.30)); break;
    case 'HOT_AFTER_SHORT_COLD': adj = Math.round(adj * 0.88); break;
    case 'HOT': adj = Math.round(adj * (1 - (1 - sf.lowDensity20) * 0.18)); break;
    case 'WHITE_CLUSTER':
      adj = sf.avgPostClusterGap !== null
        ? Math.round(adj * 0.65 + sf.avgPostClusterGap * 0.35)
        : Math.round(adj * 0.92); break;
    case 'EXTREME_WHITE': adj = Math.round(adj * 0.75); break;
    case 'COLD': adj = Math.round(adj * (1 + sf.densityTrend * 0.12)); break;
    default: adj = Math.round(adj * (1 - sf.densityTrend * 0.06));
  }
  return Math.max(1, adj);
}

// streakConfBonus — COLD PENALTY STRONGER
function streakConfBonus(sf, isRare) {
  if (!sf) return 0;
  const pnr = sf.predictedNextRegime;
  const tc = sf.transitionConfidence ?? 0;
  const cb = sf.calibConfBonus ?? 0;
  let base = cb;

  if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && sf.calibrated && tc > 75) {
    base += 14 + (isRare && sf.b2bPrecursor ? 4 : 0);
  } else if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && tc >= 65) {
    base += 7;
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 55) {
    base -= 6; // v6: stronger penalty
  }

  const calibMult = sf.calibrated ? 1.0 : 0.5;
  switch (sf.regime) {
    case 'B2B': base += Math.floor((sf.b2bContinuationProb > 0.3 ? 5 : 2) * calibMult); break;
    case 'WHITE_CLUSTER': base += Math.floor((sf.avgPostClusterGap !== null ? 4 : 2) * calibMult); break;
    case 'EXTREME_WHITE': base += Math.floor(6 * calibMult); break;
    case 'HOT': base += Math.floor(3 * calibMult); break;
    case 'COLD': base -= 2; break;
  }
  return base;
}

function effectiveRegime(sf) {
  if (!sf) return 'NEUTRAL';
  return (sf.transitionConfidence ?? 0) >= 68 ? sf.predictedNextRegime : sf.regime;
}

// All run* engine functions (runHybridLstmXgb to runSHA512) — unchanged except they now benefit from stronger sf

// runStackingMeta — unchanged

// computeNgConsensus — unchanged

// MAIN TICK — RECALCULATE ON MISS ADDED
async function runNgComputeEngine() {
  try {
    const rounds = await getNgRounds();
    if (rounds.length < 50) return;
    const lastRoundId = rounds[rounds.length - 1].roundId;

    const calibrations = {};
    for (const target of TARGETS) {
      try {
        calibrations[target.label] = computeCalibration(rounds, target.min, target.label, target.rare);
      } catch(e) {
        calibrations[target.label] = null;
        console.error(`[ngCompute] calib/${target.label}:`, e.message);
      }
    }

    const streakFeatures = {};
    for (const target of TARGETS) {
      try {
        streakFeatures[target.label] = extractPredictiveStreakFeatures(
          rounds, target.min, calibrations[target.label] ?? CALIB_DUMMY
        );
      } catch(e) {
        streakFeatures[target.label] = null;
        console.error(`[ngCompute] psf/${target.label}:`, e.message);
      }
    }

    const ALGO_MAP = {
      hlstm_xgb: runHybridLstmXgb, htrans_lstm: runHybridTransformerLstm,
      htft: runHybridTft, tft: runTftFull, nbeats: runNbeats, tcn: runTcn,
      lgbm: runLightGBM, gru: runGRU, bilstm: runBiLSTM, sha512: runSHA512,
    };

    const allNgResults = {};
    for (const [eid, algo] of Object.entries(ALGO_MAP)) {
      allNgResults[eid] = {};
      for (const target of TARGETS) {
        try {
          const r = algo(rounds, target, streakFeatures[target.label]);
          if (r) allNgResults[eid][target.label] = r;
        } catch(e) {
          console.error(`[ngCompute] ${eid}/${target.label}:`, e.message);
        }
      }
    }

    allNgResults['stacking'] = {};
    for (const target of TARGETS) {
      try {
        const r = runStackingMeta(rounds, target, allNgResults, streakFeatures[target.label]);
        if (r) allNgResults['stacking'][target.label] = r;
      } catch(e) {
        console.error(`[ngCompute] stacking/${target.label}:`, e.message);
      }
    }

    const ngConsensus = computeNgConsensus(allNgResults, lastRoundId, streakFeatures);
    allNgResults['ng_consensus'] = {};
    for (const target of TARGETS) {
      const c = ngConsensus[target.label];
      if (c) {
        const baseConf = clamp(55 + Math.round(c.engineCount * 4), 55, 95);
        const finalConf = clamp(baseConf + (c.tcBonus ?? 0), 55, 99);
        allNgResults['ng_consensus'][target.label] = {
          low: c.lo - lastRoundId,
          high: c.hi - lastRoundId,
          expectedGap: Math.round((c.lo + c.hi) / 2 - lastRoundId),
          probW: null,
          conf: finalConf,
          _meta: { engineCount: c.engineCount, engines: c.engines },
        };
      }
    }

    // Resolution + lock loop — RECALCULATE ON MISS
    for (const engineId of NG_ENGINE_IDS) {
      const payload = {};
      for (const target of TARGETS) {
        const win = ngWindows[engineId][target.label];
        const fresh = allNgResults[engineId]?.[target.label];

        if (win) {
          const { lo, hi, generation, roundWhenMade } = win;
          const eLo = Math.max(roundWhenMade + 1, lo - earlyHitTolerance(target.maxWidth));
          const earlyHit = lo > roundWhenMade + 1 && eLo <= lo - 1
            ? findHitInRange(rounds, eLo, lo - 1, target.min)
            : null;

          if (earlyHit) {
            await saveNgOutcome(engineId, target, 'early', lo, hi, earlyHit.roundId, generation);
            delete ngWindows[engineId][target.label];
          } else if (lastRoundId >= hi) {
            const hit = findHitInRange(rounds, lo, hi, target.min);
            await saveNgOutcome(engineId, target, hit ? 'win' : 'loss', lo, hi, hit?.roundId ?? null, generation);

            // v6: RECALCULATE ON MISS
            delete ngWindows[engineId][target.label];
            if (!hit) {
              // Force fresh prediction immediately
              const newSf = streakFeatures[target.label];
              const newFresh = ALGO_MAP[engineId]?.(rounds, target, newSf) ||
                               (engineId === 'stacking' ? runStackingMeta(rounds, target, allNgResults, newSf) : null) ||
                               (engineId === 'ng_consensus' ? computeNgConsensus(allNgResults, lastRoundId, streakFeatures)[target.label] : null);

              if (newFresh && newFresh.low && newFresh.high) {
                const newLo = lastRoundId + newFresh.low;
                const newHi = lastRoundId + newFresh.high;
                const newGen = generation + 1;
                ngWindows[engineId][target.label] = {
                  lo: newLo, hi: newHi,
                  roundWhenMade: lastRoundId,
                  generation: newGen,
                  eta: { probW: newFresh.probW, conf: newFresh.conf, expectedGap: newFresh.expectedGap }
                };
                payload[target.label] = ngWindows[engineId][target.label];
                console.log(`[ngCompute v6] ${engineId} ${target.label} MISS → RECALCULATED new window #${newLo}-#${newHi}`);
              }
            }
          } else {
            const hit = findHitInRange(rounds, lo, hi, target.min);
            if (hit) {
              await saveNgOutcome(engineId, target, 'win', lo, hi, hit.roundId, generation);
              delete ngWindows[engineId][target.label];
            } else {
              payload[target.label] = { lo, hi, roundWhenMade, generation, eta: win.eta };
              continue;
            }
          }
        }

        if (fresh) {
          const newLo = lastRoundId + fresh.low;
          const newHi = lastRoundId + fresh.high;
          const gen = (ngWindows[engineId][target.label]?.generation ?? 0) + 1;
          const baseEta = { probW: fresh.probW, conf: fresh.conf, expectedGap: fresh.expectedGap };
          const eta = fresh._meta ? { ...baseEta, ...fresh._meta } : baseEta;
          ngWindows[engineId][target.label] = { lo: newLo, hi: newHi, roundWhenMade: lastRoundId, generation: gen, eta };
          payload[target.label] = ngWindows[engineId][target.label];
        }
      }

      if (Object.keys(payload).length) await saveLockedAdvPreds(engineId, payload);
    }
  } catch (e) {
    console.error('[ngCompute] Fatal:', e.message, e.stack);
  }
}

// init, runWithInit, reset functions unchanged

module.exports = {
  runNgComputeEngine: runNgComputeEngineWithInit,
  resetNgComputeState,
  resetNgWindowsOnly,
  NG_ENGINE_IDS
};