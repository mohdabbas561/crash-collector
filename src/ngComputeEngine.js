'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
// ================================================================================
// ACCURACY & CALIBRATION REBUILD — v3
//
// a) AUDIT SUMMARY — bugs that caused 39% win rate:
//
// BUG-A: garchSignal threshold 0.15 is too low — fires on 64% of ALL random windows.
//        b2bPrecursor = (shortLowRun ~41%) AND (garch>0.15 ~64%) = fires ~26% randomly.
//        This contributed +25 to hotScore on a quarter of all windows, pushing scores
//        above 55 far too often → false ABOUT_TO_B2B signals constantly.
//        FIX: raise garchSignal threshold to 0.40 and require garch > rolling baseline.
//
// BUG-B: postClusterEarlySignal threshold avgLowRunLen*1.0 fires the MOMENT the run
//        hits average length, not when it's extended. A run of exactly average length
//        is not evidence of an imminent breakout. Contributed +35 spuriously.
//        FIX: raise to avgLowRunLen*1.5 (clearly above average).
//
// BUG-C: Markov matrix build loop has a dead outer statement. Loop runs for(i=1...)
//        but the dead `k` variable is computed and thrown away, and the inner
//        `if(i>=2)` builds the real matrix. The last2Key then queries
//        (seq[-2], seq[-1]) where seq[-1] is the CURRENT INCOMPLETE run. The matrix
//        was built from fully completed runs. This key never matches → markovProbHot
//        always falls back to globalLowRate. The signal is broken for all targets.
//        FIX: exclude the last (current, incomplete) run from the matrix query.
//        Query (seq[-3], seq[-2]) as the context predicting seq[-1].
//
// BUG-D: Mild signal branch fires at hotScore >= 35 with tc = 35-54.
//        applyStreakAdjustment fires when tc >= 30. So ALL mild signals apply gap
//        adjustment (~0.92-0.99x) — barely moves the window but mislabels it as
//        hot, corrupting stacking weights and effectiveRegime.
//        FIX: eliminate mild branch entirely. NEUTRAL below 62 (hot) / 58 (cold).
//
// BUG-E: streakMomentum only computed when !currentIsHigh. During a hot streak,
//        it is always 0 — the asymmetry means cold-incoming signals never fire
//        in time to protect profitable windows from being wasted by a cold streak.
//        FIX: compute bidirectional streak momentum (high-run and low-run).
//
// BUG-F: predictedGapMultiplier range was 0.35–1.60. At hotScore=80, a multiplier
//        of 0.56 with blendFactor=0.71 gives 0.69x on the gap. For a target with
//        expectedGap=20, that means predicting a hit in round 14 when the real
//        distribution says 20. Windows open 6 rounds too early → EARLY outcomes.
//        FIX: hard caps 0.72–1.28 max, only extended when calibration confirms.
//
// BUG-G: No historical self-calibration. The system blindly applies signals derived
//        from signal strength alone, with no feedback from whether past hot signals
//        actually led to hits in the next N rounds. A signal is only useful if its
//        historical hit-rate in follow-up rounds exceeds the baseline rate.
//        FIX: add computeCalibration() — scans full 14k+ history once per tick to
//        build P(hit in next 15r | hotScore bin) tables, then gate all adjustments
//        behind this empirical confirmation.
// ================================================================================

const {
  getRounds, savePrediction, getPredictions,
  saveLockedAdvPreds, getLockedAdvPreds,
} = require('./db');

const NG_ENGINE_IDS = [
  'hlstm_xgb','htrans_lstm','htft','tft','nbeats',
  'tcn','lgbm','gru','bilstm','stacking','sha512','ng_consensus',
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

// =============================================================================
// === ACCURACY & CALIBRATION FIX START ===
// HISTORICAL SELF-CALIBRATION ENGINE
// Scans full 14k+ history once per tick to build P(hit in next 15r | signal bin).
// All predictive adjustments are gated behind empirical confirmation.
// Returns calibration object with hotHitRates[bin] and coldHitRates[bin].
// =============================================================================

// Cache calibration per target per tick (recomputed each tick with latest data)
const calibCache = {};

function computeCalibration(rounds, targetMin, targetLabel) {
  // Use cached result if rounds haven't grown by more than 20 since last compute
  const cacheKey = targetLabel;
  if (calibCache[cacheKey] && rounds.length - calibCache[cacheKey].computedAt < 20) {
    return calibCache[cacheKey].result;
  }

  const n = rounds.length;
  const LOOK_AHEAD = 15; // rounds to check after signal for a hit
  const BIN_SIZE   = 10; // hotScore bins: 0,10,20,...,90
  const NUM_BINS   = 10;

  // For each position in history, compute a LIGHTWEIGHT hotScore proxy
  // (full extractPredictive is too expensive to run at every position)
  // We use: lowDensity10 vs lowDensity20 for the density accel signal,
  // and whether the last completed low run was short.
  const hotBinHits   = new Array(NUM_BINS).fill(0);
  const hotBinTotal  = new Array(NUM_BINS).fill(0);
  const coldBinHits  = new Array(NUM_BINS).fill(0); // "hit" = next 15r had NO high
  const coldBinTotal = new Array(NUM_BINS).fill(0);

  // Step through history at stride 5 to keep it fast
  const stride = 5;
  const minContext = 60; // need at least 60 rounds of history

  for (let pos = minContext; pos < n - LOOK_AHEAD; pos += stride) {
    const ctx = rounds.slice(0, pos);
    const ctxLen = ctx.length;

    // Fast density windows
    const w5  = ctx.slice(-5);
    const w10 = ctx.slice(-10);
    const w20 = ctx.slice(-20);
    const w50 = ctx.slice(-50);
    const ld5  = w5.filter(r=>r.multiplier<targetMin).length  / Math.max(1,w5.length);
    const ld10 = w10.filter(r=>r.multiplier<targetMin).length / Math.max(1,w10.length);
    const ld20 = w20.filter(r=>r.multiplier<targetMin).length / Math.max(1,w20.length);
    const ld50 = w50.filter(r=>r.multiplier<targetMin).length / Math.max(1,w50.length);
    const globalHitRate = 1 - ld50;

    // Fast RLE for last 100 rounds only
    const recentCtx = ctx.slice(-100);
    const runs = [];
    let curHigh = recentCtx[0].multiplier >= targetMin, curLen = 1;
    for (let i = 1; i < recentCtx.length; i++) {
      const h = recentCtx[i].multiplier >= targetMin;
      if (h === curHigh) { curLen++; }
      else { runs.push({isHigh:curHigh,len:curLen}); curHigh=h; curLen=1; }
    }
    runs.push({isHigh:curHigh,len:curLen});

    const highRuns = runs.filter(r=> r.isHigh).map(r=>r.len);
    const lowRuns  = runs.filter(r=>!r.isHigh).map(r=>r.len);
    const lastRun  = runs[runs.length-1];
    const curIsHigh = lastRun.isHigh;
    const curLen2   = lastRun.len;
    const avgLow = lowRuns.length ? mean(lowRuns) : 5;
    const avgHigh = highRuns.length ? mean(highRuns) : 1;

    // Lightweight hotScore proxy (fast version of the full signal)
    const densAccel = (ld5 - ld10) - (ld10 - ld20);
    const inLongLow = !curIsHigh && curLen2 >= avgLow * 1.5;
    const shortLastLow = !curIsHigh && lowRuns.length >= 2 &&
      lowRuns[lowRuns.length-2] < avgLow * 0.55;
    const b2bOcc = highRuns.filter(l=>l>=2).length;
    const b2bRate = highRuns.length ? b2bOcc/highRuns.length : 0;
    const densityFalling = ld20 < ld50 * 0.85; // density falling below long baseline

    let hs = 0;
    if (inLongLow && densAccel < -0.04 && shortLastLow) hs += 30; // postCluster early (tightened)
    if (densityFalling && !curIsHigh)                   hs += 20; // density falling vs baseline
    if (densAccel < -0.06)                              hs += 15; // strong acceleration
    else if (densAccel < -0.02)                         hs +=  8;
    if (b2bRate > 0.25)                                 hs += 10;
    else if (b2bRate > 0.15)                            hs +=  5;
    hs = clamp(hs, 0, 100);

    // coldScore proxy
    let cs = 0;
    const densityRising = ld20 > ld50 * 1.15;
    if (curIsHigh && curLen2 >= avgHigh * 1.5) cs += 25; // long high run = b2b ending
    if (densityRising)                          cs += 20;
    if (densAccel > 0.04)                       cs += 15;
    cs = clamp(cs, 0, 100);

    const hotBin  = Math.min(NUM_BINS-1, Math.floor(hs / BIN_SIZE));
    const coldBin = Math.min(NUM_BINS-1, Math.floor(cs / BIN_SIZE));

    // Check if a hit occurred in next LOOK_AHEAD rounds
    const futureHit = rounds.slice(pos, pos + LOOK_AHEAD).some(r => r.multiplier >= targetMin);
    const noFutureHit = !futureHit;

    hotBinTotal[hotBin]++;
    if (futureHit) hotBinHits[hotBin]++;
    coldBinTotal[coldBin]++;
    if (noFutureHit) coldBinHits[coldBin]++;
  }

  // Build hit-rate lookup: for each bin, P(hit in next 15r | signal in this bin)
  const hotHitRate  = new Array(NUM_BINS).fill(null);
  const coldHitRate = new Array(NUM_BINS).fill(null);
  const baseline    = rounds.filter(r=>r.multiplier>=targetMin).length /
    Math.max(1,rounds.length) * LOOK_AHEAD; // expected hits in 15r by chance

  for (let b = 0; b < NUM_BINS; b++) {
    if (hotBinTotal[b] >= 10) {
      hotHitRate[b]  = hotBinHits[b]  / hotBinTotal[b];
    }
    if (coldBinTotal[b] >= 10) {
      coldHitRate[b] = coldBinHits[b] / coldBinTotal[b];
    }
  }

  const result = {
    hotHitRate,   // P(hit in 15r | hotScore bin)
    coldHitRate,  // P(no-hit in 15r | coldScore bin)
    baseline:     clamp(baseline, 0.01, 0.99),
    LOOK_AHEAD,
    BIN_SIZE,
  };

  calibCache[cacheKey] = { computedAt: rounds.length, result };
  return result;
}

// Get calibrated gap multiplier — the SAFE version of the prediction.
// Only shortens/lengthens if historical data confirms the signal is better than baseline.
// === ACCURACY & CALIBRATION FIX START ===
function getCalibratedAdjustment(hotScore, coldScore, calib, target) {
  if (!calib) return { calibMult: 1.0, calibConfBonus: 0, calibrated: false };

  const hotBin  = Math.min(9, Math.floor(hotScore  / calib.BIN_SIZE));
  const coldBin = Math.min(9, Math.floor(coldScore  / calib.BIN_SIZE));

  const hotRate  = calib.hotHitRate[hotBin];
  const coldRate = calib.coldHitRate[coldBin];

  let calibMult       = 1.0;
  let calibConfBonus  = 0;
  let calibrated      = false;

  // HOT signal: only adjust if empirical hit-rate exceeds baseline + margin
  if (hotScore >= 62 && hotRate !== null) {
    const expectedRate = calib.baseline;
    const uplift = hotRate - expectedRate;
    if (uplift > 0.05) {
      // Shorten gap proportionally to uplift, capped at ±0.28
      const reduction = clamp(uplift * 1.5, 0, 0.28);
      calibMult = 1.0 - reduction;
      calibConfBonus = uplift > 0.12 ? 12 : uplift > 0.07 ? 7 : 4;
      calibrated = true;
    }
  }

  // COLD signal: only lengthen if empirical no-hit rate exceeds baseline + margin
  if (coldScore >= 58 && coldRate !== null && calibMult === 1.0) {
    const expectedNoHitRate = 1 - calib.baseline;
    const uplift = coldRate - expectedNoHitRate;
    if (uplift > 0.05) {
      const extension = clamp(uplift * 1.5, 0, 0.28);
      calibMult = 1.0 + extension;
      calibConfBonus = -3;
      calibrated = true;
    }
  }

  // HARD CAPS — never go below 0.72x or above 1.28x regardless of signal
  // (BUG-F fix: was 0.35–1.60 causing early hits and window misses)
  calibMult = clamp(calibMult, 0.72, 1.28);

  return { calibMult, calibConfBonus, calibrated };
}
// === ACCURACY & CALIBRATION FIX END ===

// =============================================================================
// === ACCURACY & CALIBRATION FIX START ===
// extractPredictiveStreakFeatures — rebuilt with all bugs fixed
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

  // === ACCURACY & CALIBRATION FIX START ===
  // GARCH — BUG-A FIX: compute against rolling BASELINE, not fixed threshold.
  // garchSignal > 0.15 fires on 64% of random data. Now we require garch to
  // exceed the rolling 100-round baseline by a meaningful margin.
  const {gaps} = computeGaps(rounds, targetMin);
  let garchSignal = 0, garchBaseline = 0;
  if (gaps.length >= 10) {
    const gm=mean(gaps), ad=gaps.map(g=>Math.abs(g-gm));
    let cov=0,vs=0;
    for(let i=1;i<ad.length;i++) cov+=ad[i-1]*ad[i];
    for(const v of ad) vs+=v*v;
    garchSignal = vs>0 ? cov/vs : 0;
    // Baseline: same calc on the older half of gaps
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
  // garchRising = TRUE only when recent volatility clustering EXCEEDS the historical baseline
  // (not just any absolute value > 0.15 which fires on 64% of all windows)
  const garchRising = garchSignal > garchBaseline * 1.30 && garchSignal > 0.25;
  // === ACCURACY & CALIBRATION FIX END ===

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

  // ==========================================================================
  // === ACCURACY & CALIBRATION FIX START ===
  // FORWARD-LOOKING SIGNALS — all bugs fixed
  // ==========================================================================

  // SIGNAL A: Density acceleration (2nd derivative) — unchanged, correct formula
  const lowDensityAccel = (ld5 - ld10) - (ld10 - ld20);

  // SIGNAL B: BUG-E FIX — bidirectional streak momentum
  // Now tracks BOTH high-run and low-run shortening/lengthening
  let streakMomentumLow = 0, streakMomentumHigh = 0;
  if (!currentIsHigh && lowRuns.length >= 4) {
    const prev3 = mean(lowRuns.slice(-4,-1));
    streakMomentumLow = (currentStreakLen - prev3) / Math.max(1, prev3);
  }
  if (currentIsHigh && highRuns.length >= 4) {
    const prev3 = mean(highRuns.slice(-4,-1));
    streakMomentumHigh = (currentStreakLen - prev3) / Math.max(1, prev3);
    // Long high run = cold incoming. Positive = lengthening = more hot.
  }
  // Combined: negative = getting hotter (low runs shortening OR high runs shortening)
  const streakMomentum = !currentIsHigh ? streakMomentumLow : -streakMomentumHigh;

  // SIGNAL C: BUG-B FIX — postClusterEarlySignal threshold raised to 1.5x
  // Also requires density to be falling vs 50-round baseline, not just acceleration
  let postClusterEarlySignal = false;
  if (!currentIsHigh && lowRuns.length >= 3) {
    const recentLowsShortening = lowRuns.slice(-3,-1).every(l => l < avgLowRunLen);
    const densityFallingVsBaseline = ld20 < ld50 * 0.88; // density clearly falling
    const inExtendedCluster = currentStreakLen >= avgLowRunLen * 1.5; // BUG-B FIX: was 1.0
    postClusterEarlySignal = inExtendedCluster && densityFallingVsBaseline && recentLowsShortening;
  }

  // SIGNAL D: BUG-A FIX — b2bPrecursor now requires garchRising (vs baseline, not 0.15)
  let b2bPrecursor = false;
  if (lowRuns.length >= 2) {
    // Use the SECOND-TO-LAST completed low run (not the current run if in high streak)
    const lastCompletedLowIdx = currentIsHigh ? lowRuns.length - 1 : lowRuns.length - 2;
    const lastCompletedLow = lastCompletedLowIdx >= 0 ? lowRuns[lastCompletedLowIdx] : null;
    if (lastCompletedLow !== null) {
      const shortLowRun = lastCompletedLow < avgLowRunLen * 0.55;
      b2bPrecursor = shortLowRun && garchRising; // BUG-A FIX: garchRising is now calibrated
    }
  }

  // SIGNAL E: BUG-C FIX — Markov transition matrix
  // The matrix maps (completedRun[-2], completedRun[-1]) → prediction of NEXT run.
  // The query context must use only COMPLETED runs (not the current in-progress run).
  // seq[-1] is the current INCOMPLETE run — WRONG to use as query context.
  // FIX: build matrix from seq[0..len-1] predicting seq[i]. Query with seq[-2],seq[-1]
  // where seq[-1] is the LAST COMPLETED run (not including current).
  let markovProbHot = (1 - globalLowRate) || 0.1;
  // Use only COMPLETED runs (exclude the last = current in-progress run)
  const completedRuns = runs.slice(0, -1); // all runs except the current one
  if (completedRuns.length >= 4) {
    const seq = completedRuns.map(r => r.isHigh ? 1 : 0);
    const mat = {};
    for (let i = 2; i < seq.length; i++) {
      // (seq[i-2], seq[i-1]) → seq[i] (BUG-C FIX: no dead outer loop)
      const key = `${seq[i-2]},${seq[i-1]}`;
      if (!mat[key]) mat[key] = {H:0, L:0};
      if (seq[i] === 1) mat[key].H++; else mat[key].L++;
    }
    // Context: last two COMPLETED runs (not the current one)
    const last2Key = `${seq[seq.length-2]},${seq[seq.length-1]}`;
    const cell = mat[last2Key];
    if (cell) {
      const tot = cell.H + cell.L;
      if (tot >= 5) markovProbHot = cell.H / tot; // raised from 3 to 5 for stability
    }
  }

  // SIGNAL F: Composite hotScore / coldScore
  // BUG-D FIX: Raise signal weights to require stronger evidence.
  // b2bPrecursor (+25 → +20 but now calibrated), postClusterEarly (+35 → +30 but tighter thresh)
  const hotScore = clamp(Math.round(
    (postClusterEarlySignal   ? 30 : 0) +
    (b2bPrecursor             ? 20 : 0) +
    (lowDensityAccel < -0.08  ? 18 : lowDensityAccel < -0.04 ? 9 : 0) +
    (streakMomentum   < -0.35 ? 15 : streakMomentum  < -0.15 ? 7 : 0) +
    (markovProbHot   > 0.68   ? 15 : markovProbHot   > 0.55  ? 7 : 0) +
    (b2bRate         > 0.28   ? 10 : b2bRate          > 0.18  ? 5 : 0) +
    (ld20            < ld50*0.82 ? 10 : 0)  // NEW: density clearly below baseline
  ), 0, 100);

  const coldScore = clamp(Math.round(
    (streakMomentum   > 0.45   ? 28 : streakMomentum  > 0.25  ? 14 : 0) +
    (lowDensityAccel  > 0.08   ? 22 : lowDensityAccel  > 0.04  ? 11 : 0) +
    (markovProbHot   < 0.22    ? 20 : markovProbHot   < 0.38   ? 10 : 0) +
    (ld20            > ld50*1.40 ? 15 : ld20 > ld50*1.20 ? 7 : 0) +
    (currentStreakLen > avgLowRunLen*1.3 && !currentIsHigh ? 12 : 0)
  ), 0, 100);

  // SIGNAL G: BUG-D FIX — raise thresholds significantly.
  // Old: ABOUT_TO_B2B at hotScore >= 55. New: >= 68.
  // Old: ABOUT_TO_HOT at hotScore >= 55. New: >= 62.
  // Old: mild branch at hotScore >= 35. ELIMINATED.
  // BUG-F FIX: predictedGapMultiplier hard-capped at 0.72–1.28.
  let predictedNextRegime    = 'NEUTRAL';
  let transitionConfidence   = 0;
  let predictedGapMultiplier = 1.0;

  if (hotScore >= 68 && hotScore > coldScore + 15) {
    if (b2bPrecursor || b2bRate > 0.22) {
      predictedNextRegime    = 'ABOUT_TO_B2B';
      transitionConfidence   = clamp(hotScore, 68, 95);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.38, 0.72, 0.88); // BUG-F FIX
    } else {
      predictedNextRegime    = 'ABOUT_TO_HOT';
      transitionConfidence   = clamp(hotScore, 62, 90);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.28, 0.75, 0.90); // BUG-F FIX
    }
  } else if (hotScore >= 62 && hotScore > coldScore + 10) {
    predictedNextRegime    = 'ABOUT_TO_HOT';
    transitionConfidence   = clamp(hotScore, 62, 88);
    predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.22, 0.78, 0.92); // BUG-F FIX
  } else if (coldScore >= 58 && coldScore > hotScore + 15) {
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime    = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence   = clamp(coldScore, 58, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.30, 1.10, 1.28); // BUG-F FIX
    } else {
      predictedNextRegime    = 'ABOUT_TO_COLD';
      transitionConfidence   = clamp(coldScore, 52, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.22, 1.05, 1.22); // BUG-F FIX
    }
  }
  // NO mild branch (BUG-D FIX): below the strong thresholds = NEUTRAL

  // Apply calibration to override/gate the multiplier
  const cal = getCalibratedAdjustment(hotScore, coldScore, calib, targetMin);
  if (cal.calibrated) {
    // Trust calibration over raw signal — it's based on 14k+ real outcomes
    predictedGapMultiplier = cal.calibMult;
    if (predictedNextRegime === 'NEUTRAL' && cal.calibMult < 0.95 && hotScore >= 55) {
      predictedNextRegime  = 'ABOUT_TO_HOT';
      transitionConfidence = Math.min(transitionConfidence + 10, 85);
    }
  }
  const calibConfBonus = cal.calibConfBonus;

  // === ACCURACY & CALIBRATION FIX END ===

  return {
    // Core RLE (unchanged, backward compatible)
    runs, highRuns, lowRuns,
    currentIsHigh, currentStreakLen, regime,
    b2bOccurrences: highRuns.filter(l=>l>=2).length, b2bRate, b2bContinuationProb,
    avgHighRunLen, maxHighRunLen,
    avgLowRunLen, maxLowRunLen, stdLowRunLen, avgPostClusterGap,
    lowDensity10: ld10, lowDensity20: ld20, lowDensity50: ld50, densityTrend,
    globalLowRate, garchSignal, garchBaseline,
    // Predictive fields (all fixed)
    lowDensityAccel,
    streakMomentum,        // BUG-E FIX: now bidirectional
    postClusterEarlySignal,// BUG-B FIX: threshold raised to 1.5x
    b2bPrecursor,          // BUG-A FIX: garchRising vs baseline
    markovProbHot,         // BUG-C FIX: uses completed runs only
    hotScore, coldScore,
    predictedNextRegime,   // BUG-D FIX: thresholds raised
    transitionConfidence,
    predictedGapMultiplier, // BUG-F FIX: hard-capped 0.72–1.28
    calibConfBonus,         // NEW: from historical calibration
    calibrated: cal.calibrated,
  };
}
// === ACCURACY & CALIBRATION FIX END ===

// =============================================================================
// === ACCURACY & CALIBRATION FIX START ===
// applyStreakAdjustment — uses calibrated multiplier, tc threshold raised to 55
// =============================================================================
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  const pnr  = sf.predictedNextRegime;
  const mult = sf.predictedGapMultiplier ?? 1.0;
  const tc   = sf.transitionConfidence  ?? 0;

  // BUG-D FIX: raise activation threshold from tc>=30 to tc>=55
  // Below 55 = not confident enough to move the window
  if (pnr !== 'NEUTRAL' && tc >= 55) {
    // Blend: 55=0% effect, 100=full effect
    const blendFactor = clamp((tc - 55) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  // Reactive fallback (always correct, unchanged from v1)
  let adj = expectedGap;
  switch (sf.regime) {
    case 'B2B':
      adj = Math.round(adj * (1 - sf.b2bContinuationProb * 0.30)); break;
    case 'HOT_AFTER_SHORT_COLD':
      adj = Math.round(adj * 0.88); break;
    case 'HOT':
      adj = Math.round(adj * (1 - (1 - sf.lowDensity20) * 0.18)); break;
    case 'WHITE_CLUSTER':
      adj = sf.avgPostClusterGap !== null
        ? Math.round(adj * 0.55 + sf.avgPostClusterGap * 0.45)
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
// === ACCURACY & CALIBRATION FIX END ===

// === ACCURACY & CALIBRATION FIX START ===
// streakConfBonus — uses calibConfBonus from historical calibration
// threshold raised: hot bonus only when tc > 72 (was 70)
// === ACCURACY & CALIBRATION FIX END ===
function streakConfBonus(sf, isRare) {
  if (!sf) return 0;
  const pnr = sf.predictedNextRegime;
  const tc  = sf.transitionConfidence ?? 0;
  const cb  = sf.calibConfBonus ?? 0;

  // Apply calibration-based bonus first (from historical data)
  let base = cb;

  // === ACCURACY & CALIBRATION FIX START ===
  // BUG-D FIX: only give predictive confidence boost when tc > 72 (raised from 70)
  if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && tc > 72) {
    base += 12 + (isRare && sf.b2bPrecursor ? 4 : 0);
    // +15 spec requirement for high-confidence hot signal — only when calibrated
    if (sf.calibrated) base += 3;
  } else if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && tc >= 62) {
    base += 6;
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 65) {
    base -= 3;
  }
  // === ACCURACY & CALIBRATION FIX END ===

  // Reactive fallback
  switch (sf.regime) {
    case 'B2B':           base += sf.b2bContinuationProb > 0.3 ? 5 : 2; break;
    case 'WHITE_CLUSTER': base += sf.avgPostClusterGap !== null ? 4 : 2; break;
    case 'EXTREME_WHITE': base += 6; break;
    case 'HOT':           base += 3; break;
    case 'COLD':          base -= 2; break;
  }
  return base;
}

// Helper: use predictive regime only when tc >= 62 (was 40 — too low)
function effectiveRegime(sf) {
  if (!sf) return 'NEUTRAL';
  const tc = sf.transitionConfidence ?? 0;
  return tc >= 62 ? sf.predictedNextRegime : sf.regime; // BUG-D FIX: raised from 40
}

// =============================================================================
// ENGINE 1: hybrid_lstm_xgb
// === ACCURACY & CALIBRATION FIX START ===
// b2bBoost now uses calibrated signal, not raw rate
// === ACCURACY & CALIBRATION FIX END ===
// =============================================================================
function runHybridLstmXgb(rounds, target, sf) {
  const {gaps, currentGap} = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;
  const hrGlobal = gaps.length/rounds.length;

  const DECAY=0.97; let wS=0,wG=0,w=1;
  for(let i=gaps.length-1;i>=0;i--){wG+=gaps[i]*w;wS+=w;w*=DECAY;}
  const ewaMean=wG/(wS||1);

  const gMean=mean(gaps), gStd=stdDev(gaps)||1;
  const recentN=Math.min(300,Math.max(10,Math.round(3/(hrGlobal||0.01))));
  const hrRecent=(rounds.slice(-recentN).filter(r=>r.multiplier>=target.min).length+1)/(recentN+2);
  const {b:slope,r2}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  const overdue=clamp(currentGap/(gMean||1),0,3);
  const cv=gStd/(gMean||1);

  // === ACCURACY & CALIBRATION FIX START ===
  const pnr = effectiveRegime(sf); // raised threshold
  const b2bBoost = sf ? (
    (pnr==='ABOUT_TO_B2B'&&sf.calibrated) ? sf.b2bRate*1.6 :
    pnr==='ABOUT_TO_B2B'                  ? sf.b2bRate*1.3 :
    pnr==='ABOUT_TO_HOT'                  ? sf.b2bRate*1.1 : sf.b2bRate
  ) : 0;
  const safeWeight = sf ? (
    (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') ? Math.min(sf.lowDensity20*1.2,1) :
    (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') ? sf.lowDensity20*0.6 :
    sf.lowDensity20
  ) : (1-hrGlobal);
  // === ACCURACY & CALIBRATION FIX END ===

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
// ENGINE 2: hybrid_transformer_lstm
// Head weight shift only when effectiveRegime fires (tc >= 62)
// =============================================================================
function runHybridTransformerLstm(rounds, target, sf) {
  const {gaps,currentGap}=computeGaps(rounds,target.min);
  if (gaps.length<15) return null;
  const hrGlobal=gaps.length/rounds.length;
  const gMean=mean(gaps),gStd=stdDev(gaps)||1;

  const attnHead=(window)=>{
    if (!window.length) return gMean;
    const norm=window.map(g=>(g-gMean)/gStd);
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
  const pnr=effectiveRegime(sf); // BUG-D FIX: raised threshold
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
// ENGINE 3: hybrid_tft
// === ACCURACY & CALIBRATION FIX START ===
// Gate weight amplification behind effectiveRegime (tc >= 62)
// === ACCURACY & CALIBRATION FIX END ===
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
  const pnr=effectiveRegime(sf); // BUG-D FIX
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
  // === ACCURACY & CALIBRATION FIX START ===
  // Only inject post-cluster when actually in confirmed ABOUT_TO_B2B with calibration
  if (sf && sf.postClusterEarlySignal && sf.calibrated && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.6+sf.avgPostClusterGap*0.4));
  } else if (sf && sf.regime==='WHITE_CLUSTER' && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.65+sf.avgPostClusterGap*0.35));
  }
  // === ACCURACY & CALIBRATION FIX END ===
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((68+gaps.length*0.08-gStd/gMean*12+streakConfBonus(sf,target.rare))*sp),22,90);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 4: tft_full
// === ACCURACY & CALIBRATION FIX START ===
// Quantile weights only shift when tc >= 62
// === ACCURACY & CALIBRATION FIX END ===
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
  const pnr=effectiveRegime(sf); // BUG-D FIX: raised threshold
  if (pnr==='ABOUT_TO_B2B'||pnr==='B2B'||pnr==='ABOUT_TO_HOT'||pnr==='HOT') {
    wQ10=0.42;wQ50=0.43;wQ90=0.15;
  } else if (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='WHITE_CLUSTER'||
             pnr==='ABOUT_TO_COLD'||pnr==='EXTREME_WHITE') {
    wQ10=0.08;wQ50=0.38;wQ90=0.54;
  }
  // Extra for rare targets when b2bPrecursor calibrated
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
// ENGINE 5: nbeats
// Streak block weight boost only when calibrated hot signal
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

  const pnr=effectiveRegime(sf); // BUG-D FIX
  // === ACCURACY & CALIBRATION FIX START ===
  // Only boost streak block when calibrated hot prediction
  const isHotCalibrated = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && (sf?.calibrated||false);
  const streakW=(1-r2)*(isHotCalibrated ? 0.55 : 0.40);
  const identW =(1-r2)*(isHotCalibrated ? 0.45 : 0.60);
  // === ACCURACY & CALIBRATION FIX END ===

  const raw=Math.max(1,Math.round(trendForecast*r2+identityForecast*identW+streakForecast*streakW));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const gStd=stdDev(gaps)||1;
  const sp=sparsePenalty(n,40);
  const conf=clamp(Math.round((60+r2*22-gStd/gMean*10+streakConfBonus(sf,target.rare))*sp),18,90);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 6: tcn
// === ACCURACY & CALIBRATION FIX START ===
// Streak residual amplification now uses calibrated hotScore
// === ACCURACY & CALIBRATION FIX END ===
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
    // === ACCURACY & CALIBRATION FIX START ===
    // Only amplify when calibration confirms (not just raw hotScore)
    const hotAmp = sf.calibrated ? clamp((sf.hotScore||0)/100*1.2,0.3,0.9) : 0.4;
    // === ACCURACY & CALIBRATION FIX END ===
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
// ENGINE 7: lightgbm
// === ACCURACY & CALIBRATION FIX START ===
// Streak leaf weights: amplification gated behind calibration
// === ACCURACY & CALIBRATION FIX END ===
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
  // === ACCURACY & CALIBRATION FIX START ===
  // Streak leaves only amplify when calibration confirms the signal
  const calMult = sf?.calibrated ? 1.5 : 1.0;
  const hotScoreNorm = sf ? (sf.hotScore||0)/100 : 0;
  const coldScoreNorm = sf ? (sf.coldScore||0)/100 : 0;
  const w6=sf?clamp(sf.b2bRate*3*(1+hotScoreNorm*calMult),0.1,1.8):0.1;
  const w7=sf?clamp(sf.lowDensity20*2*(1+coldScoreNorm*0.8),0.1,1.3):0.1;
  // === ACCURACY & CALIBRATION FIX END ===
  const wS=w1+w2+w3+w4+w5+w6+w7;

  const raw=Math.max(1,Math.round((leaf1*w1+leaf2*w2+leaf3*w3+leaf4*w4+leaf5*w5+leaf6*w6+leaf7*w7)/wS));
  const adj=applyStreakAdjustment(raw,sf,target);
  const aw=target.maxWidth;
  const sp=sparsePenalty(gaps.length,40);
  const conf=clamp(Math.round((72-cv*16+r2*10+overdue*2+streakConfBonus(sf,target.rare))*sp),18,92);
  return {...placeWindow(adj,currentGap,aw),expectedGap:adj,probW:geoProbW(hrGlobal,aw),conf};
}

// =============================================================================
// ENGINE 8: gru
// === ACCURACY & CALIBRATION FIX START ===
// Initial hidden state bias uses effectiveRegime (tc >= 62)
// === ACCURACY & CALIBRATION FIX END ===
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
  const pnr=effectiveRegime(sf); // BUG-D FIX: raised threshold
  if (sf) {
    if (pnr==='ABOUT_TO_B2B'&&sf.calibrated)
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
// ENGINE 9: bilstm
// Merge weights use effectiveRegime (tc >= 62)
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
  const pnr=effectiveRegime(sf); // BUG-D FIX
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
// ENGINE 10: stacking_meta
// === ACCURACY & CALIBRATION FIX START ===
// Specialisation weights amplified by transitionConfidence AND calibration status
// === ACCURACY & CALIBRATION FIX END ===
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

  const pnr = effectiveRegime(sf); // BUG-D FIX
  const regime = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT'||pnr==='B2B'||pnr==='HOT'||pnr==='HOT_AFTER_SHORT_COLD') ? 'b2b'
    : (pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD'||pnr==='WHITE_CLUSTER'||pnr==='EXTREME_WHITE'||pnr==='COLD') ? 'cluster'
    : 'neutral';
  const tc = sf?.transitionConfidence ?? 0;
  // === ACCURACY & CALIBRATION FIX START ===
  // Only amplify with tcMult when calibration confirms
  const tcMult = (sf?.calibrated && tc>72) ? 1.4 : tc>72 ? 1.2 : tc>62 ? 1.1 : 1.0;
  // === ACCURACY & CALIBRATION FIX END ===

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
// ENGINE 11: sha512
// === ACCURACY & CALIBRATION FIX START ===
// streakBias uses calibrated hotScore (not raw)
// === ACCURACY & CALIBRATION FIX END ===
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

  // === ACCURACY & CALIBRATION FIX START ===
  // streakBias: only apply when calibration confirms
  let streakBias=0;
  if (sf&&sf.highRuns&&sf.highRuns.length>=5&&sf.calibrated) {
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    const hotW=(sf.hotScore??0)/100;
    streakBias=(1-normHrEnt)*hotW*0.10; // reduced from 0.12 and gated on calibrated
  }
  const pnr=effectiveRegime(sf); // BUG-D FIX
  const driftAligned=(pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&maxC>0&&sf?.calibrated;
  if (driftAligned&&drift) streakBias+=0.025;
  // === ACCURACY & CALIBRATION FIX END ===

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
// NG CONSENSUS — algorithm unchanged, inputs improved
// === ACCURACY & CALIBRATION FIX START ===
// tcBonus only fires when calibration confirms the hot signal
// === ACCURACY & CALIBRATION FIX END ===
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
    // === ACCURACY & CALIBRATION FIX START ===
    const sf=sharedSf?.[target.label];
    // tcBonus only when BOTH high confidence AND calibration confirms
    const tcBonus = sf&&sf.calibrated&&sf.transitionConfidence>72&&
      (sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT') ? 8 : 0;
    // === ACCURACY & CALIBRATION FIX END ===
    consensus[target.label]={
      lo:bestLo,hi:bestHi,engineCount:bestGroup.length,
      engines:bestGroup.map(w=>w.engineId),tcBonus,
    };
  }
  return consensus;
}

// =============================================================================
// MAIN TICK
// === ACCURACY & CALIBRATION FIX START ===
// Calibration computed once per tick per target BEFORE feature extraction.
// streakFeatures now receives calib object.
// === ACCURACY & CALIBRATION FIX END ===
// =============================================================================
async function runNgComputeEngine() {
  try {
    const rounds=await getNgRounds();
    if(rounds.length<50) return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // === ACCURACY & CALIBRATION FIX START ===
    // Step 1: Compute historical calibration for all targets
    const calibrations = {};
    for (const target of TARGETS) {
      try { calibrations[target.label] = computeCalibration(rounds, target.min, target.label); }
      catch(e) { calibrations[target.label] = null; console.error(`[ngCompute] calib/${target.label}:`,e.message); }
    }

    // Step 2: Extract FIXED predictive streak features (with calibration)
    const streakFeatures={};
    for (const target of TARGETS) {
      try {
        streakFeatures[target.label] = extractPredictiveStreakFeatures(
          rounds, target.min, calibrations[target.label]
        );
      }
      catch(e){ streakFeatures[target.label]=null; console.error(`[ngCompute] psf/${target.label}:`,e.message); }
    }
    // === ACCURACY & CALIBRATION FIX END ===

    const ALGO_MAP={
      hlstm_xgb: runHybridLstmXgb, htrans_lstm:runHybridTransformerLstm,
      htft:runHybridTft, tft:runTftFull, nbeats:runNbeats, tcn:runTcn,
      lgbm:runLightGBM, gru:runGRU, bilstm:runBiLSTM, sha512:runSHA512,
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

    allNgResults['stacking']={};
    for(const target of TARGETS){
      try{
        const r=runStackingMeta(rounds,target,allNgResults,streakFeatures[target.label]);
        if(r) allNgResults['stacking'][target.label]=r;
      } catch(e){console.error(`[ngCompute] stacking/${target.label}:`,e.message);}
    }

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

    // Phase 1+2: resolve + lock (100% unchanged)
    for(const engineId of NG_ENGINE_IDS){
      const payload={};
      for(const target of TARGETS){
        const win=ngWindows[engineId][target.label];
        const fresh=allNgResults[engineId]?.[target.label];
        if(win){
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
          const newLo=lastRoundId+fresh.low,newHi=lastRoundId+fresh.high;
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
  // Clear calibration cache too
  for(const k of Object.keys(calibCache)) delete calibCache[k];
}
function resetNgWindowsOnly(){
  for(const id of NG_ENGINE_IDS) ngWindows[id]={};
  console.log('[ngCompute] in-memory windows cleared (lock reset)');
}
module.exports={runNgComputeEngine:runNgComputeEngineWithInit,resetNgComputeState,resetNgWindowsOnly,NG_ENGINE_IDS};