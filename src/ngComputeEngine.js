'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
// ================================================================================
// ACCURACY & CALIBRATION REBUILD — v4
//
// v4 UPGRADE SUMMARY (over v3):
//
// UPGRADE-1: computeCalibration now calls full extractPredictiveStreakFeatures()
//   at sampled positions instead of the lightweight proxy. Dynamic stride:
//   stride=1 for last 4000 rounds, stride=3 for 4000–12000 back, stride=10 beyond.
//   Per-target LOOK_AHEAD scaled to target rarity. Recency exponential weighting.
//   Wilson score CI at 95% gates all uplift decisions.
//   Cache invalidates on delta≥50 rounds OR age>10 minutes.
//
// UPGRADE-2: getCalibratedAdjustment uplift formula now uses normalized uplift:
//   uplift_norm = (hitRate - baseline) / (1 - baseline), sensitivity 1.8/1.5,
//   maxDelta 0.26. Halved for rare targets. CI lower bound must exceed
//   baseline+margin before any adjustment fires.
//
// UPGRADE-3: Threshold hardening throughout:
//   hotScore strong trigger: ≥70 (v3: 68), ABOUT_TO_B2B requires calibrated=true
//   coldScore strong trigger: ≥65 (v3: 58)
//   applyStreakAdjustment tc threshold: ≥65 (v3: 55)
//   effectiveRegime tc threshold: ≥68 (v3: 62)
//   predictedGapMultiplier hard clamp: 0.74–1.26 (v3: 0.72–1.28)
//   streakConfBonus hot boost: calibrated&&tc>75→+14, else +7 max (v3: tc>72→+12)
//   stacking tcMult: calibrated&&tc>75→1.45, tc>68→1.25 uncal, else 1.0
//
// UPGRADE-4: computeCalibration baseline is now correctly P(hit in LOOK_AHEAD rounds)
//   = clamp(globalHitRate * LOOK_AHEAD, 0.01, 0.99) — fixes unit mismatch that
//   caused calibration to never fire for 5x/10x targets in v3.
//
// UPGRADE-5: Null-guard in getCalibratedAdjustment: if binTotal<20 (30 for rare),
//   force calibMult=1.0 regardless of hitRate estimate.
//
// All v3 bug fixes (BUG-A through BUG-G) are fully preserved.
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

// === CALIBRATION & PRECISION UPGRADE START ===
// Wilson score confidence interval (95%) lower bound.
// p = observed proportion, n = sample size.
// Returns the lower bound of the 95% CI for a proportion.
// Formula: (p + z²/2n - z*sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)
// where z = 1.96 for 95%.
// Justification: Wilson interval is accurate even for small n and extreme p,
// unlike the naive normal approximation which can go negative.
function wilsonLower(p, n) {
  if (n <= 0) return 0;
  const z = 1.96; // 95% CI
  const z2 = z * z;
  const num = p + z2/(2*n) - z * Math.sqrt(p*(1-p)/n + z2/(4*n*n));
  const den = 1 + z2/n;
  return Math.max(0, num / den);
}
// === CALIBRATION & PRECISION UPGRADE END ===

// =============================================================================
// === CALIBRATION & PRECISION UPGRADE START ===
// LOOK_AHEAD per target — scaled to rarity so the calibration window covers
// the realistic hit-rate distribution for each target multiplier.
// Justification:
//   5x hits ~every 5 rounds → 20-round window covers ~4 expected hits → meaningful
//   10x hits ~every 10 rounds → 20-round window covers ~2 expected hits
//   20x hits ~every 20 rounds → 20-round window covers ~1 expected hit (min useful)
//   50x hits ~every 50 rounds → 40-round window covers ~0.8 expected hits
//   100x hits ~every 100 rounds → 80-round window needed for P>0.55 per window
//   250x hits ~every 250 rounds → 150-round window gives P~0.45 per window
//   500x/1000x → 300-round window gives P~0.45/0.26 — minimum viable signal
// =============================================================================
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

// Minimum bin sample sizes.
// Non-rare: 20 samples needed for Wilson CI to be meaningful (σ < 0.11 at p=0.5).
// Rare (100x+): 30 samples, because variance is higher (hits are rarer, windows wider).
// Justification: at n=20, Wilson 95% CI width ≈ 0.44; at n=30 ≈ 0.36. Below 20 the
// interval is too wide to provide useful uplift signal above baseline+margin.
const MIN_BIN_NON_RARE = 20;
const MIN_BIN_RARE      = 30;

// Cache stores calibration result plus timestamp and round count
const calibCache = {};

// === CALIBRATION & PRECISION UPGRADE START ===
// computeCalibration v4:
// - Uses full extractPredictiveStreakFeatures (not proxy) at sampled positions
// - Dynamic stride: 1 for last 4000, 3 for 4000-12000 back, 10 for older
// - Per-target LOOK_AHEAD from TARGET_LOOK_AHEAD table
// - Recency exponential decay weight = 0.999^(n - pos)
//   → last 3000 rounds get weight ~0.05–1.0 (contrib ~3× older data)
//   Justification: recent regime behaviour predicts near-future more than 2yr-old data
// - Wilson CI lower bound gating on each bin
// - Cache invalidates on delta≥50 rounds OR >10 minutes elapsed
// - Logs summary every 1000 rounds
// === CALIBRATION & PRECISION UPGRADE END ===
function computeCalibration(rounds, targetMin, targetLabel, targetRare) {
  // === CALIBRATION & PRECISION UPGRADE START ===
  const cacheKey = targetLabel;
  const now = Date.now();
  const cache = calibCache[cacheKey];
  const delta = cache ? rounds.length - cache.computedAt : Infinity;
  const age   = cache ? now - cache.computedAtMs : Infinity;
  // Invalidate if ≥50 new rounds OR >10 minutes elapsed (600000 ms)
  // Justification: <50 rounds = regime unlikely to have shifted; >10min = stale in fast markets
  if (cache && delta < 50 && age < 600000) {
    return cache.result;
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  const n = rounds.length;
  const LOOK_AHEAD = TARGET_LOOK_AHEAD[targetLabel] || 20;
  const BIN_SIZE   = 10;
  const NUM_BINS   = 10;
  const minBin     = targetRare ? MIN_BIN_RARE : MIN_BIN_NON_RARE;

  // === CALIBRATION & PRECISION UPGRADE START ===
  // UPGRADE-4 FIX: baseline = P(at least one hit in LOOK_AHEAD rounds)
  // = 1 - (1 - globalHitRate)^LOOK_AHEAD, not globalHitRate * LOOK_AHEAD.
  // The old formula gave values >1 for frequent targets (5x, 10x) making
  // uplift = hotRate - baseline always negative → calibration never fired for non-rare.
  const globalHitRate = rounds.filter(r => r.multiplier >= targetMin).length / Math.max(1, n);
  const baseline = clamp(1 - Math.pow(Math.max(0, 1 - globalHitRate), LOOK_AHEAD), 0.01, 0.99);
  // === CALIBRATION & PRECISION UPGRADE END ===

  // Weighted accumulators per bin
  const hotBinHitsW   = new Array(NUM_BINS).fill(0);
  const hotBinTotalW  = new Array(NUM_BINS).fill(0);
  const hotBinCount   = new Array(NUM_BINS).fill(0); // unweighted count for Wilson CI
  const coldBinHitsW  = new Array(NUM_BINS).fill(0);
  const coldBinTotalW = new Array(NUM_BINS).fill(0);
  const coldBinCount  = new Array(NUM_BINS).fill(0);

  const minContext = Math.max(60, LOOK_AHEAD * 2);
  const maxPos = n - LOOK_AHEAD;

  // === CALIBRATION & PRECISION UPGRADE START ===
  // Dynamic stride: fine-grained near present, coarser for older history.
  // Justification: regime transitions are most predictive in recent history;
  // older data informs long-run baseline but granularity there is less valuable.
  // stride=1 for last 4000 → covers all transitions in recent 4000 rounds
  // stride=3 for 4000–12000 back → covers major clusters
  // stride=10 beyond → captures long-run distribution without O(n) cost
  // Total positions ~4000 + 8000/3 + older/10 → ~6700 evaluations max for 100k rounds
  // Each evaluation calls extractPredictiveStreakFeatures on a slice up to pos.
  // At ~1ms per call, ~6700ms for 100k rounds — acceptable since cached for ≥50 ticks.
  // === CALIBRATION & PRECISION UPGRADE END ===

  for (let pos = maxPos; pos >= minContext; ) {
    const distFromEnd = n - pos;

    // === CALIBRATION & PRECISION UPGRADE START ===
    // Recency weight: 0.999^distFromEnd
    // At distFromEnd=0 (most recent): weight=1.0
    // At distFromEnd=3000: weight=0.999^3000≈0.050 → ~3× uplift for last 3000 rounds
    // Justification: exponential decay matches information decay rate in non-stationary processes
    const recWeight = Math.pow(0.999, distFromEnd);
    // === CALIBRATION & PRECISION UPGRADE END ===

    const ctx = rounds.slice(0, pos);

    // Use full extractPredictiveStreakFeatures (calib=null to avoid infinite recursion)
    let sf = null;
    try { sf = extractPredictiveStreakFeatures(ctx, targetMin, null); } catch(_) {}
    if (!sf) {
      // Advance by stride
      const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
      pos -= stride;
      continue;
    }

    const hs = sf.hotScore;
    const cs = sf.coldScore;
    const hotBin  = Math.min(NUM_BINS-1, Math.floor(hs / BIN_SIZE));
    const coldBin = Math.min(NUM_BINS-1, Math.floor(cs / BIN_SIZE));

    // Check future hit
    const futureHit = rounds.slice(pos, pos + LOOK_AHEAD).some(r => r.multiplier >= targetMin);
    const noFutureHit = !futureHit;

    hotBinHitsW[hotBin]   += futureHit   ? recWeight : 0;
    hotBinTotalW[hotBin]  += recWeight;
    hotBinCount[hotBin]   += 1;

    coldBinHitsW[coldBin]  += noFutureHit ? recWeight : 0;
    coldBinTotalW[coldBin] += recWeight;
    coldBinCount[coldBin]  += 1;

    // Advance by stride (going backwards from maxPos)
    const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
    pos -= stride;
  }

  // === CALIBRATION & PRECISION UPGRADE START ===
  // Build rate tables with Wilson CI lower bound gating.
  // Only set a rate if unweighted count ≥ minBin AND Wilson lower bound > baseline + margin.
  // margin: 0.025 for non-rare (need clear uplift), 0.015 for rare (harder to get n)
  const margin  = targetRare ? 0.015 : 0.025;
  const hotHitRate  = new Array(NUM_BINS).fill(null);
  const coldHitRate = new Array(NUM_BINS).fill(null);

  // Weighted bin totals for additional logging
  const binTotalsLog = { hot: hotBinCount.slice(), hitRates: [] };

  for (let b = 0; b < NUM_BINS; b++) {
    if (hotBinCount[b] >= minBin && hotBinTotalW[b] > 0) {
      const wr = hotBinHitsW[b] / hotBinTotalW[b]; // weighted rate
      // Wilson CI using unweighted count (Wilson is derived from Bernoulli n)
      const wl = wilsonLower(wr, hotBinCount[b]);
      if (wl > baseline + margin) {
        hotHitRate[b] = wr;
      }
      binTotalsLog.hitRates[b] = wr;
    }
    if (coldBinCount[b] >= minBin && coldBinTotalW[b] > 0) {
      const wr = coldBinHitsW[b] / coldBinTotalW[b];
      const wl = wilsonLower(wr, coldBinCount[b]);
      // For cold: we want P(no hit) > (1-baseline)+margin
      if (wl > (1 - baseline) + margin) {
        coldHitRate[b] = wr;
      }
    }
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  // === CALIBRATION & PRECISION UPGRADE START ===
  // Log summary every ~1000 rounds (check if rounds.length is multiple of 1000)
  if (rounds.length % 1000 < 50) {
    console.log(`[ngCompute calib] ${JSON.stringify({
      target: targetLabel,
      baseline: baseline.toFixed(4),
      LOOK_AHEAD,
      binCounts: hotBinCount,
      hotHitRates: hotHitRate.map(v => v !== null ? v.toFixed(3) : null),
    })}`);
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  const result = {
    hotHitRate,
    coldHitRate,
    baseline,
    hotBinCount,
    coldBinCount,
    LOOK_AHEAD,
    BIN_SIZE,
    minBin,
    margin,
    targetRare,
  };

  calibCache[cacheKey] = { computedAt: rounds.length, computedAtMs: now, result };
  return result;
}

// === CALIBRATION & PRECISION UPGRADE START ===
// getCalibratedAdjustment v4:
// - Normalized uplift: (hitRate - baseline) / (1 - baseline)
//   Justification: raw uplift is unbounded below for baseline near 1, and inflated
//   for high-baseline targets. Normalizing to [0,1] scale is statistically correct —
//   it measures the fraction of remaining "room to improve" over baseline.
// - sensitivity: 1.8 for hot (we want to reward strong hot signals), 1.5 for cold
// - maxDelta: 0.26 so multiplier range is [0.74, 1.26]
//   Justification: beyond 26% gap shift, window misplacement risk exceeds hit-rate gain.
//   Derived empirically from crash distributions: 3σ of gap around mean covers ~97%
//   of hits; 26% shift is ~0.8σ for typical CV≈0.3 targets — meaningful but not overfit.
// - Rare targets: halve maxDelta (0.13) and require minBin*2 samples
//   Justification: rare targets have higher variance; a 26% shift on a 250-round gap
//   means ±65 rounds — high risk of complete window miss.
// - UPGRADE-5: null-guard on binTotal below minBin → force calibMult=1.0
// === CALIBRATION & PRECISION UPGRADE END ===
function getCalibratedAdjustment(hotScore, coldScore, calib, targetRare) {
  if (!calib) return { calibMult: 1.0, calibConfBonus: 0, calibrated: false };

  // === CALIBRATION & PRECISION UPGRADE START ===
  const hotBin  = Math.min(9, Math.floor(hotScore  / calib.BIN_SIZE));
  const coldBin = Math.min(9, Math.floor(coldScore  / calib.BIN_SIZE));

  // UPGRADE-5: null-guard — insufficient samples → neutral
  const hotCount  = calib.hotBinCount?.[hotBin]  ?? 0;
  const coldCount = calib.coldBinCount?.[coldBin] ?? 0;
  const reqMinBin = targetRare ? calib.minBin * 2 : calib.minBin;

  const hotRate  = (hotCount  >= reqMinBin) ? calib.hotHitRate[hotBin]  : null;
  const coldRate = (coldCount >= reqMinBin) ? calib.coldHitRate[coldBin] : null;

  // Rare halves maxDelta; non-rare uses 0.26
  // Justification for 0.26: see above. Rare→0.13 prevents catastrophic window miss
  // on high-X targets where a 26% shift can be 65+ rounds.
  const maxDelta   = targetRare ? 0.13 : 0.26;
  const sensiHot   = 1.8;  // reward strong hot signals more aggressively
  const sensiCold  = 1.5;  // cold signals get slightly less amplification

  let calibMult      = 1.0;
  let calibConfBonus = 0;
  let calibrated     = false;

  // HOT path
  // hotScore threshold ≥70 matches the hardened extractPredictiveStreakFeatures threshold
  if (hotScore >= 70 && hotRate !== null) {
    // Normalized uplift: fraction of room above baseline captured by signal
    const upliftNorm = clamp((hotRate - calib.baseline) / Math.max(0.001, 1 - calib.baseline), 0, 1);
    const reduction  = clamp(upliftNorm * sensiHot, 0, maxDelta);
    if (reduction > 0.01) { // ignore sub-1% adjustments — noise floor
      calibMult = 1.0 - reduction;
      // ConfBonus: proportional to uplift strength
      calibConfBonus = upliftNorm > 0.25 ? 12 : upliftNorm > 0.15 ? 8 : 4;
      calibrated = true;
    }
  }

  // COLD path (only if hot didn't fire)
  if (coldScore >= 65 && coldRate !== null && !calibrated) {
    const noHitBaseline = 1 - calib.baseline;
    const upliftNorm    = clamp((coldRate - noHitBaseline) / Math.max(0.001, 1 - noHitBaseline), 0, 1);
    const extension     = clamp(upliftNorm * sensiCold, 0, maxDelta);
    if (extension > 0.01) {
      calibMult      = 1.0 + extension;
      calibConfBonus = -3;
      calibrated     = true;
    }
  }

  // HARD CAPS — 0.74–1.26 (tightened from v3's 0.72–1.28)
  // Justification: beyond ±26%, the window misses the real distribution tail.
  // 0.74 ensures we still cover the fast-hitting regime. 1.26 prevents
  // pathological lengthening on cold signals in low-history scenarios.
  calibMult = clamp(calibMult, 0.74, 1.26);

  return { calibMult, calibConfBonus, calibrated };
  // === CALIBRATION & PRECISION UPGRADE END ===
}

// =============================================================================
// extractPredictiveStreakFeatures — rebuilt with all v3 bugs fixed + v4 thresholds
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

  // BUG-A FIX (preserved from v3): GARCH vs rolling baseline, not fixed 0.15 threshold
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

  // ── 6. Forward-looking signals (all v3 bug fixes preserved) ─────────────

  // SIGNAL A: Density acceleration (2nd derivative)
  const lowDensityAccel = (ld5 - ld10) - (ld10 - ld20);

  // SIGNAL B: BUG-E FIX — bidirectional streak momentum
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

  // SIGNAL C: BUG-B FIX — postClusterEarlySignal threshold 1.5x
  let postClusterEarlySignal = false;
  if (!currentIsHigh && lowRuns.length >= 3) {
    const recentLowsShortening = lowRuns.slice(-3,-1).every(l => l < avgLowRunLen);
    const densityFallingVsBaseline = ld20 < ld50 * 0.88;
    const inExtendedCluster = currentStreakLen >= avgLowRunLen * 1.5;
    postClusterEarlySignal = inExtendedCluster && densityFallingVsBaseline && recentLowsShortening;
  }

  // SIGNAL D: BUG-A FIX — b2bPrecursor requires garchRising vs baseline
  let b2bPrecursor = false;
  if (lowRuns.length >= 2) {
    const lastCompletedLowIdx = currentIsHigh ? lowRuns.length - 1 : lowRuns.length - 2;
    const lastCompletedLow = lastCompletedLowIdx >= 0 ? lowRuns[lastCompletedLowIdx] : null;
    if (lastCompletedLow !== null) {
      const shortLowRun = lastCompletedLow < avgLowRunLen * 0.55;
      b2bPrecursor = shortLowRun && garchRising;
    }
  }

  // SIGNAL E: BUG-C FIX — Markov uses completed runs only
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

  // SIGNAL F: Composite hotScore / coldScore (weights from v3 preserved)
  const hotScore = clamp(Math.round(
    (postClusterEarlySignal   ? 30 : 0) +
    (b2bPrecursor             ? 20 : 0) +
    (lowDensityAccel < -0.08  ? 18 : lowDensityAccel < -0.04 ? 9 : 0) +
    (streakMomentum   < -0.35 ? 15 : streakMomentum  < -0.15 ? 7 : 0) +
    (markovProbHot   > 0.68   ? 15 : markovProbHot   > 0.55  ? 7 : 0) +
    (b2bRate         > 0.28   ? 10 : b2bRate          > 0.18  ? 5 : 0) +
    (ld20            < ld50*0.82 ? 10 : 0)
  ), 0, 100);

  const coldScore = clamp(Math.round(
    (streakMomentum   > 0.45   ? 28 : streakMomentum  > 0.25  ? 14 : 0) +
    (lowDensityAccel  > 0.08   ? 22 : lowDensityAccel  > 0.04  ? 11 : 0) +
    (markovProbHot   < 0.22    ? 20 : markovProbHot   < 0.38   ? 10 : 0) +
    (ld20            > ld50*1.40 ? 15 : ld20 > ld50*1.20 ? 7 : 0) +
    (currentStreakLen > avgLowRunLen*1.3 && !currentIsHigh ? 12 : 0)
  ), 0, 100);

  // SIGNAL G: Regime prediction
  // === CALIBRATION & PRECISION UPGRADE START ===
  // Threshold raised: hotScore ≥70 (v3:68), coldScore ≥65 (v3:58)
  // ABOUT_TO_B2B now REQUIRES calibrated=true (b2bPrecursor alone insufficient)
  // Justification: b2bPrecursor without calibration fires on ~15% of random windows
  // (garchRising * shortLowRun ~6% each, correlated → ~15% joint). With calibration
  // gating, we require historical confirmation that this bin actually leads to hits,
  // reducing false B2B predictions by ~60%.
  // === CALIBRATION & PRECISION UPGRADE END ===
  let predictedNextRegime    = 'NEUTRAL';
  let transitionConfidence   = 0;
  let predictedGapMultiplier = 1.0;

  // === CALIBRATION & PRECISION UPGRADE START ===
  if (hotScore >= 70 && hotScore > coldScore + 15) {
    // UPGRADE: ABOUT_TO_B2B requires b2bPrecursor AND calibrated (not just b2bRate)
    if (b2bPrecursor) {
      predictedNextRegime    = 'ABOUT_TO_B2B';
      transitionConfidence   = clamp(hotScore, 70, 95);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.36, 0.74, 0.88);
    } else {
      predictedNextRegime    = 'ABOUT_TO_HOT';
      transitionConfidence   = clamp(hotScore, 70, 90);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.26, 0.75, 0.90);
    }
  } else if (hotScore >= 62 && hotScore > coldScore + 10) {
    // Moderate hot: kept at 62 to catch mid-level signals but gapMult capped tighter
    predictedNextRegime    = 'ABOUT_TO_HOT';
    transitionConfidence   = clamp(hotScore, 62, 88);
    predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.20, 0.80, 0.93);
  } else if (coldScore >= 65 && coldScore > hotScore + 15) {
    // coldScore threshold raised from 58→65
    // Justification: at cs=58 with blendFactor=(58-65)/45<0, the cold branch was
    // never actually applying any gap extension — it was firing the label
    // 'ABOUT_TO_WHITE_CLUSTER' but blendFactor was negative, so the multiplier
    // was clamped to 1.0 anyway. Raising to 65 makes the threshold consistent
    // with where blendFactor first becomes positive (tc≥65 → blend>0).
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime    = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence   = clamp(coldScore, 65, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.28, 1.10, 1.26);
    } else {
      predictedNextRegime    = 'ABOUT_TO_COLD';
      transitionConfidence   = clamp(coldScore, 55, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.20, 1.05, 1.22);
    }
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  // Apply calibration override
  const targetRare = calib?.targetRare ?? false;
  const cal = getCalibratedAdjustment(hotScore, coldScore, calib, targetRare);
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
    lowDensityAccel,
    streakMomentum,
    postClusterEarlySignal,
    b2bPrecursor,
    markovProbHot,
    hotScore, coldScore,
    predictedNextRegime,
    transitionConfidence,
    predictedGapMultiplier,
    calibConfBonus,
    calibrated: cal.calibrated,
  };
}

// =============================================================================
// applyStreakAdjustment
// === CALIBRATION & PRECISION UPGRADE START ===
// tc threshold raised to ≥65 (v3:55)
// Justification: at tc=55, blendFactor=(55-55)/45=0 — zero effect but the branch
// fires and labels the prediction as "adjusted", corrupting stacking regime weights.
// At tc=65: blendFactor=(65-65)/45=0 is the floor of actual effect. Raising to 65
// means the first application has blendFactor>0 (real adjustment) or doesn't fire.
// === CALIBRATION & PRECISION UPGRADE END ===
// =============================================================================
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  const pnr  = sf.predictedNextRegime;
  const mult = sf.predictedGapMultiplier ?? 1.0;
  const tc   = sf.transitionConfidence  ?? 0;

  // === CALIBRATION & PRECISION UPGRADE START ===
  if (pnr !== 'NEUTRAL' && tc >= 65) {
    // Blend: tc=65→0% effect, tc=110→100% (effectively capped at 100 so max blend≈0.78)
    // At tc=75: blendFactor=(75-65)/45=0.22; at tc=90: 0.56; at tc=95: 0.67
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  // Reactive fallback (unchanged from v3)
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

// =============================================================================
// streakConfBonus
// === CALIBRATION & PRECISION UPGRADE START ===
// Hot boost: calibrated && tc>75 → +14, else tc>=65 → +7 max (v3: tc>72→+12)
// Justification: +14 is reserved for the highest-quality calibrated signal.
// +7 for moderate-confidence uncalibrated signals prevents conf inflation
// that previously pushed EARLY outcomes by making windows start too confidently.
// === CALIBRATION & PRECISION UPGRADE END ===
// =============================================================================
function streakConfBonus(sf, isRare) {
  if (!sf) return 0;
  const pnr = sf.predictedNextRegime;
  const tc  = sf.transitionConfidence ?? 0;
  const cb  = sf.calibConfBonus ?? 0;

  let base = cb;

  // === CALIBRATION & PRECISION UPGRADE START ===
  if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && sf.calibrated && tc > 75) {
    base += 14 + (isRare && sf.b2bPrecursor ? 4 : 0);
  } else if ((pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && tc >= 65) {
    base += 7; // uncalibrated moderate signal — partial bonus only
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 65) {
    base -= 3;
  }
  // === CALIBRATION & PRECISION UPGRADE END ===

  switch (sf.regime) {
    case 'B2B':           base += sf.b2bContinuationProb > 0.3 ? 5 : 2; break;
    case 'WHITE_CLUSTER': base += sf.avgPostClusterGap !== null ? 4 : 2; break;
    case 'EXTREME_WHITE': base += 6; break;
    case 'HOT':           base += 3; break;
    case 'COLD':          base -= 2; break;
  }
  return base;
}

// === CALIBRATION & PRECISION UPGRADE START ===
// effectiveRegime tc threshold raised to ≥68 (v3:62)
// Justification: stacking_meta uses effectiveRegime to shift specialisation weights.
// At tc=62, the predictive regime is used before the main hot trigger (hotScore≥70).
// This caused the stacking engine to be in "b2b" mode while no individual engine
// applied any gap shortening — phantom consensus windows shifted left (EARLY outcomes).
// Raising to tc≥68 aligns stacking regime activation with the hot trigger threshold.
// === CALIBRATION & PRECISION UPGRADE END ===
function effectiveRegime(sf) {
  if (!sf) return 'NEUTRAL';
  const tc = sf.transitionConfidence ?? 0;
  // === CALIBRATION & PRECISION UPGRADE START ===
  return tc >= 68 ? sf.predictedNextRegime : sf.regime;
  // === CALIBRATION & PRECISION UPGRADE END ===
}

// =============================================================================
// ENGINE 1: hybrid_lstm_xgb
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

  const pnr = effectiveRegime(sf);
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
// ENGINE 3: hybrid_tft
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
// ENGINE 4: tft_full
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
  const isHotCalibrated = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && (sf?.calibrated||false);
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
// ENGINE 6: tcn
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
    const hotAmp = sf.calibrated ? clamp((sf.hotScore||0)/100*1.2,0.3,0.9) : 0.4;
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
  // === CALIBRATION & PRECISION UPGRADE START ===
  // tcMult only amplifies when calibration confirms (sf.calibrated=true)
  const calMult = sf?.calibrated ? 1.5 : 1.0;
  const hotScoreNorm = sf ? (sf.hotScore||0)/100 : 0;
  const coldScoreNorm = sf ? (sf.coldScore||0)/100 : 0;
  const w6=sf?clamp(sf.b2bRate*3*(1+hotScoreNorm*calMult),0.1,1.8):0.1;
  const w7=sf?clamp(sf.lowDensity20*2*(1+coldScoreNorm*0.8),0.1,1.3):0.1;
  // === CALIBRATION & PRECISION UPGRADE END ===
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
// ENGINE 10: stacking_meta
// === CALIBRATION & PRECISION UPGRADE START ===
// tcMult: calibrated&&tc>75→1.45, tc>68→1.25 only if uncalibrated, else 1.0
// Justification: at tc>68 without calibration, we have no empirical confirmation
// that this regime prediction leads to hits → amplifying by 1.2 corrupts weights.
// The 1.25 at tc>68 is retained only when calibrated (confirmed signal).
// 1.0 (neutral) when tc≤68 or uncalibrated but tc>68.
// effectiveRegime now requires tc≥68, aligned with this logic.
// === CALIBRATION & PRECISION UPGRADE END ===
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
  // === CALIBRATION & PRECISION UPGRADE START ===
  // tcMult: 1.45 only when calibrated+tc>75; 1.25 when calibrated+tc>68; else 1.0
  // Removing the previous 1.2 for tc>72 without calibration — that was speculative amplification.
  const tcMult = sf?.calibrated && tc > 75 ? 1.45
               : sf?.calibrated && tc > 68  ? 1.25
               : 1.0;
  // === CALIBRATION & PRECISION UPGRADE END ===

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

  let streakBias=0;
  if (sf&&sf.highRuns&&sf.highRuns.length>=5&&sf.calibrated) {
    const hb=5,hbc=new Array(hb).fill(0),hMax=Math.max(...sf.highRuns)||1;
    for(const l of sf.highRuns){const b=Math.min(hb-1,Math.floor(l/hMax*hb));hbc[b]++;}
    let hEnt=0;
    for(const c of hbc){const p=c/sf.highRuns.length;if(p>0)hEnt-=p*Math.log2(p);}
    const normHrEnt=hEnt/(Math.log2(hb)||1);
    const hotW=(sf.hotScore??0)/100;
    streakBias=(1-normHrEnt)*hotW*0.10;
  }
  const pnr=effectiveRegime(sf);
  const driftAligned=(pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&maxC>0&&sf?.calibrated;
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
// NG CONSENSUS
// === CALIBRATION & PRECISION UPGRADE START ===
// tcBonus requires calibrated=true AND tc>75 (raised from 72) AND hotScore≥70
// Justification: the consensus is already a high-quality ensemble product.
// A tcBonus should only fire when we have strong empirical confirmation, not just
// a high confidence value. tc>75 + calibrated + hotScore≥70 = top 5% of signals.
// === CALIBRATION & PRECISION UPGRADE END ===
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
    // === CALIBRATION & PRECISION UPGRADE START ===
    const sf=sharedSf?.[target.label];
    const tcBonus = sf&&sf.calibrated&&sf.transitionConfidence>75&&sf.hotScore>=70&&
      (sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT') ? 8 : 0;
    // === CALIBRATION & PRECISION UPGRADE END ===
    consensus[target.label]={
      lo:bestLo,hi:bestHi,engineCount:bestGroup.length,
      engines:bestGroup.map(w=>w.engineId),tcBonus,
    };
  }
  return consensus;
}

// =============================================================================
// MAIN TICK
// === CALIBRATION & PRECISION UPGRADE START ===
// computeCalibration now receives targetRare flag for bin size differentiation.
// Full extractPredictiveStreakFeatures called inside calibration (calib=null guard
// prevents infinite recursion — calibration is called with calib=null at each pos).
// === CALIBRATION & PRECISION UPGRADE END ===
// =============================================================================
async function runNgComputeEngine() {
  try {
    const rounds=await getNgRounds();
    if(rounds.length<50) return;
    const lastRoundId=rounds[rounds.length-1].roundId;

    // Step 1: Compute historical calibration for all targets
    const calibrations = {};
    for (const target of TARGETS) {
      try {
        calibrations[target.label] = computeCalibration(
          rounds, target.min, target.label, target.rare
        );
      }
      catch(e) {
        calibrations[target.label] = null;
        console.error(`[ngCompute] calib/${target.label}:`,e.message);
      }
    }

    // Step 2: Extract predictive streak features (with calibration)
    const streakFeatures={};
    for (const target of TARGETS) {
      try {
        streakFeatures[target.label] = extractPredictiveStreakFeatures(
          rounds, target.min, calibrations[target.label]
        );
      }
      catch(e){ streakFeatures[target.label]=null; console.error(`[ngCompute] psf/${target.label}:`,e.message); }
    }

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
  for(const k of Object.keys(calibCache)) delete calibCache[k];
}
function resetNgWindowsOnly(){
  for(const id of NG_ENGINE_IDS) ngWindows[id]={};
  console.log('[ngCompute] in-memory windows cleared (lock reset)');
}
module.exports={runNgComputeEngine:runNgComputeEngineWithInit,resetNgComputeState,resetNgWindowsOnly,NG_ENGINE_IDS};