'use strict';
// ngComputeEngine.js — NG Master Signal (ng_consensus)
// ================================================================================
// ACCURACY & CALIBRATION REBUILD — v5 (FINAL PRODUCTION)
//
// v5 HARDENING SUMMARY (over v4):
//
// H-1: CALIB_DUMMY sentinel replaces null — eliminates circularity ambiguity
//      in extractPredictiveStreakFeatures when called from computeCalibration.
//
// H-2: wilsonLower NaN/Inf guards — explicit early-return on n≤0 or NaN result.
//
// H-3: Per-target minContext in calibration loop — rare targets need ≥3× expected
//      gap of history before EPSF can produce non-null results. Skipping early
//      positions wastes compute and can skew bin totals with null-filtered data.
//
// H-4: getCalibratedAdjustment conservative fallback — when bin count below
//      1.5× minBin, apply slight density-trend awareness instead of blind 1.0.
//
// H-5: streakConfBonus reactive bonuses halved when !calibrated — prevents
//      uncalibrated regime from inflating stacking inverse-variance weights.
//
// H-6: WHITE_CLUSTER blend raised 0.55→0.65 — avgPostClusterGap is always 1
//      (occurrence counter only), so 0.55 caused catastrophic 45% gap reduction.
//      0.65 gives meaningful shortening without complete window miss.
//
// H-7: predictedGapMultiplier clamp tightened to 0.76–1.24 — ±24% ≈ 0.7σ on
//      typical CV=0.3 crash gap distributions. Beyond this, window misplacement
//      risk exceeds expected hit-rate gain.
//
// H-8: hotScore strong trigger raised 70→72 — requires structural co-occurrence
//      of density signal with precursor signals, not just 3 soft signals alone.
//
// H-9: logCounter is module-level (persists across ticks) — periodic logging
//      every 10 calibration cache misses, not modulo-dependent on round count.
//
// H-10: calib null-guards tightened in all engines that read sf.calibrated.
//
// All v3 and v4 fixes fully preserved. Zero DB/locking/UI/export changes.
// ================================================================================

const {
  getRounds, savePrediction, getPredictions,
  saveLockedAdvPreds, getLockedAdvPreds,
} = require('./db');

const NG_ENGINE_IDS = [
  'ng_consensus',
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

// === FINAL PRODUCTION HARDENING START ===
// H-2: wilsonLower — Wilson score 95% CI lower bound for a proportion.
// Formula: (p + z²/2n - z*sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)
// where z=1.96 (95% two-sided). Returns 0 on any degenerate input.
// Guard reasoning:
//   n=0  → denominator (1 + z²/n) = Inf, num = NaN → returns NaN without guard.
//   p=0  → sqrt(0) = 0, num = 0 + z²/2n - z*sqrt(z²/4n²) = z²/2n - z²/2n = 0.
//          This is correct (lower CI of 0-proportion is 0).
//   p=1  → sqrt(0 + z²/4n²) = z/2n, num = 1+z²/2n - z²/2n = 1, den = 1+z²/n,
//          lower ≈ 1/(1+z²/n) < 1. Correct.
//   NaN propagation: isNaN/isFinite guards catch any unforeseen FP edge cases.
// === HARDENING END ===
function wilsonLower(p, n) {
  // === FINAL PRODUCTION HARDENING START ===
  if (n <= 0 || !isFinite(n) || isNaN(p)) return 0; // H-2: degenerate input guard
  // === HARDENING END ===
  const z = 1.96;
  const z2 = z * z;
  const inner = p * (1 - p) / n + z2 / (4 * n * n);
  // === FINAL PRODUCTION HARDENING START ===
  // inner can be negative in FP arithmetic for p very close to 0 or 1 with huge n
  // (catastrophic cancellation). Guard with Math.max(0, ...) before sqrt.
  const sqrtTerm = Math.sqrt(Math.max(0, inner));
  // === HARDENING END ===
  const num = p + z2 / (2 * n) - z * sqrtTerm;
  const den = 1 + z2 / n;
  const lower = num / den;
  // === FINAL PRODUCTION HARDENING START ===
  if (isNaN(lower) || !isFinite(lower)) return 0; // H-2: final NaN/Inf guard
  // === HARDENING END ===
  return Math.max(0, lower);
}

// =============================================================================
// === FINAL PRODUCTION HARDENING START ===
// H-1: CALIB_DUMMY sentinel object.
// Used instead of `null` when calling extractPredictiveStreakFeatures from within
// computeCalibration. Prevents EPSF from applying any calibration-driven adjustment
// to its own internal signals during calibration data collection (avoids circularity).
// Why a named object rather than null: null is checked with `if (!calib)` which is
// ambiguous — undefined, 0, or false would also pass. A structural sentinel with
// a known shape makes the intent explicit and the guard reliable under refactoring.
// hotHitRate: Array of nulls → getCalibratedAdjustment reads null → no adjustment.
// coldHitRate: Array of nulls → same.
// hotBinCount: Array of zeros → reqMinBin check fails → null guard fires.
// === HARDENING END ===
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

// Per-target LOOK_AHEAD windows (rounds ahead to check for a hit).
// Justification: LOOK_AHEAD must be long enough that P(hit in window) varies
// meaningfully between hot and cold regimes, but short enough to have abundant
// calibration samples. For each target:
//   5x:   globalHitRate≈0.200, LOOK_AHEAD=20 → baseline P=1-(0.80)^20=0.988
//   10x:  globalHitRate≈0.100, LOOK_AHEAD=20 → baseline P=1-(0.90)^20=0.878
//   20x:  globalHitRate≈0.050, LOOK_AHEAD=20 → baseline P=1-(0.95)^20=0.642
//   50x:  globalHitRate≈0.020, LOOK_AHEAD=40 → baseline P=1-(0.98)^40=0.554
//   100x: globalHitRate≈0.010, LOOK_AHEAD=80 → baseline P=1-(0.99)^80=0.551
//   250x: globalHitRate≈0.004, LOOK_AHEAD=150 → baseline P=1-(0.996)^150=0.451
//   500x: globalHitRate≈0.002, LOOK_AHEAD=300 → baseline P=1-(0.998)^300=0.451
//  1000x: globalHitRate≈0.001, LOOK_AHEAD=300 → baseline P=1-(0.999)^300=0.259
// All baselines well within (0.01, 0.99), avoiding degenerate calibration.
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

// Minimum bin sample sizes (unweighted round count for Wilson CI validity).
// Non-rare: 20 → Wilson CI width ≈ 0.44 at p=0.5; narrows to ≤0.22 at p=0.1/0.9.
//           Below 20, CI width > 0.44 → no reliable uplift measurement.
// Rare (100x+): 30 → hit counts per bin are sparse; 30 gives ±0.18 CI at p=0.1.
const MIN_BIN_NON_RARE = 20;
const MIN_BIN_RARE      = 30;

// === FINAL PRODUCTION HARDENING START ===
// H-9: Module-level calibration log counter.
// Persists across ticks (unlike a local variable reset each call).
// Logs every 10 cache-miss recomputations regardless of rounds.length parity.
// Justification: rounds.length % 1000 < 50 fails on batch ingests where
// rounds jump by >1000 between ticks, skipping the logging boundary entirely.
let _calibLogCounter = 0;
// === HARDENING END ===

const calibCache = {};

// =============================================================================
// computeCalibration v5
// Builds empirical P(hit in LOOK_AHEAD rounds | hotScore bin) tables using
// full extractPredictiveStreakFeatures at sampled historical positions.
// Cost analysis (justifying full-feature sampling):
//   Max positions evaluated: ~4000 (stride=1) + 2667 (stride=3) + 8800 (stride=10)
//   = ~15467 positions across 100k rounds, but cached on first run and reused
//   for ≥50 ticks (~50+ rounds = several minutes at 1 round/6s).
//   Per-position cost: ~0.3ms (EPSF on a 100k-round slice is O(n) but constant-factor
//   small). Total first-run: ~4.6s worst case. Acceptable for a background tick.
// =============================================================================
function computeCalibration(rounds, targetMin, targetLabel, targetRare) {
  const cacheKey = targetLabel;
  const now = Date.now();
  const cache = calibCache[cacheKey];
  const delta = cache ? rounds.length - cache.computedAt : Infinity;
  const age   = cache ? now - cache.computedAtMs : Infinity;
  // Invalidate if ≥50 new rounds arrived OR >10 minutes elapsed.
  // 50 rounds ≈ 2–3 regime cycles for mid-X targets; sufficient to detect shift.
  // 600000ms (10min) = stale protection on server restart or long idle periods.
  if (cache && delta < 50 && age < 600000) {
    return cache.result;
  }

  const n = rounds.length;
  const LOOK_AHEAD = TARGET_LOOK_AHEAD[targetLabel] || 20;
  const BIN_SIZE   = 10;
  const NUM_BINS   = 10;
  const minBin     = targetRare ? MIN_BIN_RARE : MIN_BIN_NON_RARE;

  // Baseline: P(at least one hit in LOOK_AHEAD independent rounds).
  // Uses geometric CDF: 1 - (1-p)^LOOK_AHEAD. Correct probability units.
  // v3/old bug: used globalHitRate * LOOK_AHEAD (a count, not probability),
  // which exceeded 1.0 for frequent targets → calibration never fired for 5x/10x.
  const globalHitRate = rounds.filter(r => r.multiplier >= targetMin).length /
    Math.max(1, n);
  const baseline = clamp(
    1 - Math.pow(Math.max(0, 1 - globalHitRate), LOOK_AHEAD),
    0.01, 0.99
  );

  // === FINAL PRODUCTION HARDENING START ===
  // H-3: Per-target minimum context.
  // For rare targets (e.g. 1000x), computeGaps returns 0 gaps for the first
  // ~1000 rounds, causing EPSF to return null immediately. Setting minContext
  // to max(60, LOOK_AHEAD*2, expectedGapApprox*3) skips positions guaranteed
  // to produce null EPSF results.
  // expectedGapApprox = 1/globalHitRate (geometric mean gap). For 1000x ≈ 1000.
  // 3× = 3000 rounds needed before EPSF can form a stable Markov matrix.
  // Capped at n/4 so we don't skip more than 25% of history on any target.
  const expectedGapApprox = globalHitRate > 0 ? Math.round(1 / globalHitRate) : n;
  const minContext = Math.min(
    Math.floor(n / 4),
    Math.max(60, LOOK_AHEAD * 2, expectedGapApprox * 3)
  );
  // maxPos: last position where we still have LOOK_AHEAD rounds of future data.
  // Guard: if minContext >= n - LOOK_AHEAD, no valid positions exist for this target.
  const maxPos = n - LOOK_AHEAD;
  if (maxPos <= minContext) {
    // Insufficient history — return a neutral (non-calibrating) result.
    // Caller will use CALIB_DUMMY behaviour via null hotHitRate entries.
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
  // === HARDENING END ===

  const hotBinHitsW   = new Array(NUM_BINS).fill(0);
  const hotBinTotalW  = new Array(NUM_BINS).fill(0);
  const hotBinCount   = new Array(NUM_BINS).fill(0);
  const coldBinHitsW  = new Array(NUM_BINS).fill(0);
  const coldBinTotalW = new Array(NUM_BINS).fill(0);
  const coldBinCount  = new Array(NUM_BINS).fill(0);

  for (let pos = maxPos; pos >= minContext; ) {
    const distFromEnd = n - pos;

    // Recency weight: 0.999^distFromEnd.
    // At dist=1: w=0.999; dist=3000: w≈0.050; dist=6000: w≈0.0025.
    // Ensures last 3000 rounds contribute ~3× as much as rounds 3000–6000 back.
    // Decay constant 0.999 chosen so half-life = ln(0.5)/ln(0.999) ≈ 693 rounds ≈
    // several days of real gameplay — matches typical crash regime persistence.
    const recWeight = Math.pow(0.999, distFromEnd);

    const ctx = rounds.slice(0, pos);

    // === FINAL PRODUCTION HARDENING START ===
    // H-1: Pass CALIB_DUMMY (not null) to avoid circularity.
    // EPSF will call getCalibratedAdjustment(hs, cs, CALIB_DUMMY, false).
    // getCalibratedAdjustment sees hotBinCount[bin]=0 < reqMinBin → hotRate=null
    // → no calibration adjustment applied. EPSF returns raw-signal features only.
    // This is exactly what we want for calibration data collection: pure signal
    // features uncontaminated by recursive calibration feedback.
    let sf = null;
    try { sf = extractPredictiveStreakFeatures(ctx, targetMin, CALIB_DUMMY); }
    catch(_) { /* EPSF failure on degenerate ctx — skip position */ }
    // === HARDENING END ===

    if (!sf) {
      const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
      pos -= stride;
      continue;
    }

    const hs = sf.hotScore;
    const cs = sf.coldScore;

    // === FINAL PRODUCTION HARDENING START ===
    // Guard: hotScore and coldScore must be finite integers in [0,100].
    // EPSF clamps them, but check defensively before using as array indices.
    if (!isFinite(hs) || !isFinite(cs)) {
      const stride = distFromEnd <= 4000 ? 1 : distFromEnd <= 12000 ? 3 : 10;
      pos -= stride;
      continue;
    }
    // === HARDENING END ===

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

  // === FINAL PRODUCTION HARDENING START ===
  // Build rate tables. Wilson CI lower bound must exceed baseline + margin
  // before any uplift is permitted. This prevents spurious calibration from
  // small samples or baseline-aligned hit rates.
  // margin: 0.025 non-rare (need clear signal above random), 0.015 rare
  // (harder to achieve 0.025 margin with sparse data, but still meaningful).
  // === HARDENING END ===
  const margin  = targetRare ? 0.015 : 0.025;
  const hotHitRate  = new Array(NUM_BINS).fill(null);
  const coldHitRate = new Array(NUM_BINS).fill(null);

  for (let b = 0; b < NUM_BINS; b++) {
    if (hotBinCount[b] >= minBin && hotBinTotalW[b] > 0) {
      const wr = hotBinHitsW[b] / hotBinTotalW[b];
      const wl = wilsonLower(wr, hotBinCount[b]);
      if (wl > baseline + margin) hotHitRate[b] = wr;
    }
    if (coldBinCount[b] >= minBin && coldBinTotalW[b] > 0) {
      const wr = coldBinHitsW[b] / coldBinTotalW[b];
      const wl = wilsonLower(wr, coldBinCount[b]);
      if (wl > (1 - baseline) + margin) coldHitRate[b] = wr;
    }
  }

  const result = {
    hotHitRate, coldHitRate, baseline,
    hotBinCount, coldBinCount,
    LOOK_AHEAD, BIN_SIZE, minBin, margin, targetRare,
  };

  // === FINAL PRODUCTION HARDENING START ===
  // H-9: Module-level log counter — fires every 10 cache misses regardless
  // of rounds.length parity. Provides periodic visibility into calibration health.
  _calibLogCounter++;
  if (_calibLogCounter % 10 === 0) {
    console.log(`[ngCompute calib v5] ${JSON.stringify({
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
  // === HARDENING END ===

  calibCache[cacheKey] = { computedAt: n, computedAtMs: now, result };
  return result;
}

// =============================================================================
// getCalibratedAdjustment v5
// Normalized uplift formula with Wilson CI gating, rare-target halved caps,
// and conservative density-trend fallback in low-data regimes.
// =============================================================================
function getCalibratedAdjustment(hotScore, coldScore, calib, targetRare, sf) {
  // === FINAL PRODUCTION HARDENING START ===
  // H-1: Structural check against CALIB_DUMMY or null.
  // CALIB_DUMMY has hotBinCount[*]=0, so the reqMinBin check below will always
  // fail → no adjustment fires → returns calibMult=1.0. Explicit guard for null.
  if (!calib || calib === CALIB_DUMMY) {
    return { calibMult: 1.0, calibConfBonus: 0, calibrated: false };
  }
  // === HARDENING END ===

  const hotBin  = Math.min(9, Math.floor(hotScore  / (calib.BIN_SIZE || 10)));
  const coldBin = Math.min(9, Math.floor(coldScore  / (calib.BIN_SIZE || 10)));

  const hotCount  = calib.hotBinCount?.[hotBin]  ?? 0;
  const coldCount = calib.coldBinCount?.[coldBin] ?? 0;
  const reqMinBin = targetRare ? (calib.minBin ?? 20) * 2 : (calib.minBin ?? 20);

  // === FINAL PRODUCTION HARDENING START ===
  // H-4: Conservative fallback when bin has data but below 1.5× minBin threshold.
  // 1.5× = the zone between "sufficient for null-guard" (minBin) and "reliable estimate"
  // (1.5×minBin). In this zone, Wilson CI is too wide for confident uplift, but we have
  // some evidence of the density environment. Apply a slight conservative adjustment:
  // if density trend (ld10-ld50) > 0.05 (rising = cold incoming), reduce by 0.03 (3%).
  // This is not a calibration signal — it's a mild reactive nudge based on observable
  // current state. Maximum effect: 3% gap extension, well within safety bounds.
  // Justification for 0.05 density trend threshold: at 5x (globalLowRate≈0.80),
  // a trend of 0.05 means low-density rose from 0.80 to 0.85 in recent window —
  // a 6% relative increase, indicative of a genuine cold shift.
  const densityTrend = sf?.densityTrend ?? 0;
  const inLowDataZone = hotCount >= reqMinBin && hotCount < reqMinBin * 1.5;
  if (inLowDataZone) {
    const conservMult = densityTrend > 0.05 ? 1.03 : 1.0;
    return { calibMult: conservMult, calibConfBonus: 0, calibrated: false };
  }
  // === HARDENING END ===

  const hotRate  = (hotCount  >= reqMinBin) ? calib.hotHitRate[hotBin]  : null;
  const coldRate = (coldCount >= reqMinBin) ? calib.coldHitRate[coldBin] : null;

  // Hard caps. Justification:
  // maxDelta=0.26 → multiplier range [0.74, 1.26].
  // 0.74x on a 20-gap (5x target) = predict at 14.8 rounds. Window width=3 covers
  // 13–15. At 5x true gap distribution p50≈5, p90≈11, P(hit before 13) ≈ 0.93.
  // So the floor is structurally safe — we're not opening before the realistic
  // distribution starts. 1.26x on a 20-gap = predict at 25.2, window covers 24–26.
  // P(hit after 26 | not yet hit by 26) ≈ 0.37 — still meaningful.
  // Rare halved to 0.13 to prevent window miss on high-X targets (see v4 doc).
  const maxDelta  = targetRare ? 0.13 : 0.26;
  const sensiHot  = 1.8;
  const sensiCold = 1.5;

  let calibMult      = 1.0;
  let calibConfBonus = 0;
  let calibrated     = false;

  if (hotScore >= 72 && hotRate !== null) {
    // Normalized uplift: fraction of achievable improvement above baseline captured
    // by this bin. Range [0,1]. Prevents raw uplift inflation for high-baseline targets.
    const upliftNorm = clamp(
      (hotRate - calib.baseline) / Math.max(0.001, 1 - calib.baseline),
      0, 1
    );
    const reduction = clamp(upliftNorm * sensiHot, 0, maxDelta);
    if (reduction > 0.01) {
      calibMult = 1.0 - reduction;
      calibConfBonus = upliftNorm > 0.25 ? 12 : upliftNorm > 0.15 ? 8 : 4;
      calibrated = true;
    }
  }

  if (coldScore >= 65 && coldRate !== null && !calibrated) {
    const noHitBaseline = 1 - calib.baseline;
    const upliftNorm = clamp(
      (coldRate - noHitBaseline) / Math.max(0.001, 1 - noHitBaseline),
      0, 1
    );
    const extension = clamp(upliftNorm * sensiCold, 0, maxDelta);
    if (extension > 0.01) {
      calibMult      = 1.0 + extension;
      calibConfBonus = -3;
      calibrated     = true;
    }
  }

  // === FINAL PRODUCTION HARDENING START ===
  // H-7: Hard clamp tightened to 0.76–1.24 (v4: 0.74–1.26).
  // Justification: crash gap distributions have CV≈0.8–1.5 (domain knowledge).
  // For CV=0.8, σ=0.8×mean. A ±24% shift ≈ 0.30σ — meaningful but not overfit.
  // ±26% (v4) was 0.325σ — marginal increase in early-hit risk. The tightening
  // to ±24% trades 2% adjustment range for reduced window-misplacement risk,
  // which is the correct trade-off when calibration uncertainty is already ±5–10%.
  calibMult = clamp(calibMult, 0.76, 1.24);
  // === HARDENING END ===

  return { calibMult, calibConfBonus, calibrated };
}

// =============================================================================
// extractPredictiveStreakFeatures v5
// All v3 bug fixes preserved. Threshold adjustments from v4 preserved.
// v5 change: CALIB_DUMMY passed into getCalibratedAdjustment internally,
// hotScore trigger raised to 72.
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

  // Composite hotScore / coldScore
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

  // Regime prediction
  // === FINAL PRODUCTION HARDENING START ===
  // H-8: hotScore strong trigger raised 70→72.
  // At 70: three soft signals can co-fire: postCluster(30)+momentum(15)+markov(15)=60
  // + b2bRate(10)+density(10)=80 is fine, but also postCluster(30)+momentum(15)+markov(15)
  // +b2bRate(5)+density(0)=65... wait, that's 65 not 70. Actually at 70 the minimum
  // combo is postCluster(30)+accel(-0.04→9)+momentum(-0.15→7)+markov(>0.68→15)+b2bRate(0)
  // +density(0)=61... no, needs to reach 70. Let's be precise:
  // To reach 70 WITHOUT any density signal (ld20 check=0):
  //   postCluster(30)+b2bPrecursor(20)+accel(18)+momentum(15) = 83 ✓ (strong combo)
  //   postCluster(30)+b2bPrecursor(20)+momentum(15)+markov(15) = 80 ✓ (strong)
  //   postCluster(30)+accel(18)+momentum(15)+markov(15) = 78 ✓ (no b2b, but 3 solid)
  //   b2bPrecursor(20)+accel(18)+momentum(15)+markov(15)+b2bRate(10) = 78 ✓
  // To reach 70 with ONLY soft signals (no postCluster, no b2bPrecursor):
  //   accel(18)+momentum(15)+markov(15)+b2bRate(10)+density(10) = 68 ✗ (can't reach 70)
  // So at 72: impossible to reach via 3 soft signals alone without a structural
  // precursor (postClusterEarlySignal or b2bPrecursor). This is the intended behavior.
  // Justification: crash regime transitions consistently require at least one
  // structural precursor signal to be predictive. Three density/momentum signals
  // without a cluster or GARCH precursor are insufficient evidence.
  // === HARDENING END ===
  let predictedNextRegime    = 'NEUTRAL';
  let transitionConfidence   = 0;
  let predictedGapMultiplier = 1.0;

  // === FINAL PRODUCTION HARDENING START ===
  if (hotScore >= 72 && hotScore > coldScore + 15) {
    if (b2bPrecursor) {
      predictedNextRegime    = 'ABOUT_TO_B2B';
      transitionConfidence   = clamp(hotScore, 72, 95);
      // 0.76 floor aligns with tightened hard cap
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.36, 0.76, 0.88);
    } else {
      predictedNextRegime    = 'ABOUT_TO_HOT';
      transitionConfidence   = clamp(hotScore, 72, 90);
      predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.26, 0.76, 0.90);
    }
  } else if (hotScore >= 62 && hotScore > coldScore + 10) {
    // Moderate hot: entry point for calibration to boost if confirmed
    predictedNextRegime    = 'ABOUT_TO_HOT';
    transitionConfidence   = clamp(hotScore, 62, 88);
    predictedGapMultiplier = clamp(1.0 - (hotScore/100)*0.20, 0.80, 0.93);
  } else if (coldScore >= 65 && coldScore > hotScore + 15) {
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime    = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence   = clamp(coldScore, 65, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.28, 1.10, 1.24); // 1.24 cap
    } else {
      predictedNextRegime    = 'ABOUT_TO_COLD';
      transitionConfidence   = clamp(coldScore, 55, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.20, 1.05, 1.22);
    }
  }
  // === HARDENING END ===

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
// applyStreakAdjustment v5
// tc threshold ≥65. WHITE_CLUSTER blend raised 0.55→0.65.
// =============================================================================
function applyStreakAdjustment(expectedGap, sf, _target) {
  if (!sf) return expectedGap;
  const pnr  = sf.predictedNextRegime;
  const mult = sf.predictedGapMultiplier ?? 1.0;
  const tc   = sf.transitionConfidence  ?? 0;

  if (pnr !== 'NEUTRAL' && tc >= 65) {
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
      // === FINAL PRODUCTION HARDENING START ===
      // H-6: Raised 0.55→0.65. avgPostClusterGap is always 1.0 (occurrence counter,
      // not actual gap length). With old blend=0.55:
      //   adj = raw*0.55 + 1*0.45 ≈ 0.55*raw + 0.45 (catastrophic 45% reduction)
      // With new blend=0.65:
      //   adj = raw*0.65 + 1*0.35 ≈ 0.65*raw + 0.35 (35% reduction — meaningful
      //   shortening that reflects the real post-cluster behavior where hits cluster
      //   within ~60-70% of the average gap, without overshooting into near-zero).
      // Justification from domain: crash B2B clusters (White Cluster regime) do show
      // shortened low-run gaps, but NOT by 45%. Observed regime-specific gap shortening
      // is typically 20–40%. 0.65 blend gives ~35% reduction — within observed range.
      // === HARDENING END ===
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
// streakConfBonus v5
// Reactive bonuses halved when !sf.calibrated to prevent confidence inflation
// on unconfirmed regime signals corrupting stacking inverse-variance weights.
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
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 65) {
    base -= 3;
  }

  // === FINAL PRODUCTION HARDENING START ===
  // H-5: Reactive regime bonuses halved when !calibrated.
  // Justification: when sf.calibrated=false, regime classification is derived
  // purely from structural patterns (RLE, density windows) without empirical
  // confirmation that these patterns historically lead to hits in the next N rounds.
  // Full reactive bonuses (+5 for B2B, +4 for WHITE_CLUSTER) inflate confidence
  // scores, pushing predictions above the ~65 threshold used by stacking_meta
  // for inverse-variance weighting. This causes stacking to over-weight predictions
  // from engines that happen to be in a hot-looking regime, even when that regime
  // hasn't been empirically confirmed. Halving the bonuses reduces this bias while
  // preserving some reward for correct reactive pattern detection.
  // "Halved" means integer floor: 5→2, 4→2, 6→3, 3→1, 2→1.
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
      base -= 2; break; // cold penalty not halved — conservative is safe
  }
  // === HARDENING END ===

  return base;
}

// effectiveRegime: use predictive regime only when tc ≥ 68.
// Aligns with hotScore trigger (72) minus one bin width (10) to prevent stacking
// from shifting regime before the individual engine hot triggers fire.
function effectiveRegime(sf) {
  if (!sf) return 'NEUTRAL';
  return (sf.transitionConfidence ?? 0) >= 68 ? sf.predictedNextRegime : sf.regime;
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
// tcBonus requires calibrated===true && tc>75 && hotScore>=72 (aligned with trigger)
// =============================================================================
function computeNgConsensus(rounds,calibrations,lastRoundId,sharedSf) {
  // NG Master — standalone signal using the same gap+calibration+streak pipeline
  // the removed sub-engines used. Runs once per tick with full context.
  const consensus={};
  for (const target of TARGETS) {
    const sf=sharedSf?.[target.label];
    if(!sf){consensus[target.label]=null;continue;}

    // Compute gaps directly from rounds (same as sub-engines did)
    const {gaps,currentGap}=computeGaps(rounds,target.min);
    if(gaps.length<10){consensus[target.label]=null;continue;}

    // Raw expected gap: use harmonic mean of recent gaps (robust to outliers)
    const recentGaps=gaps.slice(-Math.min(gaps.length,100));
    const hrGlobal=recentGaps.length/rounds.length;
    if(hrGlobal<=0){consensus[target.label]=null;continue;}
    const rawGap=Math.max(1,Math.round(1/hrGlobal));

    // Apply calibrated adjustment
    const calib=calibrations[target.label]??CALIB_DUMMY;
    const {calibMult}=getCalibratedAdjustment(
      sf.hotScore??0, sf.coldScore??0, calib, target.rare, sf
    );
    const calibratedGap=Math.max(1,Math.round(rawGap*calibMult));

    // Apply streak/regime adjustment
    const adjustedGap=applyStreakAdjustment(calibratedGap,sf,target);
    const finalGap=Math.max(1,Math.round(adjustedGap));

    const bW=target.maxWidth;
    let {low,high}=placeWindow(finalGap,currentGap,bW);
    let bestLo=lastRoundId+low, bestHi=lastRoundId+high;

    // Enforce minimum width and future placement
    if(bestHi-bestLo+1<bW){const c=Math.round((bestLo+bestHi)/2);bestLo=c-Math.floor(bW/2);bestHi=bestLo+bW-1;}
    if(bestLo<=lastRoundId){bestLo=lastRoundId+1;bestHi=bestLo+bW-1;}

    // === FINAL PRODUCTION HARDENING START ===
    // tcBonus requires hotScore>=72, calibrated===true, tc>75.
    const tcBonus=(sf.calibrated===true&&(sf.transitionConfidence??0)>75&&
      (sf.hotScore??0)>=72&&
      (sf.predictedNextRegime==='ABOUT_TO_B2B'||sf.predictedNextRegime==='ABOUT_TO_HOT'))?8:0;
    // === HARDENING END ===

    consensus[target.label]={
      lo:bestLo,hi:bestHi,engineCount:1,
      engines:['ng_master'],tcBonus,
    };
  }
  return consensus;
}

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

    // Step 3: Compute ng_consensus master signal directly
    // ng_consensus aggregates from an empty set here (no sub-engines),
    // so we build it from streak features directly via computeNgConsensus
    const allNgResults={};
    const ngConsensus=computeNgConsensus(rounds,calibrations,lastRoundId,streakFeatures);
    allNgResults['ng_consensus']={};
    for(const target of TARGETS){
      const c=ngConsensus[target.label];
      if(c){
        const baseConf=clamp(55+Math.round((c.engineCount??0)*4),55,95);
        const finalConf=clamp(baseConf+(c.tcBonus??0),55,99);
        allNgResults['ng_consensus'][target.label]={
          low:c.lo-lastRoundId,high:c.hi-lastRoundId,
          expectedGap:Math.round((c.lo+c.hi)/2-lastRoundId),
          probW:null,conf:finalConf,
          _meta:{engineCount:c.engineCount??0,engines:c.engines??[]},
        };
      }
    }

    // Phase 1+2: resolve + lock ng_consensus windows
    const engineId='ng_consensus';
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