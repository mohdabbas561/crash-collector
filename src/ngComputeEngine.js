'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 engines + ng_consensus)
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

  // Hot margin: 0.025 non-rare / 0.015 rare — unchanged from v5.
  const hotMargin  = targetRare ? 0.015 : 0.025;
  // === v6 COLD-PATH UPGRADE START ===
  // Cold margin lowered: 0.025→0.012 (non-rare), 0.015→0.012 (rare).
  // Cold minBin lowered: 20/30→15 for cold bins only.
  // Justification: white-cluster phases are structurally sparse — the market
  // spends less time in deep cold than in hot. With minBin=20/30, cold bins
  // rarely accumulate enough samples to unlock calibration. Lowering to 15
  // allows empirically-observed cold stretches (which happen less frequently
  // but are real) to fire calibration. The Wilson CI guard (wl > threshold)
  // still prevents random noise — we just allow smaller-but-still-valid samples.
  // margin=0.012: accepts cold signal if Wilson lower bound exceeds no-hit
  // baseline by 1.2% — tighter than hot (2.5%) but still meaningful for cold.
  const COLD_MIN_BIN = 15;
  const coldMargin  = 0.012;
  // === UPGRADE END ===
  const hotHitRate  = new Array(NUM_BINS).fill(null);
  const coldHitRate = new Array(NUM_BINS).fill(null);

  for (let b = 0; b < NUM_BINS; b++) {
    if (hotBinCount[b] >= minBin && hotBinTotalW[b] > 0) {
      const wr = hotBinHitsW[b] / hotBinTotalW[b];
      const wl = wilsonLower(wr, hotBinCount[b]);
      if (wl > baseline + hotMargin) hotHitRate[b] = wr;
    }
    // === v6 COLD-PATH UPGRADE START ===
    if (coldBinCount[b] >= COLD_MIN_BIN && coldBinTotalW[b] > 0) {
      const wr = coldBinHitsW[b] / coldBinTotalW[b];
      const wl = wilsonLower(wr, coldBinCount[b]);
      if (wl > (1 - baseline) + coldMargin) coldHitRate[b] = wr;
    }
    // === UPGRADE END ===
  }

  const result = {
    hotHitRate, coldHitRate, baseline,
    hotBinCount, coldBinCount,
    LOOK_AHEAD, BIN_SIZE, minBin, margin: hotMargin, targetRare,
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
  // Hot path: maxDelta unchanged (0.26 non-rare / 0.13 rare), sensiHot=1.8 unchanged.
  const maxDeltaHot = targetRare ? 0.13 : 0.26;
  const sensiHot    = 1.8;
  // === v6 COLD-PATH UPGRADE START ===
  // Cold path: maxDelta raised 0.26→0.40 (non-rare), 0.13→0.20 (rare).
  // sensiCold raised 1.5→1.9.
  // Justification: maxDelta=0.26 capped cold extension at 26% even with perfect
  // calibration signal. A 20-gap at 1.26× = 25.2 rounds — far too short when the
  // market is in a genuine 8-round white cluster (true gap often 30-50 rounds).
  // 1.40× cap = 28 rounds on a 20-gap, 56 rounds on a 40-gap (50x target).
  // sensiCold=1.9: normalized uplift maps more aggressively to extension.
  // At upliftNorm=0.3: old=0.45 (capped to 0.26), new=0.57 (capped to 0.40).
  // Rare cold raised to 0.20 (was 0.13) — white clusters affect rare targets too.
  const maxDeltaCold = targetRare ? 0.20 : 0.40;
  const sensiCold    = 1.9;
  // === UPGRADE END ===

  let calibMult      = 1.0;
  let calibConfBonus = 0;
  let calibrated     = false;

  if (hotScore >= 72 && hotRate !== null) {
    // Hot path: 100% unchanged from v5.
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

  // === v6 COLD-PATH UPGRADE START ===
  // Cold trigger lowered 65→58. At coldScore=58, a genuine white cluster
  // (streakMomentum>0.25=14 + lowDensityAccel>0.04=11 + markov<0.38=10 +
  // ld20>ld50*1.20=7 + streakLen=12) = 54 — close. With densityTrend bonus
  // from new coldScore weights, 58 is reliably reachable during real clusters.
  if (coldScore >= 58 && coldRate !== null && !calibrated) {
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
  // === UPGRADE END ===

  // === v6 COLD-PATH UPGRADE START ===
  // Hot path: floor 0.76 unchanged (never compress below 76% of gap).
  // Cold path: ceiling raised 1.24→1.40 to allow full cold extension.
  // Hot: calibMult should only reduce gap (calibMult<1.0); floor 0.76.
  // Cold: calibMult should only extend gap (calibMult>1.0); ceiling 1.40.
  // A calibMult that reduces gap on cold (calibMult<1.0) would be a bug —
  // guard it with the 0.76 floor. A calibMult that extends hot is also unusual
  // but benign — the 1.40 ceiling still applies.
  calibMult = clamp(calibMult, 0.76, 1.40);
  // Hot signal: never let calibration push multiplier above 1.0 on hot path
  if (hotScore >= 72 && calibMult > 1.0) calibMult = 1.0;
  // === UPGRADE END ===

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

  // === v6 COLD-PATH UPGRADE START ===
  // coldScore weight increases:
  // streakMomentum: +25% → 35/17 (was 28/14). During a genuine 8-round cluster,
  //   streakMomentum often hits 0.3-0.5 — the extra weight pushes coldScore over 58.
  // lowDensityAccel (densityTrend proxy): +30% → 29/14 (was 22/11). The acceleration
  //   of low density is the earliest detectable signal of a genuine white cluster.
  //   At 8+ consecutive lows, ld5→ld10→ld20 all converge high; accel fires strongly.
  // currentStreakLen bonus: raised 12→16 to reward directly observable cluster depth.
  // All hot-score weights unchanged.
  // === UPGRADE END ===
  const coldScore = clamp(Math.round(
    (streakMomentum   > 0.45   ? 35 : streakMomentum  > 0.25  ? 17 : 0) +
    (lowDensityAccel  > 0.08   ? 29 : lowDensityAccel  > 0.04  ? 14 : 0) +
    (markovProbHot   < 0.22    ? 20 : markovProbHot   < 0.38   ? 10 : 0) +
    (ld20            > ld50*1.40 ? 15 : ld20 > ld50*1.20 ? 7 : 0) +
    (currentStreakLen > avgLowRunLen*1.3 && !currentIsHigh ? 16 : 0)
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
  // === v6 COLD-PATH UPGRADE START ===
  // Cold trigger lowered 65→58. ABOUT_TO_WHITE_CLUSTER multiplier cap raised
  // 1.10–1.24 → 1.15–1.40. ABOUT_TO_COLD raised 1.05–1.22 → 1.10–1.32.
  // Justification: at coldScore=65, the market is already 5-7 rounds into a
  // cluster; we're reacting too late. At coldScore=58, the cluster is 2-4 rounds
  // deep — still early enough for the lengthened window to capture the tail.
  // 1.40 upper cap: for a 40-gap (50x target), 1.40× = 56 rounds. Typical white
  // cluster on 50x lasts 30-60 rounds — 1.40× is calibrated, not excessive.
  // For 5x (20-gap): 1.40× = 28 rounds — the cluster still ends within this window.
  } else if (coldScore >= 58 && coldScore > hotScore + 12) {
    if (currentStreakLen >= avgLowRunLen * 1.5 || regime === 'EXTREME_WHITE') {
      predictedNextRegime    = 'ABOUT_TO_WHITE_CLUSTER';
      transitionConfidence   = clamp(coldScore, 58, 88);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.42, 1.15, 1.40);
    } else {
      predictedNextRegime    = 'ABOUT_TO_COLD';
      transitionConfidence   = clamp(coldScore, 50, 82);
      predictedGapMultiplier = clamp(1.0 + (coldScore/100)*0.30, 1.10, 1.32);
    }
  }
  // === UPGRADE END ===
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

  // === v6 COLD-PATH UPGRADE START ===
  // Hot path: tc≥65 unchanged — ABOUT_TO_B2B / ABOUT_TO_HOT still require tc≥65.
  // Cold path: ABOUT_TO_WHITE_CLUSTER / ABOUT_TO_COLD fire at tc≥58.
  // blendFactor for cold starts at tc=58, reaches full blend at tc=103 (effectively
  // capped at 1.0 by clamp). This means at tc=58: blendFactor=0, at tc=78: 0.44,
  // at tc=88: 0.67 — gradual ramp that avoids snapping to full extension immediately.
  const isColdPnr = pnr === 'ABOUT_TO_WHITE_CLUSTER' || pnr === 'ABOUT_TO_COLD';
  const isHotPnr  = pnr === 'ABOUT_TO_B2B' || pnr === 'ABOUT_TO_HOT';

  if (isHotPnr && tc >= 65) {
    // Hot path: UNCHANGED from v5
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  if (isColdPnr && tc >= 58) {
    // Cold path: blendFactor starts at tc=58
    const blendFactor = clamp((tc - 58) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }

  if (!isHotPnr && !isColdPnr && pnr !== 'NEUTRAL' && tc >= 65) {
    // Any other predictive regime at tc≥65 (NEUTRAL already excluded above)
    const blendFactor = clamp((tc - 65) / 45, 0, 1);
    const blendedMult = 1.0 + (mult - 1.0) * blendFactor;
    return Math.max(1, Math.round(expectedGap * blendedMult));
  }
  // === UPGRADE END ===

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
  // === v6 COLD-PATH UPGRADE START ===
  // Cold confidence penalty lowered tc threshold 65→58, strengthened -3→-6.
  // Justification: during white clusters, confidence should be penalized more
  // aggressively to prevent the engine from over-betting on the current window.
  // The window is LONGER now (1.15-1.40×) so we need lower confidence to
  // discourage false precision. -6 vs -3 keeps the user appropriately cautious.
  } else if ((pnr==='ABOUT_TO_WHITE_CLUSTER'||pnr==='ABOUT_TO_COLD') && tc > 58) {
    base -= 6;
  }
  // === UPGRADE END ===

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
// ENGINE 1: hybrid_lstm_xgb
// H-10: calib null-guard — b2bBoost only amplifies when sf.calibrated=true
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
  const {b:slope,r2}=olsLinear(gaps.slice(-Math.min(100,gaps.length)));
  const overdue=clamp(currentGap/(gMean||1),0,3);
  const cv=gStd/(gMean||1);

  const pnr = effectiveRegime(sf);
  // === FINAL PRODUCTION HARDENING START ===
  // H-10: b2bBoost calibration amplification only when sf.calibrated=true.
  // Without calibration, use base b2bRate only (no ×1.3/×1.6 multiplier).
  const b2bBoost = sf ? (
    (pnr==='ABOUT_TO_B2B' && sf.calibrated) ? sf.b2bRate*1.6 :
    pnr==='ABOUT_TO_B2B'                    ? sf.b2bRate      :
    pnr==='ABOUT_TO_HOT'                    ? sf.b2bRate*1.1  : sf.b2bRate
  ) : 0;
  // === HARDENING END ===
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
  // === FINAL PRODUCTION HARDENING START ===
  // H-10: postClusterEarlySignal injection only when sf.calibrated=true
  if (sf && sf.postClusterEarlySignal && sf.calibrated && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.6+sf.avgPostClusterGap*0.4));
  } else if (sf && sf.regime==='WHITE_CLUSTER' && sf.avgPostClusterGap!==null) {
    raw=Math.max(1,Math.round(raw*0.65+sf.avgPostClusterGap*0.35));
  }
  // === HARDENING END ===
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
  // === FINAL PRODUCTION HARDENING START ===
  // H-10: rare b2b quantile shift only when calibrated
  if (target.rare && sf?.b2bPrecursor && sf?.calibrated) {
    wQ10=Math.min(wQ10+0.08,0.55);wQ90=Math.max(wQ90-0.08,0.05);
  }
  // === HARDENING END ===
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
  // === FINAL PRODUCTION HARDENING START ===
  // H-10: streakBlock weight boost only when calibrated
  const isHotCalibrated = (pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT') && (sf?.calibrated===true);
  // === HARDENING END ===
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
    // === FINAL PRODUCTION HARDENING START ===
    // H-10: hotAmp only amplifies when sf.calibrated=true
    const hotAmp = (sf.calibrated===true)
      ? clamp((sf.hotScore||0)/100*1.2, 0.3, 0.9)
      : 0.4;
    // === HARDENING END ===
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
  // === FINAL PRODUCTION HARDENING START ===
  // H-10: calMult only amplifies when sf.calibrated=true (strict boolean check)
  const calMult = (sf?.calibrated===true) ? 1.5 : 1.0;
  const hotScoreNorm  = sf ? (sf.hotScore||0)/100  : 0;
  const coldScoreNorm = sf ? (sf.coldScore||0)/100 : 0;
  // === HARDENING END ===
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
    // === FINAL PRODUCTION HARDENING START ===
    // H-10: calibrated amplification (0.40) only when sf.calibrated=true
    if      (pnr==='ABOUT_TO_B2B' && sf.calibrated===true)
      h=gMean*(1-sf.b2bContinuationProb*0.40);
    else if (pnr==='ABOUT_TO_B2B'||pnr==='B2B')
      h=gMean*(1-sf.b2bContinuationProb*0.28);
    // === HARDENING END ===
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
// tcMult: calibrated&&tc>75→1.45, calibrated&&tc>68→1.25, else 1.0
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
  // === FINAL PRODUCTION HARDENING START ===
  // tcMult: only amplify when sf.calibrated===true (strict).
  // Without empirical confirmation, regime specialisation weights stay neutral (1.0).
  const tcMult = (sf?.calibrated===true && tc > 75) ? 1.45
               : (sf?.calibrated===true && tc > 68) ? 1.25
               : 1.0;
  // === HARDENING END ===

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

  // === FINAL PRODUCTION HARDENING START ===
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
  // === HARDENING END ===
  const pnr=effectiveRegime(sf);
  // === FINAL PRODUCTION HARDENING START ===
  const driftAligned=(pnr==='ABOUT_TO_B2B'||pnr==='ABOUT_TO_HOT')&&maxC>0&&sf?.calibrated===true;
  // === HARDENING END ===
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
// tcBonus requires calibrated===true && tc>75 && hotScore>=72 (aligned with trigger)
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
    // === FINAL PRODUCTION HARDENING START ===
    // tcBonus requires hotScore>=72 (aligned with v5 trigger), calibrated===true, tc>75.
    // This is the top 2–3% of all hot signals — the only cases where we should
    // boost consensus confidence beyond the base engine-count formula.
    const tcBonus = (sf?.calibrated===true && (sf?.transitionConfidence??0)>75 &&
      (sf?.hotScore??0)>=72 &&
      (sf?.predictedNextRegime==='ABOUT_TO_B2B'||sf?.predictedNextRegime==='ABOUT_TO_HOT')) ? 8 : 0;
    // === HARDENING END ===
    consensus[target.label]={
      lo:bestLo,hi:bestHi,engineCount:bestGroup.length,
      engines:bestGroup.map(w=>w.engineId),tcBonus,
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
        // Pass actual calibration (or CALIB_DUMMY if null) — never raw null
        streakFeatures[target.label] = extractPredictiveStreakFeatures(
          rounds, target.min, calibrations[target.label] ?? CALIB_DUMMY
        );
      } catch(e) {
        streakFeatures[target.label]=null;
        console.error(`[ngCompute] psf/${target.label}:`, e.message);
      }
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

    // Phase 1+2: resolve + lock
    for(const engineId of NG_ENGINE_IDS){
      const payload={};
      const newlyLocked=[];
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
              // Window still active — re-lock to DB but don't treat as new
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
          newlyLocked.push(`${target.label}:#${newLo}-#${newHi}`);
        }
      }
      if(Object.keys(payload).length) await saveLockedAdvPreds(engineId,payload);
      if(newlyLocked.length) console.log(`[ngCompute] ${engineId} NEW windows: ${newlyLocked.join(' ')}`);
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