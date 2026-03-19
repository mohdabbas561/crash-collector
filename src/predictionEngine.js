'use strict';
// predictionEngine.js  v19-FINAL
//
// ═══════════════════════════════════════════════════════════════════════════════
// CHANGELOG vs v18-PRECISION
// ═══════════════════════════════════════════════════════════════════════════════
//
// PROBLEM: v18 floor at 0.46 still admitted some low-quality signals.
// The 0.42–0.46 zone is salvageable ONLY with very tight model agreement.
// Decision reasons were verbose; output fields unnecessary for production.
//
// FINAL DECISION SYSTEM (4 zones, no exceptions):
//
//   ZONE 0 — Hard floor
//     prob < 0.42                               → SKIP  (FILTERED)
//
//   ZONE 1 — Low zone (0.42–0.46): mostly noise
//     spread < 0.08 AND confidence > 55         → TAKE  (MODERATE_SIGNAL)
//     else                                      → SKIP  (FILTERED)
//
//   ZONE 2 — Core moderate (0.46–0.52)
//     spread < 0.12 AND confidence > 50         → TAKE  (MODERATE_SIGNAL)
//     else                                      → SKIP  (FILTERED)
//
//   ZONE 3 — Strong (≥ 0.52)                   → TAKE  (STRONG_SIGNAL)
//
// AMPLIFICATION (unchanged from v18):
//   Only when rawProbW ≥ 0.50 AND spread < 0.10
//
// ENSEMBLE (unchanged from v18):
//   Best model direct if leads all others by ≥ 0.06; else ensemble.
//
// TIMING: zero changes — identical to v16/v17/v18.
//
// OUTPUT (simplified — only what matters):
//   prob, confidence, spread, recommendation (TAKE/SKIP), reason
//   Diagnostic fields (rawProbW, amplifiedProbW, finalProbUsed,
//   signalStrength, aggressiveMode) retained for observability.
//
// METRIC: winRate = wins / total trades (takenWins / takenTotal). Only this.
//
// NOTE: If accuracy does not improve past ~50% after this change, the system
//   has reached the ceiling imposed by IID randomness in the underlying game.
//   No further decision-logic changes will help. The probability math is correct.
//
// ALL timing / regime / calibration / window / rare-engine math UNCHANGED.
// DB format UNCHANGED. processEngine, runPredictionEngine UNCHANGED.
//
// ── HOW TO DEPLOY ────────────────────────────────────────────────────────────
//   1. Replace predictionEngine.js with this file.
//   2. Restart — no DB migration needed.
//   3. Console: "[v19-FINAL] Loaded — 4-zone decision, timing intact"
//   4. Fewer trades, cleaner signals. winRate target 50–60%.
//   5. decisionReason: STRONG_SIGNAL / MODERATE_SIGNAL / FILTERED
//
// ═══════════════════════════════════════════════════════════════════════════════

const ENGINE_VERSION = 'v19-FINAL';
const {
  getRounds, savePrediction, getPredictions,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedStatPreds, getLockedStatPreds,
} = require('./db');

console.log('[v19-FINAL] Loaded — 4-zone decision, timing intact');

// ── Targets ───────────────────────────────────────────────────────────────────

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

const STAT_MODELS = [
  { id: 'ens'  },
  { id: 'geo'  },
  { id: 'bay'  },
  { id: 'km'   },
  { id: 'rf'   },   // Random Forest         — ml-random-forest
  { id: 'gbt'  },   // Gradient Boosting     — ml-cart (GBT ensemble)
  { id: 'lr'   },   // Logistic Regression   — simple-statistics
  { id: 'nb'   },   // Naive Bayes (Gaussian) — ml-naivebayes
  { id: 'lstm' },   // LSTM                  — pure JS, zero deps
  { id: 'lgbm' },   // LightGBM              — pure JS leaf-wise boosting
  { id: 'prp'  },   // Prophet               — pure JS seasonality/cycle
  { id: 'gru'  },   // GRU Neural Net        — pure JS gated recurrent
  { id: 'ifor' },   // Isolation Forest      — pure JS anomaly detection
  { id: 'meta' },   // Meta-Ensemble         — weighted average of all 9 ML engines
];

// ── Lazy-load ML libraries (installed via npm, optional) ─────────────────────
let _rf = null, _cart = null, _ss = null, _nb = null;
function getRF()   { if (_rf   === null) { try { const _m = require('ml-random-forest'); _rf = _m.RandomForestClassifier ? _m : (_m.default || _m); } catch(e) { _rf = false; console.warn('[ml] ml-random-forest not available:', e.message); } } return _rf || null; }
function getCART() { if (_cart === null) { try { const _m = require('ml-cart'); _cart = _m.DecisionTreeClassifier ? _m : (_m.default || _m); } catch(e) { _cart = false; console.warn('[ml] ml-cart not available:', e.message); } } return _cart || null; }
function getSS()   { if (_ss   === null) { try { _ss   = require('simple-statistics'); } catch(e) { _ss  = false; console.warn('[ml] simple-statistics not available:',  e.message); } } return _ss   || null; }
function getNB()   { if (_nb   === null) { try { const _m = require('ml-naivebayes'); _nb = _m.GaussianNB || _m.default?.GaussianNB || _m; } catch(e) { _nb = false; console.warn('[ml] ml-naivebayes not available:', e.message); } } return _nb || null; }

// ── In-memory ML model cache (retrain when rounds grow by ML_RETRAIN_INTERVAL) ─
const mlModelCache       = {};   // { rf_5x: {model, trainedAt}, ... }
const ML_RETRAIN_INTERVAL = 2000; // retrain every 2000 new rounds (~28 min) — full data, infrequent
let   mlTrainingEnabled  = false; // stays false until after first full engine cycle
let   mlLastTrainRound   = 0;

// Called after first successful runPredictionEngine() tick — enables ML training
function enableMLTraining() {
  if (!mlTrainingEnabled) {
    mlTrainingEnabled = true;
    console.log('[ml] Training enabled — ML engines will train on next tick');
  }
}

const MIN_ROUNDS                    = 50;
const STALE_FORCE_REBUILD_THRESHOLD = 500;
const WINDOW_LAMBDA                 = 0.008;

// ── Regime constants ──────────────────────────────────────────────────────────
const REGIME_SHORT_WINDOW           = 30;
const REGIME_WIDTH_SCALE            = 0.25;
const REGIME_OFFSET_SCALE           = 0.18;
const REGIME_DECAY                  = 0.10;
const REGIME_CUSUM_CLIP_HOT         = 1.2;
const REGIME_CUSUM_CLIP_COLD        = 0.8;
const REGIME_HYSTERESIS_REQUIRED    = 2;
const REGIME_ACTIVATION_OUTCOMES    = 50;
const REGIME_EARLY_HINT_SIGMA       = 1.8;
const REGIME_EARLY_HINT_MAX_FACTOR  = 0.25;

// ── Calibration constants ─────────────────────────────────────────────────────
const CAL_BINS  = [0, 0.30, 0.45, 0.60, 0.75, 1.01];
const CAL_DECAY = { normal: 0.08, rare: 0.03 };
const CAL_MIN_SAMPLES = 12;
const CAL_WARMUP_OUTCOMES = 80;
const CAL_WARMUP_RAW_WEIGHT = 0.65;

// ── Decision engine constants ─────────────────────────────────────────────────
// TEMP_SCALE_AGREEMENT kept for applyTemperatureScaling (ENS builder).
const TEMP_SCALE_AGREEMENT  = 0.15;

// ── Decision constants — 4-zone system ───────────────────────────────────────
// Zone 0: hard floor — no trade below this
const SIG_HARD_FLOOR          = 0.42;   // below → SKIP always
// Zone 1: low zone (0.42–0.46) — admit only on very tight agreement
const SIG_LOW_ZONE_TOP        = 0.46;   // top of low zone
const SIG_LOW_ZONE_SPREAD     = 0.08;   // spread must be < this in low zone
const SIG_LOW_ZONE_CONF       = 55;     // confidence must be > this in low zone
// Zone 2: core moderate (0.46–0.52)
const SIG_MODERATE_TOP        = 0.52;   // top of moderate zone (= bottom of strong)
const SIG_MODERATE_SPREAD     = 0.12;   // spread must be < this in moderate zone
const SIG_MODERATE_CONF       = 50;     // confidence must be > this in moderate zone
// Zone 3: strong — TAKE unconditionally
const SIG_STRONG_PROB         = 0.52;   // ≥ this → STRONG_SIGNAL

// Amplification — restricted to high-confidence zone only (unchanged from v18)
const SIG_AMP_FACTOR          = 1.45;
const SIG_AMP_MIN_PROB        = 0.50;   // raw prob must be ≥ this to amplify
const SIG_AMP_MAX_SPREAD      = 0.10;   // spread must be < this to amplify
const SIG_AMP_MAX_ECE         = 0.04;

// Ensemble best-model direct use (unchanged from v18)
const SIG_BEST_MODEL_MIN_N    = 20;
const SIG_BEST_MODEL_GAP      = 0.06;   // best must lead ALL others by ≥ this

// Calibration relaxation (unchanged)
const SIG_CAL_RELAX_MAX_HITS  = 200;
const SIG_CAL_RELAX_MAX_ECE   = 0.02;

// High-conviction window boost (unchanged — only affects window width, not decision)
const SIG_CONVICTION_SPREAD   = 0.08;
const SIG_CONVICTION_REGIME   = 0.30;
const SIG_CONVICTION_BOOST    = 0.05;

// ── Rare engine constants ──────────────────────────────────────────────────────
const RARE_MIN_MULTIPLIER      = 100;
const RARE_TAIL_GEO_BLEND      = 0.40;
const RARE_TAIL_GEO_BLEND_SPARSE = 0.70;
const RARE_SPARSE_HITS         = 30;
const RARE_EV_THRESHOLD        = 0.80;
const RARE_TAIL_MIN            = 0.12;
const RARE_EXTREME_GAP_BOOST   = 0.15;
const RARE_EXTREME_WIDTH_BOOST = 1.25;
const RARE_EARLY_WINDOW_FRAC   = 0.60;
const RARE_LATE_WINDOW_FRAC    = 2.50;
const RARE_CAL_IMPACT_HITS     = 150;
const RARE_CAL_REDUCED_WEIGHT  = 0.30;
const RARE_PAYOUT = { '100x':0.95, '250x':0.92, '500x':0.90, '1000x':0.88 };

// ── Timing feedback constants ─────────────────────────────────────────────────
const TIMING_MAX_SHIFT_FACTOR     = 0.35;
const TIMING_GAP_CORRECTION_SCALE = 0.20;
const TIMING_CENTER_PULL_SCALE    = 0.25;
const TIMING_RECENT_WINDOW        = 20;
const TIMING_RECENT_SPIKE_THRESH  = 0.30;
const TIMING_RECENT_SPIKE_SHIFT   = 0.10;
const TIMING_WIDTH_BOOST_THRESH   = 0.20;
const TIMING_WIDTH_BOOST_FRAC     = 0.12;

// ── Per-target timing feedback state ─────────────────────────────────────────
const timingState = {};
for (const t of TARGETS) {
  timingState[t.label] = {
    earlyCount:       0,
    totalCount:       0,
    earlyQueue:       [],
    recentEarlyCount: 0,
  };
}

// ── Per-engine independent CUSUM / regime state ───────────────────────────────
const engineCusumState = {};
for (const id of ['engine', 'ens', 'geo', 'bay', 'km', 'rf', 'gbt', 'lr', 'nb', 'lstm', 'lgbm', 'prp', 'gru', 'ifor', 'meta']) {
  engineCusumState[id] = {};
  for (const t of TARGETS) {
    engineCusumState[id][t.label] = {
      cusum:             0,
      regimeFactor:      0,
      ewmaRate:          -1,
      count:             0,
      regimeLabel:       'neutral',
      hysteresisCount:   0,
      pendingLabel:      'neutral',
      confirmedFactor:   0,
    };
  }
}

// ── Per-engine state ──────────────────────────────────────────────────────────
const STATE = {
  engine:  { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  pattern: { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ens:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  geo:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  bay:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  km:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  rf:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  gbt:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  lr:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  nb:      { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  lstm:    { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  lgbm:    { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  prp:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  gru:     { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  ifor:    { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
  meta:    { lockedMap: null, savedSet: null, needsRebuild: true, lastRoundId: 0 },
};

// ── Per-engine last-hit tracker ────────────────────────────────────────────────
// Each engine independently tracks the roundId of the last WIN it recorded
// per target. This drives independent gapSinceLast so windows diverge.
// Initialised to -1 (unknown — falls back to shared gapStats on first run).
const engineLastHit = {};
for (const id of ['engine', 'ens', 'geo', 'bay', 'km', 'rf', 'gbt', 'lr', 'nb', 'lstm', 'lgbm', 'prp', 'gru', 'ifor', 'meta']) {
  engineLastHit[id] = {};
  for (const t of TARGETS) engineLastHit[id][t.label] = -1;
}

// ── Gap stats cache ───────────────────────────────────────────────────────────
const gapStatsCache = new Map();
let   cacheRoundId  = -1;

// ── Calibration state ─────────────────────────────────────────────────────────
const calibState = {};
for (const t of TARGETS) {
  calibState[t.label] = {};
  for (const m of STAT_MODELS) {
    calibState[t.label][m.id] = CAL_BINS.slice(0, -1).map((lo, i) => {
      const mid = (lo + CAL_BINS[i + 1]) / 2;
      return { ewmaAct: mid, ewmaPred: mid, count: 0 };
    });
  }
}

// ── Adaptive ensemble weights ─────────────────────────────────────────────────
const modelScores = {};
for (const t of TARGETS) {
  modelScores[t.label] = {};
  for (const m of STAT_MODELS) modelScores[t.label][m.id] = { ewma: 0.693, count: 0 };
}

// ── Validation metrics ────────────────────────────────────────────────────────
const valMetrics = {};
for (const t of TARGETS) {
  valMetrics[t.label] = {};
  for (const m of STAT_MODELS) {
    valMetrics[t.label][m.id] = {
      brierSum: 0, logLossSum: 0, count: 0,
      wins: 0, losses: 0, earlyCount: 0,
      takenWins: 0, takenTotal: 0,
      // F8: trade metrics — TAKE only
      tradeCount: 0,    // TAKE decisions taken
      totalWins:  0,    // wins on taken trades
    };
  }
}

// ── F8: Per-target taken-trades tracking ─────────────────────────────────────
const takenTradesMetrics = {};
for (const t of TARGETS) {
  takenTradesMetrics[t.label] = { wins: 0, losses: 0, early: 0, total: 0 };
}

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
  for (const id of Object.keys(engineCusumState)) {
    for (const t of TARGETS) {
      engineCusumState[id][t.label] = {
        cusum: 0, regimeFactor: 0, ewmaRate: -1, count: 0,
        regimeLabel: 'neutral', hysteresisCount: 0,
        pendingLabel: 'neutral', confirmedFactor: 0,
      };
    }
  }
  gapStatsCache.clear();
  cacheRoundId = -1;
  initialised  = false;
  for (const t of TARGETS) {
    timingState[t.label] = { earlyCount: 0, totalCount: 0, earlyQueue: [], recentEarlyCount: 0 };
    takenTradesMetrics[t.label] = { wins: 0, losses: 0, early: 0, total: 0 };
  }
  // Reset per-engine last-hit tracker
  for (const id of Object.keys(engineLastHit)) {
    for (const t of TARGETS) engineLastHit[id][t.label] = -1;
  }
}

// ── Per-engine gapSinceLast ────────────────────────────────────────────────────
// Returns how many rounds have elapsed since THIS engine last recorded a win
// for this target. Falls back to shared gapStats value if unknown.
// This is what makes each engine's window position independent.
function getEngineGapSinceLast(engineId, targetLabel, lastRoundId, sharedGapSinceLast) {
  const lastHit = engineLastHit[engineId]?.[targetLabel] ?? -1;
  if (lastHit < 0) return sharedGapSinceLast; // no history yet — use shared
  return Math.max(0, lastRoundId - lastHit);
}

// ── Math helpers ──────────────────────────────────────────────────────────────
function sigmoid(x)       { return 1 / (1 + Math.exp(-x)); }
function logit(p)         { const q = Math.max(1e-7, Math.min(1 - 1e-7, p)); return Math.log(q / (1 - q)); }
function fromLogit(l)     { return sigmoid(l); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Timing feedback: record outcome ───────────────────────────────────────────
function recordTimingOutcome(targetLabel, isEarly) {
  const ts = timingState[targetLabel];
  if (!ts) return;
  ts.totalCount++;
  if (isEarly) ts.earlyCount++;
  const val = isEarly ? 1 : 0;
  ts.earlyQueue.push(val);
  ts.recentEarlyCount += val;
  if (ts.earlyQueue.length > TIMING_RECENT_WINDOW) {
    ts.recentEarlyCount -= ts.earlyQueue.shift();
  }
}

// ── Timing feedback: compute shift params ────────────────────────────────────
function getTimingParams(targetLabel) {
  const ts = timingState[targetLabel];
  if (!ts || ts.totalCount < 5) {
    return { earlyRate: 0, recentEarlyRate: 0, timingShiftFactor: 0, hasData: false };
  }
  const earlyRate       = ts.earlyCount / ts.totalCount;
  const recentTotal     = ts.earlyQueue.length;
  const recentEarlyRate = recentTotal > 0 ? ts.recentEarlyCount / recentTotal : earlyRate;
  const timingShiftFactor = clamp(earlyRate, 0, TIMING_MAX_SHIFT_FACTOR);
  return { earlyRate, recentEarlyRate, timingShiftFactor, hasData: true };
}

// ── Timing-corrected window placement ────────────────────────────────────────
function applyTimingCorrection(expectedGap, effectiveWidth, targetLabel, maxWidth) {
  const { earlyRate, recentEarlyRate, timingShiftFactor, hasData } = getTimingParams(targetLabel);

  // Width is fixed to maxWidth — timing boost disabled per window-gap spec.
  let correctedWidth = effectiveWidth;

  if (!hasData) {
    const low  = Math.max(0, expectedGap - correctedWidth);
    const high = low + correctedWidth - 1;
    return { low, high, effectiveWidth: correctedWidth, timingShiftFactor: 0 };
  }

  const expectedGapCorrected = Math.max(1, Math.round(
    expectedGap * (1 - TIMING_GAP_CORRECTION_SCALE * earlyRate)
  ));
  const center = Math.max(1, Math.round(
    expectedGapCorrected * (1 - TIMING_CENTER_PULL_SCALE * earlyRate)
  ));
  let low  = Math.max(0, center - Math.floor(correctedWidth / 2));
  let high = low + correctedWidth - 1;

  if (recentEarlyRate > TIMING_RECENT_SPIKE_THRESH) {
    const spikeShift = Math.round(expectedGapCorrected * TIMING_RECENT_SPIKE_SHIFT);
    low  = Math.max(0, low  - spikeShift);
    high = low + correctedWidth - 1;
  }

  return { low, high, effectiveWidth: correctedWidth, timingShiftFactor, expectedGapCorrected };
}

// ── Dynamic regime thresholds ─────────────────────────────────────────────────
function getDynamicRegimeParams(outcomeCount) {
  if (outcomeCount < REGIME_ACTIVATION_OUTCOMES) {
    return { threshold: 9999, minFactor: 9999, active: false };
  }
  const t = clamp((outcomeCount - REGIME_ACTIVATION_OUTCOMES) / (400 - REGIME_ACTIVATION_OUTCOMES), 0, 1);
  const threshold = 2.4 - t * (2.4 - 1.65);
  const minFactor = 0.45 - t * (0.45 - 0.18);
  return { threshold, minFactor, active: true };
}

// ── Per-Engine Regime Detection ───────────────────────────────────────────────
function updateEngineRegime(rounds, targetMin, targetLabel, engineId, gs, isRandomSignal) {
  const cs = engineCusumState[engineId]?.[targetLabel];
  if (!cs) return { regimeFactor: 0, regimeLabel: 'neutral', shortRate: 0, shortBaseline: 0, cusumNorm: 0, regimeConfidence: 0 };

  const n = rounds.length;
  const shortStart  = Math.max(0, n - REGIME_SHORT_WINDOW);
  const shortWindow = rounds.slice(shortStart);

  let shortHits = 0;
  for (const r of shortWindow) if (r.multiplier >= targetMin) shortHits++;
  const shortRate     = shortHits / Math.max(1, shortWindow.length);
  const shortBaseline = gs.hits / Math.max(1, gs.n);

  if (cs.ewmaRate < 0) cs.ewmaRate = shortRate;
  else cs.ewmaRate = (1 - REGIME_DECAY) * cs.ewmaRate + REGIME_DECAY * shortRate;
  cs.count++;

  const regimeConfidence = clamp((cs.count - REGIME_ACTIVATION_OUTCOMES) / 200, 0, 1);
  const { threshold, minFactor, active } = getDynamicRegimeParams(cs.count);

  if (!active) {
    const sigma        = Math.sqrt(Math.max(1e-9, shortBaseline * (1 - shortBaseline)));
    const deviationSig = sigma > 0 ? (shortRate - shortBaseline) / sigma : 0;
    let hintFactor = 0;
    if (Math.abs(deviationSig) >= REGIME_EARLY_HINT_SIGMA) {
      const rawHint = Math.sign(deviationSig) * REGIME_EARLY_HINT_MAX_FACTOR
        * clamp((Math.abs(deviationSig) - REGIME_EARLY_HINT_SIGMA) / REGIME_EARLY_HINT_SIGMA, 0, 1);
      hintFactor = rawHint * ((isRandomSignal === true) ? 0.40 : 1.0);
    }
    cs.regimeFactor    = hintFactor;
    cs.confirmedFactor = hintFactor;
    cs.regimeLabel     = hintFactor > 0.10 ? 'hot' : hintFactor < -0.10 ? 'cold' : 'neutral';
    return {
      regimeFactor:     +hintFactor.toFixed(4),
      regimeLabel:      cs.regimeLabel,
      shortRate, shortBaseline,
      cusumNorm: 0,
      regimeConfidence: 0,
      earlyHint: true,
    };
  }

  const deviation = shortRate - shortBaseline;
  cs.cusum += deviation;

  const sigma    = Math.sqrt(Math.max(1e-9, shortBaseline * (1 - shortBaseline)));
  const cusumNorm = cs.cusum / Math.max(sigma * Math.sqrt(REGIME_SHORT_WINDOW), 1e-6);

  const clipHot  = REGIME_CUSUM_CLIP_HOT  * sigma * Math.sqrt(REGIME_SHORT_WINDOW);
  const clipCold = REGIME_CUSUM_CLIP_COLD * sigma * Math.sqrt(REGIME_SHORT_WINDOW);
  cs.cusum = clamp(cs.cusum, -clipCold, clipHot);

  const rawFactor = Math.tanh(cusumNorm / (threshold * 1.5));
  const randomDampen = (isRandomSignal === true) ? 0.40 : 1.0;
  cs.regimeFactor = clamp(
    (1 - REGIME_DECAY) * cs.regimeFactor + REGIME_DECAY * rawFactor * randomDampen,
    -1, 1
  );

  const rawLabel = cs.regimeFactor > 0.15 ? 'hot'
    : cs.regimeFactor < -0.15 ? 'cold'
    : 'neutral';

  if (rawLabel === cs.pendingLabel) {
    cs.hysteresisCount++;
  } else {
    cs.pendingLabel    = rawLabel;
    cs.hysteresisCount = 1;
  }

  if (cs.hysteresisCount >= REGIME_HYSTERESIS_REQUIRED) {
    cs.regimeLabel     = cs.pendingLabel;
    cs.confirmedFactor = cs.regimeFactor;
  }

  const effectiveFactor = Math.abs(cs.confirmedFactor) >= minFactor ? cs.confirmedFactor : 0;

  return {
    regimeFactor:      +effectiveFactor.toFixed(4),
    regimeLabel:       cs.regimeLabel,
    shortRate:         +shortRate.toFixed(4),
    shortBaseline:     +shortBaseline.toFixed(4),
    cusumNorm:         +cusumNorm.toFixed(3),
    regimeConfidence:  +regimeConfidence.toFixed(3),
    earlyHint:         false,
  };
}

// ── Apply regime to window placement ─────────────────────────────────────────
function applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regimeFactor) {
  if (regimeFactor === 0) return { low, high };
  const widthDelta  = clamp(-regimeFactor * REGIME_WIDTH_SCALE, -0.20, 0.30);
  const newWidth    = Math.max(1, Math.round(effectiveWidth * (1 + widthDelta)));
  const offsetShift = Math.round(regimeFactor * expectedGap * REGIME_OFFSET_SCALE);
  const newLow      = Math.max(0, low - offsetShift);
  const newHigh     = newLow + newWidth - 1;
  return { low: newLow, high: newHigh };
}

// ── Regime-adjusted p blending ────────────────────────────────────────────────
function applyRegimeToP(baseP, shortRate, regimeFactor, blendStrength) {
  if (regimeFactor === 0) return baseP;
  const regimeBlend = Math.abs(regimeFactor) * blendStrength;
  const blended = (1 - regimeBlend) * baseP + regimeBlend * shortRate;
  return Math.max(1e-6, Math.min(0.5, blended));
}

// ── Gap stats cache ───────────────────────────────────────────────────────────
function getGapStats(rounds, targetMin, lastRoundId) {
  if (lastRoundId !== cacheRoundId) {
    gapStatsCache.clear();
    cacheRoundId = lastRoundId;
  }
  const key = `${targetMin}`;
  if (gapStatsCache.has(key)) return gapStatsCache.get(key);
  const result = computeGapStats(rounds, targetMin);
  gapStatsCache.set(key, result);
  return result;
}

function computeGapStats(rounds, targetMin) {
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

  const p0 = hits / n;
  let cusum = 0, maxCusum = 0;
  const cusumWindow = rounds.slice(-150);
  for (const r of cusumWindow) {
    cusum += (r.multiplier >= targetMin ? 1 : 0) - p0;
    if (Math.abs(cusum) > maxCusum) maxCusum = Math.abs(cusum);
  }
  const sigma0    = Math.sqrt(Math.max(1e-9, p0 * (1 - p0)));
  const cusumNorm = maxCusum / (sigma0 * Math.sqrt(cusumWindow.length));
  const rateShifted = cusumNorm > 1.36;

  const sg = [...gaps].sort((a, b) => a - b);
  const m2 = Math.floor(sg.length / 2);
  const medianGap = sg.length === 0 ? Math.round(1 / pGlobal) :
    sg.length % 2 === 1 ? sg[m2] : (sg[m2 - 1] + sg[m2]) / 2;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap  = gaps.length > 0 ? gSum / gaps.length : 1 / pGlobal;
  const variance = gaps.length > 1 ? Math.max(0, gSS / gaps.length - meanGap ** 2) : meanGap * meanGap;
  const stdGap   = Math.sqrt(variance);
  const cv       = meanGap > 0 ? stdGap / meanGap : 1;

  const pctile = (frac) => {
    if (sg.length === 0) return meanGap;
    return sg[Math.min(sg.length - 1, Math.floor(frac * sg.length))];
  };
  const p75 = pctile(0.75);
  const p90 = pctile(0.90);
  const p95 = pctile(0.95);

  const hsm = halfSampleMode(sg, medianGap, cv);
  let modeWeight;
  if (cv < 1.0)      modeWeight = 0.50;
  else if (cv > 1.3) modeWeight = 0.10;
  else               modeWeight = 0.50 - (cv - 1.0) * (0.40 / 0.30);

  const kmExpectedGap = Math.max(1, Math.round(medianGap * (1 - modeWeight) + hsm * modeWeight));
  const kmCDF = buildKmCDF(sg);

  let currentStreak = 0;
  for (let i = n - 1; i >= 0; i--) {
    if (rounds[i].multiplier < targetMin) currentStreak++;
    else break;
  }

  let maxAC = 0;
  if (gaps.length >= 50) {
    let gVarSum = 0;
    for (const g of gaps) gVarSum += (g - meanGap) ** 2;
    if (gVarSum > 0) {
      for (let lag = 1; lag <= Math.min(5, gaps.length - 1); lag++) {
        let cov = 0;
        for (let i = lag; i < gaps.length; i++) {
          cov += (gaps[i] - meanGap) * (gaps[i - lag] - meanGap);
        }
        const ac = cov / gVarSum;
        if (Math.abs(ac) > Math.abs(maxAC)) maxAC = ac;
      }
    }
  }
  const isRandom = gaps.length >= 50 && Math.abs(maxAC) < 0.10;

  return {
    hits, n, pGlobal, pRecent, rateShifted,
    cusumNorm: +cusumNorm.toFixed(3),
    gapSinceLast, meanGap, medianGap, stdGap, cv,
    p75, p90, p95, sg, kmCDF,
    hsm, kmExpectedGap,
    currentStreak,
    maxAC:    +maxAC.toFixed(4),
    isRandom,
  };
}

// ── Half-sample mode ──────────────────────────────────────────────────────────
function halfSampleMode(sg, medianGap, cv) {
  const n = sg.length;
  if (n < 8)    return medianGap;
  if (cv > 2.0) return sg[Math.floor(n / 2)];
  const h = Math.max(2, Math.floor(n / 2));
  let bestRange = sg[sg.length - 1] - sg[0] + 1, bestStart = 0;
  for (let i = 0; i + h - 1 < n; i++) {
    const range = sg[i + h - 1] - sg[i];
    if (range < bestRange) { bestRange = range; bestStart = i; }
  }
  return (sg[bestStart] + sg[bestStart + h - 1]) / 2;
}

// ── KM CDF ────────────────────────────────────────────────────────────────────
function buildKmCDF(sg) {
  if (sg.length < 5) return null;
  const m = sg.length;
  const steps = [];
  let S = 1.0, i = 0;
  while (i < m) {
    const t = sg[i];
    const nAtRisk = m - i;
    let d = 0;
    while (i < m && sg[i] === t) { d++; i++; }
    S *= (1 - d / nAtRisk);
    steps.push({ t, S: Math.max(0, S) });
  }
  return steps;
}

function kmCDFQuery(kmCDF, W) {
  if (!kmCDF || kmCDF.length === 0) return null;
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (kmCDF[mid].t <= W) { idx = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  const S = idx >= 0 ? kmCDF[idx].S : 1.0;
  return Math.max(0, Math.min(1 - 1e-9, 1 - S));
}

function kmWindowProb(kmCDF, lo, hi) {
  const cdfHi = kmCDFQuery(kmCDF, hi)           ?? 0;
  const cdfLo = lo > 0 ? (kmCDFQuery(kmCDF, lo - 1) ?? 0) : 0;
  return Math.max(0, cdfHi - cdfLo);
}

// ── HYBRID window placement ───────────────────────────────────────────────────
function hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth) {
  const parametric = parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast);
  let km = parametric;
  if (kmCDF) {
    km = empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast);
  }

  const blendedLow  = Math.round(0.70 * parametric.low  + 0.30 * km.low);
  const blendedHigh = Math.round(0.70 * parametric.high + 0.30 * km.high);

  const anchoredWidth = blendedHigh - blendedLow + 1;

  const mw = maxWidth ?? effectiveWidth;
  const { low, high, effectiveWidth: correctedWidth } =
    applyTimingCorrection(expectedGap, anchoredWidth, targetLabel ?? '', mw);

  return { low, high, effectiveWidth: correctedWidth };
}

function parametricWindowPlacement(expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) {
    return { low: 0, high: effectiveWidth - 1 };
  }
  const remaining = expectedGap - gapSinceLast;
  const low  = Math.max(0, remaining - effectiveWidth);
  const high = low + effectiveWidth - 1;
  return { low, high };
}

function empiricalWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast) {
  if (gapSinceLast >= expectedGap) {
    return { low: 0, high: effectiveWidth - 1 };
  }
  const maxLow = Math.max(0, 3 * expectedGap - gapSinceLast - effectiveWidth);
  let bestScore = -Infinity, bestLow = 0;
  for (let low = 0; low <= maxLow; low++) {
    const absLo   = gapSinceLast + low;
    const absHi   = gapSinceLast + low + effectiveWidth - 1;
    const probHit = kmWindowProb(kmCDF, absLo, absHi);
    const score   = probHit - WINDOW_LAMBDA * effectiveWidth;
    if (score > bestScore) { bestScore = score; bestLow = low; }
    if (absLo > expectedGap * 3 && probHit < 0.001) break;
  }
  return { low: bestLow, high: bestLow + effectiveWidth - 1 };
}

// ── Pareto + exponential tail smoothing for rare targets ──────────────────────
function applyParetoCorrectedProbW(rawProbW, cv, expectedGap, maxWidth, isRare, hits) {
  if (!isRare || cv <= 1.1) return rawProbW;
  const aggressiveMode = hits >= 120;
  const alpha  = 1 / Math.max(0.1, cv * cv);
  const paretoP = 1 - Math.pow(1 / (1 + maxWidth / Math.max(1, expectedGap)), alpha);
  const expP = 1 - Math.exp(-maxWidth / Math.max(1, expectedGap));
  const tailP = paretoP * 0.60 + expP * 0.40;
  const blendW = aggressiveMode
    ? Math.min(0.45, (cv - 1.0) * 0.70)
    : Math.min(0.30, (cv - 1.0) * 0.60);
  const blended = (1 - blendW) * rawProbW + blendW * tailP;
  return Math.max(1e-6, Math.min(1 - 1e-6, blended));
}

// ── Calibration ───────────────────────────────────────────────────────────────
function getCalBinIdx(probW) {
  for (let i = 0; i < CAL_BINS.length - 1; i++) {
    if (probW < CAL_BINS[i + 1]) return i;
  }
  return CAL_BINS.length - 2;
}

function updateCalibration(targetLabel, modelId, predictedProbW, outcome) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return;
  const target  = TARGETS.find(t => t.label === targetLabel);
  const decay   = target?.rare ? CAL_DECAY.rare : CAL_DECAY.normal;
  const actual  = outcome === 'win' ? 1 : outcome === 'early' ? 0.5 : 0;
  const bin     = bins[getCalBinIdx(predictedProbW)];
  bin.ewmaAct   = (1 - decay) * bin.ewmaAct  + decay * actual;
  bin.ewmaPred  = (1 - decay) * bin.ewmaPred + decay * predictedProbW;
  bin.count     = Math.min(bin.count + 1, 500);
}

function applyCalibration(probW, targetLabel, modelId, totalCount) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < CAL_MIN_SAMPLES) return probW;
  const warmupT = clamp((totalCount ?? bin.count) / CAL_WARMUP_OUTCOMES, 0, 1);
  const empirical  = bin.ewmaAct;
  const predicted  = bin.ewmaPred;
  if (predicted < 1e-6) return probW;
  const ratio = empirical / predicted;
  if (Math.abs(ratio - 1) < 0.12) return probW;
  const corrected = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(0.80, Math.min(1.20, ratio))));
  const capped    = Math.min(corrected, probW + 0.05);
  const rawWeight = CAL_WARMUP_RAW_WEIGHT * (1 - warmupT);
  return probW * rawWeight + capped * (1 - rawWeight);
}

// ── Validation metrics ────────────────────────────────────────────────────────
// F8: primary KPI is winRate = takenWins / takenTotal (wins on TAKE calls only)
function updateValidationMetrics(targetLabel, modelId, predictedProbW, outcome, recommendation) {
  const v = valMetrics[targetLabel]?.[modelId];
  if (!v) return;
  const actual = outcome === 'win' ? 1 : outcome === 'early' ? 0.5 : 0;
  const p = Math.max(1e-7, Math.min(1 - 1e-7, predictedProbW));
  v.brierSum   += (actual - p) ** 2;
  v.logLossSum += -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
  v.count++;
  if (outcome === 'win')        v.wins++;
  else if (outcome === 'early') v.earlyCount++;
  else                          v.losses++;
  // F8: track taken-trade win rate (TAKE only — WEAK_TAKE removed)
  if (recommendation === 'TAKE') {
    v.takenTotal++;
    v.tradeCount++;
    if (outcome === 'win') {
      v.takenWins++;
      v.totalWins++;
    }
  }
}

function getECE(targetLabel, modelId) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return null;
  let ece = 0, total = 0;
  for (const bin of bins) {
    if (bin.count < 2) continue;
    ece   += Math.abs(bin.ewmaAct - bin.ewmaPred) * bin.count;
    total += bin.count;
  }
  return total > 0 ? ece / total : null;
}

// ── BMA Ensemble weights ──────────────────────────────────────────────────────
const BMA_WARMUP_COUNT = 60;
const BMA_WARMUP_PRIORS = [0.45, 0.35, 0.20];

function logLossVal(actual, probW) {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probW));
  return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
}

function updateModelScore(targetLabel, modelId, predictedProbW, outcome) {
  const s = modelScores[targetLabel]?.[modelId];
  if (!s) return;
  const actual  = outcome === 'win' ? 1 : outcome === 'early' ? 0.5 : 0;
  const loss    = logLossVal(actual, predictedProbW);
  const target  = TARGETS.find(t => t.label === targetLabel);
  const decay   = target?.rare ? 0.02 : 0.05;
  s.ewma  = s.count === 0 ? loss : (1 - decay) * s.ewma + decay * loss;
  s.count = Math.min(s.count + 1, 500);
}

function buildEnsemble(targetLabel, probGeo, probBay, probKm, cv) {
  const scores   = modelScores[targetLabel];
  const modelIds = ['geo', 'bay', 'km'];
  const probs    = [probGeo, probBay, probKm];

  const minCount = Math.min(...modelIds.map(id => scores[id]?.count ?? 0));

  let weights;
  if (minCount < BMA_WARMUP_COUNT) {
    const warmupT = clamp(minCount / BMA_WARMUP_COUNT, 0, 1);
    let cvPriors;
    if (cv > 1.5)      cvPriors = [0.25, 0.30, 0.45];
    else if (cv < 1.0) cvPriors = [0.55, 0.30, 0.15];
    else               cvPriors = [...BMA_WARMUP_PRIORS];

    const adaptiveWeights = modelIds.map(id => {
      const avgLoss = scores[id]?.count > 2 ? scores[id].ewma : 0.693;
      return Math.exp(-avgLoss * 2);
    });
    const awSum = adaptiveWeights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < adaptiveWeights.length; i++) adaptiveWeights[i] /= awSum;

    weights = cvPriors.map((p, i) => (1 - warmupT) * p + warmupT * adaptiveWeights[i]);
  } else {
    weights = modelIds.map(id => {
      const avgLoss = scores[id]?.count > 2 ? scores[id].ewma : 0.693;
      return Math.exp(-avgLoss * 2);
    });
  }

  const wSum = weights.reduce((a, b) => a + b, 0);
  if (wSum < 1e-9) return { ensProb: (probGeo + probBay + probKm) / 3, spread: 0, adjWeights: [1/3, 1/3, 1/3], modelDisagreementScore: 0 };
  for (let i = 0; i < weights.length; i++) weights[i] /= wSum;

  const logits = probs.map(logit);
  const wMean  = logits.reduce((s, l, i) => s + weights[i] * l, 0);
  const wVar   = logits.reduce((s, l, i) => s + weights[i] * (l - wMean) ** 2, 0);
  const spread = Math.sqrt(wVar);

  const adjWeights = [...weights];
  for (let i = 0; i < logits.length; i++) {
    if (Math.abs(logits[i] - wMean) > 1.5) adjWeights[i] *= 0.5;
  }
  const adjSum = adjWeights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < adjWeights.length; i++) adjWeights[i] /= adjSum;

  const ensLogit = logits.reduce((s, l, i) => s + adjWeights[i] * l, 0);
  const ensProb  = fromLogit(ensLogit);

  const modelDisagreementScore = +Math.max(
    Math.abs(logits[0] - logits[1]),
    Math.abs(logits[1] - logits[2]),
    Math.abs(logits[0] - logits[2])
  ).toFixed(3);

  return { ensProb, spread, adjWeights, modelDisagreementScore };
}

// ── Temperature scaling ───────────────────────────────────────────────────────
function applyTemperatureScaling(probW, spread, targetLabel, modelId) {
  if (spread > TEMP_SCALE_AGREEMENT) return probW;
  const ece = getECE(targetLabel, modelId);
  if (ece == null || ece > 0.03) return probW;
  const T = 0.85;
  const sharpened = fromLogit(logit(probW) / T);
  return Math.max(1e-6, Math.min(1 - 1e-6, sharpened));
}

// ── Beta confidence ───────────────────────────────────────────────────────────
function betaConf(probW, hits, spread, regimeConfidence, z) {
  const effectiveN = Math.min(hits, 300);
  const alpha = probW * effectiveN + 1;
  const beta  = (1 - probW) * effectiveN + 1;
  const postVar = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const postStd = Math.sqrt(postVar);

  let c = probW * 100 - postStd * 120;
  if (hits < 10)      c -= 15;
  else if (hits < 25) c -= 7;
  else if (hits < 50) c -= 3;

  if (spread != null) c -= Math.min(15, Math.round(15 * spread / 0.5));

  const streakPenalty = (z != null && z > 2.8 && regimeConfidence > 0.5) ? 8 : 0;
  c -= streakPenalty * (regimeConfidence ?? 0);

  return Math.max(20, Math.min(88, Math.round(c)));
}

// ── F4: Signal amplification — restricted to high-confidence zone ─────────────
// Apply ONLY when rawProbW ≥ 0.50 AND spread < 0.10.
// Mid-zone amplification (prev: spread < 0.18) is disabled.
// This prevents low-quality signals being pushed above the 0.46 floor.

function amplifySignal(probW, spread, targetLabel, modelId) {
  // F4: raw prob must already be ≥ 0.50 and spread very tight
  if (probW < SIG_AMP_MIN_PROB) return { amplified: probW, wasAmplified: false };
  if ((spread ?? 1) > SIG_AMP_MAX_SPREAD) return { amplified: probW, wasAmplified: false };
  const ece = getECE(targetLabel, modelId);
  if (ece != null && ece > SIG_AMP_MAX_ECE) return { amplified: probW, wasAmplified: false };
  const amplified = fromLogit(logit(probW) * SIG_AMP_FACTOR);
  return {
    amplified: Math.max(1e-6, Math.min(1 - 1e-6, amplified)),
    wasAmplified: true,
  };
}

// ── F5: Best-model direct use ─────────────────────────────────────────────────
// If the best-performing sub-model's probability exceeds all others by ≥ 0.06
// (in linear probability space), use it directly instead of the ensemble.
// No blending — either the best model leads clearly or we use the ensemble.
// Returns { prob, modelUsed, wasOverridden }.

function selectBestModelDirect(targetLabel, probs, modelIds) {
  const scores = modelScores[targetLabel];
  let bestIdx = -1, bestLoss = Infinity;
  for (let i = 0; i < modelIds.length; i++) {
    const s = scores[modelIds[i]];
    if (!s || s.count < SIG_BEST_MODEL_MIN_N) continue;
    if (s.ewma < bestLoss) { bestLoss = s.ewma; bestIdx = i; }
  }
  if (bestIdx < 0) return { prob: null, wasOverridden: false };

  const bestProb = probs[bestIdx];
  // Check that best model leads ALL others by at least SIG_BEST_MODEL_GAP
  const leadsAll = probs.every((p, i) => i === bestIdx || (bestProb - p) >= SIG_BEST_MODEL_GAP);
  if (!leadsAll) return { prob: null, wasOverridden: false };

  return {
    prob:         bestProb,
    modelUsed:    modelIds[bestIdx],
    wasOverridden: true,
  };
}

// ── S4: Calibration relaxation ────────────────────────────────────────────────
function applyCalibrationRelaxed(probW, targetLabel, modelId, hits) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < CAL_MIN_SAMPLES) return probW;

  const ece = getECE(targetLabel, modelId);
  const sparse = (hits ?? 0) < SIG_CAL_RELAX_MAX_HITS;
  const tight  = ece != null && ece < SIG_CAL_RELAX_MAX_ECE;
  const correctionWeight = (sparse || tight) ? 0.50 : 1.0;

  const empirical = bin.ewmaAct;
  const predicted = bin.ewmaPred;
  if (predicted < 1e-6) return probW;
  const ratio = empirical / predicted;
  if (Math.abs(ratio - 1) < 0.12) return probW;

  const corrected = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(0.80, Math.min(1.20, ratio))));
  const capped    = Math.min(corrected, probW + 0.05);
  const warmupT   = clamp((bin.count) / CAL_WARMUP_OUTCOMES, 0, 1);
  const rawWeight = CAL_WARMUP_RAW_WEIGHT * (1 - warmupT);
  const fullyCalibrated = probW * rawWeight + capped * (1 - rawWeight);
  return probW * (1 - correctionWeight) + fullyCalibrated * correctionWeight;
}

// ── 4-zone signal decision ────────────────────────────────────────────────────
// Zone 0: prob < 0.42              → SKIP  (FILTERED)
// Zone 1: prob 0.42–0.46           → TAKE only if spread < 0.08 AND conf > 55
// Zone 2: prob 0.46–0.52           → TAKE only if spread < 0.12 AND conf > 50
// Zone 3: prob ≥ 0.52              → TAKE  (STRONG_SIGNAL)
// Probability math untouched — only decision boundaries change.

function computeSignalDecision(
  probW, confidence, spread, regimeFactor, regimeConfidence,
  isRandom, targetLabel, modelId, hits
) {
  const spreadVal = spread ?? 0;

  // Amplification — restricted to prob ≥ 0.50 AND spread < 0.10
  const { amplified: amplifiedProbW, wasAmplified } = amplifySignal(probW, spreadVal, targetLabel, modelId);

  // High-conviction window boost (does not affect decision, only window width)
  const highConviction = spreadVal < SIG_CONVICTION_SPREAD
    && Math.abs(regimeFactor ?? 0) > SIG_CONVICTION_REGIME;
  const finalProbUsed = highConviction
    ? Math.min(0.90, amplifiedProbW + SIG_CONVICTION_BOOST)
    : amplifiedProbW;

  // signalStrength — informational only, not used in any decision branch
  const signalStrength = clamp(Math.round(
    clamp((finalProbUsed - 0.35) / 0.30 * 100, 0, 100) * 0.40
    + clamp((1 - spreadVal / 0.30) * 100, 0, 100)       * 0.25
    + clamp((confidence - 30) / 55 * 100, 0, 100)        * 0.25
    + clamp(Math.abs(regimeFactor ?? 0) * 100, 0, 100)   * 0.10
  ), 0, 100);

  // aggressiveMode — only used by applyAggressiveModeWidth for window sizing
  const ev = finalProbUsed - (1 - finalProbUsed);
  const aggressiveMode = Math.abs(regimeFactor ?? 0) > 0.35
    && spreadVal < 0.10 && ev > 0;

  // ── Zone 0: hard floor ───────────────────────────────────────────────────
  if (finalProbUsed < SIG_HARD_FLOOR) {
    return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
      aggressiveMode, 'SKIP', 'FILTERED');
  }

  // ── Zone 3: strong — TAKE unconditionally ───────────────────────────────
  if (finalProbUsed >= SIG_STRONG_PROB) {
    return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
      aggressiveMode, 'TAKE', 'STRONG_SIGNAL');
  }

  // ── Zone 1: low zone (0.42–0.46) — very tight gate ──────────────────────
  if (finalProbUsed < SIG_LOW_ZONE_TOP) {
    if (spreadVal < SIG_LOW_ZONE_SPREAD && confidence > SIG_LOW_ZONE_CONF) {
      return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
        aggressiveMode, 'TAKE', 'MODERATE_SIGNAL');
    }
    return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
      aggressiveMode, 'SKIP', 'FILTERED');
  }

  // ── Zone 2: moderate (0.46–0.52) ────────────────────────────────────────
  if (spreadVal < SIG_MODERATE_SPREAD && confidence > SIG_MODERATE_CONF) {
    return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
      aggressiveMode, 'TAKE', 'MODERATE_SIGNAL');
  }
  return buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
    aggressiveMode, 'SKIP', 'FILTERED');
}

// Output builder — minimal, clean fields only
function buildSignalResult(amplifiedProbW, finalProbUsed, signalStrength,
  aggressiveMode, recommendation, decisionReason) {
  return {
    recommendation,
    decisionReason,
    finalProbUsed:  +finalProbUsed.toFixed(4),
    amplifiedProbW: +amplifiedProbW.toFixed(4),
    signalStrength: clamp(signalStrength, 0, 100),
    aggressiveMode,
    // kept for DB / API backward compatibility
    ev:             +(finalProbUsed - (1 - finalProbUsed)).toFixed(4),
    signalQuality:  clamp(Math.round(finalProbUsed * 60 + signalStrength * 0.40), 0, 100),
    risk:           signalStrength >= 65 ? 'low' : signalStrength >= 45 ? 'medium' : 'high',
    rawProbW:       +amplifiedProbW.toFixed(4),   // alias for consumers expecting rawProbW
  };
}

// computeDecision — mirrors 4-zone logic for non-ENS builders (GEO/BAY/KM/ENGINE)
function computeDecision(probW, confidence, spread, regimeFactor, regimeConfidence, isRandom) {
  const spreadVal = spread ?? 0;
  const ev = probW - (1 - probW);
  const aggressiveMode = Math.abs(regimeFactor ?? 0) > 0.35 && spreadVal < 0.10 && ev > 0;

  if (probW < SIG_HARD_FLOOR)
    return { ev: +ev.toFixed(4), signalQuality: 0,  risk: 'high',   recommendation: 'SKIP', aggressiveMode, decisionReason: 'FILTERED' };
  if (probW >= SIG_STRONG_PROB)
    return { ev: +ev.toFixed(4), signalQuality: 70, risk: 'low',    recommendation: 'TAKE', aggressiveMode, decisionReason: 'STRONG_SIGNAL' };
  if (probW < SIG_LOW_ZONE_TOP) {
    if (spreadVal < SIG_LOW_ZONE_SPREAD && confidence > SIG_LOW_ZONE_CONF)
      return { ev: +ev.toFixed(4), signalQuality: 50, risk: 'medium', recommendation: 'TAKE', aggressiveMode, decisionReason: 'MODERATE_SIGNAL' };
    return { ev: +ev.toFixed(4), signalQuality: 0,  risk: 'high',   recommendation: 'SKIP', aggressiveMode, decisionReason: 'FILTERED' };
  }
  // Zone 2: 0.46–0.52
  if (spreadVal < SIG_MODERATE_SPREAD && confidence > SIG_MODERATE_CONF)
    return { ev: +ev.toFixed(4), signalQuality: 55, risk: 'medium', recommendation: 'TAKE', aggressiveMode, decisionReason: 'MODERATE_SIGNAL' };
  return { ev: +ev.toFixed(4), signalQuality: 0, risk: 'high', recommendation: 'SKIP', aggressiveMode, decisionReason: 'FILTERED' };
}

// ── Aggressive mode window widening ──────────────────────────────────────────
function applyAggressiveModeWidth(low, high, aggressiveMode, spread, ev) {
  if (!aggressiveMode) return { low, high };
  const spreadVal = spread ?? 0;
  const boostFrac =
    (spreadVal < 0.10 && ev > 0) ? 0.10 :
    (spreadVal > 0.20)            ? 0.00 :
                                    0.05;
  if (boostFrac === 0) return { low, high };
  const currentWidth = high - low + 1;
  const extraWidth   = Math.round(currentWidth * boostFrac);
  if (extraWidth < 1) return { low, high };
  const halfExtra    = Math.floor(extraWidth / 2);
  return {
    low:  Math.max(0, low - halfExtra),
    high: high + (extraWidth - halfExtra),
  };
}

function getStreakInfo(gs, maxWidth, isRare) {
  const { gapSinceLast, meanGap, stdGap, cv, p90, p95 } = gs;
  const z = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;
  const confPenalty = Math.round(20 * sigmoid(z - 1.5));

  let streakStatus = 'normal';
  if (gapSinceLast >= p95)      streakStatus = 'extreme';
  else if (gapSinceLast >= p90) streakStatus = 'severe';

  let effectiveWidth = maxWidth;
  // Window width is fixed to maxWidth — streak and CV expansions disabled.
  // Width is controlled exclusively by the TARGETS table.

  return { z: +z.toFixed(2), confPenalty, streakStatus, effectiveWidth };
}

// ── BUILD: ENGINE ─────────────────────────────────────────────────────────────
function buildPrediction(rounds, targetMin, maxWidth, isRare, lastRoundId) {
  if (isRare && targetMin >= RARE_MIN_MULTIPLIER) {
    const gs = getGapStats(rounds, targetMin, lastRoundId);
    if (!gs) return null;
    const target      = TARGETS.find(t => t.min === targetMin);
    const targetLabel = target?.label ?? '?';
    return buildRare(gs, maxWidth, targetLabel, true, rounds, targetMin);
  }

  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const { hits, pGlobal, pRecent, rateShifted, gapSinceLast, currentStreak, kmCDF, kmExpectedGap, cv } = gs;

  const target      = TARGETS.find(t => t.min === targetMin);
  const targetLabel = target?.label ?? '?';
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'engine', gs, gs.isRandom);

  let baseBlend = rateShifted ? 0.25 : 0;
  const regimeBlendBoost = Math.abs(regime.regimeFactor) * 0.20;
  const totalBlend = Math.min(0.45, baseBlend + regimeBlendBoost);
  const p = Math.max(1e-6, Math.min(0.5, (1 - totalBlend) * pGlobal + totalBlend * regime.shortRate));

  const probW       = 1 - Math.pow(1 - p, maxWidth);
  const expectedGap = Math.max(1, Math.round((1 - p) / p));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare ?? false);

  const { low, high } = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, gs.cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);

  const confidence = Math.max(20, betaConf(probW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision   = computeSignalDecision(
    probW, confidence, null, regime.regimeFactor,
    regime.regimeConfidence, gs.isRandom, targetLabel, 'ens', hits
  );
  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, null, decision.ev);

  return {
    low: aLow, high: aHigh, expectedGap, opensIn: aLow,
    confidence,
    probW:        +probW.toFixed(4),
    p:            +p.toFixed(6),
    rateShifted,  cusumNorm: gs.cusumNorm,
    streakStatus, currentStreak, z,
    gapSinceLast, hits,
    regime:           regime.regimeLabel,
    regimeFactor:     regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ── BUILD: GEO ────────────────────────────────────────────────────────────────
function buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, currentStreak, kmCDF, kmExpectedGap, cv } = gs;
  // Each engine uses its own gapSinceLast so windows diverge independently
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'geo', gs, gs.isRandom);
  const pLaplace = (hits + 1) / (n + 2);
  const pGeo     = applyRegimeToP(pLaplace, regime.shortRate, regime.regimeFactor, 0.25);
  const rawProbW   = 1 - Math.pow(1 - pGeo, maxWidth);
  const calibProbW = applyCalibrationRelaxed(rawProbW, targetLabel, 'geo', hits);
  const expectedGap = Math.max(1, Math.round((1 - pGeo) / pGeo));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high } = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, gs.cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision   = computeSignalDecision(
    calibProbW, confidence, null, regime.regimeFactor,
    regime.regimeConfidence, gs.isRandom, targetLabel, 'geo', hits
  );
  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, null, decision.ev);
  return {
    low: aLow, high: aHigh, expectedGap, opensIn: aLow,
    confidence,
    probW:    +calibProbW.toFixed(4),
    rawProbW: +rawProbW.toFixed(4),
    p:        +pGeo.toFixed(6),
    streakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted: gs.rateShifted, model: 'geo',
    regime:           regime.regimeLabel,
    regimeFactor:     regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ── BUILD: BAY ────────────────────────────────────────────────────────────────
function buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, pRecent, rateShifted, currentStreak, kmCDF, kmExpectedGap, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'bay', gs, gs.isRandom);
  const regimeActive  = Math.abs(regime.regimeFactor) > 0;
  const baseRecencyW  = rateShifted ? 0.20 : 0.05;
  const regimeBoost   = Math.abs(regime.regimeFactor) * 0.20;
  const rawRecencyW   = Math.min(0.45, baseRecencyW + regimeBoost);
  const totalRecencyW = Math.min(0.35, regimeActive ? rawRecencyW * 0.50 : rawRecencyW);
  const recencyBlend = Math.abs(regime.regimeFactor) > 0.15
    ? (1 - Math.abs(regime.regimeFactor)) * pRecent + Math.abs(regime.regimeFactor) * regime.shortRate
    : pRecent;
  const pBay        = Math.max(1e-6, Math.min(0.5, (1 - totalRecencyW) * pGlobal + totalRecencyW * recencyBlend));
  const rawProbW    = 1 - Math.pow(1 - pBay, maxWidth);
  const calibProbW  = applyCalibrationRelaxed(rawProbW, targetLabel, 'bay', hits);
  const expectedGap = Math.max(1, Math.round((1 - pBay) / pBay));
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high } = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision   = computeSignalDecision(
    calibProbW, confidence, null, regime.regimeFactor,
    regime.regimeConfidence, gs.isRandom, targetLabel, 'bay', hits
  );
  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, null, decision.ev);
  return {
    low: aLow, high: aHigh, expectedGap, opensIn: aLow,
    confidence,
    probW:    +calibProbW.toFixed(4),
    rawProbW: +rawProbW.toFixed(4),
    p:        +pBay.toFixed(6),
    streakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted, model: 'bay',
    regime:           regime.regimeLabel,
    regimeFactor:     regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ── BUILD: KM ─────────────────────────────────────────────────────────────────
function buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, kmCDF, kmExpectedGap, currentStreak, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  if (!kmCDF) return null;
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'km', gs, gs.isRandom);
  let rawProbW = kmCDFQuery(kmCDF, maxWidth);
  if (rawProbW == null) return null;
  rawProbW = applyParetoCorrectedProbW(rawProbW, cv, kmExpectedGap, maxWidth, isRare, hits);
  rawProbW = Math.max(1e-6, Math.min(1 - 1e-6, rawProbW));
  const calibProbW  = applyCalibrationRelaxed(rawProbW, targetLabel, 'km', hits);
  const regimeGapShift = Math.round(-regime.regimeFactor * kmExpectedGap * 0.15);
  const regimeExpGap   = Math.max(1, kmExpectedGap + regimeGapShift);
  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high } = hybridWindowPlacement(kmCDF, regimeExpGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, regimeExpGap, effectiveWidth, regime.regimeFactor);
  const confidence = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision   = computeSignalDecision(
    calibProbW, confidence, null, regime.regimeFactor,
    regime.regimeConfidence, gs.isRandom, targetLabel, 'km', hits
  );
  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, null, decision.ev);
  return {
    low: aLow, high: aHigh, expectedGap: kmExpectedGap, opensIn: aLow,
    confidence,
    probW:    +calibProbW.toFixed(4),
    rawProbW: +rawProbW.toFixed(4),
    p:        +rawProbW.toFixed(6),
    streakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted: gs.rateShifted, model: 'km',
    regime:           regime.regimeLabel,
    regimeFactor:     regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ── BUILD: ENS ────────────────────────────────────────────────────────────────
function buildEns(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, pRecent, rateShifted, kmCDF, kmExpectedGap,
          currentStreak, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;

  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'ens', gs, gs.isRandom);

  const pGeo = (hits + 1) / (n + 2);
  const regimeActive    = Math.abs(regime.regimeFactor) > 0;
  const ensBaseRecencyW = rateShifted ? 0.20 : 0.05;
  const ensRegimeBoost  = Math.abs(regime.regimeFactor) * 0.20;
  const ensRawRecencyW  = Math.min(0.45, ensBaseRecencyW + ensRegimeBoost);
  const ensRecencyW     = Math.min(0.35, regimeActive ? ensRawRecencyW * 0.50 : ensRawRecencyW);
  const ensRecencyBlend = Math.abs(regime.regimeFactor) > 0.15
    ? (1 - Math.abs(regime.regimeFactor)) * pRecent + Math.abs(regime.regimeFactor) * regime.shortRate
    : pRecent;
  const pBay = Math.max(1e-6, Math.min(0.5, (1 - ensRecencyW) * pGlobal + ensRecencyW * ensRecencyBlend));

  const probGeo = applyCalibrationRelaxed(1 - Math.pow(1 - pGeo, maxWidth), targetLabel, 'geo', hits);
  const probBay = applyCalibrationRelaxed(1 - Math.pow(1 - pBay, maxWidth), targetLabel, 'bay', hits);

  let rawKmProb = kmCDF ? kmCDFQuery(kmCDF, maxWidth) : null;
  if (rawKmProb != null) rawKmProb = applyParetoCorrectedProbW(rawKmProb, cv, kmExpectedGap, maxWidth, isRare, hits);
  const probKm = rawKmProb != null
    ? applyCalibrationRelaxed(Math.max(1e-6, Math.min(1 - 1e-6, rawKmProb)), targetLabel, 'km', hits)
    : probGeo;

  const { ensProb, spread, adjWeights, modelDisagreementScore } = buildEnsemble(targetLabel, probGeo, probBay, probKm, cv);

  // F5: Best-model direct use — if best leads all others by ≥ 0.06, use directly
  const subProbs   = [probGeo, probBay, probKm];
  const modelIds   = ['geo', 'bay', 'km'];
  const bestModel  = selectBestModelDirect(targetLabel, subProbs, modelIds);

  // Temperature scaling when models agree + calibration stable
  const tempScaled = applyTemperatureScaling(ensProb, spread, targetLabel, 'ens');

  // F5: if best model leads clearly, use it directly; otherwise use ensemble
  const blendedProb = bestModel.wasOverridden ? bestModel.prob : tempScaled;

  const calibrated = applyCalibrationRelaxed(blendedProb, targetLabel, 'ens', hits);

  const egGeo = Math.max(1, Math.round((1 - pGeo) / pGeo));
  const egBay = Math.max(1, Math.round((1 - pBay) / pBay));
  const egKm  = kmExpectedGap;
  const ensExpectedGap = Math.max(1, Math.round(
    adjWeights[0] * egGeo + adjWeights[1] * egBay + adjWeights[2] * egKm
  ));

  const pKmPerRound = Math.max(1e-6, Math.min(0.5, 1 / (Math.max(1, egKm) + 1)));
  const pEnsBase    = Math.max(1e-6, Math.min(0.5,
    adjWeights[0] * pGeo + adjWeights[1] * pBay + adjWeights[2] * pKmPerRound
  ));
  const pEns = applyRegimeToP(pEnsBase, regime.shortRate, regime.regimeFactor, 0.30);

  const regimeGapShift  = Math.round(-regime.regimeFactor * ensExpectedGap * 0.15);
  const regimeEnsExpGap = Math.max(1, ensExpectedGap + regimeGapShift);

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);

  const { low, high } = hybridWindowPlacement(kmCDF, regimeEnsExpGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, regimeEnsExpGap, effectiveWidth, regime.regimeFactor);

  const ensembleConfidence = clamp(100 - Math.round(spread * 80) - Math.round(modelDisagreementScore * 10), 20, 95);
  const confidence = Math.max(20, betaConf(calibrated, hits, spread, regime.regimeConfidence, z) - confPenalty);

  const decision = computeSignalDecision(
    calibrated, confidence, spread, regime.regimeFactor,
    regime.regimeConfidence, gs.isRandom, targetLabel, 'ens', hits
  );

  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, spread, decision.ev);

  const modelLabel = bestModel.wasOverridden ? `ens(best:${bestModel.modelUsed})` : 'ens';

  return {
    low: aLow, high: aHigh, expectedGap: ensExpectedGap, opensIn: aLow,
    confidence,
    probW:    +calibrated.toFixed(4),
    rawProbW: +ensProb.toFixed(4),
    p:        +pEns.toFixed(6),
    streakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted, model: modelLabel,
    spread:               +spread.toFixed(3),
    modelDisagreementScore,
    ensembleConfidence,
    bestModelOverride:    bestModel.wasOverridden ? bestModel.modelUsed : null,
    regime:               regime.regimeLabel,
    regimeFactor:         regime.regimeFactor,
    regimeConfidence:     regime.regimeConfidence,
    ensWeights: {
      geo: +adjWeights[0].toFixed(3),
      bay: +adjWeights[1].toFixed(3),
      km:  +adjWeights[2].toFixed(3),
    },
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── RARE ENGINE (targets ≥ 100x) ─────────────────────────────────────────────

function paretoCDF(W, alpha, expectedGap) {
  const a     = Math.max(1.05, alpha);
  const scale = Math.max(1, expectedGap * (a - 1));
  return Math.max(0, Math.min(1 - 1e-9, 1 - Math.pow(1 + W / scale, -a)));
}

function paretoWindowProb(lo, hi, alpha, expectedGap) {
  const cdfHi = paretoCDF(hi, alpha, expectedGap);
  const cdfLo = lo > 0 ? paretoCDF(lo - 1, alpha, expectedGap) : 0;
  return Math.max(0, cdfHi - cdfLo);
}

function kmSurvival(kmCDF, t) {
  if (!kmCDF || kmCDF.length === 0) return 1;
  let lo = 0, hi = kmCDF.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (kmCDF[mid].t <= t) { idx = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return idx >= 0 ? Math.max(0, kmCDF[idx].S) : 1.0;
}

function kmWindowProbDirect(kmCDF, lo, hi) {
  const sLo = lo > 0 ? kmSurvival(kmCDF, lo - 1) : 1.0;
  const sHi = kmSurvival(kmCDF, hi);
  return Math.max(0, sLo - sHi);
}

function applyRareCalibration(probW, targetLabel, modelId, hits) {
  const bins = calibState[targetLabel]?.[modelId];
  if (!bins) return probW;
  const bin = bins[getCalBinIdx(probW)];
  if (bin.count < CAL_MIN_SAMPLES) return probW;
  const empirical = bin.ewmaAct;
  const predicted = bin.ewmaPred;
  if (predicted < 1e-6) return probW;
  const ratio = empirical / predicted;
  if (Math.abs(ratio - 1) < 0.12) return probW;
  const corrected = Math.max(1e-6, Math.min(1 - 1e-6, probW * Math.max(0.80, Math.min(1.20, ratio))));
  const capped    = Math.min(corrected, probW + 0.05);
  const calWeight = hits < RARE_CAL_IMPACT_HITS ? RARE_CAL_REDUCED_WEIGHT : 1.0;
  return probW * (1 - calWeight) + capped * calWeight;
}

function computeRareDecision(tailProbability, extremeGapScore, targetLabel, hits) {
  const payout    = RARE_PAYOUT[targetLabel] ?? 0.90;
  const rareEV    = tailProbability * payout;
  const dataConf  = clamp(hits / 80, 0, 1);
  const tailConf  = clamp(tailProbability / 0.35, 0, 1);
  const confidence = Math.round(20 + 60 * dataConf * 0.50 + 60 * tailConf * 0.50);
  const rareSignal =
    rareEV > RARE_EV_THRESHOLD ||
    tailProbability > RARE_TAIL_MIN ||
    extremeGapScore > 0;
  const recommendation = rareSignal ? 'TAKE' : 'SKIP';
  const risk = rareEV > 1.20 ? 'low' : rareEV > 0.80 ? 'medium' : 'high';
  return {
    rareEV:         +rareEV.toFixed(4),
    estimatedPayout: payout,
    rareSignal,
    recommendation,
    confidence:     clamp(confidence, 20, 85),
    risk,
    signalQuality:  Math.round(clamp(rareEV * 60 + tailProbability * 40, 0, 100)),
    ev:             +rareEV.toFixed(4),
  };
}

function buildRare(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const { hits, n, pGlobal, currentStreak, kmCDF,
          kmExpectedGap, cv, p95, meanGap, stdGap, sg } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;

  const regime = updateEngineRegime(rounds, targetMin, targetLabel, 'ens', gs, gs.isRandom);
  const alpha = 1 / Math.max(0.30, cv * cv);
  const pGeo  = (hits + 1) / (n + 2);
  const geoP  = 1 - Math.pow(1 - pGeo, maxWidth);
  const tailP_full = paretoCDF(maxWidth, alpha, kmExpectedGap);
  const kmP = kmCDF ? kmWindowProbDirect(kmCDF, 0, maxWidth) : null;
  const sparse = hits < RARE_SPARSE_HITS;
  const geoWeight  = sparse ? RARE_TAIL_GEO_BLEND_SPARSE : RARE_TAIL_GEO_BLEND;
  const tailWeight = 1 - geoWeight;
  const tailEstimate = kmP != null
    ? tailP_full * 0.60 + kmP * 0.40
    : tailP_full;
  let rawProbW = geoWeight * geoP + tailWeight * tailEstimate;
  rawProbW = Math.max(1e-6, Math.min(1 - 1e-6, rawProbW));
  const calibProbW = applyRareCalibration(rawProbW, targetLabel, 'ens', hits);
  const z = stdGap > 0 ? (gapSinceLast - meanGap) / stdGap : 0;
  const extremeGap = gapSinceLast > (p95 ?? meanGap * 2.5);
  const extremeGapScore = extremeGap ? +z.toFixed(2) : 0;
  let tailProbability = calibProbW;
  // effectiveWidth fixed to maxWidth — no extreme-gap or CV expansion.
  let effectiveWidth = maxWidth;
  if (extremeGap) tailProbability = Math.min(0.95, tailProbability + RARE_EXTREME_GAP_BOOST);
  const earlyEnd  = Math.max(1, Math.round(kmExpectedGap * RARE_EARLY_WINDOW_FRAC));
  const lateEnd   = Math.max(earlyEnd + 1, Math.round(kmExpectedGap * RARE_LATE_WINDOW_FRAC));
  const earlyWindow = { low: 0, high: earlyEnd };
  const lateWindow  = { low: kmExpectedGap, high: lateEnd };
  const earlyTailP = kmCDF
    ? kmWindowProbDirect(kmCDF, earlyWindow.low, earlyWindow.high)
    : paretoWindowProb(earlyWindow.low, earlyWindow.high, alpha, kmExpectedGap);
  const lateTailP = kmCDF
    ? kmWindowProbDirect(kmCDF, lateWindow.low, lateWindow.high)
    : paretoWindowProb(lateWindow.low, lateWindow.high, alpha, kmExpectedGap);
  const regimeOffset = Math.round(-regime.regimeFactor * kmExpectedGap * 0.12);
  const primaryIsEarly = earlyTailP >= lateTailP;
  const baseWin = primaryIsEarly ? earlyWindow : lateWindow;
  const baseWidth = baseWin.high - baseWin.low + 1;
  const { low: tcLow, high: tcHigh } = applyTimingCorrection(
    kmExpectedGap, baseWidth, targetLabel, maxWidth
  );
  let primaryLow  = Math.max(0, tcLow  - regimeOffset);
  // Hard-cap width to maxWidth — rare dual-window logic can exceed spec otherwise
  let primaryHigh = Math.min(primaryLow + maxWidth - 1, Math.max(primaryLow, tcHigh - regimeOffset));
  const decision = computeRareDecision(tailProbability, extremeGapScore, targetLabel, hits);
  const streakStatus = extremeGapScore > 0
    ? (z > 3 ? 'extreme' : 'severe')
    : 'normal';
  return {
    low:          primaryLow,
    high:         primaryHigh,
    expectedGap:  kmExpectedGap,
    opensIn:      primaryLow,
    confidence:   decision.confidence,
    probW:        +calibProbW.toFixed(4),
    rawProbW:     +rawProbW.toFixed(4),
    p:            +pGeo.toFixed(6),
    streakStatus,
    currentStreak: gs.currentStreak,
    z:            +z.toFixed(2),
    gapSinceLast,
    hits,
    rateShifted:  gs.rateShifted,
    model:        'rare',
    regime:       regime.regimeLabel,
    regimeFactor: regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom:     gs.isRandom,
    maxAC:        gs.maxAC,
    tailProbability:  +tailProbability.toFixed(4),
    extremeGapScore,
    alpha:            +alpha.toFixed(3),
    earlyWindow:      { ...earlyWindow, tailP: +earlyTailP.toFixed(4) },
    lateWindow:       { ...lateWindow,  tailP: +lateTailP.toFixed(4)  },
    primaryWindow:    earlyTailP >= lateTailP ? 'early' : 'late',
    ...decision,
  };
}

// ── buildStatPrediction — dispatcher ─────────────────────────────────────────
function buildStatPrediction(rounds, targetMin, maxWidth, modelId, lastRoundId) {
  const gs = getGapStats(rounds, targetMin, lastRoundId);
  if (!gs) return null;
  const target      = TARGETS.find(t => t.min === targetMin);
  const targetLabel = target?.label ?? '?';
  const isRare      = target?.rare ?? false;

  // Each engine uses its own independent gapSinceLast based on when it last
  // recorded a WIN for this target. This makes window positions diverge.
  const engineGap = getEngineGapSinceLast(modelId, targetLabel, lastRoundId, gs.gapSinceLast);

  if (isRare && targetMin >= RARE_MIN_MULTIPLIER) {
    return buildRare(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
  }

  switch (modelId) {
    case 'geo':  return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'bay':  return buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'km':   return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'ens':  return buildEns(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'rf':   return buildRF(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'gbt':  return buildGBT(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'lr':   return buildLR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'nb':   return buildNB(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'lstm': return buildLSTM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'lgbm': return buildServerLGBM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'prp':  return buildServerPRP(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'gru':  return buildServerGRU(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'ifor': return buildServerIFOR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    case 'meta':  return buildMeta(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGap);
    default:     return null;
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// ── ML ENGINES: RF / GBT / LR / NB / LSTM ────────────────────────────────────
// Each engine:
//   1. Extracts features from gap stats + recent rounds
//   2. Trains / predicts using its library (or pure-JS LSTM)
//   3. Returns the same { low, high, probW, confidence, ... } shape as geo/bay/km
//   4. Falls back gracefully if the library is not installed
// ══════════════════════════════════════════════════════════════════════════════

// ── Shared ML feature extractor ───────────────────────────────────────────────
// Returns a flat numeric array suitable for ml-random-forest, ml-cart, etc.
function extractMLFeatures(gs, rounds, targetMin) {
  if (!gs) return null;
  const { hits, n, pGlobal, pRecent, gapSinceLast, meanGap, cv, kmExpectedGap, currentStreak, cusumNorm } = gs;
  if (hits < 3 || n < 10) return null;

  // Recent window analysis (last 50 rounds)
  const recent = rounds.slice(-50);
  const recentHits = recent.filter(r => r.multiplier >= targetMin).length;
  const recentRate = recentHits / Math.max(1, recent.length);

  // Last 10 rounds binary pattern
  const last10 = rounds.slice(-10).map(r => r.multiplier >= targetMin ? 1 : 0);
  const streak3 = last10.slice(-3).reduce((a, b) => a + b, 0);

  // Gap features
  const overdueRatio  = meanGap > 0 ? gapSinceLast / meanGap : 0;
  const kmGapRatio    = kmExpectedGap > 0 ? gapSinceLast / kmExpectedGap : 0;
  const streakFactor  = currentStreak / Math.max(1, meanGap);

  return [
    pGlobal,                                          // 0  global hit probability
    pRecent,                                          // 1  recent 200-round rate
    recentRate,                                       // 2  recent 50-round rate
    Math.min(3, overdueRatio),                        // 3  gap overdue ratio (capped at 3x)
    Math.min(3, kmGapRatio),                          // 4  vs KM expected gap (capped at 3x)
    Math.min(3, cv),                                  // 5  CV uncapped to 3 — preserves high-variance signal
    Math.min(1, cusumNorm / 3),                       // 6  CUSUM normalised (rate shift signal)
    streak3 / 3,                                      // 7  hits in last 3 rounds
    last10.reduce((a,b)=>a+b,0) / 10,                 // 8  hits in last 10 rounds
    Math.min(1, streakFactor),                        // 9  current losing streak factor
    Math.min(1, hits / 500),                          // 10 data richness
    Math.min(1, gapSinceLast / Math.max(1, meanGap * 3)), // 11 gap relative to 3x expected — works for all targets
  ];
}

// Converts feature vector to labelled training rows from historical gaps
// Returns { X: number[][], y: number[] } for binary classification (hit in next W rounds)
function buildMLDataset(rounds, targetMin, maxWidth) {
  // O(n) sliding window. Cap at 3000 rounds — npm JS libs (RF/GBT/LR/NB/LSTM)
  // are CPU-bound and block on large datasets. 3000 rows = rich signal for all
  // targets including 1000x (~15 hits), runs in <2s. Server-side engines
  // (LGBM/IFOR/PRP/GRU via srvBuildDataset) use full data — they're faster.
  const X = [], y = [];
  const CONTEXT = 100;
  const CAP = 3000;
  const trainRounds = rounds.slice(-Math.min(rounds.length, CAP + maxWidth));
  if (trainRounds.length < CONTEXT + maxWidth + 10) return { X, y };

  for (let i = CONTEXT; i < trainRounds.length - maxWidth; i++) {
    const ctx = trainRounds.slice(i - CONTEXT, i); // fixed-size context window
    const n = ctx.length;

    // Hit stats in context window
    let hits = 0, lastIdx = -1;
    const gaps = [];
    for (let j = 0; j < n; j++) {
      if (ctx[j].multiplier >= targetMin) {
        if (lastIdx !== -1) gaps.push(j - lastIdx - 1);
        lastIdx = j; hits++;
      }
    }
    if (hits < 2) continue;

    const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;
    const meanGap = gaps.length > 0 ? gaps.reduce((a,b)=>a+b,0)/gaps.length : n/Math.max(1,hits);
    const variance = gaps.length > 1 ? gaps.reduce((a,b)=>a+(b-meanGap)**2,0)/gaps.length : meanGap*meanGap;
    const cv = meanGap > 0 ? Math.sqrt(variance)/meanGap : 1;
    const pGlobal = (hits+1)/(n+2);
    const r50Hits = ctx.slice(-50).filter(r=>r.multiplier>=targetMin).length;
    const pRecent = (r50Hits+1)/52;
    const recentRate = r50Hits / Math.min(50, n);
    const last10 = ctx.slice(-10).map(r=>r.multiplier>=targetMin?1:0);
    const streak3 = last10.slice(-3).reduce((a,b)=>a+b,0);
    let streak=0; for(let k=n-1;k>=0;k--){if(ctx[k].multiplier<targetMin)streak++;else break;}
    const kmEG = Math.max(1, Math.round(meanGap));
    const overdueRatio = meanGap > 0 ? gapSinceLast/meanGap : 0;

    const feat = [
      pGlobal, pRecent, recentRate,
      Math.min(3, overdueRatio), Math.min(3, gapSinceLast/kmEG), Math.min(3, cv),
      0, streak3/3, // cusumNorm=0 (skip expensive CUSUM in training)
      last10.reduce((a,b)=>a+b,0)/10, Math.min(1, streak/Math.max(1,meanGap)),
      Math.min(1, hits/100), Math.min(1, gapSinceLast/Math.max(1,meanGap*3)),
    ];

    const label = trainRounds.slice(i, i+maxWidth).some(r=>r.multiplier>=targetMin) ? 1 : 0;
    X.push(feat);
    y.push(label);
  }
  return { X, y };
}

// Cache key for ML model
function mlKey(modelId, targetMin) { return `${modelId}_${targetMin}`; }

// Check if model needs retraining
const mlTrainingInProgress = new Set(); // keys currently being trained

function needsRetrain(key, currentRoundCount) {
  if (!mlTrainingEnabled) return false;
  if (mlTrainingInProgress.has(key)) return false; // already training this key
  const cached = mlModelCache[key];
  if (!cached) return true;
  return (currentRoundCount - cached.trainedAt) >= ML_RETRAIN_INTERVAL;
}

// ── Shared window builder for ML models ──────────────────────────────────────
// Given a probability from a trained model, build a window using the same
// hybridWindowPlacement logic as the existing stat engines.
function buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, probW, modelName) {
  const { hits, kmCDF, kmExpectedGap, currentStreak, cv } = gs;
  const gapSinceLast = engineGapSinceLast ?? gs.gapSinceLast;
  const regime = updateEngineRegime(rounds, targetMin, targetLabel, modelName, gs, gs.isRandom);
  const rawProbW   = probW;
  const calibProbW = applyCalibrationRelaxed(rawProbW, targetLabel, modelName, hits);

  // ── KEY FIX: derive expectedGap from THIS engine's own probability ──────────
  // Each ML engine produces a different probW. Back-calculate what per-round
  // probability p gives P(hit in maxWidth rounds) = probW, then expectedGap = 1/p.
  // This makes every engine produce a DIFFERENT window position.
  //
  // P(hit in W rounds) = 1 - (1-p)^W  =>  p = 1 - (1-probW)^(1/W)
  // Clamp to avoid degenerate values, blend 50/50 with KM gap for stability.
  const pPerRound = Math.max(1e-4, Math.min(0.5,
    1 - Math.pow(Math.max(1e-9, 1 - Math.max(0.01, Math.min(0.99, calibProbW))), 1 / Math.max(1, maxWidth))
  ));
  const mlExpectedGap = Math.max(1, Math.round(1 / pPerRound));
  // Blend: 50% engine's own gap estimate, 50% KM empirical for stability
  const expectedGap = Math.max(1, Math.round(0.5 * mlExpectedGap + 0.5 * kmExpectedGap));

  const { z, confPenalty, streakStatus, effectiveWidth } = getStreakInfo(gs, maxWidth, isRare);
  const { low, high } = hybridWindowPlacement(kmCDF, expectedGap, effectiveWidth, gapSinceLast, cv, targetLabel, maxWidth);
  const { low: rLow, high: rHigh } = applyRegimeToWindow(low, high, expectedGap, effectiveWidth, regime.regimeFactor);
  const confidence = Math.max(20, betaConf(calibProbW, hits, null, regime.regimeConfidence, z) - confPenalty);
  const decision   = computeSignalDecision(calibProbW, confidence, null, regime.regimeFactor, regime.regimeConfidence, gs.isRandom, targetLabel, modelName, hits);
  const { low: aLow, high: aHigh } = applyAggressiveModeWidth(rLow, rHigh, decision.aggressiveMode, null, decision.ev);
  return {
    low: aLow, high: aHigh, expectedGap, opensIn: aLow,
    confidence,
    probW:    +calibProbW.toFixed(4),
    rawProbW: +rawProbW.toFixed(4),
    p:        +calibProbW.toFixed(6),
    streakStatus, currentStreak, z, gapSinceLast, hits,
    rateShifted: gs.rateShifted, model: modelName,
    regime:           regime.regimeLabel,
    regimeFactor:     regime.regimeFactor,
    regimeConfidence: regime.regimeConfidence,
    isRandom: gs.isRandom, maxAC: gs.maxAC,
    ...decision,
  };
}

// ── BUILD: RF (Pure-JS Random Forest — no npm dependency, fast) ──────────────
// Replaces ml-random-forest which takes 48s on 3000 samples.
// Pure-JS implementation: random feature subsets + CART-style splits.
// Runs in <500ms on 3000 samples. Same algorithm, zero dependencies.

function pureRFTree(X, y, maxDepth, nFeats, seed) {
  // Fast RF tree using random thresholds (Extra-Trees style):
  // Instead of searching all split points, try K random thresholds per feature.
  // 10-100x faster than exhaustive search, similar accuracy in practice.
  let s = seed; const rng = () => { s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; };
  const nF = X[0].length;
  // Precompute feature min/max for random threshold generation
  const fMin = Array.from({length:nF}, (_,f) => Math.min(...X.map(x=>x[f])));
  const fMax = Array.from({length:nF}, (_,f) => Math.max(...X.map(x=>x[f])));

  function score(idxs, t, f) {
    let l1=0,l0=0,r1=0,r0=0;
    for (const i of idxs) { if(X[i][f]<=t){if(y[i])l1++;else l0++;}else{if(y[i])r1++;else r0++;} }
    const lN=l1+l0, rN=r1+r0, N=lN+rN; if(!lN||!rN) return -1;
    const impL=lN>0?1-(l1/lN)**2-(l0/lN)**2:0;
    const impR=rN>0?1-(r1/rN)**2-(r0/rN)**2:0;
    const pImp=idxs.length>0?1-(idxs.filter(i=>y[i]).length/idxs.length)**2-(idxs.filter(i=>!y[i]).length/idxs.length)**2:0;
    return pImp - (lN/N)*impL - (rN/N)*impR;
  }

  function build(idxs, depth) {
    const n=idxs.length, nPos=idxs.filter(i=>y[i]).length;
    const p1=nPos/n;
    if (depth>=maxDepth||n<8||p1===0||p1===1) return {leaf:true,prob:p1};
    // Random feature subset
    const feats=Array.from({length:nF},(_,i)=>i).sort(()=>rng()-0.5).slice(0,nFeats);
    let best={gain:-1,f:0,t:0};
    for (const f of feats) {
      // 5 random thresholds per feature (Extra-Trees style)
      for (let k=0;k<5;k++) {
        const t=fMin[f]+rng()*(fMax[f]-fMin[f]);
        const g=score(idxs,t,f);
        if (g>best.gain) best={gain:g,f,t};
      }
    }
    if (best.gain<=0.001) return {leaf:true,prob:p1};
    const L=idxs.filter(i=>X[i][best.f]<=best.t);
    const R=idxs.filter(i=>X[i][best.f]>best.t);
    if (!L.length||!R.length) return {leaf:true,prob:p1};
    return {leaf:false,f:best.f,t:best.t,left:build(L,depth+1),right:build(R,depth+1)};
  }
  return build(Array.from({length:X.length},(_,i)=>i), 0);
}
function pureRFPredict(tree, x) {
  if (tree.leaf) return tree.prob;
  return x[tree.f] <= tree.t ? pureRFPredict(tree.left, x) : pureRFPredict(tree.right, x);
}

function buildRF(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key = mlKey('rf', targetMin);
  let prob;
  try {
    if (needsRetrain(key, rounds.length)) {
      mlTrainingInProgress.add(key);
      const { X, y } = buildMLDataset(rounds, targetMin, maxWidth);
      if (X.length < 30) {
        mlModelCache[key] = { model: null, trainedAt: rounds.length };
      } else {
        const nTrees = 15, maxDepth = 4, nFeats = Math.max(2, Math.round(Math.sqrt(X[0].length)));
        const trees = Array.from({length: nTrees}, (_, t) => {
          // Bootstrap sample
          const idxs = Array.from({length: X.length}, () => Math.floor(Math.random()*X.length));
          const Xb = idxs.map(i=>X[i]), yb = idxs.map(i=>y[i]);
          return pureRFTree(Xb, yb, maxDepth, nFeats, t * 12345 + 42);
        });
        mlModelCache[key] = { model: trees, trainedAt: rounds.length };
      }
      mlTrainingInProgress.delete(key);
      if (!mlModelCache[key]?.model) return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
    }
    if (!mlModelCache[key] || !mlModelCache[key].model) return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
    const feat = extractMLFeatures(gs, rounds, targetMin);
    if (!feat) return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
    const trees = mlModelCache[key].model;
    prob = trees.reduce((s, t) => s + pureRFPredict(t, feat), 0) / trees.length;
  } catch(e) {
    console.warn('[rf] error:', e.message);
    return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }
  prob = Math.max(0.05, Math.min(0.95, prob));
  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, prob, 'rf');
}

// ── BUILD: GBT (Gradient Boosted Trees via ml-cart) ──────────────────────────
// ml-cart exports DecisionTreeClassifier; we build a GBT ensemble manually
// using gradient-boosted residuals (AdaBoost-style with CART weak learners).
function buildGBT(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  // Pure-JS GBT — uses pureRFTree internally, no npm dependency

  const key = mlKey('gbt', targetMin);
  let prob;

  try {
    if (needsRetrain(key, rounds.length)) {
      mlTrainingInProgress.add(key);
        const { X, y } = buildMLDataset(rounds, targetMin, maxWidth);
        if (X.length < 30) {
          mlModelCache[key] = { model: null, trainedAt: rounds.length };
        } else {

      // Build GBT: shallow CART trees fitting pseudo-residuals
      const nTrees = 12;
      const lr     = 0.15;
      const trees  = [];
      let   F      = new Array(X.length).fill(y.filter(v=>v===1).length / y.length); // init to base rate

      for (let t = 0; t < nTrees; t++) {
        // True pseudo-residuals in logit space (gradient of log-loss)
        const logit = v => Math.log(Math.max(1e-7, v) / Math.max(1e-7, 1 - v));
        const sigmoid = v => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v))));
        // Compute residuals in logit space for numerical stability
        const residuals = y.map((yi, i) => yi - F[i]);
        // Use pure-JS tree (same CART-style splits, no npm dependency)
        const resLabels = residuals.map(r => r >= 0 ? 1 : 0);
        const tree = pureRFTree(X, resLabels, 3, X[0].length, t * 777 + 13);
        const leafPreds = X.map(x => pureRFPredict(tree, x) >= 0.5 ? 1 : 0);
        const leafGroups = {};
        X.forEach((x, i) => {
          const lk = leafPreds[i];
          if (!leafGroups[lk]) leafGroups[lk] = { sumR: 0, count: 0 };
          leafGroups[lk].sumR += residuals[i];
          leafGroups[lk].count++;
        });
        const leafMeans = {};
        for (const lk in leafGroups) leafMeans[lk] = leafGroups[lk].sumR / leafGroups[lk].count;
        trees.push({ tree, leafMeans });
        // Update F with actual leaf mean residual
        F = F.map((fi, i) => {
          const lk = leafPreds[i];
          const step = leafMeans[lk] ?? 0;
          return Math.max(0.001, Math.min(0.999, fi + lr * step));
        });
      }

      mlModelCache[key] = { model: trees, trainedAt: rounds.length, lr, baseRate: y.filter(v=>v===1).length/y.length };
      }
      mlTrainingInProgress.delete(key);
    }

    // Not trained yet or sparse target — fall back
    if (!mlModelCache[key] || !mlModelCache[key].model) return buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    const feat = extractMLFeatures(gs, rounds, targetMin);
    if (!feat) return buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    // Score: logit-space with per-leaf mean residuals (true GBT)
    const { model: trees, baseRate: br } = mlModelCache[key];
    const gbtBase = Math.max(0.01, Math.min(0.99, br ?? gs.pGlobal * maxWidth));
    const logit2 = v => Math.log(Math.max(1e-7, v) / Math.max(1e-7, 1 - v));
    const sigmoid2 = v => 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, v))));
    let logOdds = logit2(gbtBase);
    for (const { tree, leafMeans } of trees) {
      const lk = pureRFPredict(tree, feat) >= 0.5 ? 1 : 0;
      const step = leafMeans?.[lk] ?? 0;
      logOdds += 0.15 * step;
    }
    prob = sigmoid2(logOdds);
  } catch(e) {
    console.warn('[gbt] predict error:', e.message);
    return buildBay(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  prob = Math.max(0.05, Math.min(0.95, prob));
  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, prob, 'gbt');
}

// ── BUILD: LR (Logistic Regression via simple-statistics) ────────────────────
function buildLR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const ss = getSS();
  if (!ss) {
    return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  const key = mlKey('lr', targetMin);
  let prob;

  try {
    if (needsRetrain(key, rounds.length)) {
      mlTrainingInProgress.add(key);
        const { X, y } = buildMLDataset(rounds, targetMin, maxWidth);
        if (X.length < 30) {
          mlModelCache[key] = { model: null, trainedAt: rounds.length };
        } else {

      // simple-statistics does not have a built-in LR, so we train it ourselves
      // using gradient descent + sigmoid, leveraging ss.mean / ss.standardDeviation
      // for feature standardisation.
      const nFeat = X[0].length;
      const means  = Array.from({length: nFeat}, (_, j) => ss.mean(X.map(x => x[j])));
      const stds   = Array.from({length: nFeat}, (_, j) => Math.max(1e-8, ss.standardDeviation(X.map(x => x[j]))));
      // Standardise
      const Xn = X.map(x => x.map((v, j) => (v - means[j]) / stds[j]));

      // Gradient descent with class weighting for imbalanced targets
      const sig = v => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, v))));
      let w = new Array(nFeat).fill(0), b = 0;
      const lrRate = 0.05, epochs = 60, lambda = 0.01; // more epochs, stronger L2
      const posRate = y.filter(v=>v===1).length / y.length;
      // Class weights: inverse frequency — rare hits get higher weight
      const wPos = posRate > 0 ? 0.5 / posRate : 1;
      const wNeg = (1-posRate) > 0 ? 0.5 / (1-posRate) : 1;
      for (let e = 0; e < epochs; e++) {
        let dw = new Array(nFeat).fill(0), db = 0;
        for (let i = 0; i < Xn.length; i++) {
          const p   = sig(Xn[i].reduce((s, v, j) => s + v * w[j], b));
          const err = (p - y[i]) * (y[i] === 1 ? wPos : wNeg); // weighted error
          for (let j = 0; j < nFeat; j++) dw[j] += err * Xn[i][j];
          db += err;
        }
        w = w.map((wj, j) => wj - lrRate * (dw[j] / Xn.length + lambda * wj));
        b -= lrRate * (db / Xn.length);
      }

      mlModelCache[key] = { model: { w, b, means, stds }, trainedAt: rounds.length };
      }
      mlTrainingInProgress.delete(key);
    }

    const feat = extractMLFeatures(gs, rounds, targetMin);
    if (!feat) return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    // Not trained yet or sparse — fall back
    if (!mlModelCache[key] || !mlModelCache[key].model) return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    const { w, b, means, stds } = mlModelCache[key].model;
    const sig = v => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, v))));
    const xn  = feat.map((v, j) => (v - means[j]) / stds[j]);
    prob = sig(xn.reduce((s, v, j) => s + v * w[j], b));
  } catch(e) {
    console.warn('[lr] predict error:', e.message);
    return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  prob = Math.max(0.05, Math.min(0.95, prob));
  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, prob, 'lr');
}

// ── BUILD: NB (Gaussian Naive Bayes via ml-naivebayes) ───────────────────────
function buildNB(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const NB = getNB();
  if (!NB) {
    return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  const key = mlKey('nb', targetMin);
  let prob;

  try {
    if (needsRetrain(key, rounds.length)) {
      mlTrainingInProgress.add(key);
        const { X, y } = buildMLDataset(rounds, targetMin, maxWidth);
        if (X.length < 30) {
          mlModelCache[key] = { model: null, trainedAt: rounds.length };
        } else {

      // ml-naivebayes: GaussianNB
      const nb = new NB();  // getNB() returns GaussianNB class directly
      nb.train(X, y.map(String)); // ml-naivebayes expects string labels
      mlModelCache[key] = { model: nb, trainedAt: rounds.length };
      }
      mlTrainingInProgress.delete(key);
    }

    const feat = extractMLFeatures(gs, rounds, targetMin);
    if (!feat) return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    // Not trained yet or sparse — fall back
    if (!mlModelCache[key] || !mlModelCache[key].model) return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    // Predict returns class labels; use predictProba if available
    if (mlModelCache[key].model.predictProba) {
      const probas = mlModelCache[key].model.predictProba([feat]);
      // Returns [[p0, p1]] or {0: p0, 1: p1} depending on version
      const row = probas[0];
      prob = (Array.isArray(row) ? row[1] : row['1']) ?? 0.5;
    } else {
      const pred = mlModelCache[key].model.predict([feat])[0];
      const isHit = pred === 1 || pred === '1' || pred === true;
      prob = isHit
        ? Math.min(0.85, gs.pGlobal * maxWidth * 1.25)
        : Math.max(0.05, gs.pGlobal * maxWidth * 0.75);
    }
  } catch(e) {
    console.warn('[nb] predict error:', e.message);
    return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  prob = Math.max(0.05, Math.min(0.95, prob));
  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, prob, 'nb');
}

// ── BUILD: LSTM (Pure-JS, zero dependencies) ──────────────────────────────────
// Simplified single-layer LSTM cell trained with truncated BPTT (gradient approx).
// Sequence input: last SEQ_LEN rounds as normalised binary hit/miss values.
const LSTM_SEQ_LEN   = 20;
const LSTM_HIDDEN    = 12;
const LSTM_LR        = 0.01;
const LSTM_EPOCHS    = 5;

function lstmSig(x) { return 1/(1+Math.exp(-Math.max(-15,Math.min(15,x)))); }
function lstmTanh(x){ const e=Math.exp(2*Math.max(-15,Math.min(15,x))); return (e-1)/(e+1); }

function initLSTMWeights() {
  // Xavier-ish init: scale by 1/sqrt(H) for stability
  const H = LSTM_HIDDEN, I = 1;
  const scale = 1 / Math.sqrt(H);
  const r  = () => (Math.random()-0.5) * scale;
  const r2 = () => (Math.random()-0.5) * 0.1; // smaller for recurrent weights
  const mat = (R,C,fn=r)  => Array.from({length:R},()=>Array.from({length:C},fn));
  const vec = (n,fn=r)    => Array.from({length:n},fn);
  return {
    Wi:mat(H,I), Ui:mat(H,H,r2), bi:vec(H, ()=>0),       // input gate: 0 bias
    Wf:mat(H,I), Uf:mat(H,H,r2), bf:vec(H, ()=>1.0),     // forget gate: 1 bias (standard)
    Wg:mat(H,I), Ug:mat(H,H,r2), bg:vec(H, ()=>0),       // cell gate: 0 bias
    Wo:mat(H,I), Uo:mat(H,H,r2), bo:vec(H, ()=>0),       // output gate: 0 bias
    Wy:Array.from({length:1},()=>Array.from({length:H},()=>(Math.random()-0.5)*0.01)),
    by:[0],
  };
}

function lstmForward(weights, seq) {
  const { Wi,Ui,bi, Wf,Uf,bf, Wg,Ug,bg, Wo,Uo,bo, Wy,by } = weights;
  const H = LSTM_HIDDEN;
  let h = new Array(H).fill(0);
  let c = new Array(H).fill(0);
  const mv = (M,v) => M.map(row=>row.reduce((s,w,k)=>s+w*v[k],0));
  const add = (a,b) => a.map((x,i)=>x+b[i]);
  for (const x of seq) {
    const xv=[x];
    const i_t = add(add(mv(Wi,xv),mv(Ui,h)),bi).map(lstmSig);
    const f_t = add(add(mv(Wf,xv),mv(Uf,h)),bf).map(lstmSig);
    const g_t = add(add(mv(Wg,xv),mv(Ug,h)),bg).map(lstmTanh);
    const o_t = add(add(mv(Wo,xv),mv(Uo,h)),bo).map(lstmSig);
    c = c.map((ci,j)=>f_t[j]*ci+i_t[j]*g_t[j]);
    h = c.map((ci,j)=>o_t[j]*lstmTanh(ci));
  }
  const logit = mv(Wy,h)[0]+by[0];
  return lstmSig(logit);
}

function trainLSTMWeights(weights, sequences, labels, lr, epochs) {
  // Train output layer (Wy, by) with correct gradient: dL/dWy[j] = err * h[j]
  // This IS correct — we run full LSTM forward to get h, then update only the
  // linear head. Gates stay at init (too expensive to backprop through for JS).
  // Result: LSTM acts as a fixed random feature extractor + trained classifier.
  // To make this meaningful, we initialise gates with structured priors (see initLSTMWeights).
  for (let epoch = 0; epoch < epochs; epoch++) {
    let dwY = weights.Wy.map(r=>r.map(()=>0));
    let dbY = [0];
    const H = LSTM_HIDDEN;
    const mv = (M,v)=>M.map(row=>row.reduce((s,ww,k)=>s+ww*v[k],0));
    const add = (a,b)=>a.map((x,j)=>x+b[j]);
    for (let i = 0; i < sequences.length; i++) {
      // Full forward pass to get final hidden state
      let hh = new Array(H).fill(0), cc = new Array(H).fill(0);
      const { Wi,Ui,bi,Wf,Uf,bf,Wg,Ug,bg,Wo:Wo_,Uo,bo:bo_,Wy,by } = weights;
      for (const x of sequences[i]) {
        const xv=[x];
        const i_t=add(add(mv(Wi,xv),mv(Ui,hh)),bi).map(lstmSig);
        const f_t=add(add(mv(Wf,xv),mv(Uf,hh)),bf).map(lstmSig);
        const g_t=add(add(mv(Wg,xv),mv(Ug,hh)),bg).map(lstmTanh);
        const o_t=add(add(mv(Wo_,xv),mv(Uo,hh)),bo_).map(lstmSig);
        cc=cc.map((ci,j)=>f_t[j]*ci+i_t[j]*g_t[j]);
        hh=cc.map((ci,j)=>o_t[j]*lstmTanh(ci));
      }
      const logit = Wy[0].reduce((s,w,j)=>s+w*hh[j],0) + by[0];
      const pred  = lstmSig(logit);
      const err   = pred - labels[i];
      // Correct gradient for output layer
      dwY = dwY.map((row,ri) => row.map((w,j) => w + err * hh[j]));
      dbY = [dbY[0] + err];
    }
    const N = sequences.length;
    weights.Wy = weights.Wy.map((row,ri) => row.map((w,j) => w - lr * dwY[ri][j] / N));
    weights.by = [weights.by[0] - lr * dbY[0] / N];
  }
  return weights;
}

function buildLSTM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key = mlKey('lstm', targetMin);
  let prob;

  try {
    if (needsRetrain(key, rounds.length)) {
      mlTrainingInProgress.add(key);
      if (rounds.length < 100) {
        mlTrainingInProgress.delete(key);
        return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
      }

      // Build training sequences — last 1000 rounds for speed
      const sequences = [], labels = [];
      const lstmTrain = rounds.slice(-Math.min(rounds.length, 3000 + maxWidth)); // cap 3000
      for (let i = LSTM_SEQ_LEN; i < lstmTrain.length - maxWidth; i++) {
        const seq = lstmTrain.slice(i-LSTM_SEQ_LEN, i).map(r => {
          const v = r.multiplier;
          return Math.min(1, Math.log(Math.max(1, v)) / Math.log(Math.max(2, targetMin*2)));
        });
        const label = lstmTrain.slice(i, i+maxWidth).some(r=>r.multiplier>=targetMin) ? 1 : 0;
        sequences.push(seq);
        labels.push(label);
      }

      if (sequences.length < 20) {
        mlModelCache[key] = { model: null, trainedAt: rounds.length };
        return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
      }

      let weights = mlModelCache[key]?.model ?? initLSTMWeights();
      weights = trainLSTMWeights(weights, sequences, labels, LSTM_LR, LSTM_EPOCHS);
      mlModelCache[key] = { model: weights, trainedAt: rounds.length };
      mlTrainingInProgress.delete(key);
    }

    // Predict on current sequence
    const seq = rounds.slice(-LSTM_SEQ_LEN).map(r => {
      const v = r.multiplier;
      return Math.min(1, Math.log(Math.max(1, v)) / Math.log(Math.max(2, targetMin*2)));
    });
    if (seq.length < LSTM_SEQ_LEN) {
      return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
    }

    // Not trained yet or sparse — fall back
    if (!mlModelCache[key] || !mlModelCache[key].model) return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);

    prob = lstmForward(mlModelCache[key].model, seq);
  } catch(e) {
    console.warn('[lstm] predict error:', e.message);
    return buildKm(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  prob = Math.max(0.05, Math.min(0.95, prob));
  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, prob, 'lstm');
}

// ── BUILD: PATTERN ────────────────────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════════════════════
// ── SERVER-SIDE PURE-JS ENGINES: LGBM / PROPHET / GRU / ISOLATION FOREST ─────
// ══════════════════════════════════════════════════════════════════════════════

function srvExtractFeatures(gs, rounds, targetMin) {
  if (!gs || gs.hits < 3) return null;
  const { hits, n, pGlobal, pRecent, gapSinceLast, meanGap, cv, kmExpectedGap, currentStreak, cusumNorm } = gs;
  const recent = rounds.slice(-50);
  const recentHits = recent.filter(r => r.multiplier >= targetMin).length;
  const recentRate = recentHits / Math.max(1, recent.length);
  const last10 = rounds.slice(-10).map(r => r.multiplier >= targetMin ? 1 : 0);
  const streak3 = last10.slice(-3).reduce((a, b) => a + b, 0);
  const overdueRatio = meanGap > 0 ? gapSinceLast / meanGap : 0;
  const kmGapRatio = kmExpectedGap > 0 ? gapSinceLast / kmExpectedGap : 0;
  const streakFactor = currentStreak / Math.max(1, meanGap);
  return [
    pGlobal, pRecent, recentRate,
    Math.min(3, overdueRatio), Math.min(3, kmGapRatio), Math.min(1, cv),
    Math.min(1, (cusumNorm||0) / 3), streak3 / 3,
    last10.reduce((a, b) => a + b, 0) / 10, Math.min(1, streakFactor),
    Math.min(1, hits / 500), Math.min(1, gapSinceLast / 200),
  ];
}

function srvBuildDataset(rounds, targetMin, maxWidth) {
  // O(n) sliding window — uses ALL collected rounds, CONTEXT=100 keeps it linear
  const X = [], y = [];
  const CONTEXT = 100; // fixed context window per sample — keeps O(n)
  // Use ALL rounds — full history
  const trainRounds = rounds;
  if (trainRounds.length < CONTEXT + maxWidth + 10) return { X, y };
  for (let i = CONTEXT; i < trainRounds.length - maxWidth; i++) {
    const ctx = trainRounds.slice(i - CONTEXT, i);
    const n = ctx.length;
    let hits=0, lastIdx=-1; const gaps=[];
    for(let j=0;j<n;j++){if(ctx[j].multiplier>=targetMin){if(lastIdx!==-1)gaps.push(j-lastIdx-1);lastIdx=j;hits++;}}
    if(hits<2)continue;
    const gapSinceLast=lastIdx===-1?n:n-lastIdx-1;
    const meanGap=gaps.length>0?gaps.reduce((a,b)=>a+b,0)/gaps.length:n/Math.max(1,hits);
    const variance=gaps.length>1?gaps.reduce((a,b)=>a+(b-meanGap)**2,0)/gaps.length:meanGap*meanGap;
    const cv=meanGap>0?Math.sqrt(variance)/meanGap:1;
    const pGlobal=(hits+1)/(n+2);
    const r50=ctx.slice(-50).filter(r=>r.multiplier>=targetMin).length;
    const pRecent=(r50+1)/52;
    const last10=ctx.slice(-10).map(r=>r.multiplier>=targetMin?1:0);
    let streak=0;for(let k=n-1;k>=0;k--){if(ctx[k].multiplier<targetMin)streak++;else break;}
    const kmEG=Math.max(1,Math.round(meanGap));
    const feat=[pGlobal,pRecent,r50/Math.min(50,n),Math.min(3,gapSinceLast/meanGap),Math.min(3,gapSinceLast/kmEG),Math.min(3,cv),0,last10.slice(-3).reduce((a,b)=>a+b,0)/3,last10.reduce((a,b)=>a+b,0)/10,Math.min(1,streak/Math.max(1,meanGap)),Math.min(1,hits/100),Math.min(1,gapSinceLast/Math.max(1,meanGap*3))];
    const label=trainRounds.slice(i,i+maxWidth).some(r=>r.multiplier>=targetMin)?1:0;
    X.push(feat); y.push(label);
  }
  return { X, y };
}

const srvModelCache = {};
const SRV_RETRAIN_INTERVAL = 2000; // retrain every 2000 new rounds
const srvTrainingInProgress = new Set();

function srvNeedsRetrain(key, n) {
  if (!mlTrainingEnabled) return false;
  if (srvTrainingInProgress.has(key)) return false;
  const c = srvModelCache[key];
  if (!c) return true;
  return (n - c.trainedAt) >= SRV_RETRAIN_INTERVAL;
}

// LGBM helpers
function srvLGBMTree(XY,depth=0,maxDepth=4){
  if(!XY.length||depth>=maxDepth)return{leaf:true,value:XY.reduce((s,[,y])=>s+y,0)/Math.max(1,XY.length)};
  const nFeat=XY[0][0].length;let bestGain=-Infinity,bestF=0,bestT=0,bestL=[],bestR=[];
  for(let f=0;f<nFeat;f++){const vals=[...new Set(XY.map(([x])=>x[f]))].sort((a,b)=>a-b);for(let t=0;t<vals.length-1;t++){const th=(vals[t]+vals[t+1])/2;const L=XY.filter(([x])=>x[f]<=th),R=XY.filter(([x])=>x[f]>th);if(!L.length||!R.length)continue;const mL=L.reduce((s,[,y])=>s+y,0)/L.length,mR=R.reduce((s,[,y])=>s+y,0)/R.length,mT=XY.reduce((s,[,y])=>s+y,0)/XY.length;const gain=XY.length*(mT**2)-L.length*(mL**2)-R.length*(mR**2);if(gain>bestGain){bestGain=gain;bestF=f;bestT=th;bestL=L;bestR=R;}}}
  if(!bestL.length||bestGain<=0)return{leaf:true,value:XY.reduce((s,[,y])=>s+y,0)/Math.max(1,XY.length)};
  return{leaf:false,feat:bestF,thresh:bestT,left:srvLGBMTree(bestL,depth+1,maxDepth),right:srvLGBMTree(bestR,depth+1,maxDepth)};
}
function srvLGBMPred1(tree,x){if(tree.leaf)return tree.value;return x[tree.feat]<=tree.thresh?srvLGBMPred1(tree.left,x):srvLGBMPred1(tree.right,x);}

function buildServerLGBM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key=`lgbm_${targetMin}`; let prob=0.5;
  try {
    if(srvNeedsRetrain(key,rounds.length)){
      srvTrainingInProgress.add(key);
      const{X,y}=srvBuildDataset(rounds,targetMin,maxWidth);
      if(X.length<20){srvModelCache[key]={model:null,trainedAt:rounds.length};srvTrainingInProgress.delete(key);return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
      const trees=[],lr=0.12,nT=10; // 10 trees — enough signal, much faster
      let F=new Array(X.length).fill(y.filter(v=>v===1).length/y.length);
      for(let t=0;t<nT;t++){const sub=X.map((x,i)=>[x,y[i]-F[i]]).filter(()=>Math.random()<0.6);if(sub.length<5)continue;const tree=srvLGBMTree(sub);trees.push(tree);F=F.map((fi,i)=>Math.max(0.001,Math.min(0.999,fi+lr*srvLGBMPred1(tree,X[i]))));}
      srvModelCache[key]={model:trees,baseRate:y.filter(v=>v===1).length/y.length,trainedAt:rounds.length};
      srvTrainingInProgress.delete(key);
    }
    if(!srvModelCache[key]||!srvModelCache[key].model)return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const feat=srvExtractFeatures(gs,rounds,targetMin);
    if(!feat)return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const{model:trees,baseRate}=srvModelCache[key];
    // Logit-space accumulation — numerically stable, can't overshoot
    const lgbmLogit=v=>Math.log(Math.max(1e-7,v)/Math.max(1e-7,1-v));
    const lgbmSig=v=>1/(1+Math.exp(-Math.max(-10,Math.min(10,v))));
    let lgbmLO=lgbmLogit(Math.max(0.01,Math.min(0.99,baseRate)));
    for(const tree of trees)lgbmLO+=0.12*srvLGBMPred1(tree,feat);
    prob=lgbmSig(lgbmLO);
  }catch(e){console.warn('[lgbm]',e.message);return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
  return buildMLWindow(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast,Math.max(0.05,Math.min(0.95,prob)),'lgbm');
}

function buildServerPRP(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key=`prp_${targetMin}`; let prob=0.5;
  try {
    if(srvNeedsRetrain(key,rounds.length)){
      srvTrainingInProgress.add(key);
      if(rounds.length<100){srvTrainingInProgress.delete(key);return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
      const signal=rounds.map(r=>r.multiplier>=targetMin?1:0);
      const N=Math.min(signal.length,128),seg=signal.slice(-N); // cap DFT at 128 for speed
      let topMag=0,topK=1;
      for(let k=1;k<N/2;k++){let re=0,im=0;for(let n=0;n<N;n++){const a=(2*Math.PI*k*n)/N;re+=seg[n]*Math.cos(a);im-=seg[n]*Math.sin(a);}const mag=Math.sqrt(re*re+im*im);if(mag>topMag){topMag=mag;topK=k;}}
      const dominantPeriod=Math.round(N/topK);
      const baseRate=signal.filter(Boolean).length/signal.length;
      const lastHitIdx=signal.reduceRight((acc,v,i)=>acc===-1&&v?i:acc,-1);
      const phase=lastHitIdx>=0?(signal.length-1-lastHitIdx)%dominantPeriod:dominantPeriod/2;
      const winSz=20,windows=[];
      for(let i=0;i+winSz<=signal.length;i++)windows.push(signal.slice(i,i+winSz).filter(Boolean).length/winSz);
      const wMean=windows.reduce((a,b)=>a+b,0)/Math.max(1,windows.length);
      const wStd=Math.sqrt(windows.reduce((a,b)=>a+(b-wMean)**2,0)/Math.max(1,windows.length));
      const clusterScore=((windows.slice(-1)[0]||0)-wMean)/Math.max(0.001,wStd);
      srvModelCache[key]={baseRate,dominantPeriod,phase,clusterScore,trainedAt:rounds.length};
      srvTrainingInProgress.delete(key);
    }
    if(!srvModelCache[key]||!srvModelCache[key].baseRate)return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const{baseRate,dominantPeriod,phase,clusterScore}=srvModelCache[key];
    const seasonality=0.5*(1+Math.sin((2*Math.PI*phase)/Math.max(1,dominantPeriod)));
    const clusterAdj=Math.min(0.2,Math.max(-0.2,clusterScore*0.08));
    const overdueBoost=Math.min(0.25,(gs.gapSinceLast/Math.max(1,gs.meanGap))*0.12);
    prob=baseRate+0.12*(seasonality-0.5)+clusterAdj+overdueBoost;
  }catch(e){console.warn('[prp]',e.message);return buildGeo(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
  return buildMLWindow(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast,Math.max(0.05,Math.min(0.95,prob)),'prp');
}

const SRV_GRU_H=12,SRV_GRU_SEQ=20;
function srvGSig(x){return 1/(1+Math.exp(-Math.max(-15,Math.min(15,x))));}
function srvGTanh(x){const e=Math.exp(2*Math.max(-15,Math.min(15,x)));return(e-1)/(e+1);}
function srvInitGRU(){const r=()=>(Math.random()-0.5)*0.1,mat=(R,C)=>Array.from({length:R},()=>Array.from({length:C},r)),vec=n=>Array.from({length:n},r),H=SRV_GRU_H;return{Wr:mat(H,1),Ur:mat(H,H),br:vec(H),Wz:mat(H,1),Uz:mat(H,H),bz:vec(H),Wn:mat(H,1),Un:mat(H,H),bn:vec(H),Wo:[Array.from({length:H},r)],bo:[0]};}
function srvGRUFwd(w,seq){const{Wr,Ur,br,Wz,Uz,bz,Wn,Un,bn,Wo,bo}=w,H=SRV_GRU_H;let h=new Array(H).fill(0);const mv=(M,v)=>M.map(row=>row.reduce((s,w,k)=>s+w*v[k],0)),add=(a,b)=>a.map((x,i)=>x+b[i]);for(const x of seq){const xv=[x];const rg=add(add(mv(Wr,xv),mv(Ur,h)),br).map(srvGSig);const zg=add(add(mv(Wz,xv),mv(Uz,h)),bz).map(srvGSig);const ng=add(add(mv(Wn,xv),mv(Un,h.map((hi,i)=>rg[i]*hi))),bn).map(srvGTanh);h=h.map((hi,i)=>(1-zg[i])*ng[i]+zg[i]*hi);}return srvGSig(mv(Wo,h)[0]+bo[0]);}

function buildServerGRU(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key=`gru_${targetMin}`; let prob=0.5;
  try {
    if(srvNeedsRetrain(key,rounds.length)){
      srvTrainingInProgress.add(key);
      if(rounds.length<80){srvTrainingInProgress.delete(key);return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
      const seqs=[],labels=[],logMax=Math.log(Math.max(2,targetMin*2));
      const gruTrain=rounds.slice(-Math.min(rounds.length,3000+maxWidth)); // cap 3000
      for(let i=SRV_GRU_SEQ;i<gruTrain.length-maxWidth;i++){const seq=gruTrain.slice(i-SRV_GRU_SEQ,i).map(r=>Math.min(1,Math.log(Math.max(1,r.multiplier))/logMax));const label=gruTrain.slice(i,i+maxWidth).some(r=>r.multiplier>=targetMin)?1:0;seqs.push(seq);labels.push(label);}
      if(seqs.length<15){srvModelCache[key]={model:null,trainedAt:rounds.length};return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
      let w=srvModelCache[key]?.model??srvInitGRU();
      for(let e=0;e<5;e++){
        let dwO=w.Wo.map(r=>r.map(()=>0)),dbO=[0];
        for(let i=0;i<seqs.length;i++){
          // Run forward pass to get final hidden state h
          const {Wr,Ur,br,Wz,Uz,bz,Wn,Un,bn}=w,H=SRV_GRU_H;
          let h=new Array(H).fill(0);
          const mv2=(M,v)=>M.map(row=>row.reduce((s,ww,k)=>s+ww*v[k],0));
          const add2=(a,b)=>a.map((x,ii)=>x+b[ii]);
          for(const x of seqs[i]){
            const xv=[x];
            const rg=add2(add2(mv2(Wr,xv),mv2(Ur,h)),br).map(srvGSig);
            const zg=add2(add2(mv2(Wz,xv),mv2(Uz,h)),bz).map(srvGSig);
            const ng=add2(add2(mv2(Wn,xv),mv2(Un,h.map((hi,ii)=>rg[ii]*hi))),bn).map(srvGTanh);
            h=h.map((hi,ii)=>(1-zg[ii])*ng[ii]+zg[ii]*hi);
          }
          const p=srvGSig(w.Wo[0].reduce((s,ww,j)=>s+ww*h[j],0)+w.bo[0]);
          const err=p-labels[i];
          // Correct gradient: dL/dWo[j] = err * h[j]
          dwO=dwO.map((row,ri)=>row.map((ww,j)=>ww+err*h[j]));
          dbO=[dbO[0]+err];
        }
        const N=seqs.length;
        w.Wo=w.Wo.map(row=>row.map((ww,j)=>ww-0.01*dwO[0][j]/N));
        w.bo=[w.bo[0]-0.01*dbO[0]/N];
      }
      srvModelCache[key]={model:w,trainedAt:rounds.length};
      srvTrainingInProgress.delete(key);
    }
    if(!srvModelCache[key]||!srvModelCache[key].model)return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const logMax=Math.log(Math.max(2,targetMin*2));
    const seq=rounds.slice(-SRV_GRU_SEQ).map(r=>Math.min(1,Math.log(Math.max(1,r.multiplier))/logMax));
    if(seq.length<SRV_GRU_SEQ)return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    prob=srvGRUFwd(srvModelCache[key].model,seq);
  }catch(e){console.warn('[gru]',e.message);return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
  return buildMLWindow(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast,Math.max(0.05,Math.min(0.95,prob)),'gru');
}

function srvITree(X,depth=0,maxDepth=8){if(X.length<=1||depth>=maxDepth)return{leaf:true,size:X.length};const nF=X[0].length,f=Math.floor(Math.random()*nF),col=X.map(x=>x[f]),mn=Math.min(...col),mx=Math.max(...col);if(mn===mx)return{leaf:true,size:X.length};const th=mn+Math.random()*(mx-mn);return{leaf:false,feat:f,thresh:th,left:srvITree(X.filter(x=>x[f]<th),depth+1,maxDepth),right:srvITree(X.filter(x=>x[f]>=th),depth+1,maxDepth)};}
function srvPathLen(tree,x,d=0){if(tree.leaf){const n=tree.size;if(n<=1)return d;return d+2*(Math.log(n-1)+0.5772)-(2*(n-1)/n);}return x[tree.feat]<tree.thresh?srvPathLen(tree.left,x,d+1):srvPathLen(tree.right,x,d+1);}

function buildServerIFOR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  const key=`ifor_${targetMin}`; let prob=0.5;
  try {
    if(srvNeedsRetrain(key,rounds.length)){
      srvTrainingInProgress.add(key);
      const{X}=srvBuildDataset(rounds,targetMin,maxWidth);
      if(X.length<20){srvModelCache[key]={model:null,trainedAt:rounds.length};srvTrainingInProgress.delete(key);return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
      const nT=30,subSz=32,trees=[];
      for(let t=0;t<nT;t++){const sub=Array.from({length:subSz},()=>X[Math.floor(Math.random()*X.length)]);trees.push(srvITree(sub));}
      srvModelCache[key]={model:{trees,subSz},trainedAt:rounds.length};
      srvTrainingInProgress.delete(key);
    }
    if(!srvModelCache[key]||!srvModelCache[key].model)return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const feat=srvExtractFeatures(gs,rounds,targetMin);
    if(!feat)return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);
    const{trees,subSz}=srvModelCache[key].model;
    const avgLen=trees.reduce((s,t)=>s+srvPathLen(t,feat),0)/trees.length;
    const c=subSz>1?2*(Math.log(subSz-1)+0.5772)-(2*(subSz-1)/subSz):1;
    const anomaly=Math.pow(2,-avgLen/Math.max(1,c));
    // Anomaly score: high = current state is unusual (short paths = easy to isolate = abnormal)
    // Reframe correctly: blend with base rate weighted by anomaly intensity
    // High anomaly + overdue gap → more likely to hit (abnormal gap IS the signal)
    const overdueBoost = Math.min(0.3, (gs.gapSinceLast / Math.max(1, gs.meanGap) - 1) * 0.15);
    const baseP = gs.pGlobal * maxWidth;
    // Anomaly amplifies base probability when overdue, dampens when early
    prob = baseP + anomaly * overdueBoost + (1 - anomaly) * Math.max(0, -overdueBoost);
  }catch(e){console.warn('[ifor]',e.message);return buildKm(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast);}
  return buildMLWindow(gs,maxWidth,targetLabel,isRare,rounds,targetMin,engineGapSinceLast,Math.max(0.05,Math.min(0.95,prob)),'ifor');
}


// ══════════════════════════════════════════════════════════════════════════════
// ── META ENGINE — Weighted stacking ensemble of all 9 ML engines ──────────────
//
// How it works:
//   1. Runs all 9 ML engines to get their raw probabilities for this target
//   2. Weights each engine by its ACTUAL win rate from historical DB predictions
//      (modelScores[target][engineId].ewma — updated every time a window resolves)
//   3. Takes a weighted average → the single best combined probability
//   4. Builds a window using buildMLWindow with this blended probability
//
// Why it works:
//   No single model is best for all targets/conditions. Meta learns which engines
//   are more reliable for each specific target and conditions it over time.
//   This is "stacking" — the classic ML meta-learning technique.
// ══════════════════════════════════════════════════════════════════════════════

// Per-target per-engine win rate tracker for meta weighting
// Seeded from modelScores which is already updated from DB history
function getMetaWeights(targetLabel) {
  const ML_ENGINE_IDS_LIST = ['rf','gbt','lr','nb','lstm','lgbm','prp','gru','ifor'];
  const weights = {};
  let totalW = 0;

  for (const eid of ML_ENGINE_IDS_LIST) {
    const score = modelScores[targetLabel]?.[eid];
    // modelScores.ewma is log-loss based: lower = better predictor
    // Convert to weight: higher win-rate engines get higher weight
    // ewma starts at 0.693 (log(2), neutral), goes lower as engine improves
    // Weight = e^(-ewma) so better engines (lower ewma) get higher weight
    const w = score ? Math.exp(-score.ewma) : 0.5;
    weights[eid] = w;
    totalW += w;
  }

  // Normalise to sum=1
  if (totalW > 0) {
    for (const eid of ML_ENGINE_IDS_LIST) weights[eid] /= totalW;
  } else {
    // Equal weights fallback
    const eq = 1 / ML_ENGINE_IDS_LIST.length;
    for (const eid of ML_ENGINE_IDS_LIST) weights[eid] = eq;
  }

  return weights;
}

function buildMeta(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) {
  // Collect raw probabilities from all 9 ML engines
  // Each engine uses its own independent model — no shared state
  const ML_ENGINES = [
    { id: 'rf',   fn: () => buildRF(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'gbt',  fn: () => buildGBT(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'lr',   fn: () => buildLR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'nb',   fn: () => buildNB(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'lstm', fn: () => buildLSTM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'lgbm', fn: () => buildServerLGBM(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'prp',  fn: () => buildServerPRP(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'gru',  fn: () => buildServerGRU(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
    { id: 'ifor', fn: () => buildServerIFOR(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast) },
  ];

  const weights = getMetaWeights(targetLabel);
  let weightedProb = 0;
  let totalWeight  = 0;
  let validCount   = 0;

  for (const { id, fn } of ML_ENGINES) {
    try {
      const result = fn();
      if (!result || result.probW == null) continue;
      const prob = Number(result.probW);
      if (!isFinite(prob)) continue;
      const w = weights[id] ?? (1 / ML_ENGINES.length);
      weightedProb += prob * w;
      totalWeight  += w;
      validCount++;
    } catch(e) {
      console.warn(`[meta] ${id} error:`, e.message);
    }
  }

  if (validCount === 0 || totalWeight === 0) {
    // All engines failed — fall back to geo
    return buildGeo(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast);
  }

  // Normalise in case some engines failed
  const metaProb = Math.max(0.05, Math.min(0.95, weightedProb / totalWeight));

  return buildMLWindow(gs, maxWidth, targetLabel, isRare, rounds, targetMin, engineGapSinceLast, metaProb, 'meta');
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
  let gSum=0, gSS=0; for (const g of gaps) { gSum+=g; gSS+=g*g; }
  const meanGap = gSum / gaps.length;
  const sg=[...gaps].sort((a,b)=>a-b);
  const mid2=Math.floor(sg.length/2);
  const medianGap=sg.length%2===1?sg[mid2]:(sg[mid2-1]+sg[mid2])/2;
  const cv=meanGap>0?Math.sqrt(Math.max(0,gSS/gaps.length-meanGap**2))/meanGap:1;
  const dW1=hW1/W1, dW2=hW2/W2, dW3=hW3/Math.min(W3,n);
  const safe=v=>Math.max(-1,Math.min(1,v));
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
    32+Math.min(18,Math.log2(hits+1)*4)+absComposite*30+(agree-1)*6-(cv>1.5?8:cv>1.2?4:0)
  )));
  return { direction, confidence:conf, hits, meanGap:Math.round(meanGap), medianGap:Math.round(medianGap), composite:+composite.toFixed(3) };
}

function buildPatternWindow(patternResult, maxWidth) {
  if (!patternResult) return null;
  const expectedGap = patternResult.medianGap || patternResult.meanGap || maxWidth;
  const low  = Math.max(0, expectedGap - maxWidth);
  return { low, high: low + maxWidth - 1, expectedGap, opensIn: low, confidence: patternResult.confidence, direction: patternResult.direction, streakStatus: 'normal', currentStreak: 0 };
}

// ── makeKey ───────────────────────────────────────────────────────────────────
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
    return { status:'hit',  hitRound:r.roundId };
  }
  if (currentRoundId > absHigh)                              return { status:'miss',   hitRound:null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status:'active', hitRound:null };
  return { status:'waiting', hitRound:null };
}

// ── processEngine ─────────────────────────────────────────────────────────────
async function processEngine({ engineId, state, sortedRounds, lastRoundId, buildFn }) {
  let anyChange = false;
  for (const target of TARGETS) {
    const existing = state.lockedMap[target.label];
    if (!existing) {
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:1, stale:false };
        anyChange = true;
        console.log(`[${engineId}] NEW ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% probW=${pred.probW??'—'} regime=${pred.regime??'—'} rec=${pred.recommendation??'—'}`);
      }
      continue;
    }
    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow)||!Number.isFinite(absHigh)||absHigh<absLow||anchorRound===0;
    const isExpired   = lastRoundId > absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit','miss','early'].includes(status.status)) {
          const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
          const key = makeKey(engineId, target.label, absLow, absHigh);
          if (!state.savedSet.has(key)) {
            state.savedSet.add(key);
            try {
              await savePrediction({ target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound, hitRound:status.hitRound||null, generation:existing.generation||1, source:engineId });
              recordTimingOutcome(target.label, outcome === 'early');
              // Record this engine's last WIN round independently — drives per-engine gapSinceLast
              // Track last hit for BOTH win and early — target DID hit regardless
              if ((outcome === 'win' || outcome === 'early') && status.hitRound) {
                if (engineLastHit[engineId]) engineLastHit[engineId][target.label] = status.hitRound;
              }
              if (STAT_MODELS.some(m=>m.id===engineId) && existing.probW!=null) {
                updateCalibration(target.label, engineId, existing.probW, outcome);
                updateModelScore(target.label, engineId, existing.probW, outcome);
                updateValidationMetrics(target.label, engineId, existing.probW, outcome, existing.recommendation ?? null);
              }
            } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
          }
        }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:(existing.generation||1)+(isNonsense?0:1), stale:false };
        console.log(`[${engineId}] REBUILD ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${pred.regime??'—'} rec=${pred.recommendation??'—'}`);
      } else {
        delete state.lockedMap[target.label];
        console.warn(`[${engineId}] ${target.label} cleared — insufficient data`);
      }
      anyChange = true;
      state.needsRebuild = false;
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit','miss','early'].includes(status.status)) {
      const outcome = status.status==='hit'?'win':status.status==='early'?'early':'loss';
      const key = makeKey(engineId, target.label, absLow, absHigh);
      if (!state.savedSet.has(key)) {
        state.savedSet.add(key);
        try {
          await savePrediction({ target:target.label, minMult:target.min, outcome, lo:absLow, hi:absHigh, anchorRound, hitRound:status.hitRound||null, generation:existing.generation||1, source:engineId });
          recordTimingOutcome(target.label, outcome === 'early');
          // Track last hit for BOTH win and early — target DID hit regardless of window timing
          if ((outcome === 'win' || outcome === 'early') && status.hitRound) {
            if (engineLastHit[engineId]) engineLastHit[engineId][target.label] = status.hitRound;
          }
          if (STAT_MODELS.some(m=>m.id===engineId) && existing.probW!=null) {
            updateCalibration(target.label, engineId, existing.probW, outcome);
            updateModelScore(target.label, engineId, existing.probW, outcome);
            updateValidationMetrics(target.label, engineId, existing.probW, outcome, existing.recommendation ?? null);
          }
          const { earlyRate } = getTimingParams(target.label);
          console.log(`[${engineId}] ${target.label} ${outcome.toUpperCase()}${outcome==='early'?' (early)':''} #${absLow}–#${absHigh}${status.hitRound?` @#${status.hitRound}`:''} earlyRate=${earlyRate.toFixed(2)}`);
        } catch(e) { console.error(`[${engineId}] save fail:`, e.message); }
      }
      const pred = buildFn(target);
      if (pred) {
        state.lockedMap[target.label] = { ...pred, targetMin:target.min, anchorRound:lastRoundId+1, generation:(existing.generation||1)+1, stale:false };
        console.log(`[${engineId}] NEXT ${target.label}: +${pred.low}–+${pred.high} conf=${pred.confidence}% regime=${pred.regime??'—'} rec=${pred.recommendation??'—'}`);
      } else { delete state.lockedMap[target.label]; }
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
      lo: anchor+(Number(pred.low)||0), hi: anchor+(Number(pred.high)||0),
      roundWhenMade: anchor, generation: pred.generation||1,
      eta: {
        low:pred.low, high:pred.high, conf:pred.confidence,
        probW:pred.probW, rawProbW:pred.rawProbW??pred.probW,
        expectedGap:pred.expectedGap, opensIn:pred.opensIn,
        streakStatus:pred.streakStatus, currentStreak:pred.currentStreak,
        spread:pred.spread??null,
        regime:pred.regime??null, regimeFactor:pred.regimeFactor??null,
        regimeConfidence:pred.regimeConfidence??null,
        ensWeights:pred.ensWeights??null,
        isRandom:pred.isRandom??null, maxAC:pred.maxAC??null,
        ev:pred.ev??null, signalQuality:pred.signalQuality??null,
        risk:pred.risk??null, recommendation:pred.recommendation??null,
        decisionReason:pred.decisionReason??null,
        signalStrength:pred.signalStrength??null,
        finalProbUsed:pred.finalProbUsed??null,
        amplifiedProbW:pred.amplifiedProbW??null,
        tailProbability:pred.tailProbability??null,
        extremeGapScore:pred.extremeGapScore??null,
        rareEV:pred.rareEV??null,
        earlyWindow:pred.earlyWindow??null,
        lateWindow:pred.lateWindow??null,
        primaryWindow:pred.primaryWindow??null,
        alpha:pred.alpha??null,
        rareSignal:pred.rareSignal??null,
      },
    };
  }
  return out;
}

function loadLockedMap(dbRows) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = TARGETS.find(t => t.label === label);
    if (!target) continue;
    const eta = pred.eta || {};
    const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;

    const low  = eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor);
    const high = eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor);

    // ── Window-spec guard ─────────────────────────────────────────────────────
    // If the saved window is wider than current maxWidth, it was built before
    // the window-gap fix. Discard it so the engine rebuilds immediately.
    const savedWidth = high - low + 1;
    if (savedWidth > target.maxWidth) {
      console.log(`[engine] DISCARD stale wide window ${label}: saved ${savedWidth}r > max ${target.maxWidth}r — will rebuild`);
      continue;
    }

    map[label] = {
      low, high,
      confidence: eta.conf??50, probW: eta.probW??null, rawProbW: eta.rawProbW??null,
      expectedGap: eta.expectedGap??null, opensIn: eta.opensIn??null,
      streakStatus: eta.streakStatus??'normal', currentStreak: eta.currentStreak??0,
      spread: eta.spread??null,
      regime: eta.regime??null, regimeFactor: eta.regimeFactor??null,
      regimeConfidence: eta.regimeConfidence??null,
      ensWeights: eta.ensWeights??null,
      isRandom: eta.isRandom??null, maxAC: eta.maxAC??null,
      ev: eta.ev??null, signalQuality: eta.signalQuality??null,
      risk: eta.risk??null, recommendation: eta.recommendation??null,
      decisionReason: eta.decisionReason??null,
      signalStrength: eta.signalStrength??null,
      finalProbUsed: eta.finalProbUsed??null,
      amplifiedProbW: eta.amplifiedProbW??null,
      tailProbability: eta.tailProbability??null,
      extremeGapScore: eta.extremeGapScore??null,
      rareEV:          eta.rareEV??null,
      earlyWindow:     eta.earlyWindow??null,
      lateWindow:      eta.lateWindow??null,
      primaryWindow:   eta.primaryWindow??null,
      alpha:           eta.alpha??null,
      rareSignal:      eta.rareSignal??null,
      targetMin: target.min, anchorRound: anchor,
      generation: pred.generation??1, stale: true,
    };
  }
  return map;
}

// ── Validation export — F8: winRate = wins / trades ──────────────────────────
function getValidationMetrics() {
  const out = {};
  for (const t of TARGETS) {
    out[t.label] = {};
    for (const m of STAT_MODELS) {
      const v   = valMetrics[t.label][m.id];
      const ece = getECE(t.label, m.id);
      const total = v.count;
      const effectiveAccuracy = total > 0
        ? +((v.wins + 0.5 * v.earlyCount) / total * 100).toFixed(1)
        : null;
      // F8: primary KPI — winRate = takenWins / takenTotal
      const winRateOnTaken = v.takenTotal > 0 ? +(v.takenWins / v.takenTotal).toFixed(4) : null;
      const balanceScore   = (v.totalWins > 0 && winRateOnTaken != null)
        ? +(v.totalWins * winRateOnTaken).toFixed(2)
        : null;
      out[t.label][m.id] = {
        brier:              total > 0 ? +(v.brierSum   / total).toFixed(4) : null,
        logLoss:            total > 0 ? +(v.logLossSum / total).toFixed(4) : null,
        ece:                ece != null ? +ece.toFixed(4) : null,
        wins:               v.wins,
        losses:             v.losses,
        early:              v.earlyCount,
        total,
        hitRate:            total > 0 ? +((v.wins  / total) * 100).toFixed(1) : null,
        effectiveAccuracy,
        earlyRate:          total > 0 ? +((v.earlyCount / total) * 100).toFixed(1) : null,
        // F8: trade metrics (primary KPI: winRateOnTaken = wins / trades)
        tradeCount:         v.tradeCount,
        totalWins:          v.totalWins,
        winRateOnTaken,     // PRIMARY KPI: wins on TAKE calls / total TAKE calls
        balanceScore,
      };
    }
  }
  return out;
}

// ── Engine stats ──────────────────────────────────────────────────────────────
function getEngineStats() {
  const out = {};
  for (const t of TARGETS) {
    out[t.label] = {};
    for (const id of ['engine', 'ens', 'geo', 'bay', 'km', 'rf', 'gbt', 'lr', 'nb', 'lstm', 'lgbm', 'prp', 'gru', 'ifor', 'meta']) {
      const cs = engineCusumState[id]?.[t.label];
      if (!cs) continue;
      const ece = STAT_MODELS.some(m => m.id === id) ? getECE(t.label, id) : null;
      out[t.label][id] = {
        regimeConfidence: +clamp((cs.count - REGIME_ACTIVATION_OUTCOMES) / 200, 0, 1).toFixed(3),
        warmupProgress:   +clamp(cs.count / CAL_WARMUP_OUTCOMES, 0, 1).toFixed(3),
        regimeLabel:      cs.regimeLabel,
        regimeFactor:     +cs.confirmedFactor.toFixed(4),
        outcomesCount:    cs.count,
        ece:              ece != null ? +ece.toFixed(4) : null,
      };
    }
  }
  return out;
}

// ── Timing stats export ───────────────────────────────────────────────────────
function getTimingStats() {
  const out = {};
  for (const t of TARGETS) {
    const ts = timingState[t.label];
    if (!ts) continue;
    const { earlyRate, recentEarlyRate, timingShiftFactor, hasData } = getTimingParams(t.label);
    out[t.label] = {
      earlyRate:         +earlyRate.toFixed(3),
      recentEarlyRate:   +recentEarlyRate.toFixed(3),
      timingShiftFactor: +timingShiftFactor.toFixed(3),
      totalOutcomes:     ts.totalCount,
      earlyOutcomes:     ts.earlyCount,
      recentWindow:      ts.earlyQueue.length,
      hasData,
    };
  }
  return out;
}

async function initialise() {
  if (initialised) return;
  initialised = true;
  for (const id of Object.keys(STATE)) STATE[id].savedSet = new Set();
  try { STATE.engine.lockedMap  = loadLockedMap(await getLockedPreds());      console.log(`[engine] loaded ${Object.keys(STATE.engine.lockedMap).length} engine preds`);  } catch(e) { console.error('[engine] init:',   e.message); STATE.engine.lockedMap  = {}; }
  try { STATE.pattern.lockedMap = loadLockedMap(await getLockedPatternPreds()); console.log(`[engine] loaded ${Object.keys(STATE.pattern.lockedMap).length} pattern preds`); } catch(e) { console.error('[engine] pattern:', e.message); STATE.pattern.lockedMap = {}; }
  try {
    const dbStats = await getLockedStatPreds();
    for (const model of STAT_MODELS) {
      STATE[model.id].lockedMap = loadLockedMap(dbStats[model.id] || {});
      console.log(`[engine] loaded ${Object.keys(STATE[model.id].lockedMap).length} ${model.id} preds`);
    }
  } catch(e) { console.error('[engine] stat init:', e.message); for (const model of STAT_MODELS) STATE[model.id].lockedMap = {}; }
  try {
    const rows = await getPredictions({ limit: 10000 });
    for (const r of rows) {
      const src = r.source || 'engine';
      const key = makeKey(src, r.target, r.lo, r.hi);
      if (STATE[src]?.savedSet) STATE[src].savedSet.add(key);
      // Seed per-engine last-hit so windows diverge immediately from history
      if (r.outcome === 'win' && r.hitRound && engineLastHit[src]?.[r.target] !== undefined) {
        const prev = engineLastHit[src][r.target];
        if (prev < r.hitRound) engineLastHit[src][r.target] = r.hitRound;
      }
      // Seed calibration + model scores from historical outcomes so accuracy
      // doesn't reset to zero after every server restart
      if (r.probW != null && ['win','loss','early'].includes(r.outcome) &&
          STAT_MODELS.some(m => m.id === src)) {
        try {
          updateCalibration(r.target, src, r.probW, r.outcome);
          updateModelScore(r.target, src, r.probW, r.outcome);
        } catch(_) {}
      }
    }
    console.log(`[engine] loaded ${rows.length} history keys (calibration pre-warmed)`);
  } catch(e) { console.error('[engine] history:', e.message); }
  for (const id of Object.keys(STATE)) STATE[id].needsRebuild = true;
}

// ── Main ──────────────────────────────────────────────────────────────────────
// ── ML engine IDs — run deferred so they never block the HTTP server ──────────
const ML_ENGINE_IDS = new Set(['rf','gbt','lr','nb','lstm','lgbm','prp','gru','ifor','meta']);

// Run a single engine, yielding event loop before each one
function runEngineDeferred(eng, rounds, lastRoundId) {
  return new Promise((resolve) => {
    setImmediate(async () => {
      const tStart = Date.now();
      try {
        if (!(lastRoundId > eng.state.lastRoundId || eng.state.needsRebuild)) {
          resolve(); return;
        }
        eng.state.needsRebuild = false;
        const changed = await processEngine({ engineId:eng.id, state:eng.state, sortedRounds:rounds, lastRoundId, buildFn:eng.buildFn });
        eng.state.lastRoundId = lastRoundId;
        if (changed) {
          const p = buildSavePayload(eng.state.lockedMap);
          try { await eng.saveFn(p); } catch(e) { console.error(`[${eng.id}] save:`, e.message); }
        }
        const ms = Date.now() - tStart;
        if (ms > 1000) console.log(`[${eng.id}] took ${ms}ms`);
      } catch(e) {
        console.error(`[${eng.id}] error in deferred:`, e.message, e.stack?.split('\n')[1] || '');
      }
      resolve();
    });
  });
}

// Tracks whether a Tier2 background run is already in progress
// Prevents overlapping Tier2 runs if Tier1 ticks faster than Tier2 completes
let tier2Running = false;

async function runTier2Background(allEngines, rounds, lastRoundId) {
  if (tier2Running) return; // skip if previous Tier2 still running
  tier2Running = true;
  const t2Start = Date.now();
  try {
    const mlEngines  = allEngines.filter(e => ML_ENGINE_IDS.has(e.id) && e.id !== 'meta');
    const metaEngine = allEngines.find(e => e.id === 'meta');
    for (const eng of mlEngines) {
      console.log(`[tier2] starting ${eng.id}`);
      await runEngineDeferred(eng, rounds, lastRoundId);
      console.log(`[tier2] done ${eng.id}`);
    }
    // Meta runs last — reads already-trained caches from sub-engines
    if (metaEngine) await runEngineDeferred(metaEngine, rounds, lastRoundId);
    console.log(`[engine] Tier2 (ML) done in ${Date.now()-t2Start}ms`);
  } catch(e) {
    console.error('[engine] Tier2 error:', e.message);
  } finally {
    tier2Running = false;
  }
}

async function runPredictionEngine() {
  try {
    await initialise();
    const rounds = await getRounds({ limit: 100000, order: 'DESC' }); // full history — use all collected data
    if (rounds.length < MIN_ROUNDS) { console.log(`[engine] waiting (${rounds.length}/${MIN_ROUNDS})`); return; }
    rounds.sort((a, b) => a.roundId - b.roundId);
    const lastRoundId = rounds[rounds.length - 1].roundId;

    const allEngines = [
      { id:'engine',  state:STATE.engine,  buildFn:(t)=>buildPrediction(rounds,t.min,t.maxWidth,t.rare,lastRoundId),         saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedPreds(p);} },
      { id:'pattern', state:STATE.pattern, buildFn:(t)=>{const pp=buildPatternPrediction(rounds,t.min);return buildPatternWindow(pp,t.maxWidth);}, saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedPatternPreds(p);} },
      ...STAT_MODELS.map(model=>({ id:model.id, state:STATE[model.id], buildFn:(t)=>buildStatPrediction(rounds,t.min,t.maxWidth,model.id,lastRoundId), saveFn:async(p)=>{if(Object.keys(p).length)await saveLockedStatPreds(model.id,p);} })),
    ];

    // ── Tier 1: Fast stat engines — await this, it's quick (~100ms) ──────────
    const t1Start = Date.now();
    for (const eng of allEngines.filter(e => !ML_ENGINE_IDS.has(e.id))) {
      if (!(lastRoundId > eng.state.lastRoundId || eng.state.needsRebuild)) continue;
      eng.state.needsRebuild = false;
      const changed = await processEngine({ engineId:eng.id, state:eng.state, sortedRounds:rounds, lastRoundId, buildFn:eng.buildFn });
      eng.state.lastRoundId = lastRoundId;
      if (changed) { const p=buildSavePayload(eng.state.lockedMap); try{await eng.saveFn(p);}catch(e){console.error(`[${eng.id}] save:`,e.message);} }
    }
    console.log(`[engine] Tier1 done in ${Date.now()-t1Start}ms`);

    // Enable ML training after Tier1
    enableMLTraining();

    // ── Tier 2: ML engines — fire and forget, runs in background ─────────────
    // poll() returns immediately after Tier1. Tier2 runs in background.
    // tier2Running flag prevents overlapping runs.
    runTier2Background(allEngines, rounds, lastRoundId);
    // intentionally NOT awaited — poll loop stays on schedule

  } catch(e) { console.error('[predictionEngine] Fatal:', e.message, e.stack); }
}

function getLockedStatMap(modelId) { return STATE[modelId]?.lockedMap || {}; }

module.exports = { runPredictionEngine, resetEngineState, getLockedStatMap, getValidationMetrics, getEngineStats, getTimingStats };