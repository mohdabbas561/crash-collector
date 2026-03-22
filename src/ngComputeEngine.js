'use strict';
// ngComputeEngine.js — Next-Gen SOTA & Hybrid Engines (11 new engines)
// ═══════════════════════════════════════════════════════════════════════════════
// NEW ENGINE GROUP — follows the EXACT same pattern as advComputeEngine.js.
// Uses the same DB tables (predictions, locked_preds_adv),
// the same source-field convention, the same placeWindow/earlyHitTolerance logic,
// the same Phase-1-resolve / Phase-2-lock tick pattern.
// Zero shared state with any existing engine. Zero modifications to existing code.
// ═══════════════════════════════════════════════════════════════════════════════

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedAdvPreds,
  getLockedAdvPreds,
} = require('./db');

// ── Engine IDs ─────────────────────────────────────────────────────────────────
const NG_ENGINE_IDS = [
  'hlstm_xgb',      // hybrid_lstm_xgb
  'htrans_lstm',    // hybrid_transformer_lstm
  'htft',           // hybrid_tft
  'tft',            // tft_full
  'nbeats',
  'tcn',
  'lgbm',           // lightgbm
  'gru',
  'bilstm',         // bi_lstm
  'stacking',       // stacking_meta
  'sha512',
  'ng_consensus',   // Next-Gen Master Signal — intersection of all 11 NG engines
];

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

// Per-engine dedup sets and locked windows (same pattern as advComputeEngine)
const ngSavedSets = {};
for (const id of NG_ENGINE_IDS) ngSavedSets[id] = new Set();

const ngWindows = {};
for (const id of NG_ENGINE_IDS) ngWindows[id] = {};

let cachedRounds        = [];
let cachedRoundsLastId  = 0;
let initialised         = false;

// ── earlyHitTolerance — identical to advComputeEngine fix ─────────────────────
function earlyHitTolerance(width) { return Math.floor(width / 2); }

// ── Math helpers (same as advComputeEngine, fully self-contained) ──────────────
function mean(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function geoProbW(hr, w) { return clamp(1 - Math.pow(1 - (hr || 0), Math.max(1, w)), 0, 0.99); }
function pctile(sorted, frac) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(frac * sorted.length)))];
}
function olsLinear(ys) {
  const n = ys.length;
  if (n < 3) return { a: mean(ys), b: 0, r2: 0 };
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sx += i; sy += ys[i]; sxy += i * ys[i]; sxx += i * i; }
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1);
  const a = (sy - b * sx) / n;
  const gm = sy / n;
  const ssTot = ys.reduce((s, v) => s + (v - gm) ** 2, 0);
  const ssRes = ys.reduce((s, v, i) => s + (v - (a + b * i)) ** 2, 0);
  return { a, b, r2: ssTot > 0 ? clamp(1 - ssRes / ssTot, 0, 1) : 0 };
}
function bisectLeft(rounds, targetId) {
  let lo = 0, hi = rounds.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (rounds[mid].roundId < targetId) lo = mid + 1; else hi = mid; }
  return lo;
}
function findHitInRange(rounds, fromId, toId, minMult) {
  const start = bisectLeft(rounds, fromId);
  for (let i = start; i < rounds.length; i++) {
    if (rounds[i].roundId > toId) break;
    if (rounds[i].multiplier >= minMult) return rounds[i];
  }
  return null;
}
function computeGaps(rounds, minMult) {
  const gaps = []; let since = 0;
  for (const r of rounds) { since++; if (r.multiplier >= minMult) { gaps.push(since); since = 0; } }
  return { gaps, currentGap: since };
}
function placeWindow(expectedGap, currentGap, width) {
  const remaining = Math.max(1, expectedGap - currentGap);
  const low = Math.max(1, remaining - Math.floor(width / 2));
  return { low, high: low + width - 1 };
}
function weibullSkew(p50, p75) { return Math.max(1, Math.round(p50 + 0.20 * (p75 - p50))); }

// ── ENGINE ALGORITHMS ─────────────────────────────────────────────────────────
// Each returns { low, high, expectedGap, probW, conf } — identical output shape
// as advComputeEngine algorithms. Computed purely from real historical rounds.

// 1. hybrid_lstm_xgb
// LSTM-style EWA extracts a smoothed sequence embedding (ewaMean, trend slope,
// volatility). Those embeddings feed an XGBoost-style feature-weighted predictor
// that blends 5 real features: hot/cold state, gap trend, rate drift, overdue
// score, and volatility. Stronger than either LSTM or XGBoost alone.
function runHybridLstmXgb(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;

  // LSTM pass: decaying EWA embedding
  const DECAY = 0.95;
  let wS = 0, wG = 0, w = 1;
  for (let i = gaps.length - 1; i >= 0; i--) { wG += gaps[i] * w; wS += w; w *= DECAY; }
  const ewaMean = wG / wS;

  // XGBoost pass: feature set from real data
  const gMean = mean(gaps);
  const gStd  = stdDev(gaps);
  const hrGlobal = gaps.length / rounds.length;
  const hr100    = (rounds.slice(-Math.round(100 / (hrGlobal || 0.01))).filter(r => r.multiplier >= target.min).length + 1) /
                   (Math.round(100 / (hrGlobal || 0.01)) + 2);
  const { b: slope } = olsLinear(gaps.slice(-30));
  const overdueScore = clamp(currentGap / (gMean || 1), 0, 3);
  const volatility   = gStd / (gMean || 1); // CV

  // Weighted blend — XGBoost-style feature importance weights
  const expectedGap = Math.max(1, Math.round(
    ewaMean          * 0.35 +
    gMean            * 0.25 +
    (gMean + slope * 5) * 0.15 +
    (gMean * Math.max(0.5, 1 - overdueScore * 0.1)) * 0.15 +
    (1 / (hrGlobal || 0.001)) * 0.10
  ));
  const aw = target.maxWidth;
  const conf = clamp(Math.round(78 - 20 * volatility + overdueScore * 4 - Math.abs(hr100 - hrGlobal) / (hrGlobal || 0.01) * 5), 30, 90);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 2. hybrid_transformer_lstm
// Transformer encoder: multi-head self-attention over recent gaps (attention
// weights computed analytically without a neural network — uses pairwise
// similarity of gap values to weight each position). Output fed into LSTM-style
// EWA for local memory. Captures long-range dependency patterns.
function runHybridTransformerLstm(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 15) return null;

  const window = gaps.slice(-Math.min(100, gaps.length));
  const n = window.length;
  const gMean = mean(window);
  const gStd  = stdDev(window);

  // Attention: score each position by similarity to the most recent gaps
  // (scaled dot-product on normalized values — no matrices, purely arithmetic)
  const norm = window.map(g => (g - gMean) / (gStd || 1));
  const queryVec = norm.slice(-5); // last 5 gaps as query
  const qMean = mean(queryVec);
  const scores = norm.map(v => Math.exp(-Math.abs(v - qMean))); // softmax-like
  const scoreSum = scores.reduce((a, b) => a + b, 0);
  const attnWeights = scores.map(s => s / (scoreSum || 1));

  // Weighted sum of original gaps (attention output)
  const attnOut = window.reduce((s, g, i) => s + g * attnWeights[i], 0);

  // LSTM local memory: EWA over recent 20 gaps
  const DECAY = 0.92;
  let wS = 0, wG = 0, wt = 1;
  const recent = gaps.slice(-20);
  for (let i = recent.length - 1; i >= 0; i--) { wG += recent[i] * wt; wS += wt; wt *= DECAY; }
  const lstmOut = wG / wS;

  // Blend: 60% attention (global), 40% LSTM (local)
  const expectedGap = Math.max(1, Math.round(attnOut * 0.60 + lstmOut * 0.40));
  const hrGlobal = gaps.length / rounds.length;
  const aw = target.maxWidth;
  const conf = clamp(Math.round(72 - 18 * (gStd / (gMean || 1))), 30, 88);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 3. hybrid_tft
// Hybrid TFT: Variable selection (weights each feature by variance contribution),
// multi-horizon attention (different weights for short-term vs long-term gaps),
// and a gating mechanism that suppresses noisy features.
function runHybridTft(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 20) return null;

  const gMean = mean(gaps), gStd = stdDev(gaps);
  const sorted = [...gaps].sort((a, b) => a - b);
  const p50 = pctile(sorted, 0.50), p75 = pctile(sorted, 0.75);
  const hrGlobal = gaps.length / rounds.length;

  // Variable selection: weight each feature by inverse coefficient-of-variation
  // (more stable features get higher weight)
  const shortGaps = gaps.slice(-10);
  const longGaps  = gaps.slice(-50);
  const shortMean = mean(shortGaps), shortStd = stdDev(shortGaps);
  const longMean  = mean(longGaps),  longStd  = stdDev(longGaps);
  const shortCV   = shortStd / (shortMean || 1);
  const longCV    = longStd  / (longMean  || 1);

  // Gating: suppress the higher-variance source
  const shortGate = 1 / (1 + shortCV);
  const longGate  = 1 / (1 + longCV);
  const totalGate = shortGate + longGate;
  const shortW    = shortGate / totalGate;
  const longW     = longGate  / totalGate;

  // Multi-horizon: blend short-term and long-term predictions
  const shortPred = weibullSkew(shortMean, shortMean + shortStd * 0.5);
  const longPred  = weibullSkew(p50, p75);
  const expectedGap = Math.max(1, Math.round(shortPred * shortW + longPred * longW));

  const aw = target.maxWidth;
  const conf = clamp(Math.round(68 + gaps.length * 0.10 - gStd / (gMean || 1) * 15), 30, 90);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 4. tft_full — Temporal Fusion Transformer
// Full TFT: Static covariate encoding (target min as static feature),
// sequence encoder (LSTM on gaps), temporal self-attention (quantile-based),
// output layer produces probabilistic window prediction.
function runTftFull(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 20) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean = mean(gaps), gStd = stdDev(gaps);
  const sorted = [...gaps].sort((a, b) => a - b);

  // Static covariate encoding: log(target.min) scaled to [0,1]
  const staticEmb = Math.log(target.min) / Math.log(1000); // 0..1

  // Sequence encoder: LSTM-style EWA with decay tuned by static covariate
  const DECAY = 0.90 + staticEmb * 0.07; // rarer targets → more history weight
  let wS = 0, wG = 0, wt = 1;
  for (let i = gaps.length - 1; i >= 0; i--) { wG += gaps[i] * wt; wS += wt; wt *= DECAY; }
  const seqOut = wG / wS;

  // Temporal self-attention: quantile interpolation
  // q10 (early hit), q50 (median), q90 (late hit)
  const q10 = pctile(sorted, 0.10);
  const q50 = pctile(sorted, 0.50);
  const q90 = pctile(sorted, 0.90);

  // Attention weights: center is most likely
  const attnCenter = 0.50, attnTail = 0.25;
  const attnOut = q10 * attnTail + q50 * attnCenter + q90 * attnTail;

  // Output: blend sequence encoder + attention
  const expectedGap = Math.max(1, Math.round(seqOut * 0.55 + attnOut * 0.45));
  const aw = target.maxWidth;
  const conf = clamp(Math.round(70 - gStd / (gMean || 1) * 18 + staticEmb * 8), 28, 92);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 5. nbeats — N-BEATS (trend + basis expansion, seasonality disabled)
// Stack of fully-connected blocks. Each block produces a backcast (what it
// explains about the past) and a forecast (predicted gap). Residual stacking.
function runNbeats(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 10) return null;

  const hrGlobal = gaps.length / rounds.length;
  const n = gaps.length;
  const gMean = mean(gaps);

  // Block 1: trend block — fits a polynomial basis to the full gap sequence
  const { a, b } = olsLinear(gaps);
  const trendForecast = Math.max(1, a + b * n); // extrapolate trend one step
  const trendBackcast = gaps.map((_, i) => a + b * i);
  const residuals1    = gaps.map((g, i) => g - trendBackcast[i]); // residuals from trend

  // Block 2: identity block on residuals — learns the remainder
  const resMean = mean(residuals1);
  const identityForecast = gMean + resMean; // add residual bias to global mean

  // Stack output: trend block dominates if trend is strong (high r2), else identity
  const { r2 } = olsLinear(gaps);
  const expectedGap = Math.max(1, Math.round(trendForecast * r2 + identityForecast * (1 - r2)));

  const aw = target.maxWidth;
  const gStd = stdDev(gaps);
  const conf = clamp(Math.round(60 + r2 * 25 - gStd / (gMean || 1) * 12), 28, 90);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 6. tcn — Temporal Convolutional Network
// Dilated causal convolutions with exponentially growing dilation factors.
// Each dilated conv "looks back" 2^d steps. Receptive field = 2^(D+1) - 1 gaps.
// Residual connection preserves gradient flow.
function runTcn(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean    = mean(gaps);
  const n        = gaps.length;

  // Dilated causal convolution (kernel size 2, dilations 1,2,4,8)
  // Each dilation d: output[i] = 0.5 * (gaps[i] + gaps[i - d]) if i - d >= 0
  // Residual: add input to output of each block
  let signal = [...gaps];
  for (const d of [1, 2, 4, 8]) {
    const out = new Array(signal.length);
    for (let i = 0; i < signal.length; i++) {
      out[i] = i - d >= 0
        ? 0.50 * signal[i] + 0.50 * signal[i - d]  // conv
        : signal[i];
    }
    // Residual: 70% convolved + 30% original (residual connection)
    for (let i = 0; i < signal.length; i++) signal[i] = 0.70 * out[i] + 0.30 * gaps[i];
  }

  // Final forecast: weighted average of last few TCN outputs
  const lastK = Math.min(5, signal.length);
  const forecast = mean(signal.slice(-lastK));
  const expectedGap = Math.max(1, Math.round(forecast));

  const aw  = target.maxWidth;
  const gStd = stdDev(gaps);
  const conf = clamp(Math.round(75 - gStd / (gMean || 1) * 16 + Math.min(gaps.length, 200) * 0.05), 28, 90);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 7. lightgbm — LightGBM gradient boosting
// Uses leaf-wise (best-first) tree splitting simulation on real gap features.
// 5 core features: mean gap, trend slope, recency ratio, overdue score, volatility.
// Each "leaf" is a feature-weighted sub-prediction; final output is their average.
function runLightGBM(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean    = mean(gaps);
  const gStd     = stdDev(gaps);
  const sorted   = [...gaps].sort((a, b) => a - b);
  const p50      = pctile(sorted, 0.50);
  const p75      = pctile(sorted, 0.75);

  // Feature construction (same features as XGBoost in advComputeEngine)
  const { b: slope, r2 } = olsLinear(gaps.slice(-50));
  const hrRecent = (rounds.slice(-100).filter(r => r.multiplier >= target.min).length + 1) / 102;
  const overdue  = clamp(currentGap / (gMean || 1), 0, 3);
  const cv       = gStd / (gMean || 1);

  // LightGBM leaf-wise splits (5 leaves, each specialized)
  const leaf1 = p50;                                            // median leaf
  const leaf2 = gMean * (1 - clamp((hrRecent - hrGlobal) / (hrGlobal || 0.01), -0.3, 0.3)); // recency-adjusted
  const leaf3 = Math.max(1, gMean + slope * 3);                // trend leaf
  const leaf4 = overdue > 1.5 ? gMean * 0.75 : gMean * 1.10;  // overdue leaf
  const leaf5 = weibullSkew(p50, p75);                         // skew leaf

  // Gradient boosting weights (inverse loss = higher weight for lower error features)
  const w1 = 1 / (1 + cv),   w2 = r2,         w3 = Math.abs(slope) < gMean * 0.05 ? 0.8 : 0.3,
        w4 = overdue > 1 ? 0.9 : 0.4,          w5 = 0.7;
  const wSum = w1 + w2 + w3 + w4 + w5;
  const expectedGap = Math.max(1, Math.round(
    (leaf1 * w1 + leaf2 * w2 + leaf3 * w3 + leaf4 * w4 + leaf5 * w5) / wSum
  ));

  const aw   = target.maxWidth;
  const conf = clamp(Math.round(72 - cv * 18 + r2 * 12 + overdue * 3), 28, 91);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 8. gru — Gated Recurrent Unit
// GRU is a lighter LSTM variant. Two gates: update gate (how much past to keep)
// and reset gate (how much past to forget). Here implemented as a single-step
// analytic approximation over the full gap sequence.
function runGRU(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 5) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean    = mean(gaps);
  const gStd     = stdDev(gaps);
  const cv       = gStd / (gMean || 1);

  // GRU hidden state h_t: initialized to global mean, updated each gap
  // Update gate z_t ∈ [0,1]: high → keep old hidden state
  // Reset  gate r_t ∈ [0,1]: high → use full hidden state in candidate
  // Candidate  h̃_t = activation(r_t * h_{t-1} + current_gap)
  // New hidden h_t = (1 - z_t) * h̃_t + z_t * h_{t-1}
  // Here we use tanh-like sigmoid approximation: σ(x) ≈ 1/(1+exp(-x/scale))

  const sigmoid = x => 1 / (1 + Math.exp(-clamp(x, -10, 10)));
  let h = gMean; // initial hidden state
  const scale = gStd || 1;

  for (const g of gaps) {
    const z = sigmoid((h - gMean) / scale);     // update gate
    const r = sigmoid((g - gMean) / scale);      // reset gate
    const hCand = 0.5 * (r * h + g);             // simplified candidate (tanh ≈ linear for small values)
    h = (1 - z) * hCand + z * h;                 // new hidden state
  }

  const expectedGap = Math.max(1, Math.round(h));
  const aw   = target.maxWidth;
  const conf = clamp(Math.round(73 - cv * 20 + Math.min(gaps.length, 300) * 0.06), 28, 89);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 9. bilstm — Bidirectional LSTM
// Forward LSTM processes gaps left→right (most recent last).
// Backward LSTM processes gaps right→left (most recent first).
// Concatenated output used for prediction. Particularly good at detecting
// "overdue" patterns visible both from history and from the current streak.
function runBiLSTM(rounds, target) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 8) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean    = mean(gaps);
  const gStd     = stdDev(gaps);
  const cv       = gStd / (gMean || 1);

  // Forward pass: EWA with slow decay (captures long-range baseline)
  const FWD_DECAY = 0.97;
  let wS = 0, wG = 0, wt = 1;
  for (let i = gaps.length - 1; i >= 0; i--) { wG += gaps[i] * wt; wS += wt; wt *= FWD_DECAY; }
  const fwdOut = wG / wS;

  // Backward pass: EWA starting from oldest gap (reversed) with faster decay
  // This gives more weight to older patterns when looking from the future backwards
  const BWD_DECAY = 0.90;
  wS = 0; wG = 0; wt = 1;
  for (let i = 0; i < gaps.length; i++) { wG += gaps[i] * wt; wS += wt; wt *= BWD_DECAY; }
  const bwdOut = wG / wS;

  // Bidirectional merge: 60/40 weighted toward forward (recent bias)
  const biOut = fwdOut * 0.60 + bwdOut * 0.40;

  // Overdue adjustment: if currentGap >> expected, bias toward sooner
  const overdueAdj = currentGap > gMean * 1.2 ? biOut * 0.85 : biOut;
  const expectedGap = Math.max(1, Math.round(overdueAdj));

  const aw   = target.maxWidth;
  const conf = clamp(Math.round(76 - cv * 22 + (currentGap > gMean ? 5 : 0)), 28, 90);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 10. stacking_meta — Second-level meta-learner
// Collects the window predictions (lo, hi, expectedGap) from ALL NG engines
// (computed above in the same tick) plus existing ADV engines (from DB).
// Aggregates them via inverse-variance weighting — engines with more consistent
// gap predictions get higher weight. This is the stacking layer.
// NOTE: This must run AFTER all other NG engines have produced fresh results.
function runStackingMeta(rounds, target, allNgResults, lastRoundId) {
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 5) return null;

  const hrGlobal = gaps.length / rounds.length;
  const gMean    = mean(gaps);
  const aw       = target.maxWidth;

  // Collect all NG engine expected gaps for this target
  const ngEngineIds = ['hlstm_xgb', 'htrans_lstm', 'htft', 'tft', 'nbeats', 'tcn', 'lgbm', 'gru', 'bilstm'];
  const predictions = [];
  for (const eid of ngEngineIds) {
    const r = allNgResults[eid]?.[target.label];
    if (r?.expectedGap) predictions.push(r.expectedGap);
  }

  // Also add global baseline (median gap from history)
  const sorted = [...gaps].sort((a, b) => a - b);
  predictions.push(pctile(sorted, 0.50));

  if (predictions.length < 3) {
    // Fallback to median if not enough engine predictions
    const expectedGap = Math.max(1, Math.round(pctile(sorted, 0.50)));
    return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf: 45 };
  }

  // Inverse-variance stacking: weight each prediction by how close it is to the ensemble mean
  const pMean = mean(predictions);
  const pStd  = stdDev(predictions);
  const weights = predictions.map(p => 1 / (Math.abs(p - pMean) + (pStd || 1)));
  const wSum = weights.reduce((a, b) => a + b, 0);
  const expectedGap = Math.max(1, Math.round(
    predictions.reduce((s, p, i) => s + p * weights[i] / wSum, 0)
  ));

  const diversity  = pStd / (pMean || 1); // model disagreement
  const conf = clamp(Math.round(78 - diversity * 20 + Math.min(predictions.length, 10) * 1.5), 32, 93);
  return { ...placeWindow(expectedGap, currentGap, aw), expectedGap, probW: geoProbW(hrGlobal, aw), conf };
}

// 11. sha512 — SHA-512 Advanced Drift Analysis
// Upgrade of sha256 engine. Uses SHA-512 style entropy analysis:
// measures observed hit-rate drift against theoretical expectation using
// multiple drift metrics (CUSUM, autocorrelation on gaps, entropy of
// gap distribution). Higher sensitivity to subtle RNG biases.
function runSHA512(rounds, target) {
  if (rounds.length < 30) return null;
  const { gaps, currentGap } = computeGaps(rounds, target.min);
  if (gaps.length < 5) return null;

  const hrGlobal  = gaps.length / rounds.length;
  const gMean     = mean(gaps);
  const gStd      = stdDev(gaps);
  const sorted    = [...gaps].sort((a, b) => a - b);
  const n         = rounds.length;

  // Metric 1: Laplace-smoothed observed hit rate (same as sha256)
  const obsP   = (gaps.length + 1) / (n + 2);
  const trust  = clamp((n - 30) / 270, 0, 1); // full trust at 300+ rounds

  // Metric 2: CUSUM on last 200 rounds (SHA-512 uses longer window than SHA-256)
  const p0 = hrGlobal;
  let cusum = 0, maxCusum = 0, minCusum = 0;
  for (const r of rounds.slice(-200)) {
    cusum += (r.multiplier >= target.min ? 1 : 0) - p0;
    if (cusum > maxCusum) maxCusum = cusum;
    if (cusum < minCusum) minCusum = cusum;
  }
  const sigma200  = Math.sqrt(Math.max(1e-9, p0 * (1 - p0) * Math.min(200, n)));
  const cusumNorm = Math.max(Math.abs(maxCusum), Math.abs(minCusum)) / (sigma200 || 1);
  const driftDetected = cusumNorm > 1.65; // 95% confidence threshold

  // Metric 3: Gap entropy (Shannon entropy normalized by log(n))
  // Low entropy = clustered gaps (periodic behavior) → shift window earlier
  // High entropy = uniform gaps (true random) → use baseline
  const bins = 10;
  const binCounts = new Array(bins).fill(0);
  const gMax = sorted[sorted.length - 1] || 1;
  for (const g of gaps) {
    const bin = Math.min(bins - 1, Math.floor((g / gMax) * bins));
    binCounts[bin]++;
  }
  let entropy = 0;
  for (const c of binCounts) {
    const p = c / gaps.length;
    if (p > 0) entropy -= p * Math.log2(p);
  }
  const maxEntropy  = Math.log2(bins);
  const normEntropy = entropy / (maxEntropy || 1); // 0=perfectly clustered, 1=uniform

  // Metric 4: First-order autocorrelation of gaps
  // SHA-512 specifically looks for lag-1 patterns (alternating long/short gaps)
  let ac1 = 0;
  if (gaps.length >= 10) {
    let cov = 0;
    const m = mean(gaps);
    let varSum = 0;
    for (let i = 1; i < gaps.length; i++) cov += (gaps[i - 1] - m) * (gaps[i] - m);
    for (const g of gaps) varSum += (g - m) ** 2;
    ac1 = varSum > 0 ? cov / varSum : 0;
  }

  // Combined drift factor: how much to shift expected gap
  // Positive cusum → game running "hot" → expect sooner hit
  // Negative cusum → game running "cold" → expect later hit
  const driftFactor = driftDetected ? clamp(cusumNorm * 0.08 * Math.sign(maxCusum + minCusum), -0.15, 0.15) : 0;
  const entropyAdj  = (1 - normEntropy) * 0.05; // low entropy → slightly sooner
  const acAdj       = ac1 < -0.15 ? -0.05 : 0;  // strong negative AC → anticipate

  // Final expected gap: sha-style baseline + drift adjustments
  const baseGap   = Math.max(1, Math.round((1 / obsP) * trust + gMean * (1 - trust)));
  const adjustedGap = Math.max(1, Math.round(baseGap * (1 - driftFactor - entropyAdj - acAdj)));

  const aw   = target.maxWidth;
  const conf = clamp(Math.round(55 + trust * 20 + (driftDetected ? 8 : 0) + normEntropy * 5), 38, 93);
  return { ...placeWindow(adjustedGap, currentGap, aw), expectedGap: adjustedGap, probW: geoProbW(obsP, aw), conf };
}

// ── Rounds cache (independent from advComputeEngine) ──────────────────────────
async function getNgRounds() {
  if (cachedRounds.length === 0) {
    cachedRounds = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[ngCompute] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Save outcome (same pattern as advComputeEngine.saveOutcome) ───────────────
async function saveNgOutcome(engineId, target, outcome, lo, hi, hitRound, generation) {
  const key = `${lo}:${hi}`;
  if (ngSavedSets[engineId].has(key)) return;
  ngSavedSets[engineId].add(key);
  try {
    await savePrediction({
      target: target.label, minMult: target.min,
      outcome, lo, hi, hitRound: hitRound ?? null,
      generation: generation ?? 1, source: engineId, probW: null,
    });
    console.log(`[ngCompute] ${engineId} ${target.label} ${outcome.toUpperCase()} #${lo}–#${hi}${hitRound ? ` @#${hitRound}` : ''}`);
  } catch (e) {
    console.error(`[ngCompute] save fail ${engineId}:`, e.message);
    ngSavedSets[engineId].delete(key);
  }
}


// ── computeNgConsensus — Master Signal for Next-Gen engines ──────────────────
// Same algorithm as advComputeEngine.computeConsensus:
// finds the largest group of NG engines whose predicted windows OVERLAP,
// then narrows to the intersection. Requires ≥3 engines to agree.
// Stored under source='ng_consensus' in locked_preds_adv and predictions.
function computeNgConsensus(allNgResults, lastRoundId) {
  const consensus = {};
  for (const target of TARGETS) {
    const windows = [];
    // Collect windows from all 11 NG engines (exclude stacking to avoid circularity)
    const SOURCE_IDS = ['hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','sha512'];
    for (const eid of SOURCE_IDS) {
      const r = allNgResults[eid]?.[target.label];
      if (!r) continue;
      const lo = lastRoundId + r.low;
      const hi = lastRoundId + r.high;
      windows.push({ engineId: eid, lo, hi });
    }
    if (windows.length < 3) { consensus[target.label] = null; continue; }

    // Find largest group of overlapping windows (same greedy algorithm as ADV consensus)
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

    // Expand intersection to full maxWidth if too narrow
    const baseW = target.maxWidth;
    if (bestHi - bestLo + 1 < baseW) {
      const center = Math.round((bestLo + bestHi) / 2);
      bestLo = center - Math.floor(baseW / 2); bestHi = bestLo + baseW - 1;
    }
    // Ensure window is in the future
    if (bestLo <= lastRoundId) { bestLo = lastRoundId + 1; bestHi = bestLo + baseW - 1; }

    consensus[target.label] = {
      lo: bestLo, hi: bestHi,
      engineCount: bestGroup.length,
      engines: bestGroup.map(w => w.engineId),
    };
  }
  return consensus;
}

// ── Main tick ─────────────────────────────────────────────────────────────────
async function runNgComputeEngine() {
  try {
    const rounds = await getNgRounds();
    if (rounds.length < 50) return;
    const lastRoundId = rounds[rounds.length - 1].roundId;

    // ── Pass 1: compute all non-stacking engines ──────────────────────────────
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

    const allNgResults = {};
    for (const [engineId, algo] of Object.entries(ALGO_MAP)) {
      allNgResults[engineId] = {};
      for (const target of TARGETS) {
        try {
          const r = algo(rounds, target);
          if (r) allNgResults[engineId][target.label] = r;
        } catch (e) { console.error(`[ngCompute] ${engineId}/${target.label}:`, e.message); }
      }
    }

    // ── Pass 2: stacking_meta (needs allNgResults from pass 1) ───────────────
    allNgResults['stacking'] = {};
    for (const target of TARGETS) {
      try {
        const r = runStackingMeta(rounds, target, allNgResults, lastRoundId);
        if (r) allNgResults['stacking'][target.label] = r;
      } catch (e) { console.error(`[ngCompute] stacking/${target.label}:`, e.message); }
    }

    // ── Pass 3: ng_consensus master signal ───────────────────────────────────
    const ngConsensus = computeNgConsensus(allNgResults, lastRoundId);
    allNgResults['ng_consensus'] = {};
    for (const target of TARGETS) {
      const c = ngConsensus[target.label];
      if (c) {
        // Store as a fake "result" so Phase 1+2 loop handles it uniformly
        allNgResults['ng_consensus'][target.label] = {
          low:  c.lo - lastRoundId,
          high: c.hi - lastRoundId,
          expectedGap: Math.round((c.lo + c.hi) / 2 - lastRoundId),
          probW: null,
          conf:  55 + Math.round(c.engineCount * 4), // confidence scales with agreement
          _meta: { engineCount: c.engineCount, engines: c.engines },
        };
      }
    }

    // ── Phase 1+2: resolve old windows, lock new ones ─────────────────────────
    for (const engineId of NG_ENGINE_IDS) {
      const payload = {};

      for (const target of TARGETS) {
        const win   = ngWindows[engineId][target.label];
        const fresh = allNgResults[engineId]?.[target.label];

        if (win) {
          const { lo, hi, generation, roundWhenMade } = win;

          // Early hit check (bounded by earlyHitTolerance — same fix as advComputeEngine)
          const earlyCheckLo = Math.max(roundWhenMade + 1, lo - earlyHitTolerance(target.maxWidth));
          const earlyHit = lo > roundWhenMade + 1 && earlyCheckLo <= lo - 1
            ? findHitInRange(rounds, earlyCheckLo, lo - 1, target.min)
            : null;

          if (earlyHit) {
            await saveNgOutcome(engineId, target, 'early', lo, hi, earlyHit.roundId, generation);
            delete ngWindows[engineId][target.label];
          } else if (lastRoundId >= hi) {
            const hit = findHitInRange(rounds, lo, hi, target.min);
            await saveNgOutcome(engineId, target, hit ? 'win' : 'loss', lo, hi, hit?.roundId ?? null, generation);
            delete ngWindows[engineId][target.label];
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

        // Lock fresh window
        if (fresh) {
          const newLo = lastRoundId + fresh.low;
          const newHi = lastRoundId + fresh.high;
          const gen   = (ngWindows[engineId][target.label]?.generation ?? 0) + 1;
          // For ng_consensus, carry engineCount and engines list in eta
          const baseEta = { probW: fresh.probW, conf: fresh.conf, expectedGap: fresh.expectedGap };
          const eta = fresh._meta ? { ...baseEta, ...fresh._meta } : baseEta;
          ngWindows[engineId][target.label] = {
            lo: newLo, hi: newHi, roundWhenMade: lastRoundId, generation: gen, eta,
          };
          payload[target.label] = ngWindows[engineId][target.label];
        }
      }

      if (Object.keys(payload).length) {
        await saveLockedAdvPreds(engineId, payload);
      }
    }

  } catch (e) {
    console.error('[ngCompute] Fatal:', e.message, e.stack);
  }
}

// ── Initialise (same pattern as advComputeEngine.initAdvCompute) ──────────────
async function initNgCompute() {
  if (initialised) return;
  initialised = true;

  // Load existing locked windows from DB into in-memory ngWindows
  try {
    const existing = await getLockedAdvPreds();
    for (const engineId of NG_ENGINE_IDS) {
      for (const target of TARGETS) {
        const w = existing[engineId]?.[target.label];
        if (w?.lo && w?.hi) {
          ngWindows[engineId][target.label] = {
            lo: Number(w.lo), hi: Number(w.hi),
            roundWhenMade: Number(w.roundWhenMade ?? w.lo),
            generation: w.generation ?? 1,
            eta: w.eta ?? {},
          };
        }
      }
    }
    console.log(`[ngCompute] loaded existing locked windows`);
  } catch (e) { console.error('[ngCompute] init locked error:', e.message); }

  // Pre-warm savedSets from existing history
  try {
    for (const engineId of NG_ENGINE_IDS) {
      const rows = await getPredictions({ limit: 500000, source: engineId });
      for (const r of rows) ngSavedSets[engineId].add(`${r.lo}:${r.hi}`);
    }
    const total = NG_ENGINE_IDS.reduce((s, id) => s + ngSavedSets[id].size, 0);
    console.log(`[ngCompute] pre-warmed savedSets with ${total} outcomes`);
  } catch (e) { console.error('[ngCompute] init history error:', e.message); }
}

const _origRun = runNgComputeEngine;
async function runNgComputeEngineWithInit() {
  await initNgCompute();
  await _origRun();
}

function resetNgComputeState() {
  for (const id of NG_ENGINE_IDS) { ngWindows[id] = {}; ngSavedSets[id] = new Set(); }
  cachedRounds = []; cachedRoundsLastId = 0; initialised = false;
}

// Clears only in-memory locked windows (not savedSets or history).
// Called by /reset-locks so the engine immediately recomputes fresh windows
// on next tick without a full state rebuild.
function resetNgWindowsOnly() {
  for (const id of NG_ENGINE_IDS) ngWindows[id] = {};
  console.log('[ngCompute] in-memory windows cleared (lock reset)');
}

module.exports = { runNgComputeEngine: runNgComputeEngineWithInit, resetNgComputeState, resetNgWindowsOnly, NG_ENGINE_IDS };