'use strict';
// patternEngine.js — Server-side Pattern Engine
// Fully independent. Uses full round dataset.
// Two-phase per tick:
//   Phase 1: resolve any expired/hit windows → save outcome to predictions
//   Phase 2: for targets with no active window → compute & lock new window
// No sliding. No overwriting active windows. No dual code paths.

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedPatternPreds,
  getLockedPatternPreds,
} = require('./db');

const ENGINE_ID  = 'pattern';
const MIN_ROUNDS = 50;
const MIN_HITS   = 8;
const MIN_GAPS   = 6;
const STALE_THRESHOLD = 50000;

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

// In-memory state
const state = {
  windows:      {},   // { [targetLabel]: { lo, hi, generation } } — absolute round numbers
  savedSet:     null, // Set of "lo:hi" keys already saved to predictions
  lastRoundId:  0,
};

let cachedRounds       = [];
let cachedRoundsLastId = 0;
let initialised        = false;

// ── Math helpers ──────────────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

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

// ── Pattern analysis ──────────────────────────────────────────────────────────
function analysePattern(rounds, targetMin) {
  const n = rounds.length;
  if (n < MIN_ROUNDS) return null;

  const W1 = 15, W2 = 50, W3 = 200;
  const s1 = Math.max(0, n - W1), s2 = Math.max(0, n - W2), s3 = Math.max(0, n - W3);

  let hits = 0, lastIdx = -1, hW1 = 0, hW2 = 0, hW3 = 0;
  const gaps = [];
  const FA = 0.20, SA = 0.02;
  let emaFast = -1, emaSlow = -1;

  for (let i = 0; i < n; i++) {
    const isHit = rounds[i].multiplier >= targetMin ? 1 : 0;
    if (emaFast < 0) { emaFast = isHit; emaSlow = isHit; }
    else { emaFast = FA * isHit + (1 - FA) * emaFast; emaSlow = SA * isHit + (1 - SA) * emaSlow; }
    if (isHit) {
      if (lastIdx !== -1) gaps.push(i - lastIdx - 1);
      lastIdx = i; hits++;
      if (i >= s1) hW1++;
      if (i >= s2) hW2++;
      if (i >= s3) hW3++;
    }
  }

  if (hits < MIN_HITS || gaps.length < MIN_GAPS) return null;

  const globalRate   = hits / n;
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap   = gSum / gaps.length;
  const stdGap    = Math.sqrt(Math.max(0, gSS / gaps.length - meanGap ** 2));
  const cv        = meanGap > 0 ? stdGap / meanGap : 1;
  const sg        = [...gaps].sort((a, b) => a - b);
  const mid2      = Math.floor(sg.length / 2);
  const medianGap = sg.length % 2 === 1 ? sg[mid2] : (sg[mid2 - 1] + sg[mid2]) / 2;

  const dW1 = hW1 / W1, dW2 = hW2 / W2, dW3 = hW3 / Math.min(W3, n);
  const safe = v => Math.max(-1, Math.min(1, v));
  const clusterScore = safe((safe((dW1-globalRate)/Math.max(globalRate,0.001)))*0.50 + (safe((dW2-globalRate)/Math.max(globalRate,0.001)))*0.30 + (safe((dW3-globalRate)/Math.max(globalRate,0.001)))*0.20);
  const trendScore   = safe((emaSlow > 0 ? (emaFast - emaSlow) / emaSlow : 0) * 4);

  let varSum = 0;
  for (const g of gaps) varSum += (g - meanGap) ** 2;
  let bestAC = 0;
  for (let lag = 1; lag <= Math.min(5, gaps.length - 1); lag++) {
    let cov = 0;
    for (let i = lag; i < gaps.length; i++) cov += (gaps[i - lag] - meanGap) * (gaps[i] - meanGap);
    const ac = varSum > 0 ? cov / varSum : 0;
    if (Math.abs(ac) > Math.abs(bestAC)) bestAC = ac;
  }
  const patternScore = safe(bestAC * 0.9);

  const last5     = gaps.slice(-5);
  const last5Mean = last5.reduce((s, v) => s + v, 0) / last5.length;
  const momentum  = meanGap > 0 ? (meanGap - last5Mean) / meanGap : 0;

  const composite    = clusterScore * 0.40 + trendScore * 0.30 + patternScore * 0.15 + safe(momentum) * 0.15;
  const absComposite = Math.abs(composite);
  const direction    = composite > 0.08 ? 'bullish' : composite < -0.08 ? 'bearish' : 'neutral';
  const agree = Math.max(
    [clusterScore, trendScore, patternScore].filter(s => s > 0.08).length,
    [clusterScore, trendScore, patternScore].filter(s => s < -0.08).length
  );
  const conf = Math.max(25, Math.min(82,
    Math.round(32 + Math.min(18, Math.log2(hits + 1) * 4) + absComposite * 30 + (agree - 1) * 6
      - (cv > 1.5 ? 8 : cv > 1.2 ? 4 : 0)
      - (gapSinceLast > meanGap * 2 ? 5 : 0))
  ));

  return { direction, confidence: conf, hits, meanGap: Math.round(meanGap),
    medianGap: Math.round(medianGap), composite: +composite.toFixed(3),
    momentum: +momentum.toFixed(3), gapSinceLast,
    clusterScore: +clusterScore.toFixed(3), trendScore: +trendScore.toFixed(3),
    patternScore: +patternScore.toFixed(3), cv: +cv.toFixed(2) };
}

function buildWindow(pr, maxWidth, anchorRound) {
  if (!pr) return null;
  const medianGap    = pr.medianGap || pr.meanGap || maxWidth;
  const gapSinceLast = pr.gapSinceLast ?? 0;
  const momentum     = pr.momentum ?? 0;
  const meanGap      = pr.meanGap || medianGap;
  const momentumAdj  = Math.max(0.70, Math.min(1.30, 1 - momentum * 0.30));
  const overdueFactor= gapSinceLast > meanGap * 1.5 ? 0.80 : 1.0;
  const expectedGap  = Math.max(1, Math.round(medianGap * momentumAdj * overdueFactor));
  const remaining    = Math.max(1, expectedGap - gapSinceLast);
  const low          = Math.max(1, remaining - Math.floor(maxWidth / 2));
  const hi           = low + maxWidth - 1;
  return {
    lo: anchorRound + low,
    hi: anchorRound + hi,
    expectedGap,
    confidence: pr.confidence,
    direction: pr.direction,
    composite: pr.composite ?? null,
    momentum: pr.momentum ?? null,
    gapSinceLast,
    eta: {
      low, high: hi, conf: pr.confidence, expectedGap,
      direction: pr.direction, composite: pr.composite ?? null,
      momentum: pr.momentum ?? null, gapSinceLast,
    },
  };
}

// ── Rounds cache ──────────────────────────────────────────────────────────────
async function getPatternRounds() {
  if (cachedRounds.length === 0) {
    cachedRounds = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[pattern] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Save outcome ──────────────────────────────────────────────────────────────
async function saveOutcome(target, outcome, lo, hi, hitRound, generation) {
  const key = `${lo}:${hi}`;
  if (state.savedSet.has(key)) return;
  state.savedSet.add(key);
  try {
    await savePrediction({
      target: target.label, minMult: target.min,
      outcome, lo, hi,
      hitRound: hitRound ?? null,
      generation: generation ?? 1,
      source: ENGINE_ID,
      probW: null,
    });
    console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${lo}–#${hi}${hitRound ? ` @#${hitRound}` : ''}`);
  } catch(e) {
    console.error(`[pattern] save fail:`, e.message);
    state.savedSet.delete(key);
  }
}

// ── Main tick ─────────────────────────────────────────────────────────────────
async function processPatternEngine(rounds, lastRoundId) {
  const toSave = {}; // windows to persist to locked_preds_pattern
  let anyChange = false;

  for (const target of TARGETS) {
    const win = state.windows[target.label];

    // ── Phase 1: resolve if window exists and is expired or hit ──────────────
    if (win) {
      const { lo, hi, generation } = win;
      const isTooOld = lastRoundId - hi > STALE_THRESHOLD;

      // Check for early hit (before window opened)
      const earlyHit = findHitInRange(rounds, win.roundWhenMade + 1, lo - 1, target.min);
      if (earlyHit) {
        await saveOutcome(target, 'early', lo, hi, earlyHit.roundId, generation);
        delete state.windows[target.label];
        anyChange = true;
        // Fall through to Phase 2 to lock new window
      } else if (lastRoundId >= hi) {
        // Window has closed — check if hit inside
        if (!isTooOld) {
          const hit = findHitInRange(rounds, lo, hi, target.min);
          await saveOutcome(target, hit ? 'win' : 'loss', lo, hi, hit?.roundId ?? null, generation);
        }
        delete state.windows[target.label];
        anyChange = true;
        // Fall through to Phase 2
      } else {
        // Window still active — check for in-window hit (resolve early)
        const hit = findHitInRange(rounds, lo, hi, target.min);
        if (hit) {
          await saveOutcome(target, 'win', lo, hi, hit.roundId, generation);
          delete state.windows[target.label];
          anyChange = true;
          // Fall through to Phase 2
        } else {
          // Still waiting — keep window, write to save payload
          toSave[target.label] = win;
          continue;
        }
      }
    }

    // ── Phase 2: no active window — compute and lock a new one ───────────────
    const pr   = analysePattern(rounds, target.min);
    const pred = buildWindow(pr, target.maxWidth, lastRoundId);
    if (pred) {
      const generation = (win?.generation ?? 0) + 1;
      state.windows[target.label] = {
        lo: pred.lo, hi: pred.hi,
        roundWhenMade: lastRoundId,
        generation,
        eta: pred.eta,
      };
      toSave[target.label] = state.windows[target.label];
      anyChange = true;
      console.log(`[pattern] LOCK ${target.label}: #${pred.lo}–#${pred.hi} dir=${pred.direction} conf=${pred.confidence}%`);
    }
  }

  // Persist locked windows to DB
  if (anyChange && Object.keys(toSave).length) {
    const payload = {};
    for (const [label, w] of Object.entries(toSave)) {
      payload[label] = {
        lo: w.lo, hi: w.hi,
        roundWhenMade: w.roundWhenMade,
        generation: w.generation,
        eta: w.eta,
      };
    }
    try { await saveLockedPatternPreds(payload); }
    catch(e) { console.error('[pattern] save locked fail:', e.message); }
  }

  return anyChange;
}

// ── Initialise ────────────────────────────────────────────────────────────────
async function initialise() {
  if (initialised) return;
  initialised = true;
  state.savedSet = new Set();
  state.windows  = {};

  // Load existing locked windows from DB
  try {
    const dbLocked = await getLockedPatternPreds();
    for (const [label, p] of Object.entries(dbLocked)) {
      if (!p?.lo || !p?.hi) continue;
      state.windows[label] = {
        lo: Number(p.lo), hi: Number(p.hi),
        roundWhenMade: Number(p.roundWhenMade ?? p.lo),
        generation: p.generation ?? 1,
        eta: p.eta ?? {},
      };
    }
    console.log(`[pattern] loaded ${Object.keys(state.windows).length} locked windows`);
  } catch(e) {
    console.error('[pattern] init locked error:', e.message);
  }

  // Pre-warm savedSet from existing history
  try {
    const rows = await getPredictions({ limit: 500000, source: ENGINE_ID });
    for (const r of rows) state.savedSet.add(`${r.lo}:${r.hi}`);
    console.log(`[pattern] pre-warmed savedSet with ${state.savedSet.size} outcomes`);
  } catch(e) {
    console.error('[pattern] init history error:', e.message);
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetPatternEngineState() {
  console.log('[pattern] reset');
  state.windows     = {};
  state.savedSet    = null;
  state.lastRoundId = 0;
  cachedRounds       = [];
  cachedRoundsLastId = 0;
  initialised        = false;
}

// ── Entry point ───────────────────────────────────────────────────────────────
async function runPatternEngine() {
  try {
    await initialise();
    const rounds = await getPatternRounds();
    if (rounds.length < MIN_ROUNDS) {
      console.log(`[pattern] waiting (${rounds.length}/${MIN_ROUNDS})`);
      return;
    }
    const lastRoundId = rounds[rounds.length - 1].roundId;
    if (lastRoundId <= state.lastRoundId) return;
    state.lastRoundId = lastRoundId;

    const t0 = Date.now();
    await processPatternEngine(rounds, lastRoundId);
    console.log(`[pattern] tick done in ${Date.now() - t0}ms — ${Object.keys(state.windows).length} active windows`);
  } catch(e) {
    console.error('[pattern] Fatal:', e.message, e.stack);
  }
}

module.exports = { runPatternEngine, resetPatternEngineState };