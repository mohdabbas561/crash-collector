'use strict';
// patternEngine.js — STANDALONE Pattern Engine
// ═══════════════════════════════════════════════════════════════════════════════
// REQUIREMENT: This engine is 100% independent of statEngine.js.
// It shares ZERO code, helpers, constants, state, or modules with statEngine.
// It may be run, modified, or removed without touching statEngine.js.
//
// Architecture:
//   - Uses full 12k+ round dataset from DB every cycle
//   - Detects clustering / trend / autocorrelation patterns per target
//   - Stores ALL predictions and locked windows exclusively in DB
//   - No local storage, no per-user caching — DB is single source of truth
//   - Zero duplication: unique constraint (source, target, window_lo, window_hi)
// ═══════════════════════════════════════════════════════════════════════════════

const {
  getRounds,
  savePrediction,
  getPredictions,
  saveLockedPatternPreds,
  getLockedPatternPreds,
} = require('./db');

// ── Constants ─────────────────────────────────────────────────────────────────
const ENGINE_ID  = 'pattern';
const MIN_ROUNDS = 50;
const MIN_HITS   = 8;
const MIN_GAPS   = 6;
const STALE_FORCE_REBUILD_THRESHOLD = 50000;

// Targets — defined locally, no import from any shared module
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

// ── Engine state — isolated, never shared ────────────────────────────────────
const state = {
  lockedMap:    null,
  savedSet:     null,
  needsRebuild: true,
  lastRoundId:  0,
};

let cachedRounds        = [];
let cachedRoundsLastId  = 0;
let initialised         = false;

// ── Math helpers (self-contained, no imports) ─────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Pattern analysis per target ───────────────────────────────────────────────
function analysePattern(sortedRounds, targetMin) {
  const n = sortedRounds.length;
  if (n < MIN_ROUNDS) return null;

  // FIX: W3=200 (was 150) — use more data for the long-window cluster score.
  // FIX: lags up to 5 (was 3) — detect longer-range autocorrelation patterns.
  const W1 = 15, W2 = 50, W3 = 200;
  const s1 = Math.max(0, n - W1), s2 = Math.max(0, n - W2), s3 = Math.max(0, n - W3);

  let hits = 0, lastIdx = -1, hW1 = 0, hW2 = 0, hW3 = 0;
  const gaps = [];
  const FA = 0.20, SA = 0.02;
  let emaFast = -1, emaSlow = -1;

  for (let i = 0; i < n; i++) {
    const isHit = sortedRounds[i].multiplier >= targetMin ? 1 : 0;
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
  // FIX: compute gapSinceLast — essential for correct window placement
  const gapSinceLast = lastIdx === -1 ? n : n - lastIdx - 1;

  let gSum = 0, gSS = 0;
  for (const g of gaps) { gSum += g; gSS += g * g; }
  const meanGap   = gSum / gaps.length;
  const stdGap    = Math.sqrt(Math.max(0, gSS / gaps.length - meanGap ** 2));
  const cv        = meanGap > 0 ? stdGap / meanGap : 1;
  const sg        = [...gaps].sort((a, b) => a - b);
  const mid2      = Math.floor(sg.length / 2);
  const medianGap = sg.length % 2 === 1 ? sg[mid2] : (sg[mid2 - 1] + sg[mid2]) / 2;

  const dW1 = hW1 / W1;
  const dW2 = hW2 / W2;
  const dW3 = hW3 / Math.min(W3, n);
  const safe = v => Math.max(-1, Math.min(1, v));
  const rW1 = globalRate > 0 ? safe((dW1 - globalRate) / Math.max(globalRate, 0.001)) : 0;
  const rW2 = globalRate > 0 ? safe((dW2 - globalRate) / Math.max(globalRate, 0.001)) : 0;
  const rW3 = globalRate > 0 ? safe((dW3 - globalRate) / Math.max(globalRate, 0.001)) : 0;
  const clusterScore = safe(rW1 * 0.50 + rW2 * 0.30 + rW3 * 0.20);
  const trendScore   = safe((emaSlow > 0 ? (emaFast - emaSlow) / emaSlow : 0) * 4);

  let varSum = 0;
  for (const g of gaps) varSum += (g - meanGap) ** 2;
  let bestAC = 0;
  // FIX: 5 lags (was 3) — matches frontend, detects more patterns
  for (let lag = 1; lag <= Math.min(5, gaps.length - 1); lag++) {
    let cov = 0;
    for (let i = lag; i < gaps.length; i++) cov += (gaps[i - lag] - meanGap) * (gaps[i] - meanGap);
    const ac = varSum > 0 ? cov / varSum : 0;
    if (Math.abs(ac) > Math.abs(bestAC)) bestAC = ac;
  }
  const patternScore = safe(bestAC * 0.9);

  // FIX: add momentum signal — last 5 gaps vs mean (matches frontend)
  const last5     = gaps.slice(-5);
  const last5Mean = last5.reduce((s, v) => s + v, 0) / last5.length;
  const momentum  = meanGap > 0 ? (meanGap - last5Mean) / meanGap : 0;

  // FIX: weights now match frontend — 0.40/0.30/0.15/0.15 (was 0.50/0.35/0.15)
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

  return {
    direction, confidence: conf, hits,
    meanGap:      Math.round(meanGap),
    medianGap:    Math.round(medianGap),
    composite:    +composite.toFixed(3),
    momentum:     +momentum.toFixed(3),
    gapSinceLast,
    clusterScore: +clusterScore.toFixed(3),
    trendScore:   +trendScore.toFixed(3),
    patternScore: +patternScore.toFixed(3),
    cv:           +cv.toFixed(2),
  };
}

// Build a locked window from pattern analysis
function buildWindow(patternResult, maxWidth) {
  if (!patternResult) return null;

  // FIX: Use gapSinceLast + momentum + overdueFactor for window placement.
  // Old code placed window at expectedGap - maxWidth regardless of current gap —
  // this meant a window that should open NOW was placed far in the future.
  const medianGap    = patternResult.medianGap || patternResult.meanGap || maxWidth;
  const gapSinceLast = patternResult.gapSinceLast ?? 0;
  const momentum     = patternResult.momentum     ?? 0;
  const meanGap      = patternResult.meanGap      || medianGap;

  // Momentum adjustment: bullish (gaps shortening) → predict sooner
  const momentumAdj   = Math.max(0.70, Math.min(1.30, 1 - momentum * 0.30));
  // Overdue adjustment: been waiting > 1.5× mean → shift window earlier
  const overdueFactor = gapSinceLast > meanGap * 1.5 ? 0.80 : 1.0;
  const expectedGap   = Math.max(1, Math.round(medianGap * momentumAdj * overdueFactor));

  // Center window on remaining rounds until predicted hit
  const remaining = Math.max(1, expectedGap - gapSinceLast);
  const low       = Math.max(0, remaining - Math.floor(maxWidth / 2));

  return {
    low, high: low + maxWidth - 1, expectedGap,
    opensIn:   low,
    confidence: patternResult.confidence,
    direction:  patternResult.direction,
    composite:  patternResult.composite  ?? null,
    momentum:   patternResult.momentum   ?? null,
    gapSinceLast,
    streakStatus:  'normal',
    currentStreak: 0,
  };
}

// ── Dedup key ─────────────────────────────────────────────────────────────────
function makeKey(target, lo, hi) {
  return `${ENGINE_ID}-${target}-${Number(lo) || 0}-${Number(hi) || 0}`;
}

// ── Status resolution ─────────────────────────────────────────────────────────
function getStatus(sortedRounds, pred, currentRoundId) {
  const anchorRound = Number(pred.anchorRound) || 0;
  const absLow  = anchorRound + (Number(pred.low)  || 0);
  const absHigh = anchorRound + (Number(pred.high) || 0);
  if (!Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow)
    return { status: 'miss', hitRound: null };

  let lo = 0, hi = sortedRounds.length - 1, startIdx = sortedRounds.length;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedRounds[mid].roundId >= anchorRound) { startIdx = mid; hi = mid - 1; }
    else lo = mid + 1;
  }
  for (let i = startIdx; i < sortedRounds.length; i++) {
    const r = sortedRounds[i];
    if (r.roundId > absHigh) break;
    if (r.multiplier < pred.targetMin) continue;
    if (r.roundId < absLow) return { status: 'early', hitRound: r.roundId };
    return { status: 'hit', hitRound: r.roundId };
  }
  if (currentRoundId > absHigh) return { status: 'miss', hitRound: null };
  if (currentRoundId >= absLow && currentRoundId <= absHigh) return { status: 'active', hitRound: null };
  return { status: 'waiting', hitRound: null };
}

// ── buildSavePayload ──────────────────────────────────────────────────────────
function buildSavePayload(lockedMap) {
  const out = {};
  for (const [label, pred] of Object.entries(lockedMap)) {
    if (pred.stale) continue;
    const anchor = Number(pred.anchorRound);
    if (!Number.isFinite(anchor) || anchor === 0) continue;
    out[label] = {
      lo: anchor + (Number(pred.low) || 0),
      hi: anchor + (Number(pred.high) || 0),
      roundWhenMade: anchor,
      generation: pred.generation || 1,
      eta: {
        low: pred.low, high: pred.high, conf: pred.confidence,
        expectedGap: pred.expectedGap, opensIn: pred.opensIn,
        direction:   pred.direction,   streakStatus: pred.streakStatus,
        composite:   pred.composite   ?? null,
        momentum:    pred.momentum    ?? null,
        gapSinceLast: pred.gapSinceLast ?? null,
      },
    };
  }
  return out;
}

// ── loadLockedMap ─────────────────────────────────────────────────────────────
function loadLockedMap(dbRows) {
  const map = {};
  for (const [label, pred] of Object.entries(dbRows)) {
    const target = TARGETS.find(t => t.label === label);
    if (!target) continue;
    const eta    = pred.eta || {};
    const anchor = Number(pred.roundWhenMade ?? pred.lo) || 0;
    const low    = eta.low  != null ? eta.low  : Math.max(0, Number(pred.lo) - anchor);
    const high   = eta.high != null ? eta.high : Math.max(0, Number(pred.hi) - anchor);
    if (high - low + 1 > target.maxWidth) {
      console.log(`[pattern] DISCARD stale wide window ${label}: ${high - low + 1}r > max ${target.maxWidth}r`);
      continue;
    }
    map[label] = {
      low, high,
      confidence:   eta.conf         ?? 50,
      expectedGap:  eta.expectedGap  ?? null,
      opensIn:      eta.opensIn      ?? null,
      direction:    eta.direction    ?? 'neutral',
      streakStatus: eta.streakStatus ?? 'normal',
      composite:    eta.composite    ?? null,
      momentum:     eta.momentum     ?? null,
      gapSinceLast: eta.gapSinceLast ?? null,
      currentStreak: 0,
      targetMin:   target.min,
      anchorRound: anchor,
      generation:  pred.generation ?? 1,
      stale:       false,
    };
  }
  return map;
}

// ── Main processing loop ──────────────────────────────────────────────────────
async function processPatternEngine(sortedRounds, lastRoundId) {
  let anyChange = false;

  for (const target of TARGETS) {
    const existing = state.lockedMap[target.label];

    if (!existing) {
      const pr   = analysePattern(sortedRounds, target.min);
      const pred = buildWindow(pr, target.maxWidth);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min,
          anchorRound: lastRoundId, generation: 1, stale: false,
        };
        anyChange = true;
        console.log(`[pattern] NEW ${target.label}: +${pred.low}–+${pred.high} dir=${pred.direction} conf=${pred.confidence}%`);
      }
      continue;
    }

    const anchorRound = Number(existing.anchorRound) || 0;
    const absLow      = anchorRound + (Number(existing.low)  || 0);
    const absHigh     = anchorRound + (Number(existing.high) || 0);
    const isNonsense  = !Number.isFinite(absLow) || !Number.isFinite(absHigh) || absHigh < absLow || anchorRound === 0;
    const isExpired   = lastRoundId >= absHigh;
    const isStale     = !!existing.stale;
    const isTooOld    = isExpired && (lastRoundId - absHigh) > STALE_FORCE_REBUILD_THRESHOLD;

    if (isNonsense || isExpired || isStale) {
      if (!isNonsense && !isTooOld) {
        const status = getStatus(sortedRounds, existing, lastRoundId);
        if (['hit', 'miss', 'early'].includes(status.status)) {
          const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
          const key     = makeKey(target.label, absLow, absHigh);
          if (!state.savedSet.has(key)) {
            state.savedSet.add(key);
            try {
              await savePrediction({
                target: target.label, minMult: target.min,
                outcome, lo: absLow, hi: absHigh,
                hitRound: status.hitRound || null,
                generation: existing.generation || 1,
                source: ENGINE_ID,
                probW: null,
              });
            } catch(e) {
              console.error(`[pattern] save fail:`, e.message);
              state.savedSet.delete(key); // FIX: remove so retry is possible next tick
            }
          }
        }
      }
      const pr   = analysePattern(sortedRounds, target.min);
      const pred = buildWindow(pr, target.maxWidth);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min,
          anchorRound: lastRoundId,
          generation: (existing.generation || 1) + (isNonsense ? 0 : 1),
          stale: false,
        };
        console.log(`[pattern] REBUILD ${target.label}: +${pred.low}–+${pred.high} dir=${pred.direction}`);
      } else {
        delete state.lockedMap[target.label];
        console.warn(`[pattern] ${target.label} cleared — insufficient data`);
      }
      anyChange = true;
      state.needsRebuild = false;
      continue;
    }

    const status = getStatus(sortedRounds, existing, lastRoundId);
    if (['hit', 'miss', 'early'].includes(status.status)) {
      const outcome = status.status === 'hit' ? 'win' : status.status === 'early' ? 'early' : 'loss';
      const key     = makeKey(target.label, absLow, absHigh);
      if (!state.savedSet.has(key)) {
        state.savedSet.add(key);
        try {
          await savePrediction({
            target: target.label, minMult: target.min,
            outcome, lo: absLow, hi: absHigh,
            hitRound: status.hitRound || null,
            generation: existing.generation || 1,
            source: ENGINE_ID,
            probW: null,
          });
          console.log(`[pattern] ${target.label} ${outcome.toUpperCase()} #${absLow}–#${absHigh}${status.hitRound ? ` @#${status.hitRound}` : ''}`);
        } catch(e) {
          console.error(`[pattern] save fail:`, e.message);
          state.savedSet.delete(key); // FIX: remove so retry is possible next tick
        }
      }
      const pr   = analysePattern(sortedRounds, target.min);
      const pred = buildWindow(pr, target.maxWidth);
      if (pred) {
        state.lockedMap[target.label] = {
          ...pred, targetMin: target.min,
          anchorRound: lastRoundId,
          generation: (existing.generation || 1) + 1,
          stale: false,
        };
        console.log(`[pattern] NEXT ${target.label}: +${pred.low}–+${pred.high}`);
      } else {
        delete state.lockedMap[target.label];
      }
      anyChange = true;
    }
  }

  return anyChange;
}

// ── In-memory rounds cache (local to this engine, not shared) ─────────────────
async function getPatternRounds() {
  if (cachedRounds.length === 0) {
    const all = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRounds       = all;
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[pattern] loaded ${cachedRounds.length} rounds`);
  } else {
    // FIX: fetch up to 5000 new rounds per cycle (was 500) — handles large offline gaps
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Initialise (once) ─────────────────────────────────────────────────────────
async function initialise() {
  if (initialised) return;
  initialised = true;
  state.savedSet = new Set();

  try {
    state.lockedMap = loadLockedMap(await getLockedPatternPreds());
    console.log(`[pattern] loaded ${Object.keys(state.lockedMap).length} locked preds`);
  } catch(e) {
    console.error('[pattern] init error:', e.message);
    state.lockedMap = {};
  }

  // Pre-warm savedSet from history — prevents re-saving already-resolved predictions
  try {
    const rows = await getPredictions({ limit: 10000, source: ENGINE_ID });
    for (const r of rows) {
      state.savedSet.add(makeKey(r.target, r.lo, r.hi));
    }
    console.log(`[pattern] loaded ${rows.length} history keys`);
  } catch(e) {
    console.error('[pattern] history init error:', e.message);
  }

  state.needsRebuild = true;
}

// ── resetEngineState ──────────────────────────────────────────────────────────
function resetPatternEngineState() {
  console.log('[pattern] resetPatternEngineState()');
  state.lockedMap    = null;
  state.savedSet     = null;
  state.needsRebuild = true;
  state.lastRoundId  = 0;
  cachedRounds       = [];
  cachedRoundsLastId = 0;
  initialised        = false;
}

// ── Main entry point ──────────────────────────────────────────────────────────
async function runPatternEngine() {
  try {
    await initialise();
    const rounds = await getPatternRounds();
    if (rounds.length < MIN_ROUNDS) {
      console.log(`[pattern] waiting (${rounds.length}/${MIN_ROUNDS})`);
      return;
    }
    const lastRoundId = rounds[rounds.length - 1].roundId;

    if (!(lastRoundId > state.lastRoundId || state.needsRebuild)) return;
    state.needsRebuild = false;

    const t0 = Date.now();
    const changed = await processPatternEngine(rounds, lastRoundId);
    state.lastRoundId = lastRoundId;

    if (changed) {
      const payload = buildSavePayload(state.lockedMap);
      if (Object.keys(payload).length) {
        try { await saveLockedPatternPreds(payload); }
        catch(e) { console.error('[pattern] save locked:', e.message); }
      }
    }
    console.log(`[pattern] done in ${Date.now() - t0}ms — ${Object.keys(state.lockedMap).length} targets locked`);
  } catch(e) {
    console.error('[pattern] Fatal:', e.message, e.stack);
  }
}

module.exports = { runPatternEngine, resetPatternEngineState };