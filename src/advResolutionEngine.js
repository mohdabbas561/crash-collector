'use strict';
// advResolutionEngine.js — Server-side Advanced Engine Resolution
// ═══════════════════════════════════════════════════════════════════════════════
// PROBLEM SOLVED: Advanced engine windows were only resolved when a user had
// the browser tab open. If offline, windows expired with no outcome saved.
//
// SOLUTION: This server-side resolver runs every collector tick (8s).
// It reads all locked adv windows from DB, scans rounds for hits, and saves
// win/loss/early outcomes — exactly like patternEngine.js does for PTN.
//
// Architecture:
//   Frontend (AdvancedEngines.jsx) — computes windows, POSTs to /locked-adv
//   THIS FILE                      — resolves windows, saves to /predictions
//   Frontend                       — reads history from /predictions
//
// INDEPENDENCE: This file shares ZERO code with patternEngine.js or statEngine.js.
// It has its own TARGETS, savedSet, rounds access, and state.
// ═══════════════════════════════════════════════════════════════════════════════

const {
  getRounds,
  savePrediction,
  getPredictions,
  getLockedAdvPreds,
  getLockedConsensusPreds,
} = require('./db');

// ── Constants ─────────────────────────────────────────────────────────────────
const ENGINE_IDS = [
  'lstm','xgb','rf','ols','cat',
  'hardgap','softgap','markov','percentile','bayes',
  'sha256','mt','lcg',
];
const CONSENSUS_ID = 'consensus';

// Targets — defined locally, zero shared imports
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

// ── State — isolated, never shared ────────────────────────────────────────────
// savedSet: prevents double-saving the same outcome even under rapid restarts
const savedSet    = new Set();
let   initialised = false;

// Local rounds cache — independent from patternEngine and statEngine caches
let cachedRounds        = [];
let cachedRoundsLastId  = 0;

// FIX: Cache locked windows to avoid 2 full DB queries every 8s tick.
// Locked windows only change when the frontend POSTs new ones (infrequent).
// TTL=30s means at most 3-4 stale ticks after a new window is posted.
// The api.js lockedAdvCache already caches GET responses for 8s, but the
// advResolutionEngine calls db functions directly (bypassing api cache).
const LOCKED_CACHE_TTL_MS = 30000;
let lockedAdvCache      = null;
let lockedAdvCacheTs    = 0;
let lockedConsCache     = null;
let lockedConsCacheTs   = 0;

// Call this when admin resets locked windows so cache is immediately busted
function bustLockedCache() {
  lockedAdvCache   = null;
  lockedAdvCacheTs = 0;
  lockedConsCache  = null;
  lockedConsCacheTs = 0;
}

// ── Binary search — O(log n) hit detection on 12k+ rounds ─────────────────────
function bisectLeft(rounds, targetId) {
  let lo = 0, hi = rounds.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (rounds[mid].roundId < targetId) lo = mid + 1;
    else hi = mid;
  }
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

// ── Dedup key ─────────────────────────────────────────────────────────────────
function makeKey(engineId, target, lo, hi) {
  return `${engineId}:${target}:${Number(lo)||0}:${Number(hi)||0}`;
}

// ── In-memory rounds cache ────────────────────────────────────────────────────
async function getAdvRounds() {
  if (cachedRounds.length === 0) {
    const all = await getRounds({ limit: 100000, order: 'ASC' });
    cachedRounds       = all;
    cachedRoundsLastId = cachedRounds.length ? cachedRounds[cachedRounds.length - 1].roundId : 0;
    console.log(`[advRes] loaded ${cachedRounds.length} rounds`);
  } else {
    const newRounds = await getRounds({ limit: 5000, minRoundId: cachedRoundsLastId + 1 });
    if (newRounds.length) {
      cachedRounds       = [...cachedRounds, ...newRounds];
      cachedRoundsLastId = cachedRounds[cachedRounds.length - 1].roundId;
    }
  }
  return cachedRounds;
}

// ── Initialise — pre-warm savedSet from existing history ──────────────────────
async function initialise() {
  if (initialised) return;
  initialised = true;
  try {
    // Load all existing adv + consensus history into savedSet
    // This prevents re-saving outcomes that were already recorded
    // FIX: no limit — load ALL existing outcomes into savedSet.
    // With 14 engines × ~10k records each, a limit of 5000 only covered 46%
    // of history, causing the server to waste cycles re-checking already-saved
    // windows. DB constraint prevents actual duplicates but wastes DB I/O.
    // We batch-load all 14 engines in parallel to minimize startup time.
    // FIX: load savedSet in sequential batches of 4 to avoid OOM.
    // 14 engines × up to 50k rows each = potentially 700k rows in memory at once.
    // Sequential batches of 4 keep peak memory bounded while still being fast.
    const allIds   = [...ENGINE_IDS, CONSENSUS_ID];
    const BATCH    = 4;
    for (let i = 0; i < allIds.length; i += BATCH) {
      const batch   = allIds.slice(i, i + BATCH);
      const results = await Promise.all(
        batch.map(engineId => getPredictions({ limit: 500000, source: engineId }))
      );
      results.forEach((rows, j) => {
        const engineId = batch[j];
        for (const r of rows) {
          savedSet.add(makeKey(engineId, r.target, r.lo, r.hi));
        }
      });
    }
    console.log(`[advRes] pre-warmed savedSet with ${savedSet.size} existing outcomes across ${allIds.length} engines`);
  } catch(e) {
    console.error('[advRes] init error:', e.message);
  }
}

// ── Resolve one engine's locked windows ───────────────────────────────────────
async function resolveEngineWindows(engineId, lockedByTarget, rounds, lastRoundId) {
  let resolved = 0;

  for (const target of TARGETS) {
    const w = lockedByTarget[target.label];
    if (!w) continue;

    const lo  = Number(w.lo);
    const hi  = Number(w.hi);
    const rwm = Number(w.roundWhenMade ?? lo);

    // Skip windows that haven't expired yet
    if (lastRoundId <= hi) continue;

    // Skip windows with bad data
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo || lo <= 0) continue;

    const key = makeKey(engineId, target.label, lo, hi);
    if (savedSet.has(key)) continue; // already resolved

    // Check for early hit (hit before window opened)
    const earlyHit = lo > rwm
      ? findHitInRange(rounds, rwm, lo - 1, target.min)
      : null;

    if (earlyHit) {
      savedSet.add(key);
      try {
        await savePrediction({
          target:     target.label,
          minMult:    target.min,
          outcome:    'early',
          lo, hi,
          hitRound:   earlyHit.roundId,
          generation: w.generation ?? 1,
          source:     engineId,
          probW:      null,
        });
        console.log(`[advRes] ${engineId} ${target.label} EARLY @#${earlyHit.roundId} window #${lo}–#${hi}`);
        resolved++;
      } catch(e) { console.error(`[advRes] save error:`, e.message); savedSet.delete(key); }
      continue;
    }

    // Check for hit inside window
    const hit = findHitInRange(rounds, lo, hi, target.min);

    savedSet.add(key);
    try {
      await savePrediction({
        target:     target.label,
        minMult:    target.min,
        outcome:    hit ? 'win' : 'loss',
        lo, hi,
        hitRound:   hit ? hit.roundId : null,
        generation: w.generation ?? 1,
        source:     engineId,
        probW:      null,
      });
      console.log(`[advRes] ${engineId} ${target.label} ${hit ? 'WIN' : 'LOSS'} window #${lo}–#${hi}${hit ? ` @#${hit.roundId}` : ''}`);
      resolved++;
    } catch(e) { console.error(`[advRes] save error:`, e.message); savedSet.delete(key); }
  }

  return resolved;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetAdvResolutionState() {
  console.log('[advRes] reset');
  savedSet.clear();
  cachedRounds       = [];
  cachedRoundsLastId = 0;
  initialised        = false;
  bustLockedCache(); // FIX: also bust locked window cache on reset
}

// ── Main entry point — called every collector tick ────────────────────────────
async function runAdvResolutionEngine() {
  try {
    await initialise();
    const rounds = await getAdvRounds();
    if (!rounds.length) return;

    const lastRoundId = rounds[rounds.length - 1].roundId;

    // FIX: Load locked windows with TTL cache — avoids 2 full DB queries every 8s.
    // Without cache, this was hitting the DB on every single tick regardless of changes.
    const now = Date.now();
    if (!lockedAdvCache || (now - lockedAdvCacheTs) > LOCKED_CACHE_TTL_MS) {
      lockedAdvCache   = await getLockedAdvPreds();
      lockedAdvCacheTs = now;
    }
    if (!lockedConsCache || (now - lockedConsCacheTs) > LOCKED_CACHE_TTL_MS) {
      lockedConsCache   = await getLockedConsensusPreds();
      lockedConsCacheTs = now;
    }
    const advLocked      = lockedAdvCache;
    const consensusLocked = lockedConsCache;

    let totalResolved = 0;

    // Resolve all 13 ML engines
    for (const engineId of ENGINE_IDS) {
      const byTarget = advLocked[engineId] ?? {};
      if (!Object.keys(byTarget).length) continue;
      const n = await resolveEngineWindows(engineId, byTarget, rounds, lastRoundId);
      totalResolved += n;
    }

    // Resolve consensus engine separately
    if (Object.keys(consensusLocked).length) {
      const n = await resolveEngineWindows(CONSENSUS_ID, consensusLocked, rounds, lastRoundId);
      totalResolved += n;
    }

    if (totalResolved > 0) {
      console.log(`[advRes] resolved ${totalResolved} outcomes this tick`);
    }
  } catch(e) {
    console.error('[advRes] Fatal:', e.message, e.stack);
  }
}

module.exports = { runAdvResolutionEngine, resetAdvResolutionState, bustLockedCache };