'use strict';
// predictionEngine.js — Orchestrator
// ─────────────────────────────────────────────────────────────────────────────
// This file is now a thin orchestrator only.
// All logic lives in fully independent modules:
//   • patternEngine.js — Pattern Engine (clustering/trend/autocorrelation)
//   • statEngine.js    — Stat Engine (GEO / BAY / KM / ENS)
//
// These two engines share ZERO code. They have independent:
//   - TARGETS definitions
//   - State (lockedMap, savedSet, calibration, timing, CUSUM)
//   - DB table access (pattern → locked_preds_pattern, stat → locked_preds + locked_preds_stat)
//   - Rounds caches
//   - initialised flags
//   - Reset functions
// ─────────────────────────────────────────────────────────────────────────────

const { runPatternEngine, resetPatternEngineState } = require('./PatternEngine');
const { runStatEngine,    resetStatEngineState, getLockedStatMap, getValidationMetrics } = require('./statEngine');

// Run both engines every tick — each manages its own dirty state
async function runPredictionEngine() {
  // Run sequentially to avoid DB connection saturation under load.
  // Each engine has its own internal dirty/needsRebuild check.
  await runStatEngine();
  await runPatternEngine();
}

function resetEngineState() {
  resetStatEngineState();
  resetPatternEngineState();
}

module.exports = {
  runPredictionEngine,
  resetEngineState,
  // Re-export stat engine introspection for API status endpoints
  getLockedStatMap,
  getValidationMetrics,
};