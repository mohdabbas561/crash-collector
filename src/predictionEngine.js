'use strict';
// predictionEngine.js — Orchestrator
// ─────────────────────────────────────────────────────────────────────────────
// Coordinates three fully independent engine modules:
//   • patternEngine.js        — PTN (clustering/trend/autocorrelation)
//   • statEngine.js           — Stat (GEO/BAY/KM/ENS)
//   • advResolutionEngine.js  — Resolves advanced engine windows server-side
//
// CRITICAL: advResolutionEngine solves the offline history problem.
// Advanced engines (lstm/xgb/rf etc) lock windows from the browser.
// Without server-side resolution, outcomes are only saved when a user
// has the tab open. advResolutionEngine runs every tick and saves
// win/loss/early regardless of whether any browser is connected.
//
// All three modules share ZERO code with each other.
// ─────────────────────────────────────────────────────────────────────────────

const { runPatternEngine,       resetPatternEngineState  } = require('./patternEngine');
const { runStatEngine,          resetStatEngineState,
        getLockedStatMap,       getValidationMetrics     } = require('./statEngine');
const { runAdvResolutionEngine, resetAdvResolutionState  } = require('./advResolutionEngine');

// Run all three engines every tick — each manages its own dirty state
async function runPredictionEngine() {
  // Run sequentially to avoid DB connection saturation under load.
  await runStatEngine();
  await runPatternEngine();
  await runAdvResolutionEngine();
}

function resetEngineState() {
  resetStatEngineState();
  resetPatternEngineState();
  resetAdvResolutionState();
}

module.exports = {
  runPredictionEngine,
  resetEngineState,
  getLockedStatMap,
  getValidationMetrics,
};