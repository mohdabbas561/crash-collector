'use strict';
// predictionEngine.js — Orchestrator
// Runs: patternEngine, ngComputeEngine (NG SOTA + consensus), advComputeEngine (ADV consensus only)
// Removed: statEngine (geo/bay/km/ens), advResolutionEngine (only resolved removed adv engines)

const { runPatternEngine,    resetPatternEngineState } = require('./patternEngine');
const { runAdvComputeEngine, resetAdvComputeState    } = require('./advComputeEngine');
const { runNgComputeEngine,  resetNgComputeState     } = require('./ngComputeEngine');

async function runPredictionEngine() {
  await runPatternEngine();
  await runAdvComputeEngine();  // ADV consensus + pattern engines (lstm/xgb math kept, windows disabled)
  await runNgComputeEngine();   // Next-Gen SOTA + ng_consensus master signal
}

function resetEngineState() {
  resetPatternEngineState();
  resetAdvComputeState();
  resetNgComputeState();
}

module.exports = { runPredictionEngine, resetEngineState };