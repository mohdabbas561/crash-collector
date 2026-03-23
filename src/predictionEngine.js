'use strict';
// predictionEngine.js — Orchestrator
// Runs: ngComputeEngine (NG Master / ng_consensus), advComputeEngine (ADV consensus only)

const { runAdvComputeEngine, resetAdvComputeState    } = require('./advComputeEngine');
const { runNgComputeEngine,  resetNgComputeState     } = require('./ngComputeEngine');

async function runPredictionEngine() {
  await runAdvComputeEngine();  // ADV consensus
  await runNgComputeEngine();   // NG Master (ng_consensus)
}

function resetEngineState() {
  resetAdvComputeState();
  resetNgComputeState();
}

module.exports = { runPredictionEngine, resetEngineState };