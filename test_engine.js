'use strict';
const engine = require('./src/oracleEngine.js');

// Same synthetic generation
function generateRounds(n) {
  const rounds = [];
  for (let i = 1; i <= n; i++) {
    let val;
    const r = Math.random();
    if (r < 0.35) val = 1.0 + Math.random() * 0.5;
    else if (r < 0.65) val = 1.5 + Math.random() * 2.5;
    else if (r < 0.85) val = 4.0 + Math.random() * 10;
    else if (r < 0.95) val = 14 + Math.random() * 36;
    else val = 50 + Math.random() * 200;
    rounds.push({ id: i, val: Number(val.toFixed(2)) });
  }
  return rounds;
}

const rounds = generateRounds(500);
const t100 = engine.ORACLE_TARGETS[5]; // 100x
const f = engine.computeOracleForecast(rounds, t100, {});

console.log('=== 100x Detailed Breakdown ===');
console.log('Confidence:', f.confidence, '| Raw:', f.rawConfidence);
console.log('EnsembleP:', f.ensembleP, '| BaselineP:', f.baselineP, '| Edge:', f.ensembleEdge);
console.log('EV:', f.ensembleEV);
console.log('KM pHitWindow:', f.pHitWindow, '| DroughtPct:', f.droughtPct);
console.log('Markov:', f.markovProb);
console.log('Pattern:', JSON.stringify(f.patternSupport));
console.log('B2B:', f.b2bScore);
console.log('Regime:', f.regimeLabel);
console.log('White:', f.whitePhase);
console.log('RoundsSince:', f.roundsSince, '| Median:', f.med, '| Window:', f.windowSize);
console.log('Layers:', JSON.stringify(f.layerBreakdown, null, 2));
