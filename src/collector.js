// collector.js
const fetch = require('node-fetch');
const { saveRounds } = require('./db');
const { runPredictionEngine } = require('./predictionEngine');

const API_URL     = 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds?limit=100';
const POLL_MS     = 5000;  // 5s — fast enough to catch resolves quickly
const MAX_RETRIES = 5;

let lastSeenRoundId   = 0;
let consecutiveErrors = 0;

async function fetchRounds() {
  const res = await fetch(API_URL, {
    headers: { 'Accept': 'application/json' },
    timeout: 10000,
  });
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  const json = await res.json();
  const arr  = json.payload || json;
  if (!Array.isArray(arr)) throw new Error('Unexpected API response shape');
  return arr
    .filter(r => r.gameResult !== null && r.gameResult !== undefined)
    .map(r => ({
      roundId   : Number(r.roundId),
      multiplier: parseFloat(r.gameResult ?? r.multiplier ?? r.crashPoint ?? 1),
      timestamp : r.createdAt ? new Date(r.createdAt).getTime() : Date.now(),
    }))
    .filter(r => !isNaN(r.roundId) && !isNaN(r.multiplier));
}

async function poll() {
  try {
    const rounds = await fetchRounds();
    if (rounds.length) {
      const newRounds = rounds.filter(r => r.roundId > lastSeenRoundId);
      if (newRounds.length) {
        const saved = await saveRounds(newRounds);
        const maxId = Math.max(...newRounds.map(r => r.roundId));
        if (saved > 0) {
          console.log(`[${new Date().toISOString()}] Saved ${saved} new rounds. Latest: #${maxId}`);
        }
        lastSeenRoundId = maxId;
      }
    }
    consecutiveErrors = 0;
    // Always run engine every tick — engines check their own dirty flags
    // and rebuild immediately when a window resolves, no round needed
    await runPredictionEngine();
  } catch (err) {
    consecutiveErrors++;
    console.error(`[${new Date().toISOString()}] Poll error (${consecutiveErrors}): ${err.message}`);
  }
}

async function startCollector() {
  console.log('Collector: doing initial fetch...');
  try {
    const rounds = await fetchRounds();
    if (rounds.length) {
      await saveRounds(rounds);
      lastSeenRoundId = Math.max(...rounds.map(r => r.roundId));
      console.log(`Collector: bootstrapped with ${rounds.length} rounds. Latest: #${lastSeenRoundId}`);
    }
  } catch (e) {
    console.error('Collector: initial fetch failed:', e.message);
  }

  // Run engine once immediately after bootstrap
  await runPredictionEngine();

  const tick = async () => {
    await poll();
    const delay = consecutiveErrors >= MAX_RETRIES ? 60000 : POLL_MS;
    setTimeout(tick, delay);
  };
  setTimeout(tick, POLL_MS);
}

module.exports = { startCollector };