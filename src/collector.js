// collector.js
const fetch = require('node-fetch');
const { saveRounds } = require('./db');
const { runPredictionEngine } = require('./predictionEngine');

const API_URL     = 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds?limit=100';
const POLL_MS     = 8000;  // 8s — gives ML Tier2 time to complete between ticks
const MAX_RETRIES = 5;

let lastSeenRoundId   = 0;
let consecutiveErrors = 0;

async function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function fetchRounds() {
  const res = await withTimeout(
    fetch(API_URL, { headers: { 'Accept': 'application/json' } }),
    10000, 'fetchRounds'
  );
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  const json = await withTimeout(res.json(), 5000, 'fetchRounds.json');
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

// Fill gap since last run — fetches rounds missed while server was offline
// Uses offset pagination to walk backwards until we hit rounds already in DB.
async function fillGap(latestInDb) {
  if (!latestInDb) return;
  console.log(`[gap-fill] DB latest: #${latestInDb} — checking for missed rounds...`);
  let offset = 0;
  let filled = 0;
  const MAX_PAGES = 50; // cap: fill up to 5000 missed rounds max
  for (let page = 0; page < MAX_PAGES; page++) {
    try {
      const url = `${API_URL.replace('?limit=100', '')}?limit=100&offset=${offset}`;
      const res  = await withTimeout(
        fetch(url, { headers: { 'Accept': 'application/json' } }),
        10000, 'gap-fill'
      );
      if (!res.ok) break;
      const json = await withTimeout(res.json(), 5000, 'gap-fill.json');
      const arr  = (json.payload || json);
      if (!Array.isArray(arr) || !arr.length) break;
      const rounds = arr
        .filter(r => r.gameResult !== null && r.gameResult !== undefined)
        .map(r => ({
          roundId   : Number(r.roundId),
          multiplier: parseFloat(r.gameResult ?? r.multiplier ?? r.crashPoint ?? 1),
          timestamp : r.createdAt ? new Date(r.createdAt).getTime() : Date.now(),
        }))
        .filter(r => !isNaN(r.roundId) && !isNaN(r.multiplier));

      // If all rounds in this page are already in DB, we're caught up
      const minId = Math.min(...rounds.map(r => r.roundId));
      if (minId <= latestInDb) {
        // Save the new ones from this page, then stop
        const newOnes = rounds.filter(r => r.roundId > latestInDb);
        if (newOnes.length) {
          const saved = await saveRounds(newOnes);
          filled += saved;
        }
        break;
      }
      const saved = await saveRounds(rounds);
      filled += saved;
      offset += 100;
      await new Promise(r => setTimeout(r, 200)); // gentle pacing
    } catch (e) {
      console.error('[gap-fill] error:', e.message);
      break;
    }
  }
  if (filled > 0) console.log(`[gap-fill] Filled ${filled} missed rounds`);
  else console.log('[gap-fill] No gap — DB is up to date');
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
    // Fill any gap from offline downtime
    await fillGap(lastSeenRoundId);
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