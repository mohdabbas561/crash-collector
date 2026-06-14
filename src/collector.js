'use strict';
// collector.js — polls DCF API every 8s, saves new rounds.
// FIX: incremental fetch limit raised to 5000 (was 500) — prevents stale
//      data when server catches up after an offline gap.
// FIX: lastSeenRoundId is now seeded from DB at startup, not from the API
//      fetch — ensures gap-fill starts from the correct point.

const fetch = require('node-fetch');
const { saveRounds, getRounds } = require('./db');

const API_URL     = 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds?limit=100';
const POLL_MS     = 20000;
const MAX_RETRIES = 5;
// FIX: cap gap-fill at 200 pages (20k rounds) — generous but bounded
const GAP_FILL_MAX_PAGES = 200;

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

async function fetchRounds(extraParams = '') {
  const url = `${API_URL}${extraParams}`;
  const res = await withTimeout(
    fetch(url, { headers: { 'Accept': 'application/json' } }),
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
  } catch (err) {
    consecutiveErrors++;
    console.error(`[${new Date().toISOString()}] Poll error (${consecutiveErrors}): ${err.message}`);
  }
}

// fillGap — fetches rounds missed while server was offline.
// FIX: uses offset pagination and stops when DB is caught up.
// FIX: MAX_PAGES raised to 200 (20k rounds) from 50 (5k rounds).
async function fillGap(latestInDb) {
  if (latestInDb == null) return;
  console.log(`[gap-fill] DB latest: #${latestInDb} — checking for missed rounds...`);
  let offset = 0;
  let filled = 0;
  for (let page = 0; page < GAP_FILL_MAX_PAGES; page++) {
    try {
      // Strip the ?limit=100 suffix from API_URL before adding offset
      const baseUrl = API_URL.split('?')[0];
      const url = `${baseUrl}?limit=100&offset=${offset}`;
      const res  = await withTimeout(
        fetch(url, { headers: { 'Accept': 'application/json' } }),
        10000, 'gap-fill'
      );
      if (!res.ok) break;
      const json = await withTimeout(res.json(), 5000, 'gap-fill.json');
      const arr  = json.payload || json;
      if (!Array.isArray(arr) || !arr.length) break;
      const rounds = arr
        .filter(r => r.gameResult !== null && r.gameResult !== undefined)
        .map(r => ({
          roundId   : Number(r.roundId),
          multiplier: parseFloat(r.gameResult ?? r.multiplier ?? r.crashPoint ?? 1),
          timestamp : r.createdAt ? new Date(r.createdAt).getTime() : Date.now(),
        }))
        .filter(r => !isNaN(r.roundId) && !isNaN(r.multiplier));

      if (!rounds.length) break;

      const minId = Math.min(...rounds.map(r => r.roundId));
      if (minId <= latestInDb) {
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
      await new Promise(r => setTimeout(r, 200));
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
    // FIX: seed lastSeenRoundId from DB first — not from API response.
    // This ensures gap-fill starts from the real DB latest, not the API window.
    const dbRecent = await getRounds({ limit: 1, order: 'DESC' });
    const dbLatest = dbRecent.length ? dbRecent[dbRecent.length - 1].roundId : 0;
    const gapFillBaseline = dbLatest;
    if (dbLatest > 0) {
      lastSeenRoundId = dbLatest;
      console.log(`Collector: DB latest round: #${dbLatest}`);
    }

    const rounds = await fetchRounds();
    if (rounds.length) {
      await saveRounds(rounds);
      const apiLatest = Math.max(...rounds.map(r => r.roundId));
      if (apiLatest > lastSeenRoundId) lastSeenRoundId = apiLatest;
      console.log(`Collector: bootstrapped with ${rounds.length} rounds. Latest: #${lastSeenRoundId}`);
    }

    // Fill any gap from offline downtime
    await fillGap(gapFillBaseline);
  } catch (e) {
    console.error('Collector: initial fetch failed:', e.message);
  }

  const tick = async () => {
    await poll();
    const delay = consecutiveErrors >= MAX_RETRIES ? 60000 : POLL_MS;
    setTimeout(tick, delay);
  };
  setTimeout(tick, POLL_MS);
}

module.exports = { startCollector };
