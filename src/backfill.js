// backfill.js — run once to pull all historical DCF crash rounds into your DB
// Place in crash-collector-main/src/ and run: node src/backfill.js
//
// FIX: pool was referenced but never instantiated in the original.
// FIX: DATABASE_URL now reads from env var (DATABASE_URL) with a fallback
//      so credentials are not committed to source control.
// FIX: START_ROUND_ID / STOP_ROUND_ID unused — backfill walks from latest
//      offset until empty pages, no hardcoded round IDs needed.

const fetch = require('node-fetch');
const { Pool } = require('pg');

const DATABASE_URL = process.env.DATABASE_URL ||
  'postgresql://postgres:CHANGE_ME@localhost:5432/railway';

// FIX: instantiate pool (was missing in original — caused ReferenceError at runtime)
const pool = new Pool({ connectionString: DATABASE_URL, ssl: { rejectUnauthorized: false } });

const BASE  = 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds';
const LIMIT = 100;
const DELAY = 400; // ms between requests

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function saveRounds(rounds) {
  if (!rounds.length) return 0;
  const values = [], params = [];
  let idx = 1;
  for (const r of rounds) {
    values.push(`($${idx++}, $${idx++}, $${idx++})`);
    params.push(r.roundId, r.multiplier, r.timestamp ?? null);
  }
  const res = await pool.query(
    `INSERT INTO rounds (round_id, multiplier, timestamp)
     VALUES ${values.join(', ')}
     ON CONFLICT (round_id) DO NOTHING`,
    params
  );
  return res.rowCount;
}

async function fetchPage(offset) {
  const url = `${BASE}?limit=${LIMIT}&offset=${offset}`;
  const res  = await fetch(url, {
    headers: { 'Accept': 'application/json' },
    timeout: 15000,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json();
  const arr  = json.payload || json;
  if (!Array.isArray(arr)) throw new Error('Unexpected response shape');
  return arr
    .filter(r => r.gameResult !== null && r.gameResult !== undefined)
    .map(r => ({
      roundId   : Number(r.roundId),
      multiplier: parseFloat(r.gameResult ?? r.multiplier ?? r.crashPoint ?? 1),
      timestamp : r.createdAt ? new Date(r.createdAt).getTime() : null,
    }))
    .filter(r => !isNaN(r.roundId) && !isNaN(r.multiplier));
}

async function main() {
  // Verify connection first
  try {
    await pool.query('SELECT 1');
    console.log('✅ DB connected successfully');
  } catch (e) {
    console.error('❌ DB connection failed:', e.message);
    console.error('Check your DATABASE_URL — make sure it is the PUBLIC URL from Railway');
    process.exit(1);
  }

  console.log('Starting backfill...');
  let offset     = 0;
  let totalSaved = 0;
  let emptyPages = 0;
  let errors     = 0;

  while (true) {
    try {
      const rounds = await fetchPage(offset);

      if (!rounds.length) {
        emptyPages++;
        if (emptyPages >= 3) {
          console.log('3 empty pages — reached end of history.');
          break;
        }
        offset += LIMIT;
        await sleep(DELAY);
        continue;
      }

      emptyPages = 0;
      errors     = 0;
      const saved = await saveRounds(rounds);
      totalSaved += saved;

      const minId = Math.min(...rounds.map(r => r.roundId));
      const maxId = Math.max(...rounds.map(r => r.roundId));
      console.log(`offset=${offset} | rounds #${minId}–#${maxId} | saved ${saved} new | total ${totalSaved}`);

      if (rounds.length < LIMIT) {
        console.log('Partial page — beginning of history reached.');
        break;
      }

      offset += LIMIT;
      await sleep(DELAY);

    } catch (e) {
      errors++;
      console.error(`Error at offset ${offset}: ${e.message}`);
      if (errors >= 5) { console.error('5 consecutive errors — stopping.'); break; }
      await sleep(5000);
    }
  }

  console.log(`\n✅ Backfill complete. Total saved: ${totalSaved}`);
  await pool.end();
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });