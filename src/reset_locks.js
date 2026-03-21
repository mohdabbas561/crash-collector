// reset_locks.js — clears ALL stale locked prediction windows
// Run once after deploying new predictionEngine.js with updated maxWidth values
// Usage: node src/reset_locks.js

const { Pool } = require('pg');

const DATABASE_URL = process.env.DATABASE_URL ||
  'postgresql://postgres:CHANGE_ME@localhost:5432/railway';

const pool = new Pool({ connectionString: DATABASE_URL, ssl: { rejectUnauthorized: false } });

async function main() {
  console.log('Connecting to database...');
  await pool.query('SELECT 1');
  console.log('Connected.');

  const r1 = await pool.query('DELETE FROM locked_preds');
  console.log(`✅ locked_preds:           deleted ${r1.rowCount} rows`);

  const r2 = await pool.query('DELETE FROM locked_preds_pattern');
  console.log(`✅ locked_preds_pattern:   deleted ${r2.rowCount} rows`);

  const r3 = await pool.query('DELETE FROM locked_preds_stat');
  console.log(`✅ locked_preds_stat:      deleted ${r3.rowCount} rows`);

  const r4 = await pool.query('DELETE FROM locked_preds_adv');
  console.log(`✅ locked_preds_adv:       deleted ${r4.rowCount} rows`);

  const r5 = await pool.query('DELETE FROM locked_preds_consensus');
  console.log(`✅ locked_preds_consensus: deleted ${r5.rowCount} rows`);

  console.log('\nAll stale windows cleared. Restart the server now.');
  console.log('On startup you should see: [engine] NEW 100x: ..., [engine] NEW 500x: ..., etc.');

  await pool.end();
}

main().catch(e => { console.error('Error:', e.message); process.exit(1); });