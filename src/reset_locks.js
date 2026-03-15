const { Pool } = require('pg');
const DATABASE_URL = 'postgresql://postgres:XwNrpbChBnTHAmDVfCYwpmpssMlsnMtt@shortline.proxy.rlwy.net:48401/railway';
const pool = new Pool({ connectionString: DATABASE_URL, ssl: { rejectUnauthorized: false } });
async function main() {
  await pool.query('SELECT 1');
  const res = await pool.query('DELETE FROM locked_preds');
  console.log(`✅ Deleted ${res.rowCount} rows — refresh app now`);
  await pool.end();
}
main().catch(e => { console.error(e.message); process.exit(1); });