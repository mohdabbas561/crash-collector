const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

async function initDB() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS rounds (
      round_id    BIGINT PRIMARY KEY,
      multiplier  NUMERIC(12, 4) NOT NULL,
      timestamp   BIGINT,
      created_at  TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_rounds_created_at ON rounds(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_rounds_multiplier ON rounds(multiplier);

    CREATE TABLE IF NOT EXISTS predictions (
      id            SERIAL PRIMARY KEY,
      target        VARCHAR(10)  NOT NULL,
      min_mult      NUMERIC(12,4) NOT NULL,
      outcome       VARCHAR(15)  NOT NULL,
      window_lo     BIGINT       NOT NULL,
      window_hi     BIGINT       NOT NULL,
      hit_round     BIGINT,
      generation    INT          NOT NULL DEFAULT 1,
      created_at    TIMESTAMPTZ  DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_target     ON predictions(target);
    CREATE INDEX IF NOT EXISTS idx_predictions_outcome    ON predictions(outcome);
    CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
  `);
}

async function savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation }) {
  await pool.query(
    `INSERT INTO predictions (target, min_mult, outcome, window_lo, window_hi, hit_round, generation)
     VALUES ($1, $2, $3, $4, $5, $6, $7)`,
    [target, minMult, outcome, lo, hi, hitRound ?? null, generation]
  );
}

async function getPredictions({ limit = 200, target = null } = {}) {
  const conditions = [];
  const params     = [];
  let idx = 1;
  if (target) { conditions.push(`target = $${idx++}`); params.push(target); }
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit);
  const res = await pool.query(
    `SELECT id, target, min_mult, outcome, window_lo, window_hi, hit_round, generation, created_at
     FROM predictions ${where}
     ORDER BY created_at DESC
     LIMIT $${idx}`,
    params
  );
  return res.rows.map(r => ({
    id:         r.id,
    target:     r.target,
    minMult:    parseFloat(r.min_mult),
    outcome:    r.outcome,
    lo:         Number(r.window_lo),
    hi:         Number(r.window_hi),
    hitRound:   r.hit_round ? Number(r.hit_round) : null,
    generation: r.generation,
    ts:         new Date(r.created_at).getTime(),
  }));
}

async function saveRounds(rounds) {
  if (!rounds.length) return 0;
  const values = [];
  const params = [];
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

async function getRounds({ limit = 1000, offset = 0, from = null, to = null } = {}) {
  const conditions = [];
  const params = [];
  let idx = 1;
  if (from) { conditions.push(`created_at >= $${idx++}`); params.push(new Date(from)); }
  if (to)   { conditions.push(`created_at <= $${idx++}`); params.push(new Date(to)); }
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit, offset);
  const res = await pool.query(
    `SELECT round_id, multiplier, timestamp, created_at
     FROM rounds ${where}
     ORDER BY round_id ASC
     LIMIT $${idx++} OFFSET $${idx++}`,
    params
  );
  return res.rows.map(row => ({
    roundId   : Number(row.round_id),
    multiplier: parseFloat(row.multiplier),
    timestamp : row.timestamp ? Number(row.timestamp) : Number(new Date(row.created_at)),
  }));
}

async function getStorageStats() {
  const res = await pool.query(`
    SELECT
      COUNT(*)                          AS total,
      MIN(round_id)                     AS oldest,
      MAX(round_id)                     AS newest,
      AVG(multiplier)::NUMERIC(10,2)    AS avg_multiplier,
      MAX(multiplier)::NUMERIC(10,2)    AS max_multiplier,
      MIN(multiplier)::NUMERIC(10,2)    AS min_multiplier,
      MIN(created_at)                   AS oldest_date,
      MAX(created_at)                   AS newest_date
    FROM rounds
  `);
  const row = res.rows[0];
  return {
    total     : Number(row.total),
    oldest    : Number(row.oldest) || null,
    newest    : Number(row.newest) || null,
    avg       : row.avg_multiplier,
    max       : row.max_multiplier,
    min       : row.min_multiplier,
    oldestDate: row.oldest_date,
    newestDate: row.newest_date,
  };
}

async function getStats() {
  const res = await pool.query(`
    SELECT
      COUNT(*)                                                        AS total,
      AVG(multiplier)::NUMERIC(10,4)                                 AS avg,
      MAX(multiplier)::NUMERIC(10,4)                                 AS highest,
      (SELECT round_id FROM rounds ORDER BY round_id DESC LIMIT 1)  AS current_round,
      SUM(CASE WHEN multiplier < 2    THEN 1 ELSE 0 END)            AS lt2,
      SUM(CASE WHEN multiplier >= 2   AND multiplier < 5   THEN 1 ELSE 0 END) AS b2_5,
      SUM(CASE WHEN multiplier >= 5   AND multiplier < 10  THEN 1 ELSE 0 END) AS b5_10,
      SUM(CASE WHEN multiplier >= 10  AND multiplier < 20  THEN 1 ELSE 0 END) AS b10_20,
      SUM(CASE WHEN multiplier >= 20  AND multiplier < 50  THEN 1 ELSE 0 END) AS b20_50,
      SUM(CASE WHEN multiplier >= 50  AND multiplier < 100 THEN 1 ELSE 0 END) AS b50_100,
      SUM(CASE WHEN multiplier >= 100 THEN 1 ELSE 0 END)            AS gt100
    FROM rounds
  `);
  const row = res.rows[0];
  return {
    tracked     : Number(row.total),
    avg         : row.avg,
    highest     : row.highest,
    currentRound: Number(row.current_round) || null,
    distribution: {
      lt2    : Number(row.lt2),
      b2_5   : Number(row.b2_5),
      b5_10  : Number(row.b5_10),
      b10_20 : Number(row.b10_20),
      b20_50 : Number(row.b20_50),
      b50_100: Number(row.b50_100),
      gt100  : Number(row.gt100),
    }
  };
}

async function clearPredictions() {
  await pool.query(`DELETE FROM predictions`);
}


// ── ACCESS CODES ──────────────────────────────────────────────────────────────
async function initAccessCodes() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS access_codes (
      id          SERIAL PRIMARY KEY,
      code        VARCHAR(64) UNIQUE NOT NULL,
      created_at  TIMESTAMPTZ DEFAULT NOW(),
      expires_at  TIMESTAMPTZ NOT NULL,
      ip          VARCHAR(64),
      note        VARCHAR(200)
    );
    CREATE INDEX IF NOT EXISTS idx_access_codes_code    ON access_codes(code);
    CREATE INDEX IF NOT EXISTS idx_access_codes_expires ON access_codes(expires_at);
  `);
}

async function createAccessCode({ code, expiresAt, note }) {
  const res = await pool.query(
    `INSERT INTO access_codes (code, expires_at, note)
     VALUES ($1, $2, $3)
     ON CONFLICT (code) DO UPDATE SET expires_at = $2, note = $3
     RETURNING *`,
    [code, new Date(expiresAt), note || '']
  );
  return res.rows[0];
}

async function getAccessCode(code) {
  const res = await pool.query(`SELECT * FROM access_codes WHERE code = $1`, [code]);
  return res.rows[0] ?? null;
}

async function updateAccessCodeIP(code, ip) {
  await pool.query(`UPDATE access_codes SET ip = $2 WHERE code = $1`, [code, ip]);
}

async function getAllAccessCodes() {
  const res = await pool.query(`SELECT * FROM access_codes ORDER BY created_at DESC`);
  return res.rows;
}

async function deleteAccessCode(id) {
  await pool.query(`DELETE FROM access_codes WHERE id = $1`, [id]);
}

module.exports = {
  initDB, saveRounds, getRounds, getStorageStats, getStats,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
};