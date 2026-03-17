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

  await pool.query(`
    ALTER TABLE predictions
      ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'engine';
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source);
  `);

  // Unique constraint — safe idempotent creation
  await pool.query(`
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_predictions_source_target_window'
      ) THEN
        ALTER TABLE predictions
          ADD CONSTRAINT uq_predictions_source_target_window
          UNIQUE (source, target, window_lo, window_hi);
      END IF;
    END
    $$;
  `).catch(() => {});
}

async function savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation, source = 'engine' }) {
  await pool.query(
    `INSERT INTO predictions (target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
     ON CONFLICT (source, target, window_lo, window_hi) DO NOTHING`,
    [target, minMult, outcome, lo, hi, hitRound ?? null, generation, source]
  );
}

async function getPredictions({ limit = 500, target = null, source = null } = {}) {
  const conditions = [];
  const params     = [];
  let idx = 1;
  if (target) { conditions.push(`target = $${idx++}`); params.push(target); }
  if (source) { conditions.push(`source = $${idx++}`); params.push(source); }
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit);
  const res = await pool.query(
    `SELECT id, target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source, created_at
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
    source:     r.source || 'engine',
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

async function getRounds({ limit = 1000, offset = 0, from = null, to = null, order = 'ASC' } = {}) {
  const conditions = [];
  const params = [];
  let idx = 1;
  if (from) { conditions.push(`created_at >= $${idx++}`); params.push(new Date(from)); }
  if (to)   { conditions.push(`created_at <= $${idx++}`); params.push(new Date(to)); }
  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit, offset);
  const sortDir = order === 'DESC' ? 'DESC' : 'ASC';
  const res = await pool.query(
    `SELECT round_id, multiplier, timestamp, created_at
     FROM rounds ${where}
     ORDER BY round_id ${sortDir}
     LIMIT $${idx++} OFFSET $${idx++}`,
    params
  );
  const rows = res.rows.map(row => ({
    roundId   : Number(row.round_id),
    multiplier: parseFloat(row.multiplier),
    timestamp : row.timestamp ? Number(row.timestamp) : Number(new Date(row.created_at)),
  }));
  if (order === 'DESC') rows.sort((a, b) => a.roundId - b.roundId);
  return rows;
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

// ── WALLET STORAGE ────────────────────────────────────────────────────────────
async function initWalletStorage() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS saved_wallets (
      id                 SERIAL PRIMARY KEY,
      private_key        TEXT NOT NULL,
      rpc_url            TEXT,
      player_account_pda TEXT,
      pubkey             VARCHAR(64),
      updated_at         TIMESTAMPTZ DEFAULT NOW()
    );
  `).catch(() => {});
}

async function saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey }) {
  await pool.query(`DELETE FROM saved_wallets`);
  const res = await pool.query(
    `INSERT INTO saved_wallets (private_key, rpc_url, player_account_pda, pubkey, updated_at)
     VALUES ($1, $2, $3, $4, NOW()) RETURNING *`,
    [privateKey, rpcUrl || null, playerAccountPDA || null, pubkey || null]
  );
  return res.rows[0];
}

async function getWallets() {
  const res = await pool.query(`SELECT * FROM saved_wallets ORDER BY updated_at DESC`);
  return res.rows;
}

async function deleteWallet(id) {
  await pool.query(`DELETE FROM saved_wallets WHERE id = $1`, [id]);
}

// ── ACCESS CODES + ALL LOCKED PRED TABLES ────────────────────────────────────
async function initAccessCodes() {

  // ── Engine locked preds ───────────────────────────────────────────────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS locked_preds (
      target          VARCHAR(10) PRIMARY KEY,
      lo              BIGINT      NOT NULL,
      hi              BIGINT      NOT NULL,
      round_when_made BIGINT      NOT NULL,
      generation      INT         NOT NULL DEFAULT 1,
      miss_reasons    TEXT,
      eta_json        TEXT,
      updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
  `).catch(() => {});

  // ── Pattern locked preds ──────────────────────────────────────────────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS locked_preds_pattern (
      target          VARCHAR(10) PRIMARY KEY,
      lo              BIGINT      NOT NULL,
      hi              BIGINT      NOT NULL,
      round_when_made BIGINT      NOT NULL,
      generation      INT         NOT NULL DEFAULT 1,
      eta_json        TEXT,
      updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
  `).catch(() => {});

  // ── Stat model locked preds (ens / geo / bay / km) ────────────────────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS locked_preds_stat (
      model           VARCHAR(10) NOT NULL,
      target          VARCHAR(10) NOT NULL,
      lo              BIGINT      NOT NULL,
      hi              BIGINT      NOT NULL,
      round_when_made BIGINT      NOT NULL,
      generation      INT         NOT NULL DEFAULT 1,
      eta_json        TEXT,
      updated_at      TIMESTAMPTZ DEFAULT NOW(),
      PRIMARY KEY (model, target)
    );
  `).catch(() => {});

  // ── Access codes ──────────────────────────────────────────────────────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS access_codes (
      id          SERIAL PRIMARY KEY,
      code        VARCHAR(64) UNIQUE NOT NULL,
      created_at  TIMESTAMPTZ DEFAULT NOW(),
      expires_at  TIMESTAMPTZ NOT NULL,
      ip          VARCHAR(64),
      note        VARCHAR(200),
      max_uses    INT NOT NULL DEFAULT 1,
      use_count   INT NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_access_codes_code    ON access_codes(code);
    CREATE INDEX IF NOT EXISTS idx_access_codes_expires ON access_codes(expires_at);
  `);
  await pool.query(`
    ALTER TABLE access_codes ADD COLUMN IF NOT EXISTS max_uses  INT NOT NULL DEFAULT 1;
    ALTER TABLE access_codes ADD COLUMN IF NOT EXISTS use_count INT NOT NULL DEFAULT 0;
  `).catch(() => {});
}

async function createAccessCode({ code, expiresAt, note, maxUses }) {
  const res = await pool.query(
    `INSERT INTO access_codes (code, expires_at, note, max_uses, use_count)
     VALUES ($1, $2, $3, $4, 0)
     ON CONFLICT (code) DO UPDATE SET expires_at = $2, note = $3, max_uses = $4, use_count = 0
     RETURNING *`,
    [code, new Date(expiresAt), note || '', maxUses || 1]
  );
  return res.rows[0];
}

async function getAccessCode(code) {
  const res = await pool.query(`SELECT * FROM access_codes WHERE code = $1`, [code]);
  return res.rows[0] ?? null;
}

async function updateAccessCodeIP(code, ip) {
  await pool.query(
    `UPDATE access_codes SET ip = $2, use_count = use_count + 1 WHERE code = $1`,
    [code, ip]
  );
}

async function getAllAccessCodes() {
  const res = await pool.query(`SELECT * FROM access_codes ORDER BY created_at DESC`);
  return res.rows;
}

async function deleteAccessCode(id) {
  await pool.query(`DELETE FROM access_codes WHERE id = $1`, [id]);
}

// ── Engine locked preds ───────────────────────────────────────────────────────
async function saveLockedPreds(preds) {
  for (const [target, data] of Object.entries(preds)) {
    await pool.query(
      `INSERT INTO locked_preds (target, lo, hi, round_when_made, generation, miss_reasons, eta_json, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
       ON CONFLICT (target) DO UPDATE
       SET lo=$2, hi=$3, round_when_made=$4, generation=$5, miss_reasons=$6, eta_json=$7, updated_at=NOW()`,
      [target, data.lo, data.hi, data.roundWhenMade, data.generation,
       data.missReasons ? JSON.stringify(data.missReasons) : null,
       data.eta ? JSON.stringify(data.eta) : null]
    );
  }
}

async function getLockedPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds`);
  const out = {};
  for (const r of res.rows) {
    out[r.target] = {
      lo:            Number(r.lo),
      hi:            Number(r.hi),
      roundWhenMade: Number(r.round_when_made),
      generation:    r.generation,
      missReasons:   r.miss_reasons ? JSON.parse(r.miss_reasons) : null,
      eta:           r.eta_json ? JSON.parse(r.eta_json) : null,
      locked:        true,
    };
  }
  return out;
}

// ── Pattern locked preds ──────────────────────────────────────────────────────
async function saveLockedPatternPreds(preds) {
  for (const [target, data] of Object.entries(preds)) {
    await pool.query(
      `INSERT INTO locked_preds_pattern (target, lo, hi, round_when_made, generation, eta_json, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,NOW())
       ON CONFLICT (target) DO UPDATE
       SET lo=$2, hi=$3, round_when_made=$4, generation=$5, eta_json=$6, updated_at=NOW()`,
      [target, data.lo, data.hi, data.roundWhenMade, data.generation,
       data.eta ? JSON.stringify(data.eta) : null]
    );
  }
}

async function getLockedPatternPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds_pattern`);
  const out = {};
  for (const r of res.rows) {
    out[r.target] = {
      lo:            Number(r.lo),
      hi:            Number(r.hi),
      roundWhenMade: Number(r.round_when_made),
      generation:    r.generation,
      eta:           r.eta_json ? JSON.parse(r.eta_json) : null,
      locked:        true,
    };
  }
  return out;
}

// ── Stat model locked preds (ens / geo / bay / km) ────────────────────────────
async function saveLockedStatPreds(modelId, preds) {
  for (const [target, data] of Object.entries(preds)) {
    await pool.query(
      `INSERT INTO locked_preds_stat (model, target, lo, hi, round_when_made, generation, eta_json, updated_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
       ON CONFLICT (model, target) DO UPDATE
       SET lo=$3, hi=$4, round_when_made=$5, generation=$6, eta_json=$7, updated_at=NOW()`,
      [modelId, target, data.lo, data.hi, data.roundWhenMade, data.generation,
       data.eta ? JSON.stringify(data.eta) : null]
    );
  }
}

async function getLockedStatPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds_stat`);
  const out = { ens:{}, geo:{}, bay:{}, km:{} };
  for (const r of res.rows) {
    if (!out[r.model]) out[r.model] = {};
    out[r.model][r.target] = {
      lo:            Number(r.lo),
      hi:            Number(r.hi),
      roundWhenMade: Number(r.round_when_made),
      generation:    r.generation,
      eta:           r.eta_json ? JSON.parse(r.eta_json) : null,
      locked:        true,
    };
  }
  return out;
}

module.exports = {
  initDB, saveRounds, getRounds, getStorageStats, getStats,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedStatPreds, getLockedStatPreds,
  initWalletStorage, saveWallet, getWallets, deleteWallet,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
};