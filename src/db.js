'use strict';
// db.js — single source of truth for all DB operations.
// ALL history records and predictions are stored exclusively here.
// Unique constraints + atomic transactions guarantee zero duplicates.

const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
  // FIX: connection pool limits — prevent runaway connection growth under high load
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});

// FIX: surface pool errors so they don't silently crash the process
pool.on('error', (err) => {
  console.error('[db] Unexpected pool error:', err.message);
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
    -- FIX: add index on round_id for fast minRoundId range queries in getRounds()
    CREATE INDEX IF NOT EXISTS idx_rounds_round_id ON rounds(round_id);

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
    ALTER TABLE predictions
      ADD COLUMN IF NOT EXISTS prob_w NUMERIC(8,6);
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source);
  `);

  // FIX: composite index on (source, target) for fast per-engine target lookups
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_predictions_source_target ON predictions(source, target);
  `).catch(() => {});

  // Unique constraint — idempotent creation, prevents all duplicate history entries
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

// ── savePrediction ─────────────────────────────────────────────────────────────
// ON CONFLICT: updates outcome + hit_round so a retry/early→win upgrade works.
// The unique constraint (source, target, window_lo, window_hi) is the single
// dedup guard — safe under concurrent requests from multiple browser tabs.
async function savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation, source = 'engine', probW = null }) {
  // FIX: validate inputs before hitting the DB — reject nonsense windows early
  if (!target || !outcome || lo == null || hi == null) throw new Error('savePrediction: missing required fields');
  if (!Number.isFinite(Number(lo)) || !Number.isFinite(Number(hi)) || Number(hi) < Number(lo))
    throw new Error(`savePrediction: invalid window lo=${lo} hi=${hi}`);

  await pool.query(
    `INSERT INTO predictions (target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source, prob_w)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
     ON CONFLICT (source, target, window_lo, window_hi) DO UPDATE
       SET outcome    = CASE
                          WHEN predictions.outcome = 'win'                           THEN predictions.outcome
                          WHEN predictions.outcome = 'early' AND EXCLUDED.outcome = 'loss' THEN predictions.outcome
                          ELSE EXCLUDED.outcome
                        END,
           hit_round  = COALESCE(predictions.hit_round, EXCLUDED.hit_round),
           generation = GREATEST(EXCLUDED.generation, predictions.generation),
           prob_w     = COALESCE(EXCLUDED.prob_w, predictions.prob_w)`,
    [target, minMult, outcome, lo, hi, hitRound ?? null, generation ?? 1, source, probW ?? null]
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
    `SELECT id, target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source, prob_w, created_at
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
    probW:      r.prob_w != null ? parseFloat(r.prob_w) : null,
    ts:         new Date(r.created_at).getTime(),
  }));
}

// ── saveRounds ─────────────────────────────────────────────────────────────────
// Batch upsert. ON CONFLICT DO NOTHING is the dedup guard for rounds.
// FIX: chunk large batches to stay under postgres parameter limit (65535 params).
const ROUND_BATCH_CHUNK = 1000; // 3 params per row → 3000 params max per chunk

async function saveRounds(rounds) {
  if (!rounds.length) return 0;
  let totalSaved = 0;
  for (let offset = 0; offset < rounds.length; offset += ROUND_BATCH_CHUNK) {
    const chunk = rounds.slice(offset, offset + ROUND_BATCH_CHUNK);
    const values = [];
    const params = [];
    let idx = 1;
    for (const r of chunk) {
      values.push(`($${idx++}, $${idx++}, $${idx++})`);
      params.push(r.roundId, r.multiplier, r.timestamp ?? null);
    }
    const res = await pool.query(
      `INSERT INTO rounds (round_id, multiplier, timestamp)
       VALUES ${values.join(', ')}
       ON CONFLICT (round_id) DO NOTHING`,
      params
    );
    totalSaved += res.rowCount;
  }
  return totalSaved;
}

async function getRounds({ limit = 1000, offset = 0, from = null, to = null, order = 'ASC', minRoundId = null } = {}) {
  const conditions = [];
  const params = [];
  let idx = 1;
  if (from) { conditions.push(`created_at >= $${idx++}`); params.push(new Date(from)); }
  if (to)   { conditions.push(`created_at <= $${idx++}`); params.push(new Date(to)); }
  if (minRoundId) { conditions.push(`round_id >= $${idx++}`); params.push(minRoundId); }
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
  const [mainRes, gapRes] = await Promise.all([
    pool.query(`
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
    `),
    pool.query(`
      WITH latest AS (SELECT round_id FROM rounds ORDER BY round_id DESC LIMIT 1),
      last_hits AS (
        SELECT
          MAX(CASE WHEN multiplier >= 2    THEN round_id END) AS h2,
          MAX(CASE WHEN multiplier >= 5    THEN round_id END) AS h5,
          MAX(CASE WHEN multiplier >= 10   THEN round_id END) AS h10,
          MAX(CASE WHEN multiplier >= 20   THEN round_id END) AS h20,
          MAX(CASE WHEN multiplier >= 25   THEN round_id END) AS h25,
          MAX(CASE WHEN multiplier >= 30   THEN round_id END) AS h30,
          MAX(CASE WHEN multiplier >= 50   THEN round_id END) AS h50,
          MAX(CASE WHEN multiplier >= 100  THEN round_id END) AS h100,
          MAX(CASE WHEN multiplier >= 200  THEN round_id END) AS h200,
          MAX(CASE WHEN multiplier >= 500  THEN round_id END) AS h500,
          MAX(CASE WHEN multiplier >= 1000 THEN round_id END) AS h1000
        FROM rounds
      )
      SELECT
        l.round_id - COALESCE(lh.h2,    0) AS g2,
        l.round_id - COALESCE(lh.h5,    0) AS g5,
        l.round_id - COALESCE(lh.h10,   0) AS g10,
        l.round_id - COALESCE(lh.h20,   0) AS g20,
        l.round_id - COALESCE(lh.h25,   0) AS g25,
        l.round_id - COALESCE(lh.h30,   0) AS g30,
        l.round_id - COALESCE(lh.h50,   0) AS g50,
        l.round_id - COALESCE(lh.h100,  0) AS g100,
        l.round_id - COALESCE(lh.h200,  0) AS g200,
        l.round_id - COALESCE(lh.h500,  0) AS g500,
        l.round_id - COALESCE(lh.h1000, 0) AS g1000
      FROM latest l, last_hits lh
    `)
  ]);
  const row = mainRes.rows[0];
  const g   = gapRes.rows[0] || {};
  return {
    tracked     : Number(row.total),
    avg         : row.avg,
    highest     : row.highest,
    currentRound: Number(row.current_round) || null,
    gaps: {
      2: Number(g.g2)||0, 5: Number(g.g5)||0, 10: Number(g.g10)||0,
      20: Number(g.g20)||0, 25: Number(g.g25)||0, 30: Number(g.g30)||0,
      50: Number(g.g50)||0, 100: Number(g.g100)||0,
      200: Number(g.g200)||0, 500: Number(g.g500)||0, 1000: Number(g.g1000)||0,
    },
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

  // ── Consensus (Master Signal) locked preds ────────────────────────────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS locked_preds_consensus (
      target          VARCHAR(10) PRIMARY KEY,
      lo              BIGINT      NOT NULL,
      hi              BIGINT      NOT NULL,
      round_when_made BIGINT      NOT NULL,
      generation      INT         NOT NULL DEFAULT 1,
      eta_json        TEXT,
      updated_at      TIMESTAMPTZ DEFAULT NOW()
    );
  `).catch(() => {});

  // ── Advanced engine locked preds (lstm/xgb/rf/ols/cat/hardgap/...) ─────────
  await pool.query(`
    CREATE TABLE IF NOT EXISTS locked_preds_adv (
      model           VARCHAR(20) NOT NULL,
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
  const entries = Object.entries(preds);
  if (!entries.length) return;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [target, data] of entries) {
      await client.query(
        `INSERT INTO locked_preds (target, lo, hi, round_when_made, generation, miss_reasons, eta_json, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
         ON CONFLICT (target) DO UPDATE
         SET lo=$2, hi=$3, round_when_made=$4, generation=$5, miss_reasons=$6, eta_json=$7, updated_at=NOW()`,
        [target, data.lo, data.hi, data.roundWhenMade, data.generation,
         data.missReasons ? JSON.stringify(data.missReasons) : null,
         data.eta ? JSON.stringify(data.eta) : null]
      );
    }
    await client.query('COMMIT');
  } catch(e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
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
  const entries = Object.entries(preds);
  if (!entries.length) return;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [target, data] of entries) {
      await client.query(
        `INSERT INTO locked_preds_pattern (target, lo, hi, round_when_made, generation, eta_json, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,NOW())
         ON CONFLICT (target) DO UPDATE
         SET lo=$2, hi=$3, round_when_made=$4, generation=$5, eta_json=$6, updated_at=NOW()`,
        [target, data.lo, data.hi, data.roundWhenMade, data.generation,
         data.eta ? JSON.stringify(data.eta) : null]
      );
    }
    await client.query('COMMIT');
  } catch(e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
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
  const entries = Object.entries(preds);
  if (!entries.length) return;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [target, data] of entries) {
      await client.query(
        `INSERT INTO locked_preds_stat (model, target, lo, hi, round_when_made, generation, eta_json, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
         ON CONFLICT (model, target) DO UPDATE
         SET lo=$3, hi=$4, round_when_made=$5, generation=$6, eta_json=$7, updated_at=NOW()`,
        [modelId, target, data.lo, data.hi, data.roundWhenMade, data.generation,
         data.eta ? JSON.stringify(data.eta) : null]
      );
    }
    await client.query('COMMIT');
  } catch(e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function getLockedStatPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds_stat`);
  const out = { ens:{}, geo:{}, bay:{}, km:{}, rf:{}, gbt:{}, lr:{}, nb:{}, lstm:{}, lgbm:{}, prp:{}, gru:{}, ifor:{}, meta:{} };
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

// ── Advanced engine locked preds ──────────────────────────────────────────────
async function saveLockedAdvPreds(modelId, preds) {
  const entries = Object.entries(preds);
  if (!entries.length) return;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [target, data] of entries) {
      await client.query(
        `INSERT INTO locked_preds_adv (model, target, lo, hi, round_when_made, generation, eta_json, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
         ON CONFLICT (model, target) DO UPDATE
         SET lo=$3, hi=$4, round_when_made=$5, generation=$6, eta_json=$7, updated_at=NOW()`,
        [modelId, target, data.lo, data.hi, data.roundWhenMade, data.generation ?? 1,
         data.eta ? JSON.stringify(data.eta) : null]
      );
    }
    await client.query('COMMIT');
  } catch(e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function getLockedAdvPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds_adv`);
  const ADV_ENGINES = [
    'consensus',
    'hlstm_xgb','htrans_lstm','htft','tft','nbeats','tcn','lgbm','gru','bilstm','stacking','sha512','ng_consensus',
  ];
  const out = {};
  ADV_ENGINES.forEach(e => { out[e] = {}; });
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

// ── Consensus locked preds ────────────────────────────────────────────────────
async function saveLockedConsensusPreds(preds) {
  const entries = Object.entries(preds);
  if (!entries.length) return;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [target, data] of entries) {
      await client.query(
        `INSERT INTO locked_preds_consensus (target, lo, hi, round_when_made, generation, eta_json, updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,NOW())
         ON CONFLICT (target) DO UPDATE
         SET lo=$2, hi=$3, round_when_made=$4, generation=$5, eta_json=$6, updated_at=NOW()`,
        [target, data.lo, data.hi, data.roundWhenMade, data.generation ?? 1,
         data.eta ? JSON.stringify(data.eta) : null]
      );
    }
    await client.query('COMMIT');
  } catch(e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function getLockedConsensusPreds() {
  const res = await pool.query(`SELECT * FROM locked_preds_consensus`);
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

module.exports = {
  pool,
  initDB, saveRounds, getRounds, getStorageStats, getStats,
  saveLockedPreds, getLockedPreds,
  saveLockedPatternPreds, getLockedPatternPreds,
  saveLockedAdvPreds, getLockedAdvPreds,
  saveLockedStatPreds, getLockedStatPreds,
  saveLockedConsensusPreds, getLockedConsensusPreds,
  initWalletStorage, saveWallet, getWallets, deleteWallet,
  savePrediction, getPredictions, clearPredictions,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
};