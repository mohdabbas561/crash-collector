'use strict';
// db.js — single source of truth for all DB operations.
// ALL history records and predictions are stored exclusively here.
// Unique constraints + atomic transactions guarantee zero duplicates.

const { Pool } = require('pg');

function firstNonEmpty(...values) {
  for (const v of values) {
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
}

function normalizeDatabaseUrl(rawUrl) {
  if (!rawUrl) return rawUrl;
  try {
    const url = new URL(rawUrl);
    const sslMode = String(url.searchParams.get('sslmode') || '').toLowerCase();
    if (sslMode === 'require' && !url.searchParams.has('uselibpqcompat')) {
      url.searchParams.set('uselibpqcompat', 'true');
    }
    return url.toString();
  } catch {
    return rawUrl;
  }
}

function buildUrlFromParts() {
  const host = firstNonEmpty(process.env.PGHOST, process.env.POSTGRES_HOST, process.env.DB_HOST);
  const port = firstNonEmpty(process.env.PGPORT, process.env.POSTGRES_PORT, process.env.DB_PORT) || '5432';
  const user = firstNonEmpty(process.env.PGUSER, process.env.POSTGRES_USER, process.env.DB_USER);
  const pass = firstNonEmpty(process.env.PGPASSWORD, process.env.POSTGRES_PASSWORD, process.env.DB_PASSWORD);
  const name = firstNonEmpty(process.env.PGDATABASE, process.env.POSTGRES_DB, process.env.DB_NAME);
  if (!host || !user || !pass || !name) return '';
  return `postgres://${encodeURIComponent(user)}:${encodeURIComponent(pass)}@${host}:${port}/${encodeURIComponent(name)}?sslmode=require`;
}

const RAW_DATABASE_URL = firstNonEmpty(
  process.env.DATABASE_URL,
  process.env.RAILWAY_DATABASE_URL,
  process.env.DATABASE_PUBLIC_URL,
  process.env.DATABASE_PRIVATE_URL,
  process.env.POSTGRES_URL,
  process.env.POSTGRESQL_URL,
  buildUrlFromParts()
);
const DATABASE_URL = normalizeDatabaseUrl(RAW_DATABASE_URL);

const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false },
  // FIX: connection pool limits — prevent runaway connection growth under high load
  max: 5,  // reduced from 10 — saves RAM on Railway Postgres
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});

// FIX: surface pool errors so they don't silently crash the process
pool.on('error', (err) => {
  console.error('[db] Unexpected pool error:', err.message);
});

function canonicalizePredictionOutcome({ lo, hi, hitRound }) {
  const loN = Number(lo);
  const hiN = Number(hi);
  const hitN = hitRound == null ? null : Number(hitRound);
  const hasHit = Number.isFinite(hitN);
  if (!hasHit) return { outcome: 'loss', hitRound: null };
  if (hitN < loN) return { outcome: 'early', hitRound: hitN };
  if (hitN <= hiN) return { outcome: 'win', hitRound: hitN };
  return { outcome: 'loss', hitRound: hitN };
}

async function initDB() {
  if (!DATABASE_URL) {
    throw new Error(
      'DATABASE_URL is missing. Set DATABASE_URL (or Railway/Postgres env vars) in Render environment variables.'
    );
  }
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
    ALTER TABLE predictions
      ADD COLUMN IF NOT EXISTS issue_mode VARCHAR(30);
  `);

  await pool.query(`
    ALTER TABLE predictions
      ADD COLUMN IF NOT EXISTS regime_mode VARCHAR(20);
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
        WHERE conname = 'uq_predictions_source_target_window_gen'
      ) THEN
        IF EXISTS (
          SELECT 1 FROM pg_constraint
          WHERE conname = 'uq_predictions_source_target_window'
        ) THEN
          ALTER TABLE predictions
            DROP CONSTRAINT uq_predictions_source_target_window;
        END IF;
        ALTER TABLE predictions
          ADD CONSTRAINT uq_predictions_source_target_window_gen
          UNIQUE (source, target, window_lo, window_hi, generation);
      END IF;
    END
    $$;
  `).catch(() => {});

  // Repair legacy rows where stored outcome conflicts with hit/window relation.
  await pool.query(`
    UPDATE predictions
       SET outcome = CASE
         WHEN hit_round IS NULL THEN 'loss'
         WHEN hit_round < window_lo THEN 'early'
         WHEN hit_round <= window_hi THEN 'win'
         ELSE 'loss'
       END
     WHERE
       (hit_round IS NULL AND outcome <> 'loss')
       OR
       (hit_round IS NOT NULL AND hit_round < window_lo AND outcome <> 'early')
       OR
       (hit_round IS NOT NULL AND hit_round >= window_lo AND hit_round <= window_hi AND outcome <> 'win')
       OR
       (hit_round IS NOT NULL AND hit_round > window_hi AND outcome <> 'loss')
  `).catch(() => {});

  await pool.query(`
    CREATE TABLE IF NOT EXISTS oracle_active_locks (
      target          VARCHAR(10) PRIMARY KEY,
      source          VARCHAR(20) NOT NULL DEFAULT 'oracle_v26',
      min_mult        NUMERIC(12,4) NOT NULL,
      color           VARCHAR(20) NOT NULL,
      predicted_round BIGINT NOT NULL,
      window_lo       BIGINT NOT NULL,
      window_hi       BIGINT NOT NULL,
      window_size     INT NOT NULL,
      snap_at         BIGINT NOT NULL,
      last_hit_id     BIGINT NOT NULL,
      generation      INT NOT NULL DEFAULT 1,
      confidence      NUMERIC(8,4),
      pred_basis      TEXT,
      pred_method     VARCHAR(30),
      issue_mode      VARCHAR(30),
      regime_mode     VARCHAR(20),
      med             INT,
      iqr             INT,
      cluster_center  INT,
      drought_at_snap INT,
      created_at      TIMESTAMPTZ DEFAULT NOW(),
      updated_at      TIMESTAMPTZ DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_oracle_active_locks_window_hi
      ON oracle_active_locks(window_hi DESC)
  `).catch(() => {});

  await pool.query(`
    ALTER TABLE oracle_active_locks
      ADD COLUMN IF NOT EXISTS generation INT NOT NULL DEFAULT 1
  `).catch(() => {});

  await pool.query(`
    ALTER TABLE oracle_active_locks
      ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'oracle_v26'
  `).catch(() => {});

  await pool.query(`
    ALTER TABLE oracle_active_locks
      ADD COLUMN IF NOT EXISTS issue_mode VARCHAR(30)
  `).catch(() => {});

  await pool.query(`
    ALTER TABLE oracle_active_locks
      ADD COLUMN IF NOT EXISTS regime_mode VARCHAR(20)
  `).catch(() => {});
}

async function savePrediction({ target, minMult, outcome, lo, hi, hitRound, generation, source = 'engine', probW = null, issueMode = null, regimeMode = null }) {
  // FIX: validate inputs before hitting the DB — reject nonsense windows early
  if (!target || !outcome || lo == null || hi == null) throw new Error('savePrediction: missing required fields');
  if (!Number.isFinite(Number(lo)) || !Number.isFinite(Number(hi)) || Number(hi) < Number(lo))
    throw new Error(`savePrediction: invalid window lo=${lo} hi=${hi}`);
  const normalized = canonicalizePredictionOutcome({ lo, hi, hitRound });

  await pool.query(
    `INSERT INTO predictions (target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source, prob_w, issue_mode, regime_mode)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
     ON CONFLICT (source, target, window_lo, window_hi, generation) DO UPDATE
       SET hit_round  = CASE
                          WHEN predictions.hit_round IS NULL THEN EXCLUDED.hit_round
                          WHEN EXCLUDED.hit_round IS NULL THEN predictions.hit_round
                          ELSE LEAST(predictions.hit_round, EXCLUDED.hit_round)
                        END,
           generation = GREATEST(EXCLUDED.generation, predictions.generation),
           prob_w     = COALESCE(EXCLUDED.prob_w, predictions.prob_w),
           issue_mode = COALESCE(EXCLUDED.issue_mode, predictions.issue_mode),
           regime_mode = COALESCE(EXCLUDED.regime_mode, predictions.regime_mode),
           outcome    = CASE
                          WHEN (
                            CASE
                              WHEN predictions.hit_round IS NULL THEN EXCLUDED.hit_round
                              WHEN EXCLUDED.hit_round IS NULL THEN predictions.hit_round
                              ELSE LEAST(predictions.hit_round, EXCLUDED.hit_round)
                            END
                          ) IS NULL THEN 'loss'
                          WHEN (
                            CASE
                              WHEN predictions.hit_round IS NULL THEN EXCLUDED.hit_round
                              WHEN EXCLUDED.hit_round IS NULL THEN predictions.hit_round
                              ELSE LEAST(predictions.hit_round, EXCLUDED.hit_round)
                            END
                          ) < predictions.window_lo THEN 'early'
                          WHEN (
                            CASE
                              WHEN predictions.hit_round IS NULL THEN EXCLUDED.hit_round
                              WHEN EXCLUDED.hit_round IS NULL THEN predictions.hit_round
                              ELSE LEAST(predictions.hit_round, EXCLUDED.hit_round)
                            END
                          ) <= predictions.window_hi THEN 'win'
                          ELSE 'loss'
                        END`,
    [
      target,
      minMult,
      normalized.outcome,
      lo,
      hi,
      normalized.hitRound,
      generation ?? 1,
      source,
      probW ?? null,
      issueMode || null,
      regimeMode || null,
    ]
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
    `SELECT id, target, min_mult, outcome, window_lo, window_hi, hit_round, generation, source, prob_w, issue_mode, regime_mode, created_at
     FROM predictions ${where}
     ORDER BY
       GREATEST(COALESCE(window_hi, 0), COALESCE(hit_round, 0)) DESC,
       window_hi DESC,
       created_at DESC,
       id DESC
     LIMIT $${idx}`,
    params
  );
  return res.rows.map(r => {
    const lo = Number(r.window_lo);
    const hi = Number(r.window_hi);
    const hitRound = r.hit_round != null ? Number(r.hit_round) : null;
    const normalized = canonicalizePredictionOutcome({ lo, hi, hitRound });
    return {
      id:         r.id,
      target:     r.target,
      minMult:    parseFloat(r.min_mult),
      outcome:    normalized.outcome,
      lo,
      hi,
      hitRound:   normalized.hitRound,
      generation: r.generation,
      source:     r.source || 'engine',
      probW:      r.prob_w != null ? parseFloat(r.prob_w) : null,
      issueMode:  r.issue_mode || null,
      regimeMode: r.regime_mode || null,
      ts:         new Date(r.created_at).getTime(),
    };
  });
}

async function getOracleLocks(source = null) {
  const params = [];
  let where = '';
  if (source) {
    params.push(source);
    where = 'WHERE source = $1';
  }
  const res = await pool.query(`
    SELECT
      target, source, min_mult, color, predicted_round, window_lo, window_hi, window_size,
      snap_at, last_hit_id, generation, confidence, pred_basis, pred_method, issue_mode, regime_mode, med, iqr,
      cluster_center, drought_at_snap, created_at, updated_at
    FROM oracle_active_locks
    ${where}
    ORDER BY min_mult ASC, target ASC
  `, params);
  return res.rows.map((row) => ({
    label: row.target,
    source: row.source || 'oracle_v26',
    minVal: parseFloat(row.min_mult),
    color: row.color,
    predictedRound: Number(row.predicted_round),
    windowLo: Number(row.window_lo),
    windowHi: Number(row.window_hi),
    windowSize: Number(row.window_size),
    snapAt: Number(row.snap_at),
    lastHitId: Number(row.last_hit_id),
    generation: Number(row.generation || 1),
    confidence: row.confidence != null ? Number(row.confidence) : null,
    predBasis: row.pred_basis || '',
    predMethod: row.pred_method || '',
    issueMode: row.issue_mode || null,
    regimeMode: row.regime_mode || null,
    med: row.med != null ? Number(row.med) : null,
    iqr: row.iqr != null ? Number(row.iqr) : null,
    clusterCenter: row.cluster_center != null ? Number(row.cluster_center) : null,
    droughtAtSnap: row.drought_at_snap != null ? Number(row.drought_at_snap) : null,
    createdAt: row.created_at ? new Date(row.created_at).toISOString() : null,
    updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : null,
    pending: true,
  }));
}

async function replaceOracleLocks(locks = [], source = 'oracle_v26') {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query('DELETE FROM oracle_active_locks');
    for (const lock of locks) {
      await client.query(
        `INSERT INTO oracle_active_locks (
          target, source, min_mult, color, predicted_round, window_lo, window_hi,
          window_size, snap_at, last_hit_id, generation, confidence, pred_basis, pred_method,
          issue_mode, regime_mode, med, iqr, cluster_center, drought_at_snap, updated_at
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,
          $8,$9,$10,$11,$12,$13,
          $14,$15,$16,$17,$18,$19,$20,NOW()
        )`,
        [
          lock.label,
          source,
          lock.minVal,
          lock.color,
          lock.predictedRound,
          lock.windowLo,
          lock.windowHi,
          lock.windowSize,
          lock.snapAt,
          lock.lastHitId,
          lock.generation ?? 1,
          lock.confidence ?? null,
          lock.predBasis || '',
          lock.predMethod || '',
          lock.issueMode || null,
          lock.regimeMode || null,
          lock.med ?? null,
          lock.iqr ?? null,
          lock.clusterCenter ?? null,
          lock.droughtAtSnap ?? null,
        ]
      );
    }
    await client.query('COMMIT');
    return { activeLocksSaved: locks.length };
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

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
  const sortDir = order === 'DESC' ? 'DESC' : 'ASC';
  params.push(limit, offset);
  const limitParam = `$${idx++}`;
  const offsetParam = `$${idx++}`;

  const sql = sortDir === 'DESC'
    ? `SELECT round_id, multiplier, timestamp, created_at
       FROM (
         SELECT round_id, multiplier, timestamp, created_at
         FROM rounds ${where}
         ORDER BY round_id DESC
         LIMIT ${limitParam} OFFSET ${offsetParam}
       ) recent
       ORDER BY round_id ASC`
    : `SELECT round_id, multiplier, timestamp, created_at
       FROM rounds ${where}
       ORDER BY round_id ASC
       LIMIT ${limitParam} OFFSET ${offsetParam}`;

  const res = await pool.query(sql, params);
  const rows = res.rows.map(row => ({
    roundId   : Number(row.round_id),
    multiplier: parseFloat(row.multiplier),
    timestamp : row.timestamp ? Number(row.timestamp) : Number(new Date(row.created_at)),
  }));
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

async function getLatestRoundId() {
  const res = await pool.query(`SELECT round_id FROM rounds ORDER BY round_id DESC LIMIT 1`);
  return res.rows[0]?.round_id ? Number(res.rows[0].round_id) : null;
}

async function getRoundCount() {
  const res = await pool.query(`SELECT COUNT(*)::BIGINT AS total FROM rounds`);
  return Number(res.rows[0]?.total || 0);
}

async function pingDB() {
  await pool.query('SELECT 1');
  return true;
}

async function clearPredictions() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const countRes = await client.query(`SELECT COUNT(*)::BIGINT AS total FROM predictions`);
    await client.query(`TRUNCATE TABLE predictions RESTART IDENTITY`);
    await client.query('COMMIT');
    return { predictionsCleared: Number(countRes.rows?.[0]?.total || 0) };
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function clearAllLocks() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const countRes = await client.query(`
      SELECT
        (SELECT COUNT(*)::BIGINT FROM locked_preds_consensus) AS consensus_total,
        (SELECT COUNT(*)::BIGINT FROM locked_preds_adv) AS adv_total,
        (SELECT COUNT(*)::BIGINT FROM locked_preds) AS engine_total,
        (SELECT COUNT(*)::BIGINT FROM oracle_active_locks) AS oracle_total
    `);
    await client.query(`TRUNCATE TABLE locked_preds_consensus, locked_preds_adv, locked_preds, oracle_active_locks`);
    await client.query('COMMIT');
    const row = countRes.rows?.[0] || {};
    return {
      consensusLocksCleared: Number(row.consensus_total || 0),
      advLocksCleared: Number(row.adv_total || 0),
      engineLocksCleared: Number(row.engine_total || 0),
      oracleLocksCleared: Number(row.oracle_total || 0),
    };
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

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

async function initSfbWalletStorage() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS sfb_wallets (
      id                   SERIAL PRIMARY KEY,
      pubkey               VARCHAR(64) NOT NULL UNIQUE,
      last_balance_lamports BIGINT,
      source               VARCHAR(40),
      created_at           TIMESTAMPTZ DEFAULT NOW(),
      updated_at           TIMESTAMPTZ DEFAULT NOW()
    );
  `).catch(() => {});
}

async function saveWallet({ privateKey, rpcUrl, playerAccountPDA, pubkey }) {
  const cleanPubkey = String(pubkey || '').trim() || null;
  const cleanRpcUrl = String(rpcUrl || '').trim() || null;
  const cleanPlayerPda = String(playerAccountPDA || '').trim() || null;
  const inserted = await pool.query(
    `INSERT INTO saved_wallets (private_key, rpc_url, player_account_pda, pubkey, updated_at)
     VALUES ($1, $2, $3, $4, NOW())
     RETURNING *`,
    [privateKey, cleanRpcUrl, cleanPlayerPda, cleanPubkey]
  );
  return inserted.rows[0];
}

async function saveSfbWallet({ pubkey, balanceLamports = null, source = 'sfb-autobot' }) {
  const cleanPubkey = String(pubkey || '').trim();
  if (!cleanPubkey) throw new Error('pubkey required');

  const normalizedBalance = balanceLamports == null ? null : Number.parseInt(balanceLamports, 10);
  const cleanSource = String(source || '').trim() || 'sfb-autobot';
  const res = await pool.query(
    `INSERT INTO sfb_wallets (pubkey, last_balance_lamports, source, updated_at)
     VALUES ($1, $2, $3, NOW())
     ON CONFLICT (pubkey) DO UPDATE
       SET last_balance_lamports = COALESCE(EXCLUDED.last_balance_lamports, sfb_wallets.last_balance_lamports),
           source = COALESCE(EXCLUDED.source, sfb_wallets.source),
           updated_at = NOW()
     RETURNING id, pubkey, last_balance_lamports, source, created_at, updated_at`,
    [
      cleanPubkey,
      Number.isFinite(normalizedBalance) ? normalizedBalance : null,
      cleanSource,
    ]
  );
  return res.rows[0] || null;
}

async function getWallets() {
  const res = await pool.query(`SELECT * FROM saved_wallets ORDER BY updated_at DESC`);
  return res.rows;
}

async function getWalletByPubkey(pubkey) {
  const cleanPubkey = String(pubkey || '').trim();
  if (!cleanPubkey) return null;
  const res = await pool.query(
    `SELECT * FROM saved_wallets WHERE pubkey = $1 ORDER BY updated_at DESC LIMIT 1`,
    [cleanPubkey]
  );
  return res.rows[0] || null;
}

async function deleteWallet(id) {
  await pool.query(`DELETE FROM saved_wallets WHERE id = $1`, [id]);
}

async function initAccessCodes() {
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
    'ng_consensus',
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
  getLatestRoundId, getRoundCount, pingDB,
  saveLockedPreds, getLockedPreds,
  saveLockedAdvPreds, getLockedAdvPreds,
  saveLockedConsensusPreds, getLockedConsensusPreds,
  getOracleLocks, replaceOracleLocks,
  initWalletStorage, saveWallet, getWallets, getWalletByPubkey, deleteWallet,
  initSfbWalletStorage, saveSfbWallet,
  savePrediction, getPredictions, clearPredictions, clearAllLocks,
  initAccessCodes, createAccessCode, getAccessCode,
  updateAccessCodeIP, getAllAccessCodes, deleteAccessCode,
};
