-- ============================================================
--  APU Predictive Maintenance — Cloud SQL Schema
--  Database: apu_predictions (PostgreSQL 14)
--  Run: psql -h 127.0.0.1 -U apu_user -d apu_predictions -f infra/schema.sql
-- ============================================================

-- ── Table 1: Per-cycle predictions ───────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    engine_id       INTEGER       NOT NULL,
    cycle           INTEGER       NOT NULL,
    predicted_rul   FLOAT         NOT NULL,
    true_rul        FLOAT         NOT NULL,
    timestamp       TIMESTAMPTZ   DEFAULT NOW(),
    model_version   VARCHAR(50),
    filename        VARCHAR(255)
);

-- ── Table 2: Drift reports ────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_reports (
    id                      SERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ   DEFAULT NOW(),
    filename                VARCHAR(255),
    drifted_sensors         TEXT,             -- comma-separated sensor names
    drift_details           JSONB,            -- full per-sensor z-score details
    prediction_drift_flag   BOOLEAN,
    model_version           VARCHAR(50)
);

-- ── Table 3: Model promotions log ─────────────────────────────
CREATE TABLE IF NOT EXISTS model_promotions (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ   DEFAULT NOW(),
    old_version     VARCHAR(50),
    new_version     VARCHAR(50),
    old_r2          FLOAT,
    new_r2          FLOAT,
    promoted        BOOLEAN
);

-- ── Indexes for query performance ─────────────────────────────
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_engine    ON predictions(engine_id);
CREATE INDEX IF NOT EXISTS idx_predictions_filename  ON predictions(filename);
CREATE INDEX IF NOT EXISTS idx_drift_timestamp       ON drift_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_promotions_timestamp  ON model_promotions(timestamp);
