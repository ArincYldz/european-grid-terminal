-- =====================================================================
--  European Electricity Market — Time-Series Schema (TimescaleDB)
-- =====================================================================
--  Design philosophy: BITEMPORAL + LONG (narrow) format.
--
--  Why long (entity, variable, value) format, not wide?
--    - Sources carry different variables (weather: 5 variables, generation:
--      ~15 fuel types, price: 1). In a wide table each new variable =
--      ALTER TABLE (schema migration). In long format a new variable = just
--      a new row. In production, schema stability is worth gold.
--
--  Why bitemporal (target_time + available_at)?
--    - To make look-ahead bias impossible at the DATABASE level. The
--      question "what did I know as of T?" can only be answered if the
--      available_at column exists. This guarantees the validity of the
--      backtest.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Source/variable metadata (reference table, not a hypertable).
CREATE TABLE IF NOT EXISTS series_meta (
    series_key   TEXT PRIMARY KEY,          -- e.g. 'de_lu.wind_speed_100m'
    entity       TEXT NOT NULL,             -- e.g. 'de_lu', 'bremen'
    variable     TEXT NOT NULL,             -- e.g. 'wind_speed_100m', 'price_eur_mwh'
    unit         TEXT,                      -- e.g. 'm/s', 'EUR/MWh', 'MW'
    description  TEXT
);

-- =====================================================================
--  Main measurements table (hypertable)
-- =====================================================================
CREATE TABLE IF NOT EXISTS measurements (
    target_time   TIMESTAMPTZ      NOT NULL,   -- the instant the data BELONGS TO
    available_at  TIMESTAMPTZ      NOT NULL,   -- the instant the data was LEARNED
    series_key    TEXT             NOT NULL REFERENCES series_meta(series_key),
    value         DOUBLE PRECISION,            -- may stay NULL for NaN/missing
    is_imputed    BOOLEAN          NOT NULL DEFAULT FALSE,
    source        TEXT             NOT NULL,
    ingested_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),

    -- Composite primary key: same target hour + same knowledge instant +
    -- same series = ONE record. This preserves forecast snapshots (different
    -- available_at) while providing idempotent upserts when the same row is
    -- re-fetched.
    PRIMARY KEY (series_key, target_time, available_at)
);

-- Hypertable on target_time — TimescaleDB automatically splits data into
-- time-range "chunks" (default 7 days). Benefit: time-range queries touch
-- only the relevant chunks (chunk exclusion), and old chunks are
-- compressed/dropped in a single operation.
SELECT create_hypertable('measurements', 'target_time', if_not_exists => TRUE);

-- as-of queries (the most recent value up to a given knowledge instant)
-- filter on available_at, so this index is critical:
CREATE INDEX IF NOT EXISTS idx_meas_series_avail
    ON measurements (series_key, available_at DESC, target_time DESC);

-- =====================================================================
--  Compression — storage + read optimization
-- =====================================================================
--  Chunks older than 7 days are compressed automatically. Time-series data
--  within one series is highly repetitive, so 10x+ compression is typical.
ALTER TABLE measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'series_key',
    timescaledb.compress_orderby   = 'target_time DESC, available_at DESC'
);

SELECT add_compression_policy('measurements', INTERVAL '7 days', if_not_exists => TRUE);

-- =====================================================================
--  Continuous Aggregate — "realized" (latest revision) hourly view
-- =====================================================================
--  Backtests and dashboards often want "the LAST KNOWN value for each target
--  hour". Computing this on every query is expensive; we precompute it with a
--  materialized view.
--  NOTE: this is a "latest, not as-of" view — for reporting, not for live
--  forecasting. The leakage-safe as-of join is still done at decision time
--  with merge_asof / SQL LATERAL (see storage.read_as_of).
CREATE MATERIALIZED VIEW IF NOT EXISTS measurements_latest_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', target_time) AS bucket,
    series_key,
    last(value, available_at)          AS latest_value
FROM measurements
GROUP BY bucket, series_key
WITH NO DATA;

SELECT add_continuous_aggregate_policy('measurements_latest_hourly',
    start_offset => INTERVAL '30 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);
