"""TimescaleDB storage adapter: idempotent writes + leakage-safe reads.

This class hides the existence of the database from the layers above
(repository pattern). If a different Postgres/Timescale setup replaces
TimescaleDB tomorrow, this is the only file that changes.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class TimescaleStore:
    def __init__(self, database_url: str | None = None):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError(
                "DATABASE_URL is not set. Add it to your .env file (see .env.example) "
                "and start TimescaleDB with `docker compose up -d`."
            )
        # pool_pre_ping: catches dead connections in the pool before a query
        # (in long-running bots, if the DB restarts this gives a clean
        # reconnect instead of a silent failure).
        self.engine: Engine = create_engine(url, pool_pre_ping=True)

    # ---------------------------------------------------------------
    #  WRITE
    # ---------------------------------------------------------------
    def upsert_series_meta(self, meta: pd.DataFrame) -> None:
        """Writes series_meta (updates if present). Required for FK integrity."""
        rows = meta.to_dict("records")
        stmt = text(
            """
            INSERT INTO series_meta (series_key, entity, variable, unit, description)
            VALUES (:series_key, :entity, :variable, :unit, :description)
            ON CONFLICT (series_key) DO UPDATE
                SET unit = EXCLUDED.unit,
                    description = EXCLUDED.description
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, rows)

    def upsert_measurements(self, long_df: pd.DataFrame) -> int:
        """Idempotently writes long-format measurements.

        Why ON CONFLICT DO UPDATE (upsert) instead of a plain INSERT?
          1. IDEMPOTENCE: if the pipeline crashes and re-fetches the same hour
             twice, no duplicate row must appear. The composite PK + upsert
             guarantee this.
          2. REVISION: ENTSO-E revises realized data later (same target_time +
             same available_at window, corrected value). DO UPDATE handles
             this cleanly.

        NOTE: the same target_time with a DIFFERENT available_at does NOT
        conflict — i.e. different forecast runs are preserved (available_at is
        in the PK). Only the exact same (series, target, availability) triple
        is updated.
        """
        if long_df.empty:
            return 0

        # NaN -> None (SQL NULL). pandas NaN blows up in psycopg.
        records = long_df.where(pd.notna(long_df), None).to_dict("records")
        stmt = text(
            """
            INSERT INTO measurements
                (target_time, available_at, series_key, value, is_imputed, source)
            VALUES
                (:target_time, :available_at, :series_key, :value, :is_imputed, :source)
            ON CONFLICT (series_key, target_time, available_at) DO UPDATE
                SET value = EXCLUDED.value,
                    is_imputed = EXCLUDED.is_imputed,
                    source = EXCLUDED.source,
                    ingested_at = now()
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, records)
        logger.info("%d measurement rows written/updated.", len(records))
        return len(records)

    # ---------------------------------------------------------------
    #  READ
    # ---------------------------------------------------------------
    def read_as_of(
        self,
        series_keys: list[str],
        decision_time: pd.Timestamp,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """LEAKAGE-SAFE read: 'what did I know as of decision_time?'

        For each (series_key, target_time), returns the MOST RECENT value
        satisfying available_at <= decision_time. This is the database-side
        counterpart of the quality.merge_asof logic (DISTINCT ON).

        Calling this while sliding decision_time forward at each backtest step
        makes peeking into the future PHYSICALLY impossible: the
        WHERE available_at <= :decision_time filter cuts leakage out.

        Why DISTINCT ON ... ORDER BY available_at DESC?
          In Postgres it is the fastest idiom for "last row per group"; it uses
          the idx_meas_series_avail index directly.
        """
        stmt = text(
            """
            SELECT DISTINCT ON (series_key, target_time)
                   target_time, series_key, value, is_imputed, available_at
            FROM measurements
            WHERE series_key = ANY(:series_keys)
              AND target_time >= :start
              AND target_time <  :end
              AND available_at <= :decision_time      -- << leakage shield
            ORDER BY series_key, target_time, available_at DESC
            """
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(
                stmt,
                conn,
                params={
                    "series_keys": series_keys,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "decision_time": decision_time.isoformat(),
                },
                parse_dates=["target_time", "available_at"],
            )
        return df

    def read_latest_hourly(
        self, series_keys: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """'Last known value' from the continuous aggregate (for reporting).

        CAUTION: this is NOT as-of, it is latest — do NOT use it as a backtest
        feature (it contains leakage). For dashboards/analysis only.
        """
        stmt = text(
            """
            SELECT bucket AS target_time, series_key, latest_value AS value
            FROM measurements_latest_hourly
            WHERE series_key = ANY(:series_keys)
              AND bucket >= :start AND bucket < :end
            ORDER BY series_key, bucket
            """
        )
        with self.engine.connect() as conn:
            return pd.read_sql(
                stmt, conn,
                params={"series_keys": series_keys, "start": start.isoformat(), "end": end.isoformat()},
                parse_dates=["target_time"],
            )

    def ping(self) -> bool:
        """Connection liveness check."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 — ping intentionally catches broadly
            logger.error("DB ping failed: %s", exc)
            return False
