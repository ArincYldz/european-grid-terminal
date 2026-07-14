"""Missing-data policy and leakage-safe join helpers.

PRODUCTION RULE: raw data is NEVER silently filled.
  1. DETECT and measure the gap (how many intervals, how long).
  2. LABEL the gap (is_imputed / is_stale columns) — so the model and risk
     layer can see "this value is imputed, not real".
  3. Fill short gaps ONLY on physical, slowly-varying signals (like
     temperature) with time-weighted interpolation.
  4. On long gaps DO NOT FILL: leave NaN. When the decision engine sees NaN
     it takes no position (NO_SIGNAL). In trading, not trading on missing
     data is always cheaper than trading on made-up data.
  5. bfill is FORBIDDEN in production: it carries a future value into the
     past (look-ahead bias itself).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Gaps shorter than this threshold may be closed by interpolation (for
# physical signals). For price series this threshold is 0: price is NOT
# interpolated — one hour's price cannot be derived from another, it is a
# separate auction result.
MAX_INTERPOLATE_GAP = 2  # number of consecutive missing intervals


def detect_gaps(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Reindexes the series onto a regular time grid, making gaps visible.

    On an outage, APIs usually do not return an error but SILENTLY drop rows
    (jumping from 10:00 to 14:00). Without reindexing we never notice the gap
    — that is the most dangerous scenario.
    """
    df = df.set_index("target_time").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "target_time"

    n_missing = int(df.drop(columns=["source"], errors="ignore").isna().all(axis=1).sum())
    if n_missing:
        logger.warning("Detected %d missing intervals in the series (%s grid).", n_missing, freq)
    return df.reset_index()


def impute_physical_signal(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Cautious filling + audit trail for physical signals.

    - Short gap (<= MAX_INTERPOLATE_GAP): time-weighted interpolation.
      Rationale: temperature/radiation change roughly continuously between
      two points; interpolation uses BOTH sides of the gap, not just the past
      and current neighbours — which is why we do this ONLY when preparing
      TRAINING data. In a live (real-time) forecast the right side of the gap
      does not exist yet; there this function naturally cannot fill and NaN
      remains — which is also the correct behaviour.
    - Long gap: NaN remains, the decision engine opens no position.
    - Every filled cell is marked with `is_imputed`.
    """
    df = df.copy()
    df["is_imputed"] = df[value_cols].isna().any(axis=1)

    df = df.set_index("target_time")
    df[value_cols] = df[value_cols].interpolate(
        method="time",
        limit=MAX_INTERPOLATE_GAP,
        limit_area="inside",  # do not fabricate the start/end: interior gaps only
    )
    df = df.reset_index()

    still_missing = int(df[value_cols].isna().any(axis=1).sum())
    if still_missing:
        logger.warning(
            "%d intervals exceeded the interpolation threshold, left as NaN (NO_SIGNAL zone).",
            still_missing,
        )
    return df


def leakage_safe_join(
    base: pd.DataFrame,
    feature: pd.DataFrame,
    decision_time_col: str = "decision_time",
    feature_cols: list[str] | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Joins features by the "knowable at decision time" rule.

    Classic mistake: a plain merge on target_time. That assumes "I knew every
    datum belonging to that hour at that hour", which is wrong (actual
    generation is published ~1 hour late, day-ahead price is announced in a
    batch the day before, weather forecast is updated run by run...).

    The right way: for each row, at decision time T, take the MOST RECENT
    record with available_at <= T. In pandas this is called merge_asof
    (backward search). If there are several forecast runs for the same target
    hour, it automatically picks the last run before T.
    """
    feature = feature.sort_values("available_at")
    base = base.sort_values(decision_time_col)

    cols = feature_cols or [
        c for c in feature.columns if c not in ("available_at", "source")
    ]
    merged = pd.merge_asof(
        base,
        feature[["available_at", *cols]].rename(
            columns={c: f"{c}{suffix}" for c in cols}
        ),
        left_on=decision_time_col,
        right_on="available_at",
        direction="backward",  # look ONLY to the past — the leakage-proof direction
    )
    return merged.drop(columns=["available_at"])
