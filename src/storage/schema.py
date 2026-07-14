"""Converts the WIDE data from the ingestion layer into the DB's LONG schema.

This module is DB-independent (pure pandas) — so it can be unit-tested
without a database. It is the "translation" layer between ingestion (Step 1)
and storage (Step 2).
"""

from __future__ import annotations

import pandas as pd

# Required meta columns of the measurements schema (everything except the value)
_META_COLS = ("target_time", "available_at", "source", "is_imputed")


def to_long_format(
    df: pd.DataFrame,
    entity: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Wide (one column per variable) -> long (series_key, value) format.

    Input example:
        target_time | available_at | wind_speed_100m | temperature_2m | source
    Output example:
        target_time | available_at | series_key            | value | is_imputed | source
        ...         | ...          | de_lu.wind_speed_100m | 6.02  | False      | ...
        ...         | ...          | de_lu.temperature_2m  | 16.9  | False      | ...

    Why `melt` here? Because the long format is schema-stable: when a new
    weather variable arrives we do not alter the table, we just add rows with
    a new series_key.
    """
    missing = [c for c in ("target_time", "available_at", "source") if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from the input frame: {missing}")

    present_values = [c for c in value_cols if c in df.columns]
    if not present_values:
        raise ValueError(f"None of the specified value columns exist: {value_cols}")

    # is_imputed is optional; if absent all are False (real measurement).
    has_flag = "is_imputed" in df.columns
    id_vars = ["target_time", "available_at", "source"] + (["is_imputed"] if has_flag else [])

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=present_values,
        var_name="variable",
        value_name="value",
    )
    long_df["series_key"] = entity + "." + long_df["variable"]
    if not has_flag:
        long_df["is_imputed"] = False

    # NaN values are written as NULL (not dropped!) — the fact that "there was
    # no data at this hour" is itself a record; the decision engine reads it
    # as NO_SIGNAL. We only drop rows lacking a target/available timestamp.
    long_df = long_df.dropna(subset=["target_time", "available_at"])

    return long_df[
        ["target_time", "available_at", "series_key", "value", "is_imputed", "source", "variable"]
    ]


def series_meta_from_long(long_df: pd.DataFrame, entity: str, units: dict[str, str] | None = None) -> pd.DataFrame:
    """Extracts the unique series list for the series_meta table from a long frame."""
    units = units or {}
    meta = long_df[["series_key", "variable"]].drop_duplicates().copy()
    meta["entity"] = entity
    meta["unit"] = meta["variable"].map(units)
    meta["description"] = None
    return meta[["series_key", "entity", "variable", "unit", "description"]]
