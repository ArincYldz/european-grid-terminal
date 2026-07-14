"""Storage transform-layer tests (run without a DB).

The logic here is database-independent, so it can be validated without a live
Timescale. The as-of SQL query is separately tested against a live DB via
run_storage_demo.py.
"""

import pandas as pd

from src.storage.schema import series_meta_from_long, to_long_format


def _sample_wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_time": pd.to_datetime(
                ["2025-06-01T00:00Z", "2025-06-01T01:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2025-06-06T00:00Z", "2025-06-06T01:00Z"], utc=True
            ),
            "wind_speed_100m": [6.02, 6.38],
            "temperature_2m": [16.9, float("nan")],  # one missing value
            "source": ["open_meteo_era5", "open_meteo_era5"],
        }
    )


def test_melt_produces_series_keys():
    long_df = to_long_format(_sample_wide(), "de_lu", ["wind_speed_100m", "temperature_2m"])
    keys = set(long_df["series_key"])
    assert keys == {"de_lu.wind_speed_100m", "de_lu.temperature_2m"}
    # 2 target hours x 2 variables = 4 rows
    assert len(long_df) == 4


def test_nan_value_preserved_as_null_not_dropped():
    # A missing VALUE row is not dropped (kept as NULL); only rows without a
    # timestamp are dropped. The fact "there was no data" is itself a record.
    long_df = to_long_format(_sample_wide(), "de_lu", ["temperature_2m"])
    assert len(long_df) == 2
    assert long_df["value"].isna().sum() == 1


def test_is_imputed_defaults_false_when_absent():
    long_df = to_long_format(_sample_wide(), "de_lu", ["wind_speed_100m"])
    assert long_df["is_imputed"].eq(False).all()


def test_is_imputed_preserved_when_present():
    wide = _sample_wide()
    wide["is_imputed"] = [False, True]
    long_df = to_long_format(wide, "de_lu", ["wind_speed_100m"])
    assert long_df.sort_values("target_time")["is_imputed"].tolist() == [False, True]


def test_series_meta_extraction():
    long_df = to_long_format(_sample_wide(), "de_lu", ["wind_speed_100m", "temperature_2m"])
    meta = series_meta_from_long(long_df, "de_lu", units={"wind_speed_100m": "m/s"})
    assert set(meta["series_key"]) == {"de_lu.wind_speed_100m", "de_lu.temperature_2m"}
    row = meta.set_index("series_key").loc["de_lu.wind_speed_100m"]
    assert row["unit"] == "m/s"


def test_missing_required_columns_raises():
    bad = pd.DataFrame({"target_time": [pd.Timestamp("2025-01-01", tz="UTC")], "x": [1]})
    try:
        to_long_format(bad, "de_lu", ["x"])
        assert False, "expected an error with missing available_at/source"
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)
