"""Step 3A demo: weather -> features -> LightGBM generation forecast.

    pip install -r requirements.txt
    python run_generation_demo.py

Data source: real ENTSO-E generation + Open-Meteo weather if ENTSOE_API_KEY is
set, otherwise a clearly-labelled synthetic fallback (see
src/pipeline/data_assembly.py). The feature and validation layers are
identical either way.
"""

import logging

from dotenv import load_dotenv

from src.features import build_generation_feature_matrix
from src.models import GenerationForecaster, time_series_cv_score
from src.pipeline import DatasetConfig, assemble_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo3a")

WEATHER_COLS = ["target_time", "temperature_2m", "wind_speed_100m",
                "wind_direction_100m", "shortwave_radiation", "cloud_cover"]


def main() -> None:
    df = assemble_dataset(DatasetConfig())

    # Features from WEATHER ONLY (never pass price/demand here or they leak in).
    feat = build_generation_feature_matrix(df[WEATHER_COLS].copy())
    feat["generation_mw"] = df["generation_mw"].to_numpy()

    # lag/rolling produce NaNs in the first rows; LightGBM tolerates NaN, but we
    # drop rows with no target for a clean start.
    n_before = len(feat)
    feat = feat.dropna(subset=["generation_mw"]).reset_index(drop=True)
    log.info("Feature matrix: %d rows, %d raw -> ready.", len(feat), n_before)

    # --- Time-series-appropriate CV (walk-forward) ---
    cv = time_series_cv_score(feat, n_splits=5)
    print("\n=== TimeSeriesSplit CV (walk-forward) ===")
    print(f"MAE : {cv['mae_mean']:.1f} +/- {cv['mae_std']:.1f} MW")
    print(f"RMSE: {cv['rmse_mean']:.1f} MW")
    print(f"Number of features: {cv['n_features']}")

    mean_gen = feat["generation_mw"].mean()
    print(f"Mean generation: {mean_gen:.0f} MW -> normalized MAE ~{cv['mae_mean']/mean_gen*100:.1f}%")

    # --- Fit the final model + feature importance ---
    model = GenerationForecaster().fit(feat)
    print("\n=== Top 12 features (gain) ===")
    print(model.feature_importance(top=12).to_string(index=False))


if __name__ == "__main__":
    load_dotenv()
    main()
