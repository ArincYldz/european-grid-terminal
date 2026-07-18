"""Explainability for the price forecast — real SHAP, not a proxy.

The dashboard promises to open the black box, so this has to be the genuine
article. LightGBM ships exact TreeSHAP through `predict(pred_contrib=True)`,
so we get true Shapley values with no extra dependency and no sampling error:
every prediction decomposes exactly as

    prediction = base_value + sum(contribution_i)

Feature-importance bar charts (`model.feature_importances_`) would have been
cheaper, but they describe the MODEL globally ("this column was split on a
lot") and cannot say why THIS hour is priced the way it is. SHAP is per-row and
signed, which is what "explain this prediction" actually means.

Also here: a nearest-neighbour search for historical hours that looked like the
one being forecast, so the explanation can point at real precedent instead of
asserting confidence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature names come from the engineering layer and are not reader-friendly.
# Anything not listed falls back to a de-underscored version of its own name.
_LABELS = {
    "residual_load": "Residual load (demand minus renewables)",
    "residual_load_lag1": "Residual load, 1 h earlier",
    "residual_load_lag24": "Residual load, same hour yesterday",
    "residual_load_roll24": "Residual load, 24 h average",
    "residual_load_ramp": "Residual load, rate of change",
    "demand_forecast_mw": "Forecast demand",
    "demand_forecast_mw_lag1": "Demand, 1 h earlier",
    "demand_forecast_mw_lag24": "Demand, same hour yesterday",
    "demand_forecast_mw_roll24": "Demand, 24 h average",
    "demand_forecast_mw_ramp": "Demand, rate of change",
    "predicted_generation_mw": "Forecast renewable generation",
    "predicted_generation_mw_lag1": "Renewables, 1 h earlier",
    "predicted_generation_mw_lag24": "Renewables, same hour yesterday",
    "predicted_generation_mw_roll24": "Renewables, 24 h average",
    "predicted_generation_mw_ramp": "Renewables, rate of change",
    "renewable_penetration": "Renewable share of demand",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "dow_sin": "Day of week",
    "dow_cos": "Day of week",
    "month_sin": "Season",
    "month_cos": "Season",
    "is_weekend": "Weekend",
    "price_lag24": "Price, same hour yesterday",
    "price_lag48": "Price, same hour two days ago",
    "price_roll24": "Price, 24 h average",
    "doy_sin": "Time of year",
    "doy_cos": "Time of year",
    "predicted_generation_mw_ramp4": "Renewables, 4 h swing",
    "demand_forecast_mw_ramp4": "Demand, 4 h swing",
    "residual_load_ramp4": "Residual load, 4 h swing",
    "renewable_penetration_lag1": "Renewable share, 1 h earlier",
    "renewable_penetration_lag24": "Renewable share, same hour yesterday",
}

# Which driver each feature belongs to, for the compact "key drivers" readout.
_DRIVER = {
    "solar": ("predicted_generation", "renewable_penetration"),
    "demand": ("demand_forecast", "residual_load"),
    "calendar": ("hour_", "dow_", "month_", "is_weekend"),
    "price_history": ("price_lag", "price_roll"),
}


def humanise(feature: str) -> str:
    return _LABELS.get(feature, feature.replace("_", " ").capitalize())


def explain_rows(model, X: pd.DataFrame, feature_names: list[str],
                 top_n: int = 5) -> list[dict]:
    """Exact per-row TreeSHAP for a fitted LightGBM regressor.

    Returns one dict per row: base value, prediction, and the top_n features by
    absolute contribution, each with its signed effect in EUR/MWh and its share
    of the total explained movement.
    """
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("explain_rows needs a fitted LightGBM model.")

    # Shape (n_rows, n_features + 1); the last column is the base (expected) value.
    contrib = booster.predict(X[feature_names], pred_contrib=True)
    contrib = np.asarray(contrib, dtype=float)

    out = []
    for row in contrib:
        base, effects = float(row[-1]), row[:-1]
        order = np.argsort(np.abs(effects))[::-1][:top_n]
        total = float(np.sum(np.abs(effects))) or 1.0
        out.append({
            "base_eur_mwh": round(base, 1),
            "prediction_eur_mwh": round(base + float(np.sum(effects)), 1),
            "features": [{
                "name": feature_names[i],
                "label": humanise(feature_names[i]),
                "effect_eur_mwh": round(float(effects[i]), 1),
                "share_pct": round(abs(float(effects[i])) / total * 100.0, 1),
            } for i in order],
        })
    return out


def aggregate_explanation(per_row: list[dict], top_n: int = 5) -> dict:
    """Collapse per-hour SHAP into one explanation for the whole horizon.

    Averages the signed contribution per feature across hours, so a driver that
    pushes prices up in the morning and down at night correctly nets out rather
    than looking important in both directions.
    """
    if not per_row:
        return {}
    totals: dict[str, list[float]] = {}
    for row in per_row:
        for f in row["features"]:
            totals.setdefault(f["name"], []).append(f["effect_eur_mwh"])

    rows = []
    for name, effects in totals.items():
        mean_signed = float(np.mean(effects))
        rows.append({
            "name": name,
            "label": humanise(name),
            "effect_eur_mwh": round(mean_signed, 1),
            "abs_effect": abs(mean_signed),
            "hours_present": len(effects),
        })
    rows.sort(key=lambda r: r["abs_effect"], reverse=True)
    rows = rows[:top_n]

    denom = sum(r["abs_effect"] for r in rows) or 1.0
    for r in rows:
        r["share_pct"] = round(r["abs_effect"] / denom * 100.0, 1)
        r.pop("abs_effect")

    base = float(np.mean([r["base_eur_mwh"] for r in per_row]))
    return {"base_eur_mwh": round(base, 1), "features": rows}


def similar_days(train_df: pd.DataFrame, target_row: pd.Series,
                 feature_names: list[str], k: int = 3,
                 price_col: str = "price_eur_mwh") -> list[dict]:
    """Historical hours whose drivers looked most like the hour being forecast.

    Standardised Euclidean distance over the model's own feature space, so the
    comparison uses the same variables the model uses. Scaling matters: without
    it, megawatt-scale columns would drown out the sin/cos calendar terms.

    Returns the k closest, each with what the price ACTUALLY did — precedent the
    user can check, rather than a claim about confidence.
    """
    have = [f for f in feature_names if f in train_df.columns]
    if not have or price_col not in train_df.columns or len(train_df) < k + 1:
        return []

    M = train_df[have].to_numpy(dtype=float)
    v = target_row[have].to_numpy(dtype=float)
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)

    d = np.sqrt(np.nansum(((M - mu) / sd - (v - mu) / sd) ** 2, axis=1))
    idx = np.argsort(d)[:k]

    out = []
    for i in idx:
        r = train_df.iloc[int(i)]
        out.append({
            "time": pd.Timestamp(r["target_time"]).isoformat(),
            "price_eur_mwh": round(float(r[price_col]), 1),
            "distance": round(float(d[int(i)]), 2),
        })
    return out


def invalidators(p10: float, p50: float, p90: float, neg_risk: float,
                 top_features: list[dict]) -> list[str]:
    """What would make this forecast wrong — stated before the fact, not after.

    Derived from the forecast's own shape: a wide band, a live negative-price
    tail, or a heavy dependence on one driver each imply a different failure
    mode. Any model is only as good as the assumptions it cannot see breaking.
    """
    out = []
    width = p90 - p10
    if width > max(40.0, abs(p50) * 0.8):
        out.append(f"The band is wide ({round(width)} EUR/MWh between P10 and P90) — "
                   "the model itself is unsure, so treat the midpoint loosely.")
    if neg_risk > 0.15:
        out.append(f"There is a {round(neg_risk*100)}% modelled chance of negative prices; "
                   "a renewables surge would push the outcome below the band's midpoint.")
    if top_features:
        lead = top_features[0]
        out.append(f"{lead['label']} carries {lead['share_pct']}% of this forecast. "
                   "A weather revision that moves it invalidates the rest.")
    out.append("Unplanned outages, interconnector limits and intraday news are not "
               "inputs to this model, so any of them can override it.")
    return out
