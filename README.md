# European Electricity Market — Intraday Price & Risk Forecasting Bot

An AI system that forecasts renewable generation and intraday electricity
prices, scores negative-price risk, and turns forecasts into STORE / SELL /
CURTAIL / ARBITRAGE signals — validated in a cost-aware backtest.

## Architecture
[Data Ingestion (APIs)] -> [Time-Series DB] -> [AI/Forecast Engine] -> [Decision/Signal Bot]

## Status — all four steps complete
- [x] **Step 1 — Data Ingestion**: Open-Meteo (historical + forecast), ENTSO-E
      (day-ahead price, actual generation, load), a data-quality layer, and a
      leakage-safe join (`merge_asof`).
- [x] **Step 2 — Time-series database (TimescaleDB)**: bitemporal long-format
      hypertable, compression + continuous aggregate, idempotent upsert
      (revision + idempotence), leakage-safe `read_as_of` (DISTINCT ON).
      `docker compose up -d` -> `python run_storage_demo.py`
- [x] **Step 3A — Generation forecast (LightGBM)**: leakage-safe feature
      engineering (power curve, cyclical calendar, past-looking lag/rolling),
      `TimeSeriesSplit` walk-forward CV, LightGBM + early stopping.
      `python run_generation_demo.py`
- [x] **Step 3B — Price + negative-price risk**: model cascade (3A->3B,
      leakage-safe via out-of-fold), residual-load feature, quantile
      regression (P10/P50/P90) + calibrated negative-price probability
      (isotonic). `python run_price_demo.py`
- [x] **Step 4 — Decision/signal engine + backtest**: expected-value argmax
      policy (SELL/STORE/CURTAIL/DISCHARGE), thresholds derived from economics
      (marginal cost, -subsidy floor, storage opportunity cost), walk-forward
      backtest with realistic costs (spread + imbalance penalty + round-trip
      efficiency). Parameters are calibrated from TRAIN history
      (`calibrate_params_from_history`) and the asset is scaled to a single
      producer, so the strategy is meaningful on real data (+~13% vs naive on
      real German 2024, driven by battery arbitrage + selective curtailment).
      `python run_decision_demo.py`

### Final additions
- [x] **Real-data integration**: `src/pipeline/data_assembly.py` assembles one
      aligned hourly frame using the best available source, in priority order:
      **ENTSO-E** (if `ENTSOE_API_KEY` is set) → **Energy-Charts** (Fraunhofer
      ISE, REAL data with NO key) → **synthetic** (offline last resort). So the
      project runs on REAL German electricity data by default, no key needed.
      Verified end-to-end on real 2024 data (8783 hourly rows).
- [x] **Risk reporting**: Sharpe ratio, max drawdown, and CVaR (`src/backtest/
      risk.py`), plus a P&L equity-curve + drawdown plot saved to
      `reports/equity_curve.png`.
- [x] **Conformal calibration**: split CQR and Adaptive Conformal Inference
      (ACI) close the quantile coverage gap (~69% -> ~80%). ACI is used because
      seasonal distribution shift breaks the exchangeability that plain CQR
      assumes.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # fill in ENTSOE_API_KEY (optional; synthetic fallback otherwise)
python run_decision_demo.py
```
Without an ENTSO-E key everything still runs on the synthetic fallback. With a
key, the same pipeline runs on real data. Start TimescaleDB (`docker compose up
-d`) only for the Step 2 storage demo.

## Core design rules (interview summary)
1. **Bitemporal schema**: every row carries `target_time` (the instant the data
   belongs to) and `available_at` (the instant it was learned). For decision
   time T, features join with `available_at <= T` only, via
   `pd.merge_asof(direction="backward")`.
2. **No silent filling**: bfill is forbidden in production (look-ahead), ffill
   is forbidden on price series. Short gaps are interpolated only on physical
   signals with an `is_imputed` flag; long gaps stay NaN -> the decision engine
   emits NO_SIGNAL.
3. **Typed errors**: transient (retry) / permanent (alert) / data-quality
   (stop the pipeline) are distinguished.
4. **Leakage defence is one thread through every layer**: bitemporal
   `available_at`, `shift`/`rolling` direction, out-of-fold cascade,
   `TimeSeriesSplit`, and backtest settlement all repeat the same principle.

## Web dashboard (ENTSO-E-style map)

`webapp/` turns the engine into a public dashboard: a Europe map; click a
country to see its generation forecast, price uncertainty band, negative-price
risk and hourly trading signals.

- `webapp/precompute.py` — runs the engine for EVERY country Energy-Charts
  serves (dynamic discovery, failures skipped) and writes static JSON to
  `webapp/site/data/`. Per country: real gen/load/price + weather window,
  three LightGBM models (wind / solar / DEMAND), CQR price quantiles,
  calibrated negative-price risk, hourly signals with Turkish rationale,
  plus a 14-day HOLDOUT evaluation (generation MAE%, price-band coverage%,
  strategy-vs-naive backtest edge% and Sharpe) and a 48h forecast-vs-actual
  replay.
- `webapp/site/` — fully static frontend (D3 choropleth map colored by a
  selectable live metric, model-credibility panel, forecast-vs-actual replay
  charts, signal tooltips), no build step. Deployable anywhere static files
  are served.

Local run:
```bash
python webapp/precompute.py          # all countries (~10-20 min, rate-limited API)
python -m http.server 8942 --directory webapp/site
# open http://localhost:8942
```

### Deploy (Vercel)
1. Push this repo to GitHub.
2. In Vercel: New Project → import the repo → set **Root Directory** to
   `webapp/site` → Framework preset: **Other** (static). Deploy.
3. Optional freshness: `.github/workflows/refresh-data.yml` re-runs the
   precompute daily and commits the JSON; Vercel auto-redeploys on push.
   `.github/workflows/ci.yml` runs the full test suite on every push/PR.

## Tests
```bash
set PYTHONPATH=.        # Windows: $env:PYTHONPATH="."
python tests/test_feature_leakage.py
python tests/test_price_models.py
python tests/test_decision.py
python tests/test_conformal.py
python tests/test_storage_transform.py
```
