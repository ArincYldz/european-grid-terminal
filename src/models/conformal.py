"""Conformalized Quantile Regression (CQR) — finite-sample coverage guarantee.

Problem it solves: raw quantile regression (QuantileForecaster) is only
*asymptotically* calibrated. In practice the P10-P90 band was too narrow
(~69% empirical coverage instead of the nominal 80%): the model is
overconfident. Point forecasts hide the tail; miscalibrated intervals give
a FALSE sense of how wide the tail is — arguably worse.

CQR (Romano, Patterson & Candes, 2019) fixes this with a distribution-free,
finite-sample guarantee. On a held-out calibration set it measures how far
reality falls outside the predicted band, then inflates the band by exactly
that margin:

    conformity score   E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )
    correction          Q  = ceil((n+1)(1-alpha)) / n   empirical quantile of {E_i}
    calibrated band     [ q_lo(x) - Q ,  q_hi(x) + Q ]

This guarantees marginal coverage >= 1 - alpha regardless of whether the
base quantile model was any good. The calibration set is split off
TEMPORALLY (never random) so the guarantee holds under time-series ordering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .price_model import QuantileForecaster

logger = logging.getLogger(__name__)


class ConformalizedQuantileForecaster:
    """Wraps QuantileForecaster and calibrates its interval width via CQR.

    lo/hi define the target interval (default 0.1..0.9 = 80% nominal).
    The median (p50) is passed through from the base model unchanged.
    """

    def __init__(self, lo: float = 0.1, hi: float = 0.9, calib_fraction: float = 0.25,
                 params: dict | None = None):
        self.lo, self.hi = lo, hi
        self.alpha = (lo) + (1.0 - hi)  # total miscoverage, e.g. 0.1+0.1 = 0.2
        self.calib_fraction = calib_fraction
        self.base = QuantileForecaster(quantiles=(lo, 0.5, hi), params=params)
        self.correction_: float = 0.0

    def fit(self, df: pd.DataFrame, target: str = "price_eur_mwh"):
        df = df.sort_values("target_time").reset_index(drop=True)
        cut = int(len(df) * (1 - self.calib_fraction))
        train, calib = df.iloc[:cut], df.iloc[cut:]

        # 1) Fit base quantile models on the earlier slice.
        self.base.fit(train, target=target)

        # 2) Measure conformity scores on the temporal calibration slice.
        q = self.base.predict(calib)
        lo_col, hi_col = f"p{int(self.lo*100)}", f"p{int(self.hi*100)}"
        y = calib[target].to_numpy()
        scores = np.maximum(q[lo_col].to_numpy() - y, y - q[hi_col].to_numpy())

        # 3) Finite-sample corrected quantile of the scores.
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.correction_ = float(np.quantile(scores, level, method="higher"))
        logger.info(
            "CQR correction Q = %.2f EUR/MWh (n_calib=%d, target coverage %.0f%%)",
            self.correction_, n, (1 - self.alpha) * 100,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated p_lo / p50 / p_hi columns (band widened by Q)."""
        q = self.base.predict(df).copy()
        lo_col, hi_col = f"p{int(self.lo*100)}", f"p{int(self.hi*100)}"
        q[lo_col] = q[lo_col] - self.correction_
        q[hi_col] = q[hi_col] + self.correction_
        # Re-sort rows to keep p_lo <= p50 <= p_hi after widening (widening
        # can only help monotonicity, but stay defensive).
        vals = np.sort(q[[lo_col, "p50", hi_col]].to_numpy(), axis=1)
        q[[lo_col, "p50", hi_col]] = vals
        return q


class AdaptiveConformalForecaster:
    """Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) for time series.

    Why not plain CQR here? CQR's coverage guarantee assumes exchangeability.
    Electricity prices are seasonal and regime-switching: the calibration
    window (e.g. autumn) does not represent the test window (e.g. winter),
    so a single fixed correction under-covers. ACI is built exactly for this
    non-exchangeable setting.

    Mechanism: it keeps an effective miscoverage level alpha_t and updates it
    online from realized coverage feedback (yesterday's price IS known today):

        interval_t = [ q_lo(x) - Q(1-alpha_t), q_hi(x) + Q(1-alpha_t) ]
        err_t      = 1 if y_t fell outside the interval else 0
        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

    When we under-cover (err=1 often), alpha_t drops -> wider band; when we
    over-cover, it narrows. Long-run coverage converges to 1 - alpha_target
    regardless of distribution shift. gamma is the adaptation step size.
    """

    def __init__(self, lo: float = 0.1, hi: float = 0.9, calib_fraction: float = 0.25,
                 gamma: float = 0.03, params: dict | None = None):
        self.lo, self.hi = lo, hi
        self.alpha_target = lo + (1.0 - hi)
        self.calib_fraction = calib_fraction
        self.gamma = gamma
        self.base = QuantileForecaster(quantiles=(lo, 0.5, hi), params=params)
        self.scores_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, target: str = "price_eur_mwh"):
        df = df.sort_values("target_time").reset_index(drop=True)
        cut = int(len(df) * (1 - self.calib_fraction))
        train, calib = df.iloc[:cut], df.iloc[cut:]
        self.base.fit(train, target=target)

        q = self.base.predict(calib)
        lo_col, hi_col = f"p{int(self.lo*100)}", f"p{int(self.hi*100)}"
        y = calib[target].to_numpy()
        self.scores_ = np.maximum(q[lo_col].to_numpy() - y, y - q[hi_col].to_numpy())
        return self

    def _q(self, alpha_t: float) -> float:
        """Score quantile at level (1 - alpha_t), clipped to a valid range."""
        level = float(np.clip(1.0 - alpha_t, 0.0, 1.0))
        if level >= 1.0:
            return float(np.max(self.scores_)) if len(self.scores_) else 0.0
        if level <= 0.0:
            return float(np.min(self.scores_))
        return float(np.quantile(self.scores_, level, method="higher"))

    def predict_adaptive(self, df: pd.DataFrame, target: str = "price_eur_mwh") -> pd.DataFrame:
        """Walk the test set in time order, updating alpha_t from feedback.

        Returns the base quantile frame with calibrated p_lo/p_hi columns.
        Requires the realized target column to be present (backtest / eval),
        because ACI consumes coverage feedback as it goes.
        """
        df = df.sort_values("target_time").reset_index(drop=True)
        q = self.base.predict(df).copy()
        lo_col, hi_col = f"p{int(self.lo*100)}", f"p{int(self.hi*100)}"
        y = df[target].to_numpy()
        base_lo, base_hi = q[lo_col].to_numpy(), q[hi_col].to_numpy()

        alpha_t = self.alpha_target
        out_lo, out_hi, alphas = np.empty(len(df)), np.empty(len(df)), np.empty(len(df))
        for t in range(len(df)):
            Qt = self._q(alpha_t)
            lo_t, hi_t = base_lo[t] - Qt, base_hi[t] + Qt
            out_lo[t], out_hi[t], alphas[t] = lo_t, hi_t, alpha_t
            covered = (y[t] >= lo_t) and (y[t] <= hi_t)
            alpha_t = float(np.clip(alpha_t + self.gamma * (self.alpha_target - (0.0 if covered else 1.0)), 0.0, 1.0))

        q[lo_col], q[hi_col] = out_lo, out_hi
        q["alpha_t"] = alphas
        return q


def empirical_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of realized values that fall inside [lo, hi]."""
    return float(((y_true >= lo) & (y_true <= hi)).mean())
