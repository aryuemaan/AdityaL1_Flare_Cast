"""Feature engineering for the forecaster.

We turn the fused light-curve frame into rolling, physically-interpretable
features that a model can use to anticipate a flare. The design is driven by
solar-flare phenomenology:

* **Levels & excess** — current soft flux relative to background (pre-heating).
* **Rise dynamics** — slope and curvature of the soft flux over several
  timescales (gradual pre-flare rise is a known precursor).
* **Hard-X-ray activity** — count-rate excess and short-window bursts; hard
  emission leads soft emission (Neupert effect), so a rising hard signal is an
  early-warning feature.
* **Hardness ratio** — hard/soft ratio and its trend; impulsive precursors are
  spectrally hard.
* **Neupert proxy** — correlation between hard flux and d(soft)/dt.
* **Variability** — rolling std / fluctuation power capturing micro-flaring.

All features are computed causally (only past/current data within each window)
so they are valid for real-time forecasting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _win(cadence_s: float, minutes: float) -> int:
    return max(2, int(round(minutes * 60.0 / cadence_s)))


def build_features(df: pd.DataFrame, cadence_s: float | None = None) -> pd.DataFrame:
    """Compute the causal feature matrix from a preprocessed frame.

    Parameters
    ----------
    df:
        Output of :func:`aditya_flarecast.preprocessing.pipeline.preprocess`.
    """
    cadence_s = cadence_s or df.attrs.get("cadence_s", 10.0)
    soft_band = df.attrs.get("solexs_flux_band", "flux_1_8A")
    hard_low = df.attrs.get("hel1os_low_band", "counts_10_30keV")

    soft = df[f"solexs_{soft_band}"].astype(float)
    soft_bg = df[f"solexs_{soft_band}_bg"].astype(float)
    soft_exc = df[f"solexs_{soft_band}_excess"].astype(float)
    hard = df[f"hel1os_{hard_low}"].astype(float)
    hard_exc = df[f"hel1os_{hard_low}_excess"].astype(float)
    # High hard band if present.
    hard_hi_col = next(
        (c for c in df.columns if c.startswith("hel1os_") and "30_70" in c and c.endswith("keV")),
        None,
    )
    hard_hi = df[hard_hi_col].astype(float) if hard_hi_col else hard

    feats = pd.DataFrame(index=df.index)

    # --- Levels (log for many-decade dynamic range) ---------------------- #
    feats["log_soft"] = np.log10(soft.clip(lower=1e-10))
    feats["log_soft_bg"] = np.log10(soft_bg.clip(lower=1e-10))
    feats["soft_over_bg"] = (soft / soft_bg.clip(lower=1e-12)).clip(0, 1e3)
    feats["log_soft_excess"] = np.log10(soft_exc.clip(lower=1e-12) + 1e-12)

    feats["hard_excess"] = hard_exc
    feats["hard_hi_excess"] = (hard_hi - hard_hi.rolling(
        _win(cadence_s, 30), min_periods=5).median()).clip(lower=0)

    # --- Hardness ratio -------------------------------------------------- #
    hardness = hard / (soft * 1e8 + 1.0)  # scale soft into a comparable range
    feats["hardness"] = hardness
    feats["hardness_trend_10m"] = hardness.diff(_win(cadence_s, 10))

    # --- Rise dynamics of soft flux over several timescales -------------- #
    for m in (5, 10, 20, 30):
        w = _win(cadence_s, m)
        feats[f"soft_slope_{m}m"] = soft.diff(w) / (w * cadence_s)
        feats[f"soft_logslope_{m}m"] = feats["log_soft"].diff(w)
        feats[f"soft_std_{m}m"] = soft.rolling(w, min_periods=3).std()
        feats[f"hard_slope_{m}m"] = hard.diff(w) / (w * cadence_s)
        feats[f"hard_max_{m}m"] = hard_exc.rolling(w, min_periods=3).max()

    # Curvature (acceleration) of the soft rise.
    w10 = _win(cadence_s, 10)
    feats["soft_curvature_10m"] = feats["soft_slope_10m"].diff(w10)

    # --- Neupert proxy: rolling corr(hard, d(soft)/dt) ------------------- #
    dsoft = soft.diff().fillna(0.0)
    w15 = _win(cadence_s, 15)
    feats["neupert_corr_15m"] = (
        hard.rolling(w15, min_periods=5).corr(dsoft).fillna(0.0)
    )

    # --- Short-window burst indicators (micro-flare precursors) ---------- #
    med5 = hard.rolling(_win(cadence_s, 5), min_periods=3).median()
    mad5 = (hard - med5).abs().rolling(_win(cadence_s, 5), min_periods=3).median()
    feats["hard_burst_sigma"] = (
        (hard - med5) / (1.4826 * mad5 + 1e-9)
    ).clip(-10, 50)
    feats["hard_burst_count_30m"] = (
        (feats["hard_burst_sigma"] > 3.0).rolling(_win(cadence_s, 30),
                                                  min_periods=1).sum()
    )

    # --- Time since last significant hard burst (recency) ---------------- #
    burst = (feats["hard_burst_sigma"] > 3.0).astype(int).to_numpy()
    since = np.zeros(len(burst))
    counter = 1e4
    for i, b in enumerate(burst):
        counter = 0.0 if b else counter + cadence_s / 60.0  # minutes
        since[i] = counter
    feats["mins_since_hard_burst"] = np.clip(since, 0, 720)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Causal fill: only forward-fill (never peek ahead), then zero-fill the head.
    feats = feats.ffill().fillna(0.0)
    feats.attrs["cadence_s"] = cadence_s
    return feats


def feature_names(df_features: pd.DataFrame) -> list[str]:
    return list(df_features.columns)
