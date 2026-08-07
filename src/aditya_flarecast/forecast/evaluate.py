"""Forecast evaluation: skill scores + operational lead time.

Beyond point-wise classification metrics, the operationally important quantity
is **lead time**: for each real flare, how long before its peak did the
forecaster first raise (and sustain) an alert? We compute it by scanning the
predicted-probability series and, for each ground-truth flare peak, finding the
earliest alert within the horizon window preceding that peak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aditya_flarecast.metrics.skill_scores import (
    best_threshold_by_tss,
    brier_score,
    contingency,
    lead_time_stats,
)
from aditya_flarecast.nowcast.classifier import class_rank


def compute_lead_times(
    times: pd.DatetimeIndex,
    probs: np.ndarray,
    catalogue: pd.DataFrame,
    threshold: float,
    horizon_min: float,
    min_class: str,
    peak_col: str = "peak_time",
    class_col: str = "goes_class",
) -> tuple[list[float], dict]:
    """Return per-flare lead times (minutes) and a hit/miss summary."""
    from aditya_flarecast.timeutils import minutes_to_ns, to_ns_array

    alert = probs >= threshold

    if len(times) == 0:
        return [], {"n_flares": 0, "n_hit": 0, "hit_rate": 0.0}
    t_ns = to_ns_array(times)

    cat = catalogue.copy()
    peaks = pd.to_datetime(cat[peak_col], utc=True)
    if class_col in cat.columns:
        keep = cat[class_col].map(lambda c: class_rank(str(c)) >= class_rank(min_class))
        peaks = peaks[keep]
    peak_ns = np.sort(to_ns_array(peaks)) if len(peaks) else np.array([], dtype=np.int64)

    # Restrict to flares whose peak falls inside the evaluation time span.
    lo_span, hi_span = t_ns.min(), t_ns.max()
    peak_ns = peak_ns[(peak_ns >= lo_span) & (peak_ns <= hi_span)]

    horizon_ns = minutes_to_ns(horizon_min)
    lead_times: list[float] = []
    n_hit = 0
    for pk in peak_ns:
        lo = pk - horizon_ns
        window = (t_ns >= lo) & (t_ns <= pk)
        if not window.any():
            lead_times.append(np.nan)
            continue
        fired = alert & window
        if fired.any():
            first_alert_ns = t_ns[fired][0]
            lead_min = (pk - first_alert_ns) / 60e9
            lead_times.append(float(lead_min))
            n_hit += 1
        else:
            lead_times.append(np.nan)

    n_peaks = int(len(peak_ns))
    summary = {
        "n_flares": n_peaks,
        "n_hit": int(n_hit),
        "hit_rate": float(n_hit / n_peaks) if n_peaks else 0.0,
        "lead_time_min": lead_time_stats(np.array(lead_times, dtype=float)),
    }
    return lead_times, summary


def evaluate_forecast(
    times: pd.DatetimeIndex,
    y_true: np.ndarray,
    probs: np.ndarray,
    catalogue: pd.DataFrame,
    horizon_min: float,
    min_class: str,
    threshold: float | None = None,
) -> dict:
    """Full forecast evaluation report (point metrics + lead time)."""
    if threshold is None:
        threshold, _ = best_threshold_by_tss(y_true, probs)

    cm = contingency(y_true, probs >= threshold)
    brier = brier_score(y_true, probs)
    _, lead = compute_lead_times(
        times, probs, catalogue, threshold, horizon_min, min_class
    )

    return {
        "threshold": float(threshold),
        "point_metrics": cm.as_dict(),
        "brier_score": brier,
        "flare_level": lead,
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
