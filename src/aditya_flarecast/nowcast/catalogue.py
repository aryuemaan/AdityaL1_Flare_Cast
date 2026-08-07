"""Nowcast orchestration + catalogue evaluation.

:func:`run_nowcast` executes the full detection stage on a preprocessed frame
and returns the fused master catalogue. :func:`evaluate_catalogue` compares a
detected catalogue against a ground-truth catalogue (available for synthetic
data or from an external event list) using time-tolerance matching, reporting
detection completeness by flare class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aditya_flarecast.config import Settings
from aditya_flarecast.logging_utils import get_logger
from aditya_flarecast.nowcast.classifier import class_letter
from aditya_flarecast.nowcast.detector import detect_hard, detect_soft
from aditya_flarecast.nowcast.fusion import fuse, to_dataframe

logger = get_logger(__name__)


def run_nowcast(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Detect flares in both channels and fuse them into a master catalogue."""
    cadence = df.attrs.get("cadence_s", settings.instrument.analysis_cadence_s)
    soft = detect_soft(df, settings.nowcast, cadence)
    hard = detect_hard(df, settings.nowcast, cadence)
    logger.info(
        "Nowcast raw detections: %d soft, %d hard bursts", len(soft), len(hard)
    )
    fused = fuse(soft, hard, settings.nowcast)
    cat = to_dataframe(fused)
    n_flares = int((cat["channel"] == "fused").sum()) if not cat.empty else 0
    n_prec = int(cat.get("candidate_precursor", pd.Series(dtype=bool)).sum()) \
        if not cat.empty else 0
    logger.info(
        "Master catalogue: %d flares, %d candidate precursors", n_flares, n_prec
    )
    return cat


def evaluate_catalogue(
    detected: pd.DataFrame,
    truth: pd.DataFrame,
    tolerance_s: float = 300.0,
) -> dict:
    """Match detected flares to ground truth within a time tolerance.

    Returns per-class recall/precision and overall counts. Only flare-type
    detections (``channel == 'fused'`` or ``'soft'``) are scored against the
    soft-flare ground truth; isolated hard bursts are excluded.
    """
    from aditya_flarecast.timeutils import seconds_to_ns, to_ns_array

    tol_ns = seconds_to_ns(tolerance_s)

    det = detected[detected["channel"].isin(["fused", "soft"])].copy()
    # Use int64 nanoseconds throughout to avoid tz-aware datetime64 pitfalls.
    det_ns = (
        np.array([], dtype=np.int64) if det.empty else to_ns_array(det["peak_time"])
    )
    truth_ns = to_ns_array(truth["soft_peak_time"])

    matched_truth = np.zeros(len(truth), dtype=bool)
    matched_det = np.zeros(len(det_ns), dtype=bool)

    for ti in range(len(truth_ns)):
        if len(det_ns) == 0:
            break
        dt = np.abs(det_ns - truth_ns[ti])
        j = int(np.argmin(dt))
        if not matched_det[j] and dt[j] <= tol_ns:
            matched_truth[ti] = True
            matched_det[j] = True

    truth = truth.copy()
    truth["_matched"] = matched_truth
    truth["_letter"] = truth["goes_class"].map(class_letter)

    per_class = {}
    for letter in ["A", "B", "C", "M", "X"]:
        sub = truth[truth["_letter"] == letter]
        if len(sub) == 0:
            continue
        per_class[letter] = {
            "n_truth": int(len(sub)),
            "n_detected": int(sub["_matched"].sum()),
            "recall": float(sub["_matched"].mean()),
        }

    tp = int(matched_truth.sum())
    fn = int((~matched_truth).sum())
    fp = int((~matched_det).sum()) if len(matched_det) else 0

    def safe(a, b):
        return a / b if b else 0.0

    return {
        "n_truth": int(len(truth)),
        "n_detected": int(len(det)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": safe(tp, tp + fn),
        "precision": safe(tp, tp + fp),
        "per_class": per_class,
        "tolerance_s": tolerance_s,
    }
