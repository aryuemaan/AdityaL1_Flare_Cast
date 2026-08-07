"""Forecast verification metrics used in operational space weather.

Binary flare forecasting is a rare-event problem, so accuracy is misleading.
We report the standard contingency-table skill scores used by NOAA SWPC and the
solar-flare-prediction literature, plus detection-focused rates and a lead-time
summary. All functions take integer/boolean arrays and are dependency-light.

Contingency table
------------------
::

              observed yes   observed no
    pred yes      TP             FP
    pred no       FN             TN
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ContingencyMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    pod: float          # Probability of Detection == TPR == recall
    far: float          # False Alarm Ratio = FP / (TP + FP)
    pofd: float         # Probability of False Detection = FP / (FP + TN)
    precision: float
    f1: float
    csi: float          # Critical Success Index (Threat Score)
    tss: float          # True Skill Statistic (Peirce) = POD - POFD
    hss: float          # Heidke Skill Score
    accuracy: float
    base_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> ContingencyMetrics:
    """Compute the full contingency-table metric set."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n = tp + fp + fn + tn

    def safe(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    pod = safe(tp, tp + fn)
    far = safe(fp, tp + fp)
    pofd = safe(fp, fp + tn)
    precision = safe(tp, tp + fp)
    f1 = safe(2 * precision * pod, precision + pod)
    csi = safe(tp, tp + fp + fn)
    tss = pod - pofd

    # Heidke skill score.
    expected_correct = safe((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp), n)
    hss = safe((tp + tn) - expected_correct, n - expected_correct) if n else 0.0

    return ContingencyMetrics(
        tp=tp, fp=fp, fn=fn, tn=tn,
        pod=pod, far=far, pofd=pofd,
        precision=precision, f1=f1, csi=csi,
        tss=tss, hss=hss,
        accuracy=safe(tp + tn, n),
        base_rate=safe(tp + fn, n),
    )


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def best_threshold_by_tss(
    y_true: np.ndarray, y_prob: np.ndarray, n_steps: int = 101
) -> tuple[float, ContingencyMetrics]:
    """Grid-search the probability threshold maximising the TSS."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_m, best_tss = 0.5, None, -np.inf
    for t in thresholds:
        m = contingency(y_true, y_prob >= t)
        if m.tss > best_tss:
            best_tss, best_t, best_m = m.tss, float(t), m
    return best_t, best_m  # type: ignore[return-value]


def lead_time_stats(lead_times_min: np.ndarray) -> dict:
    """Summarise per-flare lead times (minutes before peak the alert fired)."""
    lt = np.asarray([x for x in lead_times_min if x is not None and np.isfinite(x)])
    if lt.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": int(lt.size),
        "mean": float(np.mean(lt)),
        "median": float(np.median(lt)),
        "p10": float(np.percentile(lt, 10)),
        "p90": float(np.percentile(lt, 90)),
        "max": float(np.max(lt)),
    }
