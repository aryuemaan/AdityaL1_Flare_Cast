"""Tests for the forecast verification metrics."""
from __future__ import annotations

import numpy as np

from aditya_flarecast.metrics.skill_scores import (
    best_threshold_by_tss,
    brier_score,
    contingency,
    lead_time_stats,
)


def test_perfect_forecast():
    y = np.array([0, 0, 1, 1, 1])
    m = contingency(y, y)
    assert m.pod == 1.0
    assert m.far == 0.0
    assert m.tss == 1.0
    assert m.hss == 1.0


def test_all_wrong():
    y = np.array([0, 1, 0, 1])
    m = contingency(y, 1 - y)
    assert m.pod == 0.0
    assert m.tss <= 0.0


def test_brier_bounds():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == 0.0
    assert 0.0 <= brier_score(y, np.full(4, 0.5)) <= 1.0


def test_best_threshold_recovers_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    # Probabilities correlated with y -> a threshold should give positive TSS.
    p = np.clip(0.2 * rng.random(500) + 0.6 * y, 0, 1)
    t, m = best_threshold_by_tss(y, p)
    assert 0.0 < t < 1.0
    assert m.tss > 0.3


def test_lead_time_stats_empty():
    s = lead_time_stats(np.array([np.nan, np.nan]))
    assert s["n"] == 0
