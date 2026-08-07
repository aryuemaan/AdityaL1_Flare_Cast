"""Tests for the preprocessing pipeline and feature builder."""
from __future__ import annotations

import numpy as np

from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.preprocessing.background import subtract_background
from aditya_flarecast.preprocessing.pipeline import preprocess


def test_preprocess_frame(settings, synthetic):
    solexs, hel1os, _ = synthetic
    df = preprocess(solexs, hel1os, settings)
    assert "quality" in df.columns
    assert df["quality"].mean() > 0.5
    # Both instruments present, on a common grid.
    assert any(c.startswith("solexs_") for c in df.columns)
    assert any(c.startswith("hel1os_") for c in df.columns)
    assert df.attrs["cadence_s"] == settings.instrument.analysis_cadence_s


def test_background_below_signal(synthetic):
    solexs, _, _ = synthetic
    exc, bg = subtract_background(solexs.series("flux_1_8A"), window_min=30)
    assert (bg <= solexs.series("flux_1_8A") + 1e-12).mean() > 0.9
    assert (exc >= 0).all()


def test_features_causal_no_nan(settings, synthetic):
    solexs, hel1os, _ = synthetic
    df = preprocess(solexs, hel1os, settings)
    feats = build_features(df)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()
    # Expect the physically-motivated features to exist.
    for f in ("hardness", "neupert_corr_15m", "soft_logslope_10m",
              "hard_burst_sigma", "mins_since_hard_burst"):
        assert f in feats.columns
