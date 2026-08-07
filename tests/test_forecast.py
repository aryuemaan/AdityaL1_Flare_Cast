"""Tests for the forecasting stage: dataset, training, serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.forecast.dataset import build_dataset, chronological_split
from aditya_flarecast.forecast.predict import Forecaster
from aditya_flarecast.forecast.train import train_forecaster
from aditya_flarecast.preprocessing.pipeline import preprocess


def _features_and_labels(settings, synthetic):
    solexs, hel1os, truth = synthetic
    df = preprocess(solexs, hel1os, settings)
    feats = build_features(df)
    labels = truth.rename(columns={"soft_peak_time": "peak_time"})[
        ["peak_time", "goes_class"]
    ]
    return df, feats, labels


def test_dataset_and_split(settings, synthetic):
    _, feats, labels = _features_and_labels(settings, synthetic)
    ds = build_dataset(feats, labels, settings.forecast)
    assert len(ds.X) == len(ds.y) == len(ds.times)
    assert ds.y.sum() > 0  # at least some positive windows
    splits = chronological_split(ds, 0.25, 0.15)
    # Chronological: test times come after train times.
    assert splits["train"].times.max() <= splits["test"].times.min()


def test_train_and_serve(settings, synthetic):
    df, feats, labels = _features_and_labels(settings, synthetic)
    result = train_forecaster(feats, labels, settings, quality=df["quality"])
    assert (result.artifact_dir / "model.pkl").exists()
    assert (result.artifact_dir / "metadata.json").exists()
    assert 0.0 <= result.threshold <= 1.0

    fc = Forecaster.load(result.artifact_dir)
    out = fc.predict(feats)
    assert out.probability.shape[0] == len(feats)
    assert ((out.probability >= 0) & (out.probability <= 1)).all()

    latest = fc.predict_latest(feats)
    assert set(latest) == {"time", "probability", "alert", "threshold", "horizon_min"}


def test_label_horizon_semantics(settings, synthetic):
    from aditya_flarecast.forecast.dataset import build_labels

    _, feats, labels = _features_and_labels(settings, synthetic)
    times = feats.index[::20]
    y = build_labels(times, labels, horizon_min=60, min_class="C")
    assert y.dtype.kind in ("i", "u")
    assert set(np.unique(y)).issubset({0, 1})
