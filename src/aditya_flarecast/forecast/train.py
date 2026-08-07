"""Train the forecasting model and persist a self-contained artifact.

The saved artifact directory contains:

* ``model.pkl``     — the fitted estimator (any back-end).
* ``metadata.json`` — feature columns, config, tuned alert threshold, and the
  held-out test metrics, so the model can be loaded and served without the
  training code.

Threshold tuning is done on a chronological validation split by maximising the
True Skill Statistic (TSS), the standard operational criterion; the reported
scores come from an untouched future test split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aditya_flarecast.config import Settings
from aditya_flarecast.forecast.dataset import build_dataset, chronological_split
from aditya_flarecast.forecast.evaluate import evaluate_forecast
from aditya_flarecast.forecast.models import BaseForecastModel, build_model
from aditya_flarecast.logging_utils import get_logger
from aditya_flarecast.metrics.skill_scores import best_threshold_by_tss

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model: BaseForecastModel
    threshold: float
    metadata: dict
    artifact_dir: Path


def train_forecaster(
    features: pd.DataFrame,
    catalogue: pd.DataFrame,
    settings: Settings,
    quality: pd.Series | None = None,
    artifact_dir: str | Path | None = None,
) -> TrainingResult:
    """Build the dataset, train, tune threshold, evaluate, and persist."""
    cfg = settings.forecast
    cadence = features.attrs.get("cadence_s", settings.instrument.analysis_cadence_s)

    ds = build_dataset(features, catalogue, cfg, quality=quality, cadence_s=cadence)
    logger.info(
        "Forecast dataset: %d samples, positive rate %.3f",
        len(ds.X), float(np.mean(ds.y)) if len(ds.y) else 0.0,
    )
    if ds.y.sum() < 5:
        logger.warning(
            "Very few positive samples (%d). Consider a longer timeline, a "
            "lower min_class, or a larger horizon.", int(ds.y.sum())
        )

    splits = chronological_split(ds, cfg.test_fraction, cfg.val_fraction)
    tr, va, te = splits["train"], splits["val"], splits["test"]

    model = build_model(cfg.model, feature_cols=ds.feature_cols)
    logger.info("Training back-end: %s", model.name)
    model.fit(tr.X, tr.y)

    # Tune threshold on validation (fallback to train if val is empty/degenerate).
    if len(va.y) > 0 and va.y.sum() > 0:
        va_prob = model.predict_proba(va.X)
        threshold, _ = best_threshold_by_tss(va.y, va_prob)
    else:
        threshold = cfg.alert_threshold
    logger.info("Tuned alert threshold (max TSS on val): %.3f", threshold)

    # Evaluate on the untouched test split.
    te_prob = model.predict_proba(te.X) if len(te.X) else np.array([])
    report = (
        evaluate_forecast(
            te.times, te.y, te_prob, catalogue,
            cfg.horizon_min, cfg.min_class, threshold=threshold,
        )
        if len(te.X)
        else {}
    )

    importances = model.feature_importance()

    metadata = {
        "model_backend": model.name,
        "feature_cols": ds.feature_cols,
        "cadence_s": cadence,
        "forecast_config": cfg.model_dump(),
        "instrument_config": settings.instrument.model_dump(),
        "alert_threshold": float(threshold),
        "n_train": int(len(tr.X)),
        "n_val": int(len(va.X)),
        "n_test": int(len(te.X)),
        "test_report": report,
        "top_features": dict(list(importances.items())[:15]),
    }

    artifact_dir = Path(artifact_dir or (settings.paths.models / "forecaster"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / "model.pkl")
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    if report:
        pm = report["point_metrics"]
        fl = report["flare_level"]
        logger.info(
            "TEST | TSS=%.3f HSS=%.3f POD=%.3f FAR=%.3f | flares hit %d/%d "
            "(%.0f%%) | median lead=%.1f min",
            pm["tss"], pm["hss"], pm["pod"], pm["far"],
            fl["n_hit"], fl["n_flares"], 100 * fl["hit_rate"],
            fl["lead_time_min"]["median"],
        )

    return TrainingResult(
        model=model, threshold=threshold, metadata=metadata, artifact_dir=artifact_dir
    )
