"""Load a trained artifact and produce forecasts for new data.

:class:`Forecaster` is the serving-side object used by the API, the dashboard,
and the streaming simulator. It loads ``model.pkl`` + ``metadata.json``, aligns
incoming features to the training feature schema, and returns calibrated flare
probabilities and a boolean alert at the tuned threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aditya_flarecast.forecast.models import BaseForecastModel
from aditya_flarecast.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastOutput:
    times: pd.DatetimeIndex
    probability: np.ndarray
    alert: np.ndarray
    threshold: float
    horizon_min: float


class Forecaster:
    """Serving wrapper around a trained forecasting artifact."""

    def __init__(self, model: BaseForecastModel, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.feature_cols: list[str] = metadata["feature_cols"]
        self.threshold: float = float(metadata.get("alert_threshold", 0.5))
        self.horizon_min: float = float(
            metadata.get("forecast_config", {}).get("horizon_min", 60.0)
        )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "Forecaster":
        artifact_dir = Path(artifact_dir)
        model = BaseForecastModel.load(artifact_dir / "model.pkl")
        with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        logger.info("Loaded forecaster (%s) from %s", model.name, artifact_dir)
        return cls(model, metadata)

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in features.columns]
        for c in missing:
            features[c] = 0.0
        return features[self.feature_cols]

    def predict(self, features: pd.DataFrame) -> ForecastOutput:
        X = self._align(features.copy())
        prob = self.model.predict_proba(X)
        return ForecastOutput(
            times=features.index,
            probability=prob,
            alert=prob >= self.threshold,
            threshold=self.threshold,
            horizon_min=self.horizon_min,
        )

    def predict_latest(self, features: pd.DataFrame) -> dict:
        """Return the single most recent forecast (for real-time serving)."""
        out = self.predict(features.tail(1))
        return {
            "time": str(out.times[-1]),
            "probability": float(out.probability[-1]),
            "alert": bool(out.alert[-1]),
            "threshold": out.threshold,
            "horizon_min": out.horizon_min,
        }
