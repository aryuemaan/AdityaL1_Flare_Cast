"""Supervised dataset construction for flare forecasting.

We frame forecasting as: *given features summarising the last `lookback_min`
minutes, will a flare of at least `min_class` PEAK within the next
`horizon_min` minutes?* Labels are built from a flare catalogue with a
``peak_time`` (and optional ``goes_class``) column — this can be the synthetic
ground truth, the nowcast master catalogue (self-supervised), or an external
event list.

Splitting is strictly chronological (train in the past, test in the future) to
avoid look-ahead leakage and to mirror operational deployment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aditya_flarecast.config import ForecastConfig
from aditya_flarecast.nowcast.classifier import class_rank


@dataclass
class ForecastDataset:
    X: pd.DataFrame            # feature matrix indexed by decision time
    y: np.ndarray             # binary labels
    times: pd.DatetimeIndex   # decision times (index of X)
    feature_cols: list[str]


def build_labels(
    times: pd.DatetimeIndex,
    catalogue: pd.DataFrame,
    horizon_min: float,
    min_class: str,
    peak_col: str = "peak_time",
    class_col: str = "goes_class",
) -> np.ndarray:
    """Label each decision time 1 if a qualifying flare peaks within horizon."""
    from aditya_flarecast.timeutils import minutes_to_ns, to_ns_array

    if catalogue.empty:
        return np.zeros(len(times), dtype=int)

    cat = catalogue.copy()
    peaks = pd.to_datetime(cat[peak_col], utc=True)
    if class_col in cat.columns:
        keep = cat[class_col].map(lambda c: class_rank(str(c)) >= class_rank(min_class))
        peaks = peaks[keep]
    peak_ns = np.sort(to_ns_array(peaks)) if len(peaks) else np.array([], dtype=np.int64)

    labels = np.zeros(len(times), dtype=int)
    if peak_ns.size == 0:
        return labels
    t_ns = to_ns_array(times)
    horizon_ns = minutes_to_ns(horizon_min)
    # For each decision time, is there a qualifying peak in (t, t+horizon]?
    left = np.searchsorted(peak_ns, t_ns, side="right")
    right = np.searchsorted(peak_ns, t_ns + horizon_ns, side="right")
    labels[right > left] = 1
    return labels


def build_dataset(
    features: pd.DataFrame,
    catalogue: pd.DataFrame,
    cfg: ForecastConfig,
    quality: pd.Series | None = None,
    cadence_s: float | None = None,
) -> ForecastDataset:
    """Assemble the sliding-window forecasting dataset."""
    cadence_s = cadence_s or features.attrs.get("cadence_s", 10.0)
    stride = max(1, int(round(cfg.stride_min * 60.0 / cadence_s)))

    # Drop the initial lookback region where rolling features are unreliable.
    warmup = int(round(cfg.lookback_min * 60.0 / cadence_s))
    sel = np.arange(warmup, len(features), stride)

    Xdf = features.iloc[sel].copy()
    if quality is not None:
        good = quality.iloc[sel].to_numpy().astype(bool)
        Xdf = Xdf.loc[good]
    times = Xdf.index

    y = build_labels(
        times, catalogue, cfg.horizon_min, cfg.min_class
    )
    return ForecastDataset(
        X=Xdf, y=y, times=times, feature_cols=list(Xdf.columns)
    )


def chronological_split(
    ds: ForecastDataset, test_fraction: float, val_fraction: float
) -> dict[str, ForecastDataset]:
    """Split into train/val/test by time (no shuffling)."""
    n = len(ds.X)
    n_test = int(round(n * test_fraction))
    n_trainval = n - n_test
    n_val = int(round(n_trainval * val_fraction))
    n_train = n_trainval - n_val

    def slice_ds(a: int, b: int) -> ForecastDataset:
        return ForecastDataset(
            X=ds.X.iloc[a:b],
            y=ds.y[a:b],
            times=ds.times[a:b],
            feature_cols=ds.feature_cols,
        )

    return {
        "train": slice_ds(0, n_train),
        "val": slice_ds(n_train, n_train + n_val),
        "test": slice_ds(n_train + n_val, n),
    }
