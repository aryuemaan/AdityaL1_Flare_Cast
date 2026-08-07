"""Typed, layered configuration.

Configuration resolves in this order (later overrides earlier):

  1. Field defaults defined below.
  2. A YAML file (``configs/default.yaml`` by default).
  3. Environment variables prefixed with ``AFC_`` (e.g. ``AFC_RANDOM_SEED=7``).

Everything downstream depends only on :class:`Settings`, so behaviour is
reproducible and there are no magic constants scattered through the code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Sub-models
# --------------------------------------------------------------------------- #


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths are resolved against the repo root."""

    root: Path = Path(".")
    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    catalogues: Path = Path("data/catalogues")
    models: Path = Path("models")

    def resolve(self, base: Path) -> "PathsConfig":
        def r(p: Path) -> Path:
            return p if p.is_absolute() else (base / p)

        return PathsConfig(
            root=base,
            raw=r(self.raw),
            interim=r(self.interim),
            processed=r(self.processed),
            catalogues=r(self.catalogues),
            models=r(self.models),
        )

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.catalogues, self.models):
            p.mkdir(parents=True, exist_ok=True)


class InstrumentConfig(BaseModel):
    """Instrument-specific constants (cadence, bands, level-1 column names)."""

    # Native cadence in seconds for the two payloads.
    solexs_cadence_s: float = 1.0
    hel1os_cadence_s: float = 1.0
    # Common analysis cadence the two streams are resampled onto.
    analysis_cadence_s: float = 10.0

    # SoLEXS soft-X-ray light-curve columns (GOES-equivalent bands).
    solexs_flux_band: str = "flux_1_8A"       # W m^-2, drives GOES class
    solexs_hard_soft_band: str = "flux_0_5_4A"  # W m^-2, harder soft band

    # HEL1OS hard-X-ray count-rate columns.
    hel1os_low_band: str = "counts_10_30keV"   # counts s^-1
    hel1os_high_band: str = "counts_30_70keV"  # counts s^-1


class NowcastConfig(BaseModel):
    """Onset detection + GOES classification thresholds."""

    # Background estimate = rolling low percentile over this window (minutes).
    background_window_min: float = 30.0
    background_percentile: float = 10.0

    # Soft-X-ray onset: flux must exceed background * factor AND rise for
    # `min_rise_s` seconds with a positive smoothed derivative.
    soft_threshold_factor: float = 1.4
    soft_min_rise_s: float = 60.0
    soft_min_flux: float = 1.0e-8  # ignore anything below A-class noise floor

    # Hard-X-ray onset: impulsive spike detection on the count rate.
    hard_sigma: float = 4.0            # counts above running median in MAD units
    hard_min_duration_s: float = 20.0

    # Fusion: soft & hard detections are matched if their onset times fall
    # within this coincidence window (seconds).
    fusion_window_s: float = 300.0

    # GOES class flux thresholds (W m^-2) on the 1-8 A band.
    goes_classes: dict[str, float] = Field(
        default_factory=lambda: {
            "A": 1.0e-8,
            "B": 1.0e-7,
            "C": 1.0e-6,
            "M": 1.0e-5,
            "X": 1.0e-4,
        }
    )


class ForecastConfig(BaseModel):
    """Windowing + label definition + model selection for forecasting."""

    # Feature window: use the last `lookback_min` minutes to predict.
    lookback_min: float = 60.0
    # Positive label = a flare of at least `min_class` PEAKS within the next
    # `horizon_min` minutes.
    horizon_min: float = 60.0
    min_class: str = "C"
    # Sliding-window stride (minutes) when building the training matrix.
    stride_min: float = 5.0

    # Model back-end: "hist_gbm" (sklearn, always available), "lightgbm", or
    # "lstm" (PyTorch). The pipeline auto-falls back to hist_gbm if the
    # requested back-end is not installed.
    model: str = "hist_gbm"

    # Decision threshold applied to the predicted probability to raise an alert.
    alert_threshold: float = 0.5

    # Fraction of the timeline (chronological) held out for testing.
    test_fraction: float = 0.25
    # Chronological validation split (of the training portion).
    val_fraction: float = 0.15


class SyntheticConfig(BaseModel):
    """Parameters of the physics-motivated synthetic data generator."""

    n_days: float = 20.0
    cadence_s: float = 1.0
    # Mean number of flares per day (Poisson).
    flares_per_day: float = 6.0
    # Peak-flux power-law: dN/dF ~ F^-alpha between f_min and f_max (W m^-2).
    flux_power_law_alpha: float = 1.8
    flux_min: float = 3.0e-7   # ~ B3
    flux_max: float = 5.0e-4   # ~ X5
    # Fraction of flares preceded by a detectable hard-X-ray precursor.
    precursor_fraction: float = 0.45
    background_flux: float = 8.0e-9  # quiescent A-class background


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="AFC_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    random_seed: int = 42
    log_level: str = "INFO"

    paths: PathsConfig = Field(default_factory=PathsConfig)
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    nowcast: NowcastConfig = Field(default_factory=NowcastConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    def resolved(self, base: Path | None = None) -> "Settings":
        """Return a copy with all paths resolved against ``base`` (repo root)."""
        base = (base or Path.cwd()).resolve()
        new = self.model_copy(deep=True)
        new.paths = self.paths.resolve(base)
        return new


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    config_path: str | Path | None = "configs/default.yaml",
    overrides: dict[str, Any] | None = None,
    base: Path | None = None,
) -> Settings:
    """Load :class:`Settings` from YAML + env + explicit overrides.

    Parameters
    ----------
    config_path:
        Path to a YAML file. Missing file is tolerated (defaults are used).
    overrides:
        A nested dict merged last (highest precedence after env).
    base:
        Repo root used to resolve relative paths. Defaults to CWD.
    """
    data: dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)

    settings = Settings(**data)
    return settings.resolved(base)
