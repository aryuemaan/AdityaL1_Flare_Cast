"""Shared pytest fixtures: a small synthetic dataset and settings."""
from __future__ import annotations

import pytest

from aditya_flarecast.config import load_settings
from aditya_flarecast.io.synthetic import generate as generate_synthetic


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    base = tmp_path_factory.mktemp("afc")
    s = load_settings(
        None,
        overrides={
            "random_seed": 3,
            "instrument": {"analysis_cadence_s": 30.0},
            "synthetic": {"n_days": 3.0, "flares_per_day": 12.0},
            "forecast": {"min_class": "C", "stride_min": 5.0},
        },
        base=base,
    )
    s.paths.ensure()
    return s


@pytest.fixture(scope="session")
def synthetic(settings):
    solexs, hel1os, truth = generate_synthetic(settings.synthetic, seed=3)
    return solexs, hel1os, truth
