"""Tests for synthetic generation, schemas, and table readers."""
from __future__ import annotations

import pandas as pd

from aditya_flarecast.io.readers import read_table, write_table
from aditya_flarecast.io.schemas import LightCurve, goes_class_geq


def test_synthetic_shapes(synthetic):
    solexs, hel1os, truth = synthetic
    assert isinstance(solexs, LightCurve)
    assert isinstance(hel1os, LightCurve)
    assert "flux_1_8A" in solexs.columns
    assert "counts_10_30keV" in hel1os.columns
    assert len(solexs) == len(hel1os)
    assert len(truth) > 0
    # Ground truth carries the physics we rely on downstream.
    for col in ("soft_peak_time", "hard_peak_time", "peak_flux_1_8A", "goes_class"):
        assert col in truth.columns


def test_neupert_lead_present(synthetic):
    # Hard peak should, on average, lead the soft peak (Neupert effect).
    _, _, truth = synthetic
    lead = (
        pd.to_datetime(truth["soft_peak_time"])
        - pd.to_datetime(truth["hard_peak_time"])
    ).dt.total_seconds()
    assert lead.mean() >= 0  # soft peaks at or after hard peak on average


def test_lightcurve_is_utc_sorted(synthetic):
    solexs, _, _ = synthetic
    assert solexs.data.index.tz is not None
    assert solexs.data.index.is_monotonic_increasing


def test_table_roundtrip(tmp_path, synthetic):
    solexs, _, _ = synthetic
    p = write_table(solexs, tmp_path / "solexs.csv")
    lc2 = read_table(p, "solexs")
    assert lc2.columns == solexs.columns
    assert len(lc2) == len(solexs)


def test_goes_ordering():
    assert goes_class_geq("M2.0", "C1.0")
    assert not goes_class_geq("B5.0", "C1.0")
