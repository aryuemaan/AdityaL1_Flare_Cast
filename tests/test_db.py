"""Tests for catalogue persistence."""
from __future__ import annotations

from aditya_flarecast.db.repository import CatalogueRepository
from aditya_flarecast.nowcast.catalogue import run_nowcast
from aditya_flarecast.preprocessing.pipeline import preprocess


def test_catalogue_upsert_idempotent(settings, synthetic, tmp_path):
    solexs, hel1os, _ = synthetic
    df = preprocess(solexs, hel1os, settings)
    cat = run_nowcast(df, settings)

    repo = CatalogueRepository(tmp_path / "test.db")
    n1 = repo.upsert_catalogue(cat)
    n2 = repo.upsert_catalogue(cat)  # second time inserts nothing new
    assert n1 > 0
    assert n2 == 0

    out = repo.query_flares(limit=1000)
    assert len(out) == n1


def test_alert_log(tmp_path):
    repo = CatalogueRepository(tmp_path / "test2.db")
    repo.log_alert("2024-01-01T00:00:00Z", 0.8, True, 0.5, 60.0, "hist_gbm")
    df = repo.recent_alerts()
    assert len(df) == 1
    assert bool(df.iloc[0]["alert"]) is True
