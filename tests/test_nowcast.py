"""Tests for detectors, GOES classification, fusion, and catalogue eval."""
from __future__ import annotations

import pandas as pd

from aditya_flarecast.nowcast.catalogue import evaluate_catalogue, run_nowcast
from aditya_flarecast.nowcast.classifier import classify_peak_flux, meets_min_class
from aditya_flarecast.preprocessing.pipeline import preprocess


def test_goes_classification():
    assert classify_peak_flux(1.0e-4).startswith("X")
    assert classify_peak_flux(3.2e-5).startswith("M")
    assert classify_peak_flux(1.4e-6).startswith("C")
    assert classify_peak_flux(5e-8) in ("A5.0", "sub-A")
    assert meets_min_class("M1.0", "C")
    assert not meets_min_class("B2.0", "C")


def test_nowcast_detects_flares(settings, synthetic):
    solexs, hel1os, truth = synthetic
    df = preprocess(solexs, hel1os, settings)
    cat = run_nowcast(df, settings)
    assert not cat.empty
    # There should be fused (soft) flares and some hard bursts.
    assert (cat["channel"] == "fused").sum() > 0
    assert set(cat["channel"].unique()).issubset({"fused", "hard", "soft"})


def test_nowcast_recall_reasonable(settings, synthetic):
    solexs, hel1os, truth = synthetic
    df = preprocess(solexs, hel1os, settings)
    cat = run_nowcast(df, settings)
    report = evaluate_catalogue(cat, truth, tolerance_s=300.0)
    # On synthetic data with strong flares, recall should be high.
    assert report["recall"] > 0.7
    assert report["precision"] > 0.7


def test_neupert_lead_recorded(settings, synthetic):
    solexs, hel1os, _ = synthetic
    df = preprocess(solexs, hel1os, settings)
    cat = run_nowcast(df, settings)
    fused = cat[cat["channel"] == "fused"]
    # At least some fused flares should carry a Neupert lead measurement.
    assert fused["neupert_lead_s"].notna().sum() > 0
