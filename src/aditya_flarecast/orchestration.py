"""High-level orchestration used by the CLI and scripts.

These functions compose the stage modules into the end-to-end workflows the
challenge asks for, while keeping each stage independently testable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aditya_flarecast.config import Settings
from aditya_flarecast.db.repository import CatalogueRepository
from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.forecast.predict import Forecaster
from aditya_flarecast.forecast.train import train_forecaster
from aditya_flarecast.io.readers import load_lightcurve, write_table
from aditya_flarecast.io.schemas import LightCurve
from aditya_flarecast.io.synthetic import generate as generate_synthetic
from aditya_flarecast.logging_utils import get_logger
from aditya_flarecast.nowcast.catalogue import evaluate_catalogue, run_nowcast
from aditya_flarecast.preprocessing.pipeline import preprocess

logger = get_logger(__name__)

_TRUTH_NAME = "ground_truth_catalogue.csv"


def make_synthetic(settings: Settings, seed: int | None = None) -> dict[str, Path]:
    """Generate synthetic SoLEXS/HEL1OS + ground truth and write to raw/."""
    settings.paths.ensure()
    solexs, hel1os, truth = generate_synthetic(
        settings.synthetic, seed=seed if seed is not None else settings.random_seed
    )
    p_sol = write_table(solexs, settings.paths.raw / "solexs_l1.csv")
    p_hel = write_table(hel1os, settings.paths.raw / "hel1os_l1.csv")
    p_truth = settings.paths.raw / _TRUTH_NAME
    truth.to_csv(p_truth, index=False)
    logger.info("Wrote synthetic data: %s, %s, %s", p_sol, p_hel, p_truth)
    return {"solexs": p_sol, "hel1os": p_hel, "truth": p_truth}


def load_inputs(
    settings: Settings,
    solexs_path: str | Path | None = None,
    hel1os_path: str | Path | None = None,
) -> tuple[LightCurve, LightCurve]:
    """Load SoLEXS + HEL1OS light curves (defaults to raw/ CSVs)."""
    inst = settings.instrument
    solexs_path = solexs_path or settings.paths.raw / "solexs_l1.csv"
    hel1os_path = hel1os_path or settings.paths.raw / "hel1os_l1.csv"
    sol_map = {inst.solexs_flux_band: inst.solexs_flux_band,
               inst.solexs_hard_soft_band: inst.solexs_hard_soft_band}
    hel_map = {inst.hel1os_low_band: inst.hel1os_low_band,
               inst.hel1os_high_band: inst.hel1os_high_band}
    solexs = load_lightcurve(solexs_path, "solexs", column_map=sol_map)
    hel1os = load_lightcurve(hel1os_path, "hel1os", column_map=hel_map)
    return solexs, hel1os


def build_processed(
    settings: Settings,
    solexs: LightCurve,
    hel1os: LightCurve,
    save: bool = True,
) -> pd.DataFrame:
    """Preprocess + persist the fused analysis frame."""
    df = preprocess(solexs, hel1os, settings)
    if save:
        out = settings.paths.processed / "fused.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Store attrs alongside as JSON (parquet drops attrs).
        df.to_parquet(out)
        (settings.paths.processed / "fused_attrs.json").write_text(
            json.dumps(dict(df.attrs), default=str)
        )
        logger.info("Saved processed frame -> %s", out)
    return df


def load_processed(settings: Settings) -> pd.DataFrame:
    out = settings.paths.processed / "fused.parquet"
    df = pd.read_parquet(out)
    attrs_path = settings.paths.processed / "fused_attrs.json"
    if attrs_path.exists():
        df.attrs.update(json.loads(attrs_path.read_text()))
    return df


def nowcast_stage(
    settings: Settings,
    df: pd.DataFrame,
    persist_db: bool = True,
) -> pd.DataFrame:
    """Run detection + fusion, save the catalogue (CSV + SQLite)."""
    catalogue = run_nowcast(df, settings)
    out = settings.paths.catalogues / "master_catalogue.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(out, index=False)
    logger.info("Saved master catalogue -> %s (%d rows)", out, len(catalogue))

    if persist_db:
        repo = CatalogueRepository(settings.paths.catalogues / "flarecast.db")
        repo.upsert_catalogue(catalogue)

    # If ground truth exists, report detection completeness.
    truth_path = settings.paths.raw / _TRUTH_NAME
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        report = evaluate_catalogue(catalogue, truth,
                                    tolerance_s=settings.nowcast.fusion_window_s)
        (settings.paths.catalogues / "nowcast_eval.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        logger.info(
            "Nowcast eval: recall=%.2f precision=%.2f (per-class: %s)",
            report["recall"], report["precision"],
            {k: round(v["recall"], 2) for k, v in report["per_class"].items()},
        )
    return catalogue


def _labels_catalogue(settings: Settings, nowcast_cat: pd.DataFrame) -> pd.DataFrame:
    """Prefer ground truth for labels if present; else use nowcast catalogue."""
    truth_path = settings.paths.raw / _TRUTH_NAME
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        truth = truth.rename(columns={"soft_peak_time": "peak_time"})
        return truth[["peak_time", "goes_class"]]
    # Self-supervised: use fused flare detections.
    flares = nowcast_cat[nowcast_cat["channel"].isin(["fused", "soft"])]
    return flares[["peak_time", "goes_class"]].dropna(subset=["peak_time"])


def forecast_stage(
    settings: Settings,
    df: pd.DataFrame,
    nowcast_cat: pd.DataFrame,
):
    """Build features, train the forecaster, persist artifact."""
    features = build_features(df, cadence_s=df.attrs.get("cadence_s"))
    features.to_parquet(settings.paths.processed / "features.parquet")

    labels = _labels_catalogue(settings, nowcast_cat)
    result = train_forecaster(
        features, labels, settings,
        quality=df["quality"] if "quality" in df else None,
    )
    return result, features


def run_full_pipeline(
    settings: Settings,
    use_synthetic: bool = True,
    seed: int | None = None,
) -> dict:
    """Generate/load -> preprocess -> nowcast -> forecast -> persist."""
    settings.paths.ensure()
    if use_synthetic:
        make_synthetic(settings, seed=seed)
    solexs, hel1os = load_inputs(settings)
    df = build_processed(settings, solexs, hel1os)
    catalogue = nowcast_stage(settings, df)
    result, features = forecast_stage(settings, df, catalogue)
    return {
        "n_catalogue": int(len(catalogue)),
        "artifact_dir": str(result.artifact_dir),
        "test_report": result.metadata.get("test_report", {}),
    }


def load_forecaster(settings: Settings) -> Forecaster:
    return Forecaster.load(settings.paths.models / "forecaster")
