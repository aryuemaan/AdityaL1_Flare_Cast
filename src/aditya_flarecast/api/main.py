"""FastAPI service for real-time nowcast/forecast serving.

Endpoints
---------
* ``GET  /health``            — liveness + whether a model is loaded.
* ``GET  /forecast/latest``   — latest forecast from the processed frame.
* ``POST /forecast``          — forecast from a posted batch of samples.
* ``GET  /catalogue``         — query the nowcast master catalogue.
* ``GET  /alerts``            — recent logged forecast alerts.

Run with::

    aditya-flarecast serve-api
    # or
    uvicorn aditya_flarecast.api.main:app --reload
"""
from __future__ import annotations

import os

import pandas as pd

from aditya_flarecast import __version__
from aditya_flarecast.api.schemas import (
    CatalogueResponse,
    FlareOut,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from aditya_flarecast.config import load_settings
from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.io.schemas import LightCurve
from aditya_flarecast.logging_utils import get_logger
from aditya_flarecast.preprocessing.pipeline import preprocess

try:
    from fastapi import FastAPI, HTTPException, Query
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The API requires FastAPI. Install serving extras: pip install '.[serve]'"
    ) from exc

logger = get_logger("api")

app = FastAPI(
    title="Aditya-FlareCast API",
    version=__version__,
    description="Solar-flare nowcasting & forecasting from Aditya-L1 X-ray data.",
)


class _State:
    settings = None
    forecaster = None
    repo = None


state = _State()


@app.on_event("startup")
def _startup() -> None:
    config = os.environ.get("AFC_CONFIG", "configs/default.yaml")
    state.settings = load_settings(config)
    # Model / DB are optional at startup; endpoints report if missing.
    try:
        from aditya_flarecast.orchestration import load_forecaster

        state.forecaster = load_forecaster(state.settings)
    except Exception as exc:  # pragma: no cover
        logger.warning("No trained forecaster found yet: %s", exc)
    try:
        from aditya_flarecast.db.repository import CatalogueRepository

        state.repo = CatalogueRepository(
            state.settings.paths.catalogues / "flarecast.db"
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Catalogue DB unavailable: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    fc = state.forecaster
    return HealthResponse(
        status="ok",
        version=__version__,
        model_backend=fc.model.name if fc else None,
        model_loaded=fc is not None,
    )


@app.get("/forecast/latest", response_model=ForecastResponse)
def forecast_latest() -> ForecastResponse:
    if state.forecaster is None:
        raise HTTPException(503, "No trained forecaster loaded. Train first.")
    from aditya_flarecast.orchestration import load_processed

    df = load_processed(state.settings)
    feats = build_features(df, cadence_s=df.attrs.get("cadence_s"))
    out = state.forecaster.predict_latest(feats)
    if state.repo is not None:
        state.repo.log_alert(
            out["time"], out["probability"], out["alert"],
            out["threshold"], out["horizon_min"], state.forecaster.model.name,
        )
    return ForecastResponse(**out)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if state.forecaster is None:
        raise HTTPException(503, "No trained forecaster loaded. Train first.")
    if not req.samples:
        raise HTTPException(400, "No samples provided.")

    rows = [s.model_dump() for s in req.samples]
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")

    inst = state.settings.instrument
    sol_cols = [inst.solexs_flux_band, inst.solexs_hard_soft_band]
    hel_cols = [inst.hel1os_low_band, inst.hel1os_high_band]
    for c in sol_cols + hel_cols:
        if c not in df.columns:
            df[c] = df[df.columns[0]] * 0.0  # tolerate missing optional bands

    solexs = LightCurve("solexs", df[sol_cols].copy())
    hel1os = LightCurve("hel1os", df[hel_cols].copy())
    proc = preprocess(solexs, hel1os, state.settings)
    feats = build_features(proc, cadence_s=proc.attrs.get("cadence_s"))
    out = state.forecaster.predict_latest(feats)
    return ForecastResponse(**out)


@app.get("/catalogue", response_model=CatalogueResponse)
def catalogue(
    goes_min: str | None = Query(None, description="Minimum GOES class, e.g. C"),
    channel: str | None = Query(None, description="soft|hard|fused"),
    limit: int = Query(200, le=2000),
) -> CatalogueResponse:
    if state.repo is None:
        raise HTTPException(503, "Catalogue DB not available.")
    df = state.repo.query_flares(goes_min=goes_min, channel=channel, limit=limit)
    flares = [
        FlareOut(
            channel=r["channel"],
            peak_time=str(r["peak_time"]),
            goes_class=r.get("goes_class"),
            peak_value=float(r["peak_value"]),
            neupert_lead_s=(
                None if pd.isna(r.get("neupert_lead_s")) else float(r["neupert_lead_s"])
            ),
            candidate_precursor=bool(r.get("candidate_precursor", False)),
        )
        for _, r in df.iterrows()
    ] if not df.empty else []
    return CatalogueResponse(count=len(flares), flares=flares)


@app.get("/alerts")
def alerts(limit: int = Query(100, le=1000)) -> dict:
    if state.repo is None:
        raise HTTPException(503, "Alert log not available.")
    df = state.repo.recent_alerts(limit=limit)
    return {"count": len(df), "alerts": df.to_dict(orient="records")}
