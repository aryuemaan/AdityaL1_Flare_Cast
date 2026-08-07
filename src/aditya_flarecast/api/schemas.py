"""Pydantic schemas for the REST API."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    model_backend: str | None = None
    model_loaded: bool


class ForecastResponse(BaseModel):
    time: str
    probability: float
    alert: bool
    threshold: float
    horizon_min: float


class FlareOut(BaseModel):
    channel: str
    peak_time: str
    goes_class: str | None
    peak_value: float
    neupert_lead_s: float | None = None
    candidate_precursor: bool = False


class CatalogueResponse(BaseModel):
    count: int
    flares: list[FlareOut]


class Sample(BaseModel):
    """One time sample of combined soft + hard measurements."""

    time: str
    flux_1_8A: float
    flux_0_5_4A: float | None = None
    counts_10_30keV: float
    counts_30_70keV: float | None = None


class ForecastRequest(BaseModel):
    samples: list[Sample]
