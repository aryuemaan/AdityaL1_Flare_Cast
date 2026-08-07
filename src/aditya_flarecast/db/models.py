"""SQLAlchemy ORM models for the nowcast catalogue and forecast alerts.

A lightweight SQLite database is the durable "automated database of nowcasted
solar flares" required by the challenge, plus a log of forecast alerts for
audit and post-hoc verification.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FlareRecord(Base):
    """One nowcasted flare / impulsive burst in the master catalogue."""

    __tablename__ = "flares"
    __table_args__ = (UniqueConstraint("peak_time", "channel", name="uq_peak_channel"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    onset_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    peak_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    peak_value: Mapped[float] = mapped_column(Float)
    goes_class: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    rise_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    hard_peak_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    soft_peak_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    neupert_lead_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_precursor: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )


class ForecastAlert(Base):
    """A logged forecast probability / alert at a decision time."""

    __tablename__ = "forecast_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    probability: Mapped[float] = mapped_column(Float)
    alert: Mapped[bool] = mapped_column(Boolean, index=True)
    threshold: Mapped[float] = mapped_column(Float)
    horizon_min: Mapped[float] = mapped_column(Float)
    model_backend: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
