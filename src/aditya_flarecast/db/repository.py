"""Repository: a thin persistence API over the ORM models.

Hides SQLAlchemy session handling from callers and provides idempotent
catalogue upserts and simple queries used by the API and dashboard.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from aditya_flarecast.db.models import Base, FlareRecord, ForecastAlert
from aditya_flarecast.logging_utils import get_logger

logger = get_logger(__name__)


class CatalogueRepository:
    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self.db_path = db_path

    # -- Catalogue -------------------------------------------------------- #
    def upsert_catalogue(self, catalogue: pd.DataFrame) -> int:
        """Insert catalogue rows, skipping duplicates on (peak_time, channel)."""
        if catalogue.empty:
            return 0
        from aditya_flarecast.timeutils import to_ns_scalar

        inserted = 0
        with Session(self.engine) as session:
            # Normalise keys to (ns_int, channel) so naive (SQLite) and tz-aware
            # timestamps compare consistently.
            existing = {
                (to_ns_scalar(r.peak_time), r.channel)
                for r in session.execute(
                    select(FlareRecord.peak_time, FlareRecord.channel)
                ).all()
            }
            for _, row in catalogue.iterrows():
                pk = pd.to_datetime(row["peak_time"], utc=True).to_pydatetime()
                key = (to_ns_scalar(pk), row["channel"])
                if key in existing:
                    continue
                rec = FlareRecord(
                    channel=row["channel"],
                    onset_time=pd.to_datetime(row["onset_time"], utc=True).to_pydatetime(),
                    peak_time=pk,
                    end_time=pd.to_datetime(row["end_time"], utc=True).to_pydatetime(),
                    peak_value=float(row["peak_value"]),
                    goes_class=row.get("goes_class"),
                    rise_time_s=float(row.get("rise_time_s", 0.0) or 0.0),
                    duration_s=float(row.get("duration_s", 0.0) or 0.0),
                    hard_peak_time=_maybe_dt(row.get("hard_peak_time")),
                    soft_peak_time=_maybe_dt(row.get("soft_peak_time")),
                    neupert_lead_s=_maybe_float(row.get("neupert_lead_s")),
                    candidate_precursor=bool(row.get("candidate_precursor", False)),
                    detected_by=str(row.get("detected_by", "")),
                )
                session.add(rec)
                existing.add(key)  # prevent in-batch duplicates too
                inserted += 1
            session.commit()
        logger.info("Upserted %d new flare records into %s", inserted, self.db_path)
        return inserted

    def query_flares(
        self,
        goes_min: str | None = None,
        channel: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        with Session(self.engine) as session:
            stmt = select(FlareRecord).order_by(FlareRecord.peak_time.desc())
            if channel:
                stmt = stmt.where(FlareRecord.channel == channel)
            stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()
        df = pd.DataFrame([_row_to_dict(r) for r in rows])
        if goes_min and not df.empty and "goes_class" in df:
            from aditya_flarecast.nowcast.classifier import class_rank

            df = df[df["goes_class"].fillna("sub-A").map(
                lambda c: class_rank(str(c)) >= class_rank(goes_min))]
        return df

    # -- Alerts ----------------------------------------------------------- #
    def log_alert(
        self,
        decision_time,
        probability: float,
        alert: bool,
        threshold: float,
        horizon_min: float,
        model_backend: str = "",
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ForecastAlert(
                    decision_time=pd.to_datetime(decision_time, utc=True).to_pydatetime(),
                    probability=float(probability),
                    alert=bool(alert),
                    threshold=float(threshold),
                    horizon_min=float(horizon_min),
                    model_backend=model_backend,
                )
            )
            session.commit()

    def recent_alerts(self, limit: int = 200) -> pd.DataFrame:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ForecastAlert)
                .order_by(ForecastAlert.decision_time.desc())
                .limit(limit)
            ).scalars().all()
        return pd.DataFrame(
            [
                {
                    "decision_time": r.decision_time,
                    "probability": r.probability,
                    "alert": r.alert,
                    "threshold": r.threshold,
                    "horizon_min": r.horizon_min,
                }
                for r in rows
            ]
        )


def _maybe_dt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    return pd.to_datetime(v, utc=True).to_pydatetime()


def _maybe_float(v):
    if v is None or pd.isna(v):
        return None
    return float(v)


def _row_to_dict(r: FlareRecord) -> dict:
    return {
        "id": r.id,
        "channel": r.channel,
        "onset_time": r.onset_time,
        "peak_time": r.peak_time,
        "end_time": r.end_time,
        "peak_value": r.peak_value,
        "goes_class": r.goes_class,
        "rise_time_s": r.rise_time_s,
        "duration_s": r.duration_s,
        "hard_peak_time": r.hard_peak_time,
        "soft_peak_time": r.soft_peak_time,
        "neupert_lead_s": r.neupert_lead_s,
        "candidate_precursor": r.candidate_precursor,
        "detected_by": r.detected_by,
    }
