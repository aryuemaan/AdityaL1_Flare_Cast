"""Fuse independent soft- and hard-X-ray detections into one master catalogue.

Soft (thermal) and hard (impulsive) detectors see different physical phases of
the same event. We match them by temporal coincidence: a hard burst whose peak
falls within ``fusion_window_s`` of a soft flare's rise/peak is treated as the
impulsive phase of that flare. Matching lets us:

* attach the impulsive (hard) peak time to the thermal flare, exposing the
  Neupert lead time;
* keep isolated hard bursts (candidate precursors / micro-flares) and isolated
  soft events (slow thermal flares) in the catalogue with clear provenance.
"""
from __future__ import annotations

import pandas as pd

from aditya_flarecast.config import NowcastConfig
from aditya_flarecast.nowcast.detector import FlareEvent


def fuse(
    soft_events: list[FlareEvent],
    hard_events: list[FlareEvent],
    cfg: NowcastConfig,
) -> list[FlareEvent]:
    """Return a fused, chronologically sorted list of flare events."""
    window = pd.Timedelta(seconds=cfg.fusion_window_s)
    hard_used = [False] * len(hard_events)
    fused: list[FlareEvent] = []

    for se in soft_events:
        # Candidate hard bursts near the soft flare's onset..peak interval.
        matched_idx = None
        best_dt = None
        lo = se.onset_time - window
        hi = se.peak_time + window
        for j, he in enumerate(hard_events):
            if hard_used[j]:
                continue
            if lo <= he.peak_time <= hi:
                dt = abs((he.peak_time - se.peak_time).total_seconds())
                if best_dt is None or dt < best_dt:
                    best_dt, matched_idx = dt, j

        ev = FlareEvent(
            channel="fused",
            onset_time=se.onset_time,
            peak_time=se.peak_time,
            end_time=se.end_time,
            peak_value=se.peak_value,
            goes_class=se.goes_class,
            rise_time_s=se.rise_time_s,
            duration_s=se.duration_s,
            soft_peak_time=se.soft_peak_time,
            detected_by=["soft"],
        )
        if matched_idx is not None:
            he = hard_events[matched_idx]
            hard_used[matched_idx] = True
            ev.hard_peak_time = he.peak_time
            ev.detected_by = ["soft", "hard"]
            # Neupert lead time: hard peak minus soft peak (negative => leads).
            ev.meta["neupert_lead_s"] = (
                se.peak_time - he.peak_time
            ).total_seconds()
        fused.append(ev)

    # Remaining unmatched hard bursts -> candidate precursors / micro-flares.
    for j, he in enumerate(hard_events):
        if not hard_used[j]:
            he.channel = "hard"
            he.detected_by = ["hard"]
            he.meta["candidate_precursor"] = True
            fused.append(he)

    fused.sort(key=lambda e: e.peak_time)
    return fused


def to_dataframe(events: list[FlareEvent]) -> pd.DataFrame:
    """Serialise fused events to a tidy catalogue DataFrame."""
    if not events:
        return pd.DataFrame(
            columns=[
                "channel", "onset_time", "peak_time", "end_time", "peak_value",
                "goes_class", "rise_time_s", "duration_s", "hard_peak_time",
                "soft_peak_time", "detected_by", "neupert_lead_s",
            ]
        )
    rows = []
    for e in events:
        r = e.as_record()
        r["neupert_lead_s"] = e.meta.get("neupert_lead_s")
        r["candidate_precursor"] = e.meta.get("candidate_precursor", False)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("peak_time").reset_index(drop=True)
    return df
