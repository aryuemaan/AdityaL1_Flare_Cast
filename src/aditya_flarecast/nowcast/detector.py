"""Independent flare detection in the soft and hard X-ray channels.

Two physically distinct detectors run on the preprocessed frame:

* :func:`detect_soft` — the classic GOES-style thermal-flare detector. A flare
  is an interval where the soft flux rises significantly above its slowly
  varying background and keeps rising for a minimum time. Onset, peak time,
  peak flux and end are recorded, and a GOES class is assigned.

* :func:`detect_hard` — an impulsive-phase detector on the hard-X-ray count
  rate. It flags excursions several MAD above a running median that persist for
  a minimum duration — capturing the non-thermal impulsive bursts (and
  micro-flare precursors) that leave little soft-X-ray signature.

Each returns a list of :class:`FlareEvent`. The two lists are later fused into
a single master catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from aditya_flarecast.config import NowcastConfig
from aditya_flarecast.nowcast.classifier import classify_peak_flux


@dataclass
class FlareEvent:
    """A detected flare (or impulsive burst) in one or both channels."""

    channel: str                      # "soft", "hard", or "fused"
    onset_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_value: float                 # flux (W/m^2) for soft, counts/s for hard
    goes_class: Optional[str] = None  # only for soft/fused
    rise_time_s: float = 0.0
    duration_s: float = 0.0
    hard_peak_time: Optional[pd.Timestamp] = None
    soft_peak_time: Optional[pd.Timestamp] = None
    detected_by: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "channel": self.channel,
            "onset_time": self.onset_time,
            "peak_time": self.peak_time,
            "end_time": self.end_time,
            "peak_value": self.peak_value,
            "goes_class": self.goes_class,
            "rise_time_s": self.rise_time_s,
            "duration_s": self.duration_s,
            "hard_peak_time": self.hard_peak_time,
            "soft_peak_time": self.soft_peak_time,
            "detected_by": ",".join(self.detected_by),
        }


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of contiguous True runs."""
    if not mask.any():
        return []
    idx = np.flatnonzero(
        np.diff(np.concatenate(([0], mask.view(np.int8), [0]))) != 0
    )
    return list(zip(idx[0::2], idx[1::2]))


def detect_soft(df: pd.DataFrame, cfg: NowcastConfig, cadence_s: float) -> list[FlareEvent]:
    """Detect thermal flares in the SoLEXS soft-X-ray band."""
    band = df.attrs.get("solexs_flux_band", "flux_1_8A")
    flux = df[f"solexs_{band}"].astype(float)
    bg = df[f"solexs_{band}_bg"].astype(float)

    # Smoothed derivative to test "still rising".
    sm = max(1, int(round(60.0 / cadence_s)))
    smooth = flux.rolling(sm, center=True, min_periods=1).mean()
    dflux = smooth.diff().fillna(0.0)

    above = (flux > bg * cfg.soft_threshold_factor) & (flux > cfg.soft_min_flux)
    events: list[FlareEvent] = []
    ts = flux.index

    for s, e in _contiguous_runs(above.to_numpy()):
        seg = slice(s, e)
        # Require a sustained rise somewhere in the run.
        rising_len = int((dflux.iloc[seg] > 0).sum()) * cadence_s
        if rising_len < cfg.soft_min_rise_s:
            continue
        seg_flux = flux.iloc[seg]
        pk_pos = int(seg_flux.values.argmax())
        peak_i = s + pk_pos
        peak_val = float(seg_flux.iloc[pk_pos])
        onset_i = s
        events.append(
            FlareEvent(
                channel="soft",
                onset_time=ts[onset_i],
                peak_time=ts[peak_i],
                end_time=ts[e - 1],
                peak_value=peak_val,
                soft_peak_time=ts[peak_i],
                goes_class=classify_peak_flux(peak_val),
                rise_time_s=float((peak_i - onset_i) * cadence_s),
                duration_s=float((e - 1 - s) * cadence_s),
                detected_by=["soft"],
            )
        )
    return events


def detect_hard(df: pd.DataFrame, cfg: NowcastConfig, cadence_s: float) -> list[FlareEvent]:
    """Detect impulsive bursts in the HEL1OS hard-X-ray band."""
    low = df.attrs.get("hel1os_low_band", "counts_10_30keV")
    counts = df[f"hel1os_{low}"].astype(float)

    win = max(5, int(round(cfg.background_window_min * 60.0 / cadence_s)))
    med = counts.rolling(win, center=True, min_periods=win // 4).median()
    mad = (counts - med).abs().rolling(win, center=True, min_periods=win // 4).median()
    scaled = 1.4826 * mad + 1e-9
    sigma = (counts - med) / scaled

    above = (sigma > cfg.hard_sigma).to_numpy()
    events: list[FlareEvent] = []
    ts = counts.index
    min_len = max(1, int(round(cfg.hard_min_duration_s / cadence_s)))

    for s, e in _contiguous_runs(above):
        if (e - s) < min_len:
            continue
        seg = counts.iloc[s:e]
        pk_pos = int(seg.values.argmax())
        peak_i = s + pk_pos
        events.append(
            FlareEvent(
                channel="hard",
                onset_time=ts[s],
                peak_time=ts[peak_i],
                end_time=ts[e - 1],
                peak_value=float(seg.iloc[pk_pos]),
                hard_peak_time=ts[peak_i],
                rise_time_s=float((peak_i - s) * cadence_s),
                duration_s=float((e - 1 - s) * cadence_s),
                detected_by=["hard"],
                meta={"peak_sigma": float(sigma.iloc[peak_i])},
            )
        )
    return events
