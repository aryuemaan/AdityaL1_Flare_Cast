"""Resampling and gap handling.

SoLEXS and HEL1OS have independent cadences and occasional data gaps (e.g.
during eclipses, station handovers, or telemetry drops). To fuse them we place
both on a common regular grid and record where samples were interpolated so
downstream code can flag low-confidence regions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aditya_flarecast.io.schemas import LightCurve


def resample_lightcurve(
    lc: LightCurve,
    cadence_s: float,
    agg: str = "mean",
    max_gap_s: float | None = None,
) -> LightCurve:
    """Resample a light curve onto a regular ``cadence_s`` grid.

    Parameters
    ----------
    agg:
        Aggregation used when downsampling (``"mean"``, ``"max"``, ``"sum"``).
        Count rates should typically use ``"mean"``; flux uses ``"mean"``.
    max_gap_s:
        Interpolate gaps up to this size (seconds). Larger gaps remain NaN and
        are marked ``quality=False``. Defaults to ``5 * cadence_s``.
    """
    rule = f"{int(round(cadence_s))}s"
    df = lc.data.copy()

    resampler = df.resample(rule, label="left", closed="left")
    out = getattr(resampler, agg)()

    max_gap_s = max_gap_s if max_gap_s is not None else 5 * cadence_s
    limit = max(1, int(round(max_gap_s / cadence_s)))

    quality = out.notna().all(axis=1)
    out = out.interpolate(method="time", limit=limit, limit_direction="both")
    out = out.ffill().bfill()
    out["quality"] = quality.reindex(out.index, fill_value=False).astype(bool)

    meta = dict(lc.meta)
    meta.update({"resampled_to_s": cadence_s, "agg": agg})
    return LightCurve(lc.instrument, out, meta=meta)


def align(
    a: LightCurve,
    b: LightCurve,
) -> tuple[LightCurve, LightCurve]:
    """Restrict two light curves to their overlapping, common time grid."""
    idx = a.data.index.intersection(b.data.index)
    if len(idx) == 0:
        raise ValueError(
            "SoLEXS and HEL1OS light curves do not overlap in time; "
            "cannot fuse."
        )
    return (
        LightCurve(a.instrument, a.data.loc[idx], meta=a.meta),
        LightCurve(b.instrument, b.data.loc[idx], meta=b.meta),
    )
