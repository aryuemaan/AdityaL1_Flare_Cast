"""End-to-end preprocessing: raw SoLEXS + HEL1OS -> fused analysis frame.

Output is a single tidy DataFrame on a common time grid containing, for both
instruments: raw signals, estimated backgrounds, background-subtracted
excess, and a joint quality flag. This frame is the shared substrate for both
the nowcast detectors and the forecast feature builder.
"""
from __future__ import annotations

import pandas as pd

from aditya_flarecast.config import Settings
from aditya_flarecast.io.schemas import LightCurve
from aditya_flarecast.logging_utils import get_logger
from aditya_flarecast.preprocessing.background import subtract_background
from aditya_flarecast.preprocessing.quality import quality_flags
from aditya_flarecast.preprocessing.resample import align, resample_lightcurve

logger = get_logger(__name__)


def preprocess(
    solexs: LightCurve,
    hel1os: LightCurve,
    settings: Settings,
) -> pd.DataFrame:
    """Fuse and clean the two streams into one analysis-ready frame."""
    inst = settings.instrument
    cad = inst.analysis_cadence_s

    logger.info("Resampling both instruments to %.0fs common grid", cad)
    solexs_r = resample_lightcurve(solexs, cad, agg="mean")
    hel1os_r = resample_lightcurve(hel1os, cad, agg="mean")
    solexs_r, hel1os_r = align(solexs_r, hel1os_r)

    df = pd.DataFrame(index=solexs_r.data.index)

    # Soft-X-ray channels ------------------------------------------------- #
    soft_cols = [inst.solexs_flux_band, inst.solexs_hard_soft_band]
    for col in soft_cols:
        df[f"solexs_{col}"] = solexs_r.data[col]
        excess, bg = subtract_background(
            solexs_r.data[col],
            settings.nowcast.background_window_min,
            settings.nowcast.background_percentile,
            cadence_s=cad,
        )
        df[f"solexs_{col}_bg"] = bg
        df[f"solexs_{col}_excess"] = excess

    # Hard-X-ray channels ------------------------------------------------- #
    hard_cols = [inst.hel1os_low_band, inst.hel1os_high_band]
    for col in hard_cols:
        df[f"hel1os_{col}"] = hel1os_r.data[col]
        excess, bg = subtract_background(
            hel1os_r.data[col],
            settings.nowcast.background_window_min,
            settings.nowcast.background_percentile,
            cadence_s=cad,
        )
        df[f"hel1os_{col}_bg"] = bg
        df[f"hel1os_{col}_excess"] = excess

    # Joint quality flag -------------------------------------------------- #
    base_cols = [f"solexs_{c}" for c in soft_cols] + [
        f"hel1os_{c}" for c in hard_cols
    ]
    q = quality_flags(df, columns=base_cols)
    q &= solexs_r.data.get("quality", pd.Series(True, index=df.index)).reindex(
        df.index, fill_value=False
    )
    q &= hel1os_r.data.get("quality", pd.Series(True, index=df.index)).reindex(
        df.index, fill_value=False
    )
    df["quality"] = q.astype(bool)

    df.attrs["cadence_s"] = cad
    df.attrs["solexs_flux_band"] = inst.solexs_flux_band
    df.attrs["hel1os_low_band"] = inst.hel1os_low_band
    logger.info(
        "Preprocessed frame: %d rows, %.1f%% good quality",
        len(df),
        100.0 * df["quality"].mean(),
    )
    return df
