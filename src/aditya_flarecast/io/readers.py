"""Readers that turn on-disk products into :class:`LightCurve` objects.

A single :func:`load_lightcurve` entry point dispatches on file extension:

* ``.csv`` / ``.parquet`` — the canonical interchange format this pipeline
  writes and reads (used for the synthetic data and for cached real data).
* ``.fits`` — SoLEXS/HEL1OS Level-1 FITS light curves (needs ``astropy``).
* ``.cdf``  — CDF products (needs ``cdflib``).

The FITS/CDF paths are written defensively: because the exact ISSDC Level-1
column layout can evolve, column names are resolved via a configurable
mapping and the reader raises a clear, actionable error if a required column
is missing. If ``astropy``/``cdflib`` are not installed the reader raises an
``ImportError`` telling the user which extra to install — the rest of the
pipeline still works on CSV/parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from aditya_flarecast.io.schemas import LightCurve
from aditya_flarecast.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Canonical CSV / parquet
# --------------------------------------------------------------------------- #


def read_table(path: str | Path, instrument: str, time_col: str = "time") -> LightCurve:
    """Read a canonical CSV/parquet light curve written by this pipeline."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if time_col not in df.columns:
        # Assume the first column is the timestamp.
        time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col)
    return LightCurve(instrument, df, meta={"source": str(path)})


def write_table(lc: LightCurve, path: str | Path) -> Path:
    """Persist a :class:`LightCurve` to CSV or parquet (by extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = lc.data.copy()
    df.index.name = "time"
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)
    return path


# --------------------------------------------------------------------------- #
# FITS (SoLEXS / HEL1OS Level-1)
# --------------------------------------------------------------------------- #


def read_fits_lightcurve(
    path: str | Path,
    instrument: str,
    column_map: dict[str, str],
    time_column: str = "TIME",
    reference_epoch: str | None = None,
    hdu: int = 1,
) -> LightCurve:
    """Read a Level-1 FITS light curve.

    Parameters
    ----------
    column_map:
        Mapping of ``{output_name: fits_column_name}``. Only listed columns
        are extracted.
    time_column:
        FITS column holding the time axis. If it is a floating "seconds since
        epoch" axis, pass ``reference_epoch`` (ISO string) to convert.
    """
    try:
        from astropy.io import fits  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Reading FITS requires astropy. Install with: pip install astropy"
        ) from exc

    path = Path(path)
    with fits.open(path) as hdul:
        table = hdul[hdu].data
        cols = {c.upper() for c in table.columns.names}

        if time_column.upper() not in cols:
            raise KeyError(
                f"Time column {time_column!r} not found in {path.name}. "
                f"Available: {sorted(cols)}"
            )
        raw_time = table[time_column]
        if reference_epoch is not None:
            times = pd.Timestamp(reference_epoch, tz="UTC") + pd.to_timedelta(
                raw_time, unit="s"
            )
        else:
            times = pd.to_datetime(raw_time, utc=True)

        data = {}
        for out_name, fits_name in column_map.items():
            if fits_name.upper() not in cols:
                raise KeyError(
                    f"Column {fits_name!r} (for {out_name!r}) missing in "
                    f"{path.name}. Available: {sorted(cols)}"
                )
            data[out_name] = table[fits_name].astype(float)

    df = pd.DataFrame(data, index=pd.DatetimeIndex(times))
    logger.info("Read %d rows from FITS %s", len(df), path.name)
    return LightCurve(instrument, df, meta={"source": str(path), "level": "L1"})


# --------------------------------------------------------------------------- #
# CDF
# --------------------------------------------------------------------------- #


def read_cdf_lightcurve(
    path: str | Path,
    instrument: str,
    column_map: dict[str, str],
    time_var: str = "Time",
) -> LightCurve:
    """Read a Level-1 CDF light curve (requires ``cdflib``)."""
    try:
        import cdflib  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Reading CDF requires cdflib. Install with: pip install cdflib"
        ) from exc

    path = Path(path)
    cdf = cdflib.CDF(str(path))
    epoch = cdf.varget(time_var)
    times = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch), utc=True)
    data = {out: cdf.varget(var).astype(float) for out, var in column_map.items()}
    df = pd.DataFrame(data, index=pd.DatetimeIndex(times))
    logger.info("Read %d rows from CDF %s", len(df), path.name)
    return LightCurve(instrument, df, meta={"source": str(path), "level": "L1"})


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def load_lightcurve(
    path: str | Path,
    instrument: str,
    column_map: dict[str, str] | None = None,
    **kwargs,
) -> LightCurve:
    """Dispatch to the right reader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".parquet"}:
        return read_table(path, instrument)
    if suffix in {".fits", ".fit", ".fts"}:
        if column_map is None:
            raise ValueError("column_map is required for FITS light curves")
        return read_fits_lightcurve(path, instrument, column_map, **kwargs)
    if suffix == ".cdf":
        if column_map is None:
            raise ValueError("column_map is required for CDF light curves")
        return read_cdf_lightcurve(path, instrument, column_map, **kwargs)
    raise ValueError(f"Unsupported light-curve format: {suffix!r}")
