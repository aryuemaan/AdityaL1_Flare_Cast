"""Resolution-safe conversions between datetimes and int64 nanoseconds.

pandas >= 3.0 can use microsecond-resolution datetimes and removed
``Series.view``, so ad-hoc ``.view("int64")`` conversions are fragile. These
helpers normalise any datetime-like input to naive-UTC ``datetime64[ns]`` and
return int64 nanoseconds, guaranteeing consistent arithmetic regardless of the
input resolution or timezone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_ns_array(obj) -> np.ndarray:
    """Return int64 nanoseconds-since-epoch for a datetime-like array/Series/Index."""
    dt = pd.to_datetime(obj, utc=True)
    idx = pd.DatetimeIndex(np.atleast_1d(dt))
    idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.values.astype("datetime64[ns]").astype("int64")


def to_ns_scalar(ts) -> int:
    """Return int64 nanoseconds-since-epoch for a single timestamp."""
    return int(to_ns_array([ts])[0])


def seconds_to_ns(seconds: float) -> int:
    return int(round(seconds * 1_000_000_000))


def minutes_to_ns(minutes: float) -> int:
    return int(round(minutes * 60 * 1_000_000_000))
