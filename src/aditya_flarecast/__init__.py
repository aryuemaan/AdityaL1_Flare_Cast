"""Aditya-FlareCast: Solar-flare nowcasting & forecasting from Aditya-L1 X-ray data.

A production-oriented pipeline that fuses SoLEXS (soft X-ray) and HEL1OS
(hard X-ray) light curves to:

  * **Nowcast** flares in real time (onset detection + GOES-class labelling)
    and build a fused master catalogue.
  * **Forecast** the probability of a flare in the next *N* minutes using
    physically-motivated features (Neupert effect, hard/soft ratio, rise
    dynamics) and a supervised time-series model.

The package is import-safe with a minimal dependency set (numpy / pandas /
scipy / scikit-learn). Heavier back-ends (LightGBM, PyTorch, FastAPI,
Streamlit) are optional and loaded lazily.
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Aditya-FlareCast contributors"

from aditya_flarecast.config import Settings, load_settings  # noqa: E402

__all__ = ["Settings", "load_settings", "__version__"]
