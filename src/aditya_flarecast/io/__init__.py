"""I/O layer: data schemas, synthetic generator, and product readers."""
from aditya_flarecast.io.readers import (
    load_lightcurve,
    read_cdf_lightcurve,
    read_fits_lightcurve,
    read_table,
    write_table,
)
from aditya_flarecast.io.schemas import GOES_ORDER, LightCurve, goes_class_geq
from aditya_flarecast.io.synthetic import generate as generate_synthetic

__all__ = [
    "LightCurve",
    "GOES_ORDER",
    "goes_class_geq",
    "load_lightcurve",
    "read_table",
    "write_table",
    "read_fits_lightcurve",
    "read_cdf_lightcurve",
    "generate_synthetic",
]
