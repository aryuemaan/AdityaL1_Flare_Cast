#!/usr/bin/env python
"""Helper + guide for ingesting real Aditya-L1 Level-1 data from ISSDC PRADAN.

Aditya-L1 SoLEXS and HEL1OS Level-1 products are distributed through ISRO's
ISSDC **PRADAN** portal (https://pradan.issdc.gov.in/) and require an
authenticated account; there is no anonymous bulk API, so this repository does
**not** attempt to scrape them. Instead:

1. Register / log in at PRADAN and select the Aditya-L1 mission.
2. Download the SoLEXS and HEL1OS **Level-1** light-curve / spectrum products
   for your date range (typically FITS, sometimes CDF).
3. Place the files under ``data/raw/`` (or anywhere) and convert them to the
   pipeline's canonical CSV/parquet with :func:`convert`, or point the readers
   at the FITS/CDF directly (see ``docs/DATA.md``).

Once converted, run the pipeline on the real data with, e.g.::

    aditya-flarecast preprocess --config configs/real.yaml
    aditya-flarecast nowcast    --config configs/real.yaml
    aditya-flarecast train      --config configs/real.yaml

This script's :func:`convert` reads a FITS/CDF light curve using the project
readers and writes the canonical table the rest of the pipeline expects.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def convert(path: str, instrument: str, out: str,
            time_column: str, columns: dict[str, str],
            reference_epoch: str | None) -> None:
    from aditya_flarecast.io.readers import load_lightcurve, write_table

    kwargs = {"time_column": time_column}
    if reference_epoch:
        kwargs["reference_epoch"] = reference_epoch
    lc = load_lightcurve(path, instrument, column_map=columns, **kwargs)
    p = write_table(lc, out)
    print(f"Converted {path} -> {p} ({len(lc)} rows, cadence ~{lc.cadence_s:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="Input FITS/CDF light curve.")
    ap.add_argument("--instrument", required=True, choices=["solexs", "hel1os"])
    ap.add_argument("--out", required=True, help="Output CSV/parquet path.")
    ap.add_argument("--time-column", default="TIME",
                    help="Name of the time column/variable in the product.")
    ap.add_argument("--reference-epoch", default=None,
                    help="ISO epoch if the time axis is 'seconds since epoch'.")
    ap.add_argument("--map", nargs="+", metavar="OUT=SRC", required=True,
                    help="Column mapping, e.g. flux_1_8A=RATE1 flux_0_5_4A=RATE2")
    args = ap.parse_args()

    columns = dict(kv.split("=", 1) for kv in args.map)
    if not Path(args.path).exists():
        raise SystemExit(f"Input not found: {args.path}")
    convert(args.path, args.instrument, args.out,
            args.time_column, columns, args.reference_epoch)


if __name__ == "__main__":
    main()
