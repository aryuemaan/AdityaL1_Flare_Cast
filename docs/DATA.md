# Data

## Why synthetic data ships by default

Real Aditya-L1 Level-1 products (SoLEXS soft X-ray, HEL1OS hard X-ray) are
distributed through ISRO's **ISSDC PRADAN** portal, which requires an
authenticated account and offers no anonymous bulk API. So that the entire
pipeline is runnable, testable, and reproducible out of the box — with no
credentials and no network — this project ships a **physics-motivated synthetic
data generator** (`io/synthetic.py`).

The synthetic stream is not random noise. It models:

- a quiescent background with realistic quiet-Sun fluctuation,
- flares with fast exponential rise and slower decay,
- **power-law peak-flux distribution** (many small flares, few large) matching
  real flare statistics,
- **Neupert-effect hard X-rays** (hard emission ∝ d/dt of the soft rise, peaking
  *before* the soft peak),
- **pre-flare precursor micro-bursts** on a configurable fraction of events,
- GOES-equivalent classification of each flare.

It also emits a **ground-truth catalogue** (`ground_truth_catalogue.csv`) with
onset, soft-peak and hard-peak times, peak flux, GOES class, rise/decay
timescales, and precursor metadata — used to score nowcast recall/precision and
to label the forecaster.

Tune it in the `synthetic:` block of a config (days, flares/day, power-law slope,
precursor fraction, background level).

## Canonical table format

Every downstream stage consumes a simple UTC-indexed table. Columns expected:

| instrument | required columns                         | units        |
|------------|------------------------------------------|--------------|
| SoLEXS     | `flux_1_8A`, `flux_0_5_4A` (optional)    | W m⁻²        |
| HEL1OS     | `counts_10_30keV`, `counts_30_70keV`     | counts s⁻¹   |

The index is timezone-aware UTC. Band/column names are configurable in the
`instrument:` block, so you can map to whatever your files use.

## Plugging in real ISSDC data

1. **Get the data.** Log in to [PRADAN](https://pradan.issdc.gov.in/), select the
   Aditya-L1 mission, and download SoLEXS and HEL1OS **Level-1** light curves for
   your date range (FITS, sometimes CDF).

2. **Convert to the canonical format.** Use the helper, which reads FITS/CDF via
   the project readers (`pip install ".[fits]"` first):

   ```bash
   python scripts/download_issdc_data.py \
       --path /path/to/solexs_l1.fits --instrument solexs \
       --time-column TIME --map flux_1_8A=RATE_1_8 flux_0_5_4A=RATE_0_5_4 \
       --out data/raw/solexs_l1.csv

   python scripts/download_issdc_data.py \
       --path /path/to/hel1os_l1.fits --instrument hel1os \
       --time-column TIME --map counts_10_30keV=CH1 counts_30_70keV=CH2 \
       --out data/raw/hel1os_l1.csv
   ```

   (Adjust `--time-column` and `--map` to match the actual product columns;
   pass `--reference-epoch` if the time axis is "seconds since epoch".)

3. **Run the pipeline on real data** — skip synthesis with `--no-synthetic`:

   ```bash
   aditya-flarecast preprocess --config configs/default.yaml
   aditya-flarecast nowcast    --config configs/default.yaml
   aditya-flarecast train      --config configs/default.yaml
   ```

   Without a ground-truth catalogue, the forecaster automatically falls back to
   **self-supervised labels** derived from its own nowcast catalogue, so training
   still works on real data.

## Where files land

```
data/
  raw/         solexs_l1.csv, hel1os_l1.csv, ground_truth_catalogue.csv
  interim/     (scratch)
  processed/   fused.parquet, fused_attrs.json
  catalogues/  master_catalogue.csv, flarecast.db, nowcast_eval.json
models/
  forecaster/  model.pkl, metadata.json
```
