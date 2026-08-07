# Architecture

Aditya-FlareCast is organised as a set of composable stages, each a self-contained
subpackage under `src/aditya_flarecast/`. Data flows in one direction — from raw
light curves to a persisted catalogue and a trained forecaster — so any stage can
be run, tested, or swapped independently.

```
                 ┌──────────────┐
  raw L1 data →  │ io           │  readers (CSV / parquet / FITS / CDF)
  (or synthetic) │              │  + physics-based synthetic generator
                 └──────┬───────┘
                        │  LightCurve (UTC-indexed)
                 ┌──────▼───────┐
                 │ preprocessing│  background subtraction · resampling to a
                 │              │  common grid · quality flags · soft/hard fusion
                 └──────┬───────┘
                        │  fused analysis frame (parquet)
              ┌─────────┴──────────┐
      ┌───────▼───────┐    ┌───────▼────────┐
      │ features      │    │ nowcast        │  soft + hard detectors →
      │ (causal only) │    │                │  GOES classification → fusion
      └───────┬───────┘    └───────┬────────┘
              │                    │  master catalogue (CSV + SQLite)
      ┌───────▼───────┐    ┌───────▼────────┐
      │ forecast      │    │ db             │  idempotent upsert · alert log
      │ dataset/train │    └────────────────┘
      │ /models/eval  │
      └───────┬───────┘
              │  model.pkl + metadata.json
      ┌───────▼──────────────────────────────┐
      │ serving:  cli · api (FastAPI) ·       │
      │           dashboard (Streamlit) ·     │
      │           streaming (replay)          │
      └───────────────────────────────────────┘
```

## Design principles

**Everything is configured, nothing is magic.** All thresholds, cadences, band
names, and model choices live in a single Pydantic `Settings` tree (`config.py`),
loadable from YAML and overridable with `AFC_`-prefixed environment variables. No
tuning constant is hidden in code.

**Causal features only.** Every feature in `features/engineering.py` is computed
from a trailing window ending at the decision time — there is no look-ahead. This
is what makes the forecast operationally honest: at inference time the model only
ever sees what a live pipeline would have.

**Chronological splits.** Train/validation/test are split by time, never shuffled
(`forecast/dataset.py`). A model is trained on the past and evaluated on the
future, which is the only split that reflects real forecasting skill and avoids
temporal leakage.

**Optional dependencies degrade gracefully.** LightGBM, PyTorch, FastAPI,
Streamlit, Astropy, and cdflib are all optional. The core pipeline runs on
NumPy/pandas/SciPy/scikit-learn alone; `forecast/models.py` falls back to the
always-available HistGradientBoosting back-end if a requested one is missing.

## The physics that drives the forecast

The forecast's lead time comes from two well-established solar-flare phenomena,
both modelled in the synthetic generator and exploited in the features:

- **The Neupert effect** — hard X-ray (HEL1OS) emission from non-thermal
  electrons tends to *precede and drive* the soft X-ray (SoLEXS) rise, so the
  hard-channel time derivative and the hard/soft correlation are early indicators.
- **Pre-flare precursors** — small hard-X-ray micro-bursts that occur minutes
  before the main impulsive phase of many flares. The nowcast keeps unmatched hard
  bursts as `candidate_precursor` rows, and the forecaster learns their predictive
  value via the "minutes since hard burst" feature.

Because a fraction of flares have no precursor and rise abruptly, perfect
forecasting is physically impossible — the skill scores reflect a genuine,
not a synthetic, ceiling.

## Extending

- **New model back-end:** subclass `BaseForecastModel` in `forecast/models.py`,
  register it in `build_model()`.
- **Real instrument reader:** add a branch in `io/readers.py::load_lightcurve`
  and a column map; the rest of the pipeline is agnostic to data origin.
- **New feature:** add it to `features/engineering.py::build_features` (keep it
  causal) — it flows automatically into training and serving.
