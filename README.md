<div align="center">

# ☀️ Aditya-FlareCast

**Nowcasting & forecasting of solar flares from combined soft + hard X-ray data (Aditya-L1: SoLEXS + HEL1OS)**

Detect and classify flares in real time, build an automated flare catalogue, and
forecast the probability of an upcoming flare **with quantifiable lead time** —
served through a CLI, a REST API, and a live dashboard.

</div>

---

## Table of contents

1. [What this does](#what-this-does)
2. [Why it works — the science](#why-it-works--the-science)
3. [Quick start (2 minutes)](#quick-start-2-minutes)
4. [Installation](#installation)
5. [Running the pipeline](#running-the-pipeline)
6. [The interfaces: API & dashboard](#the-interfaces-api--dashboard)
7. [Using real Aditya-L1 data](#using-real-aditya-l1-data)
8. [Configuration](#configuration)
9. [Evaluation & metrics](#evaluation--metrics)
10. [Project layout](#project-layout)
11. [Testing & development](#testing--development)
12. [Docker](#docker)
13. [Results on the bundled dataset](#results-on-the-bundled-dataset)
14. [FAQ & notes](#faq--notes)

---

## What this does

Aditya-FlareCast addresses the challenge of **forecasting and nowcasting solar
flares** by fusing the two X-ray views Aditya-L1 provides:

- **SoLEXS** — *soft* X-rays (the thermal, GOES-like flare signature).
- **HEL1OS** — *hard* X-rays (the non-thermal, impulsive signature that often
  leads the soft rise).

It delivers all three challenge outcomes:

| Outcome | Where |
|---|---|
| **Automated flare catalogue** (detect + GOES-classify, low → high class) | `nowcast/` → `data/catalogues/master_catalogue.csv` + `flarecast.db` |
| **Trained forecasting model** (probability of a flare in the next *N* min, with lead time) | `forecast/` → `models/forecaster/` |
| **Visualisation interface with alerts** | Streamlit dashboard + FastAPI + `scripts/generate_report.py` |

Everything runs **out of the box with zero credentials** thanks to a
physics-based synthetic data generator, and the exact same code path runs on real
ISSDC data (see [below](#using-real-aditya-l1-data)).

---

## Why it works — the science

The forecast's lead time is not statistical magic; it comes from two established
solar-physics phenomena:

- **The Neupert effect.** Hard X-rays (HEL1OS) from accelerated electrons rise
  and peak *before* the soft X-ray (SoLEXS) emission — the soft flux behaves like
  the time-integral of the hard flux. So a hard-channel surge is an early warning
  of a soft-channel flare. The pipeline measures the hard/soft lead directly and
  stores it as `neupert_lead_s` on every fused event.
- **Pre-flare precursors.** Many flares are preceded by small hard-X-ray
  micro-bursts minutes before the main impulsive phase. Unmatched hard bursts are
  retained as `candidate_precursor` events and fed to the forecaster via a
  "minutes-since-hard-burst" feature.

Both effects are modelled in the synthetic generator and exploited by the
[causal features](docs/ARCHITECTURE.md). Because some flares rise abruptly with
no precursor, they are fundamentally unpredictable — the skill scores reflect a
real physical ceiling, not an artefact.

---

## Quick start (2 minutes)

```bash
# 1. Install (core deps only — runs the whole pipeline)
pip install -e .

# 2. Generate data, build the catalogue, and train the forecaster end-to-end
aditya-flarecast pipeline --config configs/default.yaml

# 3. Get the latest live forecast
aditya-flarecast predict --config configs/default.yaml
```

That's it. You now have a flare catalogue in `data/catalogues/`, a trained model
in `models/forecaster/`, and a printed JSON forecast with an alert flag and
horizon.

Want the visuals immediately?

```bash
pip install -e ".[dashboard]"
aditya-flarecast dashboard          # opens the live Streamlit dashboard
# or, for a static evaluation figure:
python scripts/generate_report.py   # writes reports/summary.png + report.json
```

---

## Installation

Requires **Python ≥ 3.10**. The core install is deliberately lean (NumPy, pandas,
SciPy, scikit-learn) so it works anywhere; heavier back-ends and interfaces are
optional extras.

```bash
# Core pipeline only
pip install -e .

# Add what you need:
pip install -e ".[serve]"      # FastAPI REST service
pip install -e ".[dashboard]"  # Streamlit dashboard + matplotlib
pip install -e ".[boost]"      # LightGBM back-end
pip install -e ".[deep]"       # PyTorch LSTM back-end
pip install -e ".[fits]"       # Astropy + cdflib for real FITS/CDF data
pip install -e ".[dev]"        # pytest, ruff, black, mypy

# Everything at once
pip install -e ".[all,dev]"
```

Optional dependencies **degrade gracefully** — if you request the `lightgbm`
model without installing it, the trainer logs a warning and falls back to the
always-available gradient-boosting back-end.

---

## Running the pipeline

Each stage is an independent CLI command. Run them separately to inspect
intermediate outputs, or use `pipeline` to run all of them.

```bash
# 1. Synthesise SoLEXS + HEL1OS data + ground truth  →  data/raw/
aditya-flarecast synth --config configs/default.yaml

# 2. Fuse + clean into one analysis frame            →  data/processed/fused.parquet
aditya-flarecast preprocess --config configs/default.yaml

# 3. Detect + classify flares → master catalogue     →  data/catalogues/
aditya-flarecast nowcast --config configs/default.yaml

# 4. Train the forecaster (chronological split)      →  models/forecaster/
aditya-flarecast train --config configs/default.yaml

# — or all four at once —
aditya-flarecast pipeline --config configs/default.yaml

# Latest real-time forecast (loads the trained model)
aditya-flarecast predict --config configs/default.yaml
```

Handy flags: `--seed N` (reproducibility), `--days N` on `synth`,
`--no-synthetic` on `pipeline` (use existing `data/raw/` instead of regenerating).

Prefer `make`? `make pipeline`, `make dashboard`, `make test`, `make report`,
`make help`.

---

## The interfaces: API & dashboard

### REST API (FastAPI)

```bash
pip install -e ".[serve]"
aditya-flarecast serve-api --config configs/default.yaml   # http://localhost:8000
```

Interactive docs at `http://localhost:8000/docs`. Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness + whether a model is loaded + back-end name |
| `GET /forecast/latest` | Latest forecast from the processed frame (logs an alert) |
| `POST /forecast` | Forecast from a posted batch of soft+hard samples |
| `GET /catalogue?goes_min=C&channel=fused` | Query the nowcast catalogue |
| `GET /alerts` | Recent logged forecast alerts |

```bash
curl "http://localhost:8000/forecast/latest"
curl "http://localhost:8000/catalogue?goes_min=C&limit=20"
```

### Dashboard (Streamlit)

```bash
pip install -e ".[dashboard]"
aditya-flarecast dashboard --config configs/default.yaml    # http://localhost:8501
```

The dashboard plots the soft and hard light curves, overlays nowcasted flares,
shows the forecast-probability track, and **flashes a visual alert banner**
whenever the probability crosses the tuned threshold — the challenge's
"interface that triggers with visual alerts" deliverable.

---

## Using real Aditya-L1 data

The bundled synthetic generator means you can try everything immediately, but the
pipeline is built for the real thing. Real Aditya-L1 Level-1 SoLEXS/HEL1OS
products come from ISRO's **ISSDC PRADAN** portal (authenticated; no anonymous
bulk API — which is why synthetic data ships by default).

In short:

```bash
pip install -e ".[fits]"

# Convert downloaded FITS/CDF into the canonical table
python scripts/download_issdc_data.py \
    --path solexs_l1.fits --instrument solexs \
    --time-column TIME --map flux_1_8A=RATE_1_8 --out data/raw/solexs_l1.csv

# Then run the pipeline without regenerating synthetic data
aditya-flarecast pipeline --config configs/default.yaml --no-synthetic
```

Without a ground-truth catalogue, training automatically falls back to
**self-supervised labels** from the nowcast catalogue. Full details, column
mappings, and directory layout are in **[docs/DATA.md](docs/DATA.md)**.

---

## Configuration

All behaviour is controlled by a single typed `Settings` tree (see `config.py`),
loaded from YAML and overridable by environment variables. No tuning constants
are hidden in code.

Bundled configs in `configs/`:

| File | Use |
|---|---|
| `default.yaml` | Full 20-day run — the reference configuration |
| `quick.yaml` | Short, coarse — fast smoke tests / CI |
| `lightgbm.yaml` | LightGBM back-end (needs `[boost]`) |
| `deep_lstm.yaml` | PyTorch LSTM back-end (needs `[deep]`) |

Override any value with an `AFC_`-prefixed env var (double underscore = nesting):

```bash
AFC_FORECAST__HORIZON_MIN=90 AFC_FORECAST__MODEL=lightgbm \
    aditya-flarecast train --config configs/default.yaml
```

Key knobs: `instrument.analysis_cadence_s`, `nowcast.soft_threshold_factor`,
`nowcast.hard_sigma`, `nowcast.fusion_window_s`, `forecast.horizon_min`,
`forecast.min_class`, `forecast.model`, `forecast.alert_threshold`.

---

## Evaluation & metrics

Forecasting is scored with the standard **space-weather verification metrics**
(in `metrics/skill_scores.py`), not plain accuracy — because flares are rare and
accuracy is misleading on imbalanced data:

- **TSS** (True Skill Statistic = POD − POFD) — the primary space-weather score.
- **HSS** (Heidke Skill Score) — skill over random chance.
- **POD** (Probability of Detection / recall) and **FAR** (False Alarm Ratio) —
  the challenge's "high true-positive, low false-alarm" objective.
- **Brier score** — probabilistic calibration.
- **Lead-time statistics** — per-flare minutes between first alert and flare
  peak (mean / median / p10 / p90 / max), directly answering "how much warning?".

The nowcast is scored on **per-class recall & precision** (A/B/C/M/X) against the
ground-truth catalogue, demonstrating detection of both low- and high-class
flares. The alert threshold is tuned on a **validation split by maximising TSS**,
then reported on a held-out **future test split** — an honest, leakage-free
estimate of operational skill.

---

## Project layout

```
aditya-flarecast/
├── src/aditya_flarecast/
│   ├── config.py              # typed Settings (YAML + env)
│   ├── io/                    # readers (CSV/parquet/FITS/CDF) + synthetic generator
│   ├── preprocessing/         # background, resample, quality, fusion
│   ├── features/              # causal feature engineering
│   ├── nowcast/               # detectors, GOES classifier, fusion, catalogue
│   ├── forecast/              # dataset, models, train, evaluate, predict
│   ├── metrics/               # TSS/HSS/POD/FAR/Brier/lead-time
│   ├── db/                    # SQLAlchemy catalogue + alert log
│   ├── api/                   # FastAPI service
│   ├── dashboard/             # Streamlit UI
│   ├── streaming.py           # real-time replay generator
│   ├── orchestration.py       # stage wiring
│   └── cli.py                 # Typer CLI
├── configs/                   # default / quick / lightgbm / deep_lstm
├── scripts/                   # generate_report.py, download_issdc_data.py
├── tests/                     # pytest suite (io, preprocessing, nowcast, forecast, metrics, db)
├── docs/                      # ARCHITECTURE.md, DATA.md
├── Dockerfile, docker-compose.yml
├── pyproject.toml, requirements*.txt, Makefile
└── README.md
```

Architecture deep-dive: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Testing & development

```bash
pip install -e ".[dev]"
pytest -q          # full suite
make lint          # ruff
make format        # black + ruff --fix
```

The suite covers synthetic generation, preprocessing & causal features, nowcast
detection/classification/fusion, the metric implementations, forecast
dataset/train/serve, and DB idempotency — including an end-to-end
synthetic → nowcast → train smoke path. CI (GitHub Actions) runs lint + tests on
Python 3.10–3.12 plus a pipeline smoke test.

---

## Docker

```bash
# One-shot image (pipeline + API + dashboard)
docker build -t aditya-flarecast .

# Full stack: bootstraps data+model, then serves API (:8000) + dashboard (:8501)
docker compose up --build
```

The `bootstrap` service generates data and trains the model into a shared volume
before the API and dashboard start.

---

## Results on the bundled dataset

Running `aditya-flarecast pipeline --config configs/default.yaml` (20 synthetic
days, seed 42) gives, on a held-out future test split:

- **Nowcast:** recall ≈ 0.92, precision ≈ 0.98 — per-class recall B ≈ 0.90,
  C ≈ 0.97, M ≈ 1.00 (detects both low- and high-class flares).
- **Forecast:** catches every predictable test flare with a **median lead time of
  ~10 minutes**; positive TSS/HSS. Flares with no precursor are correctly
  reflected as an irreducible limit rather than hidden.

Generate the figure below yourself with `python scripts/generate_report.py`
(writes `reports/summary.png` + `reports/report.json`): soft & hard light curves
with nowcasted flares, candidate precursors, and the forecast-probability track
with alert markers firing *ahead* of each flare.

> Exact numbers vary with seed and config; the `quick.yaml` config uses far less
> data and will show weaker ML metrics by design.

---

## FAQ & notes

**Is the data real?** The bundled data is **synthetic but physically motivated**
(power-law flare fluxes, Neupert-effect hard X-rays, precursors). This keeps the
project fully reproducible with no credentials. The identical pipeline runs on
real ISSDC data — see [docs/DATA.md](docs/DATA.md).

**Why gradient boosting by default?** It is strong on tabular, imbalanced
space-weather data, trains in seconds, needs no GPU, and has zero heavy
dependencies. LightGBM and an LSTM are available as optional back-ends via a
one-line config change.

**Can I forecast without a labelled catalogue?** Yes — on unlabelled real data the
forecaster self-supervises from its own nowcast catalogue.

**License:** MIT (see [LICENSE](LICENSE)).
