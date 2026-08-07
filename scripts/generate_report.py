#!/usr/bin/env python
"""Generate an evaluation report: multi-panel figure + metrics JSON.

Produces, under ``reports/``:

* ``summary.png`` — soft & hard light curves with nowcasted flares overlaid, the
  forecast probability track, and alert markers (the "visual alerts" deliverable
  as a static artefact).
* ``report.json`` — nowcast detection metrics + forecast skill scores + lead
  time statistics.

Run after ``aditya-flarecast pipeline``::

    python scripts/generate_report.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--window-hours", type=float, default=48.0,
                    help="Width of the light-curve panel window (busiest region).")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from aditya_flarecast.config import load_settings
    from aditya_flarecast.features.engineering import build_features
    from aditya_flarecast.forecast.evaluate import evaluate_forecast
    from aditya_flarecast.nowcast.catalogue import evaluate_catalogue
    from aditya_flarecast.orchestration import load_forecaster, load_processed

    settings = load_settings(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed(settings)
    soft_band = df.attrs.get("solexs_flux_band", "flux_1_8A")
    hard_low = df.attrs.get("hel1os_low_band", "counts_10_30keV")

    cat = pd.read_csv(settings.paths.catalogues / "master_catalogue.csv")
    cat["peak_time"] = pd.to_datetime(cat["peak_time"], utc=True)
    truth_path = settings.paths.raw / "ground_truth_catalogue.csv"
    truth = pd.read_csv(truth_path) if truth_path.exists() else pd.DataFrame()

    forecaster = load_forecaster(settings)
    feats = build_features(df, cadence_s=df.attrs.get("cadence_s"))
    fc_out = forecaster.predict(feats)
    prob = pd.Series(fc_out.probability, index=feats.index)

    # ---- Metrics -------------------------------------------------------- #
    report = {"config": args.config}
    if not truth.empty:
        report["nowcast"] = evaluate_catalogue(cat, truth, tolerance_s=300.0)

    labels = (
        truth.rename(columns={"soft_peak_time": "peak_time"})[["peak_time", "goes_class"]]
        if not truth.empty
        else cat[cat["channel"].isin(["fused", "soft"])][["peak_time", "goes_class"]]
    )
    y = _labels_for_times(feats.index, labels, settings)
    report["forecast"] = evaluate_forecast(
        feats.index, y, fc_out.probability, labels,
        settings.forecast.horizon_min, settings.forecast.min_class,
        threshold=forecaster.threshold,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report.get("forecast", {}).get("flare_level", {}), indent=2))

    # ---- Figure --------------------------------------------------------- #
    # Pick the busiest window (most catalogue peaks) for the panels.
    center = _busiest_center(cat, df.index)
    half = pd.Timedelta(hours=args.window_hours / 2)
    start, end = center - half, center + half
    view = df.loc[start:end]
    pv = prob.loc[start:end]
    win_cat = cat[(cat["peak_time"] >= start) & (cat["peak_time"] <= end)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    ax.semilogy(view.index, view[f"solexs_{soft_band}"], lw=0.8, color="#c0392b")
    ax.set_ylabel("SoLEXS soft flux\n(W m$^{-2}$, 1-8 Å)")
    ax.set_title("Aditya-FlareCast — nowcast + forecast (busiest window)")
    for gl, y0 in [("C", 1e-6), ("M", 1e-5), ("X", 1e-4)]:
        ax.axhline(y0, color="grey", ls=":", lw=0.6)
        ax.text(view.index[0], y0 * 1.1, gl, color="grey", fontsize=8)
    for _, r in win_cat[win_cat["channel"] == "fused"].iterrows():
        ax.axvline(r["peak_time"], color="#2980b9", alpha=0.35, lw=1.0)

    ax = axes[1]
    ax.plot(view.index, view[f"hel1os_{hard_low}"], lw=0.7, color="#8e44ad")
    ax.set_ylabel("HEL1OS hard\n(counts s$^{-1}$)")
    prec = win_cat[win_cat.get("candidate_precursor", False) == True]  # noqa: E712
    if not prec.empty:
        ax.scatter(prec["peak_time"], [ax.get_ylim()[1]] * len(prec),
                   marker="v", color="#e67e22", s=30, label="candidate precursor")
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.fill_between(pv.index, 0, pv.values, color="#16a085", alpha=0.4)
    ax.plot(pv.index, pv.values, lw=0.8, color="#16a085")
    ax.axhline(forecaster.threshold, color="red", ls="--", lw=1.0,
               label=f"alert threshold = {forecaster.threshold:.2f}")
    alerts = pv[pv >= forecaster.threshold]
    ax.scatter(alerts.index, alerts.values, color="red", s=6, zorder=5)
    ax.set_ylabel(f"P(flare ≤ {int(settings.forecast.horizon_min)} min)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=130)
    print(f"Wrote {out_dir/'summary.png'} and {out_dir/'report.json'}")


def _labels_for_times(times, labels, settings):
    from aditya_flarecast.forecast.dataset import build_labels

    return build_labels(
        times, labels, settings.forecast.horizon_min, settings.forecast.min_class
    )


def _busiest_center(cat: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Timestamp:
    fused = cat[cat["channel"] == "fused"]
    if fused.empty:
        return index[len(index) // 2]
    # Densest 1-day cluster of peaks.
    peaks = fused["peak_time"].sort_values()
    best_t, best_n = peaks.iloc[0], 0
    for t in peaks:
        n = int(((peaks >= t) & (peaks < t + pd.Timedelta(days=1))).sum())
        if n > best_n:
            best_n, best_t = n, t
    return best_t + pd.Timedelta(hours=12)


if __name__ == "__main__":
    main()
