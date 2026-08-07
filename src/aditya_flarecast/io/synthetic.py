"""Physics-motivated synthetic SoLEXS + HEL1OS generator.

Real Aditya-L1 Level-1 products live behind the ISSDC PRADAN portal and
require authenticated download, so the repository ships a generator that
produces *realistic* light curves with a known ground-truth catalogue. This
lets the entire pipeline (nowcast + forecast + evaluation) run end-to-end and
be tested in CI without external data.

Physical ingredients modelled
-----------------------------
* **Quiescent background** near the A-class floor (~1e-8 W/m^2) with slow
  drift and photon noise.
* **Flares** with a fast exponential rise and slow exponential decay in the
  soft band (the canonical GOES 1-8 A shape).
* **Neupert effect**: the hard-X-ray (HEL1OS) time profile tracks the *time
  derivative* of the soft-X-ray flux, so the hard peak leads the soft peak.
  This is the physical basis for a non-trivial forecast lead time.
* **Pre-flare precursors**: a configurable fraction of flares are preceded by
  a weak impulsive hard-X-ray micro-burst minutes before soft onset — the
  signal a forecaster can learn.
* **Peak-flux power law** so small flares vastly outnumber large ones, as
  observed.

The generator returns two :class:`LightCurve` objects and a ground-truth
flare catalogue (:class:`pandas.DataFrame`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aditya_flarecast.config import SyntheticConfig
from aditya_flarecast.io.schemas import LightCurve
from aditya_flarecast.logging_utils import get_logger

logger = get_logger(__name__)

_GOES = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def _goes_class(peak_flux: float) -> str:
    for cls in ("X", "M", "C", "B", "A"):
        if peak_flux >= _GOES[cls]:
            sub = peak_flux / _GOES[cls]
            return f"{cls}{sub:.1f}"
    return "A0.1"


def _sample_peak_flux(rng: np.random.Generator, cfg: SyntheticConfig, n: int) -> np.ndarray:
    """Inverse-transform sample from a truncated power law dN/dF ~ F^-alpha."""
    a = cfg.flux_power_law_alpha
    lo, hi = cfg.flux_min, cfg.flux_max
    u = rng.random(n)
    if abs(a - 1.0) < 1e-6:
        return lo * (hi / lo) ** u
    exp = 1.0 - a
    return (lo**exp + u * (hi**exp - lo**exp)) ** (1.0 / exp)


def generate(
    cfg: SyntheticConfig,
    seed: int = 42,
    start: str = "2024-01-01T00:00:00Z",
) -> tuple[LightCurve, LightCurve, pd.DataFrame]:
    """Generate synthetic SoLEXS, HEL1OS light curves + ground-truth catalogue.

    Returns
    -------
    (solexs, hel1os, catalogue)
    """
    rng = np.random.default_rng(seed)
    dt = cfg.cadence_s
    n = int(round(cfg.n_days * 86400 / dt))
    t0 = pd.Timestamp(start)
    index = t0 + pd.to_timedelta(np.arange(n) * dt, unit="s")
    t = np.arange(n) * dt  # seconds since start

    logger.info(
        "Synthesising %.1f days at %.0fs cadence (%d samples)", cfg.n_days, dt, n
    )

    # --- Quiescent soft background: slow sinusoidal drift + lognormal noise -- #
    drift = cfg.background_flux * (
        1.0 + 0.25 * np.sin(2 * np.pi * t / (86400 * 1.7) + rng.random() * 6.28)
    )
    soft_18 = drift * np.exp(rng.normal(0, 0.06, n))          # 1-8 A band
    soft_04 = 0.35 * soft_18 * np.exp(rng.normal(0, 0.08, n))  # harder soft band

    # Hard-X-ray quiescent count rate (Poisson around a low baseline).
    hard_base_lo = 12.0
    hard_base_hi = 4.0
    hard_lo = rng.poisson(hard_base_lo, n).astype(float)
    hard_hi = rng.poisson(hard_base_hi, n).astype(float)

    # --- Flare occurrence times (Poisson process) ------------------------- #
    n_flares = rng.poisson(cfg.flares_per_day * cfg.n_days)
    onset_t = np.sort(rng.uniform(0, t[-1] * 0.98, n_flares))
    peaks = _sample_peak_flux(rng, cfg, n_flares)

    records = []
    for k in range(n_flares):
        onset = onset_t[k]
        fpk = peaks[k]

        # Timescales scale weakly with size (bigger flares last longer).
        size = np.log10(fpk / cfg.flux_min + 1.0)
        tau_rise = rng.uniform(60, 240) * (1 + 0.4 * size)      # s
        tau_decay = rng.uniform(400, 1400) * (1 + 0.6 * size)   # s

        i0 = int(onset / dt)
        # Evaluate the flare over a generous window (10 decay times).
        span = int(min(n - i0, (tau_rise + 10 * tau_decay) / dt))
        if span <= 5:
            continue
        idx = np.arange(i0, i0 + span)
        tt = (idx - i0) * dt

        # Canonical soft profile: (1 - e^{-t/rise}) * e^{-t/decay}, normalised
        # so its maximum equals fpk.
        rise = 1.0 - np.exp(-tt / tau_rise)
        decay = np.exp(-tt / tau_decay)
        prof = rise * decay
        prof /= prof.max() + 1e-30
        soft_add = fpk * prof

        soft_18[idx] += soft_add
        soft_04[idx] += 0.5 * soft_add  # flares are relatively harder at peak

        # Neupert effect: hard flux ~ d(soft)/dt (rectified). Peaks before soft.
        dsoft = np.gradient(soft_add, dt)
        neupert = np.clip(dsoft, 0, None)
        neupert /= neupert.max() + 1e-30
        # Hard amplitude grows super-linearly with flare size (harder flares
        # are more impulsive / non-thermal).
        hard_amp_lo = 600.0 * (fpk / 1e-6) ** 0.55
        hard_amp_hi = 220.0 * (fpk / 1e-6) ** 0.65
        hard_lo[idx] += rng.poisson(np.clip(hard_amp_lo * neupert, 0, None))
        hard_hi[idx] += rng.poisson(np.clip(hard_amp_hi * neupert, 0, None))

        # Hard peak time (impulsive phase) = argmax of neupert profile.
        hard_peak_i = i0 + int(np.argmax(neupert))
        soft_peak_i = i0 + int(np.argmax(soft_add))

        # --- Optional pre-flare precursor micro-burst --------------------- #
        has_precursor = rng.random() < cfg.precursor_fraction
        precursor_lead_s = 0.0
        if has_precursor:
            lead = rng.uniform(180, 900)  # 3-15 min before onset
            precursor_lead_s = lead
            pj = int((onset - lead) / dt)
            pdur = int(rng.uniform(30, 120) / dt)
            if pj > 0 and pj + pdur < n:
                pk_idx = np.arange(pj, pj + pdur)
                bump = np.hanning(pdur) if pdur > 1 else np.array([1.0])
                amp_lo = rng.uniform(40, 160)
                amp_hi = rng.uniform(15, 60)
                hard_lo[pk_idx] += rng.poisson(np.clip(amp_lo * bump, 0, None))
                hard_hi[pk_idx] += rng.poisson(np.clip(amp_hi * bump, 0, None))
                # A faint soft "pre-heating" ramp too.
                soft_18[pk_idx] += 0.05 * fpk * bump

        records.append(
            {
                "flare_id": f"SYN{k:05d}",
                "onset_time": index[i0],
                "soft_peak_time": index[min(soft_peak_i, n - 1)],
                "hard_peak_time": index[min(hard_peak_i, n - 1)],
                "peak_flux_1_8A": float(fpk),
                "goes_class": _goes_class(fpk),
                "tau_rise_s": float(tau_rise),
                "tau_decay_s": float(tau_decay),
                "has_precursor": bool(has_precursor),
                "precursor_lead_s": float(precursor_lead_s),
            }
        )

    # Guard against tiny negatives from noise.
    soft_18 = np.clip(soft_18, 1e-10, None)
    soft_04 = np.clip(soft_04, 1e-10, None)

    solexs_df = pd.DataFrame(
        {"flux_1_8A": soft_18, "flux_0_5_4A": soft_04}, index=index
    )
    hel1os_df = pd.DataFrame(
        {"counts_10_30keV": hard_lo, "counts_30_70keV": hard_hi}, index=index
    )

    catalogue = pd.DataFrame.from_records(records)
    if not catalogue.empty:
        catalogue = catalogue.sort_values("soft_peak_time").reset_index(drop=True)

    solexs = LightCurve(
        "solexs",
        solexs_df,
        meta={"source": "synthetic", "level": "L1-sim", "cadence_s": dt},
    )
    hel1os = LightCurve(
        "hel1os",
        hel1os_df,
        meta={"source": "synthetic", "level": "L1-sim", "cadence_s": dt},
    )
    logger.info("Generated %d flares (ground truth)", len(catalogue))
    return solexs, hel1os, catalogue
