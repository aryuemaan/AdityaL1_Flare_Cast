"""Streamlit dashboard: light curves, nowcast catalogue, and live alerts.

Run with::

    aditya-flarecast dashboard
    # or
    streamlit run src/aditya_flarecast/dashboard/app.py

The dashboard visualises the fused SoLEXS (soft) and HEL1OS (hard) light curves,
overlays nowcasted flares and forecast probability, and flashes a visual alert
whenever the forecaster's probability crosses the tuned threshold — satisfying
the challenge's "interface that visualises the X-ray light curves and triggers
with visual alerts" outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when launched via ``streamlit run``.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aditya_flarecast.config import load_settings  # noqa: E402
from aditya_flarecast.features.engineering import build_features  # noqa: E402

st.set_page_config(page_title="Aditya-FlareCast", layout="wide", page_icon="☀️")


@st.cache_data(show_spinner=True)
def _load(config_path: str):
    settings = load_settings(config_path)
    from aditya_flarecast.orchestration import load_processed

    df = load_processed(settings)
    cat_path = settings.paths.catalogues / "master_catalogue.csv"
    cat = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    if not cat.empty:
        cat["peak_time"] = pd.to_datetime(cat["peak_time"], utc=True)
    return settings, df, cat


@st.cache_resource(show_spinner=False)
def _forecast(config_path: str):
    settings = load_settings(config_path)
    from aditya_flarecast.orchestration import load_forecaster, load_processed

    df = load_processed(settings)
    feats = build_features(df, cadence_s=df.attrs.get("cadence_s"))
    fc = load_forecaster(settings)
    out = fc.predict(feats)
    return pd.Series(out.probability, index=feats.index), fc.threshold, fc.horizon_min


def main() -> None:
    st.title("☀️ Aditya-FlareCast — Solar Flare Nowcasting & Forecasting")
    st.caption(
        "Combined SoLEXS (soft X-ray) + HEL1OS (hard X-ray) from Aditya-L1 · "
        "nowcast catalogue + forecast with lead time"
    )

    config_path = st.sidebar.text_input("Config", "configs/default.yaml")
    try:
        settings, df, cat = _load(config_path)
    except Exception as exc:
        st.error(f"Could not load processed data: {exc}")
        st.info("Run `aditya-flarecast pipeline` first to generate + process data.")
        return

    soft_band = df.attrs.get("solexs_flux_band", "flux_1_8A")
    hard_low = df.attrs.get("hel1os_low_band", "counts_10_30keV")

    # Time window control.
    t0, t1 = df.index[0], df.index[-1]
    st.sidebar.markdown("### Time window")
    span_hours = st.sidebar.slider("Window (hours)", 2, 72, 24)
    end = st.sidebar.select_slider(
        "End of window",
        options=list(df.index[::max(1, len(df) // 200)]),
        value=df.index[-1],
    )
    start = max(t0, end - pd.Timedelta(hours=span_hours))
    view = df.loc[start:end]

    # Forecast probability (optional).
    prob = None
    threshold = settings.forecast.alert_threshold
    horizon = settings.forecast.horizon_min
    try:
        prob_full, threshold, horizon = _forecast(config_path)
        prob = prob_full.loc[start:end]
    except Exception:
        st.sidebar.warning("No trained model — run `aditya-flarecast train`.")

    # --- Live alert banner ---------------------------------------------- #
    latest_p = float(prob.iloc[-1]) if prob is not None and len(prob) else 0.0
    cols = st.columns(4)
    cols[0].metric("Latest flare probability", f"{latest_p:.0%}")
    cols[1].metric("Alert threshold", f"{threshold:.0%}")
    cols[2].metric("Forecast horizon", f"{int(horizon)} min")
    n_flares = int((cat["channel"] == "fused").sum()) if not cat.empty else 0
    cols[3].metric("Flares in catalogue", n_flares)

    if latest_p >= threshold:
        st.error(
            f"🚨 FLARE ALERT — probability {latest_p:.0%} ≥ {threshold:.0%}. "
            f"A flare is forecast within the next {int(horizon)} minutes.",
            icon="🚨",
        )
    else:
        st.success("✅ No flare forecast in the current horizon.", icon="✅")

    # --- Light curves ---------------------------------------------------- #
    st.subheader("Soft X-ray (SoLEXS)")
    soft_df = pd.DataFrame({"soft_flux_W_m2": view[f"solexs_{soft_band}"]})
    st.line_chart(soft_df, height=240)

    st.subheader("Hard X-ray (HEL1OS)")
    hard_df = pd.DataFrame({"hard_counts_s": view[f"hel1os_{hard_low}"]})
    st.line_chart(hard_df, height=200)

    if prob is not None:
        st.subheader("Forecast probability (next horizon)")
        st.area_chart(pd.DataFrame({"flare_probability": prob}), height=180)

    # --- Nowcast catalogue table ---------------------------------------- #
    st.subheader("Nowcast master catalogue (window)")
    if not cat.empty:
        win_cat = cat[(cat["peak_time"] >= start) & (cat["peak_time"] <= end)]
        show_cols = [
            c for c in [
                "channel", "peak_time", "goes_class", "peak_value",
                "neupert_lead_s", "candidate_precursor", "detected_by",
            ] if c in win_cat.columns
        ]
        st.dataframe(win_cat[show_cols], use_container_width=True, height=280)
    else:
        st.info("No catalogue found. Run `aditya-flarecast nowcast`.")


if __name__ == "__main__":
    main()
