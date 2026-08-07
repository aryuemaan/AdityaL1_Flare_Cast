"""``aditya-flarecast`` command-line interface.

Sub-commands cover the whole lifecycle::

    aditya-flarecast synth              # generate synthetic SoLEXS+HEL1OS data
    aditya-flarecast preprocess         # fuse + clean into an analysis frame
    aditya-flarecast nowcast            # detect flares -> master catalogue + DB
    aditya-flarecast train              # train the forecaster
    aditya-flarecast pipeline           # do all of the above end-to-end
    aditya-flarecast predict            # print the latest live forecast
    aditya-flarecast serve-api          # launch the FastAPI service
    aditya-flarecast dashboard          # launch the Streamlit dashboard
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from aditya_flarecast import orchestration as orch
from aditya_flarecast.config import load_settings
from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.logging_utils import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    help="Solar-flare nowcasting & forecasting from Aditya-L1 X-ray data.",
    no_args_is_help=True,
)
logger = get_logger("cli")

_CONFIG_OPT = typer.Option("configs/default.yaml", "--config", "-c",
                           help="Path to a YAML config file.")


def _settings(config: str, seed: Optional[int]):
    overrides = {"random_seed": seed} if seed is not None else None
    settings = load_settings(config, overrides=overrides)
    configure_logging(settings.log_level)
    return settings


@app.command()
def synth(
    config: str = _CONFIG_OPT,
    seed: Optional[int] = typer.Option(None, help="Override the random seed."),
    days: Optional[float] = typer.Option(None, help="Override number of days."),
):
    """Generate synthetic SoLEXS + HEL1OS Level-1 data and ground truth."""
    settings = _settings(config, seed)
    if days is not None:
        settings.synthetic.n_days = days
    paths = orch.make_synthetic(settings, seed=seed)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


@app.command()
def preprocess(config: str = _CONFIG_OPT, seed: Optional[int] = None):
    """Preprocess raw light curves into the fused analysis frame."""
    settings = _settings(config, seed)
    solexs, hel1os = orch.load_inputs(settings)
    df = orch.build_processed(settings, solexs, hel1os)
    typer.echo(f"Processed {len(df)} rows -> {settings.paths.processed/'fused.parquet'}")


@app.command()
def nowcast(config: str = _CONFIG_OPT, seed: Optional[int] = None):
    """Detect flares in both channels and build the master catalogue."""
    settings = _settings(config, seed)
    df = orch.load_processed(settings)
    cat = orch.nowcast_stage(settings, df)
    typer.echo(f"Master catalogue: {len(cat)} rows -> "
               f"{settings.paths.catalogues/'master_catalogue.csv'}")


@app.command()
def train(config: str = _CONFIG_OPT, seed: Optional[int] = None,
          model: Optional[str] = typer.Option(None, help="hist_gbm|lightgbm|lstm")):
    """Train the flare forecaster on the processed data + catalogue."""
    settings = _settings(config, seed)
    if model:
        settings.forecast.model = model
    df = orch.load_processed(settings)
    import pandas as pd

    cat_path = settings.paths.catalogues / "master_catalogue.csv"
    cat = pd.read_csv(cat_path) if cat_path.exists() else orch.nowcast_stage(settings, df)
    result, _ = orch.forecast_stage(settings, df, cat)
    typer.echo(json.dumps(result.metadata.get("test_report", {}), indent=2, default=str))


@app.command()
def pipeline(
    config: str = _CONFIG_OPT,
    seed: Optional[int] = None,
    synthetic: bool = typer.Option(True, help="Generate synthetic data first."),
):
    """Run the entire pipeline end-to-end."""
    settings = _settings(config, seed)
    summary = orch.run_full_pipeline(settings, use_synthetic=synthetic, seed=seed)
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def predict(config: str = _CONFIG_OPT, seed: Optional[int] = None):
    """Print the latest real-time flare forecast."""
    settings = _settings(config, seed)
    df = orch.load_processed(settings)
    forecaster = orch.load_forecaster(settings)
    feats = build_features(df, cadence_s=df.attrs.get("cadence_s"))
    typer.echo(json.dumps(forecaster.predict_latest(feats), indent=2))


@app.command("serve-api")
def serve_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: str = _CONFIG_OPT,
):
    """Launch the FastAPI service (requires fastapi + uvicorn)."""
    import os

    os.environ.setdefault("AFC_CONFIG", config)
    try:
        import uvicorn
    except ImportError:
        typer.echo("Install serving extras: pip install '.[serve]'", err=True)
        raise typer.Exit(1)
    uvicorn.run("aditya_flarecast.api.main:app", host=host, port=port, reload=False)


@app.command()
def dashboard(config: str = _CONFIG_OPT):
    """Launch the Streamlit dashboard (requires streamlit)."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], check=True
        )
    except FileNotFoundError:
        typer.echo("Install viz extras: pip install '.[dashboard]'", err=True)
        raise typer.Exit(1)


def main() -> None:  # entry point
    app()


if __name__ == "__main__":
    main()
