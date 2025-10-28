from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import typer, yaml
from rich import print
from .segmentation.segmenter import Segmenter
from .models.labor import LaborForecaster
from .models.fnb import FNBConsumptionForecaster
from .models.service import ServiceLoadForecaster
from .models.departments import DepartmentForecaster
from .models.airline import AirlineForecaster
from .models.learning import HKLearner
from .calibration.labor_calibrator import LaborCalibrator
from .dq.validators import check_reservations_basic
from .config.loader import load_config

app = typer.Typer(add_completion=False, help="HospOps-Forecast CLI")

def _load_config(path: Optional[Path]) -> dict:
    return load_config(path)

@app.command()
def segment(input: Path = typer.Option(..., "--input"), output: Path = typer.Option(..., "--output"), use_unsupervised: bool = False):
    df = pd.read_csv(input)
    seg = Segmenter(use_unsupervised=use_unsupervised)
    enriched = seg.enrich(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    print(f"[bold green]Wrote enriched ->[/] {output}")

@app.command()
def forecast(
    sub: str = typer.Argument(..., help="'labor' | 'fnb' | 'service' | 'dept' | 'airline'"),
    enriched: Optional[Path] = typer.Option(None, "--enriched", help="Enriched CSV for hotel domains"),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
    area: Optional[str] = typer.Option(None, "--area", help="service: reception|breakfast; airline: gate|boarding|lounge"),
    meal: Optional[str] = typer.Option("breakfast", "--meal", help="fnb: breakfast|lunch|dinner"),
    dept: Optional[str] = typer.Option(None, "--dept", help="dept: spa|concierge|valet|engineering"),
    flights: Optional[Path] = typer.Option(None, "--flights", help="Airline flights CSV"),
    utilization: float = 0.85,
    target_wait_min: Optional[float] = typer.Option(None, "--target-wait-min", help="SLA target wait (minutes)"),
):
    cfg = _load_config(config)
    if sub == "labor":
        df = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
        out = LaborForecaster(cfg).predict(df, utilization=utilization)
    elif sub == "fnb":
        df = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
        out = FNBConsumptionForecaster(cfg).predict(df, meal=meal)
    elif sub == "service":
        if area is None: raise typer.BadParameter("--area reception|breakfast required")
        df = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
        out = ServiceLoadForecaster(cfg).predict(df, area=area, utilization=utilization, target_wait_min=target_wait_min)
    elif sub == "dept":
        if dept is None: raise typer.BadParameter("--dept required")
        df = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
        out = DepartmentForecaster(cfg).predict(df, dept=dept)
    elif sub == "airline":
        if flights is None: raise typer.BadParameter("--flights CSV required")
        df = pd.read_csv(flights)
        out = AirlineForecaster(cfg).predict(df, area=area or "boarding")
    else:
        raise typer.BadParameter("Unknown subcommand.")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[bold green]Wrote forecast ->[/] {output}")

@app.command()
def calibrate(sub: str = typer.Argument(..., help="'labor'"), enriched: Path = typer.Option(..., "--enriched"), actual: Path = typer.Option(..., "--actual"), output: Path = typer.Option(..., "--output"), config: Optional[Path] = None, min_mult: float = 0.5, max_mult: float = 3.0):
    if sub != "labor":
        raise typer.BadParameter("Only 'labor' is supported.")
    base_cfg = _load_config(config)
    en = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
    ac = pd.read_csv(actual, parse_dates=["date"])
    tuned = LaborCalibrator(base_cfg).fit_multipliers(en, ac, min_mult=min_mult, max_mult=max_mult)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False, allow_unicode=True)
    print(f"[bold green]Wrote tuned config ->[/] {output}")

@app.command()
def learn(sub: str = typer.Argument(..., help="'hk'"), enriched: Path = typer.Option(..., "--enriched"), actual: Path = typer.Option(..., "--actual"), output: Path = typer.Option(..., "--output")):
    if sub != "hk":
        raise typer.BadParameter("Only 'hk' learner implemented.")
    en = pd.read_csv(enriched, parse_dates=["arrival_date","departure_date"])
    ac = pd.read_csv(actual, parse_dates=["date"])
    tuned = HKLearner().fit(en, ac)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False, allow_unicode=True)
    print(f"[bold green]Wrote learned config ->[/] {output}")

@app.command()
def dq(sub: str = typer.Argument(..., help="'check'"), input: Path = typer.Option(..., "--input")):
    if sub != "check":
        raise typer.BadParameter("Only 'check' supported for now.")
    df = pd.read_csv(input)
    res = check_reservations_basic(df)
    print(res)

if __name__ == "__main__":
    app()
