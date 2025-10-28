from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
from io import BytesIO
from ..segmentation.segmenter import Segmenter
from ..models.labor import LaborForecaster
from ..models.fnb import FNBConsumptionForecaster
from ..models.service import ServiceLoadForecaster
from ..models.departments import DepartmentForecaster
from ..models.airline import AirlineForecaster
from ..calibration.labor_calibrator import LaborCalibrator
from ..models.learning import HKLearner
from ..dq.validators import check_reservations_basic

app = FastAPI(title="HospOps-Forecast API", version="0.3.0")

def _read_csv(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    return pd.read_csv(BytesIO(content))

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    df = _read_csv(file)
    seg = Segmenter()
    out = seg.enrich(df)
    return {"rows": len(out), "preview": out.head(50).to_dict(orient="records")}

@app.post("/forecast/labor")
async def forecast_labor(file: UploadFile = File(...)):
    df = _read_csv(file)
    seg = Segmenter(); enriched = seg.enrich(df)
    out = LaborForecaster().predict(enriched)
    return out.to_dict(orient="records")

@app.post("/forecast/fnb")
async def forecast_fnb(meal: str = Query("breakfast"), file: UploadFile = File(...)):
    df = _read_csv(file)
    seg = Segmenter(); enriched = seg.enrich(df)
    out = FNBConsumptionForecaster().predict(enriched, meal=meal)
    return out.to_dict(orient="records")

@app.post("/forecast/service")
async def forecast_service(area: str = Query(...), target_wait_min: float = Query(5.0), file: UploadFile = File(...)):
    df = _read_csv(file)
    seg = Segmenter(); enriched = seg.enrich(df)
    out = ServiceLoadForecaster().predict(enriched, area=area, target_wait_min=target_wait_min)
    return out.to_dict(orient="records")

@app.post("/forecast/dept")
async def forecast_dept(dept: str = Query(...), file: UploadFile = File(...)):
    df = _read_csv(file)
    seg = Segmenter(); enriched = seg.enrich(df)
    out = DepartmentForecaster().predict(enriched, dept=dept)
    return out.to_dict(orient="records")

@app.post("/forecast/airline")
async def forecast_airline(area: str = Query(...), flights: UploadFile = File(...)):
    df = _read_csv(flights)
    out = AirlineForecaster().predict(df, area=area)
    return out.to_dict(orient="records")

@app.post("/calibrate/labor")
async def calibrate_labor(enriched: UploadFile = File(...), actual: UploadFile = File(...)):
    en = _read_csv(enriched)
    ac = _read_csv(actual)
    from ..config.loader import load_config
    cal = LaborCalibrator(load_config(None))
    tuned = cal.fit_multipliers(en, ac)
    return tuned

@app.post("/learn/hk")
async def learn_hk(enriched: UploadFile = File(...), actual: UploadFile = File(...)):
    en = _read_csv(enriched)
    ac = _read_csv(actual)
    learner = HKLearner()
    tuned = learner.fit(en, ac)
    return tuned

@app.post("/dq/check")
async def dq_check(file: UploadFile = File(...)):
    df = _read_csv(file)
    res = check_reservations_basic(df)
    return res
