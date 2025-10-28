import pandas as pd
from hospops_forecast.segmentation import Segmenter
from hospops_forecast.models.labor import LaborForecaster

def test_labor_forecast_runs():
    df = pd.DataFrame([
        {"reservation_id":"R1","arrival_date":"2025-01-06","departure_date":"2025-01-08","adults":1,"children":0,"room_type":"Standard","channel":"Corp","company":"ACME"},
        {"reservation_id":"R2","arrival_date":"2025-01-11","departure_date":"2025-01-13","adults":2,"children":0,"room_type":"Deluxe","channel":"OTA","company":""},
        {"reservation_id":"R3","arrival_date":"2025-01-11","departure_date":"2025-01-14","adults":2,"children":2,"room_type":"Suite","channel":"OTA","company":""},
    ])
    seg = Segmenter()
    enriched = seg.enrich(df)
    lf = LaborForecaster()
    out = lf.predict(enriched)
    assert not out.empty
    assert set(["date","hk_man_hours","recommended_headcount"]).issubset(out.columns)
