import pandas as pd
from hospops_forecast.segmentation import Segmenter
from hospops_forecast.models.service import ServiceLoadForecaster

def test_service_reception_forecast_runs():
    df = pd.DataFrame([
        {"reservation_id":"R1","arrival_date":"2025-01-06","departure_date":"2025-01-08","adults":1,"children":0,"room_type":"Standard","channel":"Corp","company":"ACME"},
        {"reservation_id":"R2","arrival_date":"2025-01-11","departure_date":"2025-01-13","adults":2,"children":0,"room_type":"Deluxe","channel":"OTA","company":""},
        {"reservation_id":"R3","arrival_date":"2025-01-11","departure_date":"2025-01-14","adults":2,"children":2,"room_type":"Suite","channel":"OTA","company":""},
    ])
    seg = Segmenter()
    enriched = seg.enrich(df)
    svc = ServiceLoadForecaster()
    out = svc.predict(enriched, area="reception")
    assert not out.empty
    assert set(["datetime","expected_transactions","recommended_staff","load_status"]).issubset(out.columns)

def test_service_breakfast_forecast_runs():
    df = pd.DataFrame([
        {"reservation_id":"R1","arrival_date":"2025-01-06","departure_date":"2025-01-08","adults":1,"children":0,"room_type":"Standard","channel":"Corp","company":"ACME"},
        {"reservation_id":"R2","arrival_date":"2025-01-11","departure_date":"2025-01-13","adults":2,"children":0,"room_type":"Deluxe","channel":"OTA","company":""},
        {"reservation_id":"R3","arrival_date":"2025-01-11","departure_date":"2025-01-14","adults":2,"children":2,"room_type":"Suite","channel":"OTA","company":""},
    ])
    seg = Segmenter()
    enriched = seg.enrich(df)
    svc = ServiceLoadForecaster()
    out = svc.predict(enriched, area="breakfast")
    assert not out.empty
    assert set(["datetime","expected_covers","recommended_staff","load_status"]).issubset(out.columns)
