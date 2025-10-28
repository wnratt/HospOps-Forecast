import pandas as pd
from hospops_forecast.segmentation.segmenter import Segmenter
from hospops_forecast.models.service import ServiceLoadForecaster

def test_service_sla_runs():
    df = pd.read_csv("examples/data/sample_reservations.csv", parse_dates=["arrival_date","departure_date"])
    en = Segmenter().enrich(df)
    out = ServiceLoadForecaster().predict(en, area="reception", target_wait_min=5.0)
    assert not out.empty and "recommended_staff" in out.columns
