import pandas as pd
from hospops_forecast.segmentation.segmenter import Segmenter
from hospops_forecast.models.fnb import FNBConsumptionForecaster

def test_fnb_lunch_runs():
    df = pd.read_csv("examples/data/sample_reservations.csv", parse_dates=["arrival_date","departure_date"])
    en = Segmenter().enrich(df)
    out = FNBConsumptionForecaster().predict(en, meal="lunch")
    assert not out.empty and "main_course_portions" in out.columns
