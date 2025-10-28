import pandas as pd
from hospops_forecast.segmentation.segmenter import Segmenter
from hospops_forecast.models.departments import DepartmentForecaster

def test_dept_spa_runs():
    df = pd.read_csv("examples/data/sample_reservations.csv", parse_dates=["arrival_date","departure_date"])
    en = Segmenter().enrich(df)
    out = DepartmentForecaster().predict(en, dept="spa")
    assert not out.empty and "recommended_headcount" in out.columns
