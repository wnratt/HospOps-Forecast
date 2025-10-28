import pandas as pd
from hospops_forecast.models.airline import AirlineForecaster

def test_airline_boarding_runs():
    fl = pd.read_csv("examples/data/sample_flights.csv")
    out = AirlineForecaster().predict(fl, area="boarding")
    assert not out.empty and "recommended_staff" in out.columns
