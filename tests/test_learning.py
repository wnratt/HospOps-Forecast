import pandas as pd
from hospops_forecast.segmentation.segmenter import Segmenter
from hospops_forecast.models.learning import HKLearner
from hospops_forecast.models.labor import LaborForecaster

def test_hk_learner_runs():
    df = pd.read_csv("examples/data/sample_reservations.csv", parse_dates=["arrival_date","departure_date"])
    en = Segmenter().enrich(df)
    hk = LaborForecaster().predict(en)
    ac = hk[["date","hk_man_hours"]]
    tuned = HKLearner().fit(en, ac)
    assert "housekeeping" in tuned and "archetype_multipliers" in tuned["housekeeping"]
