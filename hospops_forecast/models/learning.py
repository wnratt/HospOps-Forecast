from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from ..features import expand_reservations_daily, count_inhouse_by_day_and_archetype
from .base import BaseForecaster
from ..segmentation.archetypes import Archetype

@dataclass
class HKLearner(BaseForecaster):
    def fit(self, enriched_reservations: pd.DataFrame, hk_actual_daily: pd.DataFrame) -> dict:
        df = self._ensure_dates(enriched_reservations)
        daily = expand_reservations_daily(df)
        grp = count_inhouse_by_day_and_archetype(daily)

        # build daily features: counts per archetype + average LOS
        pivot = grp.pivot_table(index="date", columns="archetype", values="guest_nights", aggfunc="sum").fillna(0.0)
        pivot.columns = [f"gn_{c}" for c in pivot.columns]
        # simple feature: total checkouts per day
        dep = df.copy(); dep["date"] = dep["departure_date"].dt.date
        checkouts = dep.groupby("date").reservation_id.count().rename("checkouts")
        X = pivot.join(checkouts, how="left").fillna(0.0)

        y = hk_actual_daily.copy()
        y["date"] = pd.to_datetime(y["date"]).dt.date
        y = y.groupby("date").hk_man_hours.sum()

        X, y = X.align(y, join="inner")
        if len(X) == 0:
            raise ValueError("No overlapping dates for learning.")

        model = LinearRegression()
        model.fit(X.values, y.values)
        coefs = dict(zip(X.columns.tolist(), model.coef_.tolist()))

        # Translate coefficients into a simple multiplier suggestion for archetypes
        # Baseline minutes per guest-night surrogate
        gn_cols = [c for c in X.columns if c.startswith("gn_")]
        mults = {}
        baseline = np.mean([coefs.get(c, 0.0) for c in gn_cols]) or 1.0
        for a in [a.value for a in Archetype]:
            key = f"gn_{a}"
            val = coefs.get(key, baseline)
            mults[a] = float(max(0.5, min(3.0, val / baseline)))
        out = self.config.copy()
        out.setdefault("housekeeping", {})
        out["housekeeping"].setdefault("archetype_multipliers", {})
        out["housekeeping"]["archetype_multipliers"].update(mults)
        return out
