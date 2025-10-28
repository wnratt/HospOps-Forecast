from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from ..features import expand_reservations_daily, count_inhouse_by_day_and_archetype
from .base import BaseForecaster

@dataclass
class FNBConsumptionForecaster(BaseForecaster):
    def predict(self, enriched_reservations: pd.DataFrame, meal: str = "breakfast") -> pd.DataFrame:
        df = self._ensure_dates(enriched_reservations)
        meals = self.config.get("fnb_meals", {})
        if meal not in meals:
            raise ValueError(f"Meal '{meal}' not configured. Available: {list(meals.keys())}")
        cfg = meals[meal]
        base = cfg.get("base", {})
        mults = cfg.get("multipliers", {})
        items = list(base.keys())

        daily = expand_reservations_daily(df)
        agg = count_inhouse_by_day_and_archetype(daily)

        rows = []
        for date, g in agg.groupby("date"):
            row = {"date": date}
            for _, r in g.iterrows():
                arche = r["archetype"]
                a = r["adults"]; c = r["children"]
                for item in items:
                    b_a = float(base[item]["adult"]); b_c = float(base[item]["child"])
                    m = float(mults.get(arche, {}).get(item, 1.0))
                    row[item] = row.get(item, 0.0) + (a*b_a + c*b_c) * m
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("date")
        return out[["date"] + items]
