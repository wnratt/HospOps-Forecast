from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict
from .base import BaseForecaster
from ..features import expand_reservations_daily, count_inhouse_by_day_and_archetype

@dataclass
class LaborForecaster(BaseForecaster):
    def predict(self, enriched_reservations: pd.DataFrame, shift_hours: float = 8.0, utilization: float = 0.85) -> pd.DataFrame:
        df = self._ensure_dates(enriched_reservations)
        cfg = self.config.get("housekeeping", {})
        min_checkout = float(cfg.get("minutes_per_checkout", 45))
        min_stay = float(cfg.get("minutes_per_stayover", 20))
        mults: Dict[str, float] = cfg.get("archetype_multipliers", {})
        if utilization is None:
            utilization = float(cfg.get("target_utilization", 0.85))

        daily = expand_reservations_daily(df)
        dep = df.copy()
        dep["date"] = dep["departure_date"].dt.date
        checkouts = dep.groupby(["date","archetype"]).reservation_id.count().rename("checkouts").reset_index()
        agg = count_inhouse_by_day_and_archetype(daily)
        out = agg.merge(checkouts, on=["date","archetype"], how="left").fillna({"checkouts":0})
        out["stayovers"] = (out["rooms"] - out["checkouts"]).clip(lower=0)

        def a_mult(a): return float(mults.get(a, 1.0))
        out["hk_minutes"] = (
            out["checkouts"] * min_checkout * out["archetype"].map(a_mult)
            + out["stayovers"] * min_stay * out["archetype"].map(a_mult)
        )
        day = out.groupby("date").agg(
            hk_minutes=("hk_minutes","sum"),
            rooms_total=("rooms_total","max"),
            guest_nights_total=("guest_nights_total","max"),
        ).reset_index()
        day["hk_man_hours"] = day["hk_minutes"] / 60.0
        day["recommended_headcount"] = np.ceil(day["hk_man_hours"] / (shift_hours * utilization)).astype(int).clip(lower=0)
        day["shift_hours"] = shift_hours
        day["utilization"] = utilization
        return day[["date","hk_man_hours","recommended_headcount","shift_hours","utilization","rooms_total","guest_nights_total"]].sort_values("date")
