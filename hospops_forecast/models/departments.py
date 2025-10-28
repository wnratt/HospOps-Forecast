from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from ..features import expand_reservations_daily, count_inhouse_by_day_and_archetype
from .base import BaseForecaster

@dataclass
class DepartmentForecaster(BaseForecaster):
    def predict(self, enriched_reservations: pd.DataFrame, dept: str) -> pd.DataFrame:
        df = self._ensure_dates(enriched_reservations)
        dept = dept.lower().strip()
        cfg = self.config.get("departments", {})
        if dept not in cfg:
            raise ValueError(f"Unknown department '{dept}'. Available: {list(cfg.keys())}")
        dcfg = cfg[dept]

        daily = expand_reservations_daily(df)
        agg = count_inhouse_by_day_and_archetype(daily)

        results = []
        for date, g in agg.groupby("date"):
            row = {"date": date}
            if dept == "spa":
                m_per_treat = float(dcfg["minutes_per_treatment"])
                util = float(dcfg.get("utilization", 0.8))
                total_minutes = 0.0
                for _, r in g.iterrows():
                    rate = float(dcfg["treatments_per_guest_day"].get(r["archetype"], 0.03))
                    total_minutes += (r["adults"] + r["children"]) * rate * m_per_treat
                staff_hours = total_minutes / 60.0
                headcount = int(np.ceil(staff_hours / (8 * util)))
                row.update(work_minutes=total_minutes, staff_hours=staff_hours, recommended_headcount=headcount)
            elif dept == "concierge":
                m_per_req = float(dcfg["minutes_per_request"])
                util = float(dcfg.get("utilization", 0.85))
                total_minutes = 0.0
                for _, r in g.iterrows():
                    rate = float(dcfg["requests_per_guest_day"].get(r["archetype"], 0.1))
                    total_minutes += (r["adults"] + r["children"]) * rate * m_per_req
                staff_hours = total_minutes / 60.0
                headcount = int(np.ceil(staff_hours / (8 * util)))
                row.update(work_minutes=total_minutes, staff_hours=staff_hours, recommended_headcount=headcount)
            elif dept == "valet":
                m_per_tx = float(dcfg["minutes_per_transaction"])
                util = float(dcfg.get("utilization", 0.85))
                cars = 0.0
                for _, r in g.iterrows():
                    rate = float(dcfg["cars_per_room_day"].get(r["archetype"], 0.15))
                    cars += r["rooms"] * rate
                total_minutes = cars * m_per_tx
                staff_hours = total_minutes / 60.0
                headcount = int(np.ceil(staff_hours / (8 * util)))
                row.update(transactions=cars, work_minutes=total_minutes, staff_hours=staff_hours, recommended_headcount=headcount)
            elif dept == "engineering":
                tickets_per_room_day = float(dcfg["tickets_per_room_day"])
                m_per_ticket = float(dcfg["minutes_per_ticket"])
                util = float(dcfg.get("utilization", 0.8))
                tickets = g["rooms"].sum() * tickets_per_room_day
                total_minutes = tickets * m_per_ticket
                staff_hours = total_minutes / 60.0
                headcount = int(np.ceil(staff_hours / (8 * util)))
                row.update(tickets=tickets, work_minutes=total_minutes, staff_hours=staff_hours, recommended_headcount=headcount)
            results.append(row)
        return pd.DataFrame(results).sort_values("date")
