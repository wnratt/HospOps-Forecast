from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict
from .base import BaseForecaster
from .queueing import erlang_c_wait_minutes

def _rel_to_abs_times(base_date: pd.Timestamp, base_time: pd.Timestamp, rel: str) -> pd.Timestamp:
    # rel format: "-HH:MM" relative to base_time on base_date
    hh, mm = rel.replace("-", "").split(":")
    delta = timedelta(hours=int(hh), minutes=int(mm))
    return pd.Timestamp.combine(pd.Timestamp(base_date).date(), pd.to_datetime(base_time).time()) - delta

@dataclass
class AirlineForecaster(BaseForecaster):
    def predict(self, flights: pd.DataFrame, area: str) -> pd.DataFrame:
        area = area.lower().strip()
        cfg = self.config.get("airline", {})
        if area not in cfg:
            raise ValueError(f"Unknown airline area '{area}'. Available: {list(cfg.keys())}")
        acfg = cfg[area]
        pax_per_agent = float(acfg["pax_per_agent_per_hour"])
        target_wait = float(acfg["sla_target_wait_min"])
        dist = acfg["distributions"]

        df = flights.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["gate_time"] = pd.to_datetime(df["gate_time"]).dt.time

        buckets = {}
        for _, r in df.iterrows():
            for rel, w in dist.items():
                ts = _rel_to_abs_times(pd.Timestamp(r["date"]), pd.Timestamp.combine(pd.Timestamp(r["date"]), r["gate_time"]), rel)
                buckets[ts] = buckets.get(ts, 0.0) + float(r["pax_count"]) * float(w)

        if not buckets:
            return pd.DataFrame(columns=["datetime","expected_pax","recommended_staff"])
        out = pd.DataFrame({"datetime": list(buckets.keys()), "expected_pax": list(buckets.values())}).sort_values("datetime")
        # SLA staffing via Erlang C
        rec = []
        for lam in out["expected_pax"]:
            if lam <= 0: rec.append(0); continue
            mu = pax_per_agent
            c = 1
            while c < 500:
                wq = erlang_c_wait_minutes(lam, mu, c)
                rho = lam/(c*mu)
                if (wq <= target_wait) and (rho < 0.9):
                    break
                c += 1
            rec.append(c)
        out["recommended_staff"] = rec
        out["area"] = area
        return out
