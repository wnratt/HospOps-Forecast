from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import time
from typing import Dict
from .base import BaseForecaster
from ..features import arrivals_by_day_and_archetype, expand_reservations_daily, count_inhouse_by_day_and_archetype
from .queueing import erlang_c_wait_minutes

def _hours_to_bins(distribution: Dict[str, float], date: pd.Timestamp) -> Dict[pd.Timestamp, float]:
    out = {}
    base_date = pd.Timestamp(date).date()
    for hhmm, w in distribution.items():
        h, m = hhmm.split(":")
        ts = pd.Timestamp.combine(base_date, time(int(h), int(m)))
        out[ts] = float(w)
    s = sum(out.values())
    if s > 0: out = {k: v / s for k, v in out.items()}
    return out

@dataclass
class ServiceLoadForecaster(BaseForecaster):
    def predict(self, enriched_reservations: pd.DataFrame, area: str, utilization: float = 0.85, target_wait_min: float | None = None) -> pd.DataFrame:
        df = self._ensure_dates(enriched_reservations)
        area = area.lower().strip()
        if area not in {"reception","breakfast"}:
            raise ValueError("area must be 'reception' or 'breakfast'")
        if area == "reception":
            return self._reception(df, utilization, target_wait_min)
        else:
            return self._breakfast(df, utilization, target_wait_min)

    def _reception(self, df: pd.DataFrame, utilization: float, target_wait_min: float | None) -> pd.DataFrame:
        cfg = self.config["service_load"]["reception"]
        t_per_agent = float(cfg["transactions_per_agent_per_hour"])
        util = utilization if utilization is not None else float(cfg.get("utilization", 0.85))
        dist_cfg = cfg["distributions"]
        if target_wait_min is None:
            target_wait_min = float(self.config.get("service_sla", {}).get("default_target_wait_min", 5.0))

        arrivals = arrivals_by_day_and_archetype(df)
        frames = []
        for date, g in arrivals.groupby("date"):
            hourly = {}
            for _, r in g.iterrows():
                arche = r["archetype"]; n = r["arrivals"]
                dist = _hours_to_bins(dist_cfg.get(arche, dist_cfg["Other"]), pd.Timestamp(date))
                for ts, w in dist.items():
                    hourly[ts] = hourly.get(ts, 0.0) + n * w
            if not hourly: continue
            tmp = pd.DataFrame({"datetime": list(hourly.keys()), "expected_transactions": list(hourly.values())}).sort_values("datetime")
            tmp["area"] = "reception"
            # Two staffing modes
            # Utilization-based:
            tmp["staff_util"] = np.ceil(tmp["expected_transactions"] / (t_per_agent * util)).astype(int).clip(lower=1)
            # SLA-based (Erlang C); solve minimal c s.t. E[Wq] <= target
            staff_sla = []
            for x in tmp["expected_transactions"]:
                lam = x  # arrivals per hour
                mu = t_per_agent  # service rate per agent per hour
                c = 1
                if lam == 0:
                    staff_sla.append(0)
                    continue
                while c < 200:
                    wq = erlang_c_wait_minutes(lam, mu, c)
                    rho = lam/(c*mu)
                    if (wq <= target_wait_min) and (rho < util):
                        break
                    c += 1
                staff_sla.append(c)
            tmp["staff_sla"] = staff_sla
            tmp["recommended_staff"] = np.maximum(tmp["staff_util"], tmp["staff_sla"]).astype(int)
            tmp["load_status"] = np.where(
                tmp["expected_transactions"] <= t_per_agent * util * tmp["recommended_staff"] * 0.7, "Green",
                np.where(tmp["expected_transactions"] <= t_per_agent * util * tmp["recommended_staff"], "Amber","Red")
            )
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime","expected_transactions","area","recommended_staff","load_status"])

    def _breakfast(self, df: pd.DataFrame, utilization: float, target_wait_min: float | None) -> pd.DataFrame:
        cfg = self.config["service_load"]["breakfast"]
        covers_per_staff = float(cfg["covers_per_staff_per_hour"])
        util = utilization if utilization is not None else float(cfg.get("utilization", 0.85))
        dist_cfg = cfg["distributions"]
        if target_wait_min is None:
            target_wait_min = float(self.config.get("service_sla", {}).get("default_target_wait_min", 5.0))

        daily = expand_reservations_daily(df)
        agg = count_inhouse_by_day_and_archetype(daily)

        frames = []
        for date, g in agg.groupby("date"):
            hourly = {}
            for _, r in g.iterrows():
                arche = r["archetype"]
                covers = r["adults"] + r["children"]
                dist = _hours_to_bins(dist_cfg.get(arche, dist_cfg["Other"]), pd.Timestamp(date))
                for ts, w in dist.items():
                    hourly[ts] = hourly.get(ts, 0.0) + covers * w
            if not hourly: continue
            tmp = pd.DataFrame({"datetime": list(hourly.keys()), "expected_covers": list(hourly.values())}).sort_values("datetime")
            tmp["area"] = "breakfast"
            tmp["staff_util"] = np.ceil(tmp["expected_covers"] / (covers_per_staff * util)).astype(int).clip(lower=1)
            # SLA using Erlang C on covers as "transactions"
            staff_sla = []
            for x in tmp["expected_covers"]:
                lam = x
                mu = covers_per_staff
                c = 1
                if lam == 0:
                    staff_sla.append(0); continue
                while c < 200:
                    wq = erlang_c_wait_minutes(lam, mu, c)
                    rho = lam/(c*mu)
                    if (wq <= target_wait_min) and (rho < util):
                        break
                    c += 1
                staff_sla.append(c)
            tmp["staff_sla"] = staff_sla
            tmp["recommended_staff"] = np.maximum(tmp["staff_util"], tmp["staff_sla"]).astype(int)
            tmp["load_status"] = np.where(
                tmp["expected_covers"] <= covers_per_staff * util * tmp["recommended_staff"] * 0.7, "Green",
                np.where(tmp["expected_covers"] <= covers_per_staff * util * tmp["recommended_staff"], "Amber","Red")
            )
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime","expected_covers","area","recommended_staff","load_status"])
