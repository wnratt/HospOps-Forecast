from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..features import expand_reservations_daily
from ..segmentation.archetypes import Archetype
from ..models.base import BaseForecaster

@dataclass
class LaborCalibrator(BaseForecaster):
    def _daily_decomposition(self, enriched: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_dates(enriched)
        daily = expand_reservations_daily(df)
        dep = df.copy(); dep["date"] = dep["departure_date"].dt.date
        checkouts = dep.groupby(["date","archetype"]).reservation_id.count().rename("checkouts").reset_index()
        inhouse = daily.groupby(["date","archetype"]).reservation_id.nunique().rename("rooms").reset_index()
        base = inhouse.merge(checkouts, on=["date","archetype"], how="left").fillna({"checkouts":0})
        base["stayovers"] = (base["rooms"] - base["checkouts"]).clip(lower=0)
        return base

    def fit_multipliers(self, enriched_reservations: pd.DataFrame, actuals: pd.DataFrame, min_mult: float = 0.5, max_mult: float = 3.0):
        cfg = self.config.get("housekeeping", {})
        min_checkout = float(cfg.get("minutes_per_checkout", 45))
        min_stay = float(cfg.get("minutes_per_stayover", 20))

        base = self._daily_decomposition(enriched_reservations)
        base["coeff"] = (base["checkouts"] * min_checkout + base["stayovers"] * min_stay) / 60.0
        Xw = base.pivot_table(index="date", columns="archetype", values="coeff", aggfunc="sum").fillna(0.0)
        y = actuals.copy(); y["date"] = pd.to_datetime(y["date"]).dt.date
        y = y.groupby("date").hk_man_hours.sum()
        X, y_vec = Xw.align(y, join="inner")
        if len(X) == 0: raise ValueError("No overlapping dates.")

        A = X.to_numpy(); b = y_vec.to_numpy()
        lam = 1e-3
        ATA = A.T @ A + lam * np.eye(A.shape[1]); ATb = A.T @ b
        m = np.linalg.solve(ATA, ATb)
        m = np.clip(m, min_mult, max_mult)

        out = self.config.copy()
        mults = {col: float(val) for col, val in zip(X.columns.tolist(), m)}
        for a in [a.value for a in Archetype]:
            mults.setdefault(a, float(cfg.get("archetype_multipliers", {}).get(a, 1.0)))
        out.setdefault("housekeeping", {})
        out["housekeeping"].setdefault("archetype_multipliers", {})
        out["housekeeping"]["archetype_multipliers"].update(mults)
        return out
