from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import yaml

@dataclass
class BaseForecaster:
    config: Optional[dict] = None
    def __post_init__(self):
        if self.config is None:
            from pathlib import Path
            cfg_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

    @staticmethod
    def _ensure_dates(df: pd.DataFrame, cols=("arrival_date","departure_date")) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
        return out
