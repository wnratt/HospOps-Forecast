from __future__ import annotations
import pandas as pd

REQUIRED_RES_COLS = ["reservation_id","arrival_date","departure_date","adults","children","room_type","channel","company","nationality"]

def check_reservations_basic(df: pd.DataFrame) -> dict:
    issues = []
    for c in REQUIRED_RES_COLS:
        if c not in df.columns:
            issues.append(f"Missing column: {c}")
    if "arrival_date" in df.columns and "departure_date" in df.columns:
        a = pd.to_datetime(df["arrival_date"], errors="coerce")
        d = pd.to_datetime(df["departure_date"], errors="coerce")
        bad = (d <= a).sum()
        if bad > 0:
            issues.append(f"{bad} rows have departure_date <= arrival_date")
    return {"ok": len(issues)==0, "issues": issues}
