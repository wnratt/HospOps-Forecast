from __future__ import annotations
from datetime import timedelta
import numpy as np
import pandas as pd

def ensure_datetime(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")
    return s

def expand_reservations_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["arrival_date"] = ensure_datetime(df["arrival_date"]).dt.date
    df["departure_date"] = ensure_datetime(df["departure_date"]).dt.date
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["arrival_date"]) or pd.isna(r["departure_date"]):
            continue
        d = r["arrival_date"]
        while d < r["departure_date"]:
            rr = r.to_dict()
            rr["date"] = d
            rows.append(rr)
            d = d + timedelta(days=1)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def count_inhouse_by_day_and_archetype(daily: pd.DataFrame) -> pd.DataFrame:
    grp = daily.groupby(["date","archetype"]).agg(
        rooms=("reservation_id","nunique"),
        adults=("adults","sum"),
        children=("children","sum"),
        guest_nights=("reservation_id","count"),
    ).reset_index()
    totals = grp.groupby("date").agg(
        rooms_total=("rooms","sum"),
        adults_total=("adults","sum"),
        children_total=("children","sum"),
        guest_nights_total=("guest_nights","sum"),
    ).reset_index()
    return grp.merge(totals, on="date", how="left")

def arrivals_by_day_and_archetype(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"]).dt.date
    grp = df.groupby(["arrival_date","archetype"]).agg(
        arrivals=("reservation_id","count"),
        arriving_adults=("adults","sum"),
        arriving_children=("children","sum"),
    ).reset_index().rename(columns={"arrival_date":"date"})
    return grp
