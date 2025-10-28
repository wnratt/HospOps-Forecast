from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

from .archetypes import Archetype

@dataclass
class Segmenter:
    use_unsupervised: bool = False
    random_state: int = 42
    n_clusters: int = 4

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def pick(*names, default=None):
            for n in names:
                if n in df.columns: return n
            return default

        c_res = pick("reservation_id","rezervasyon_id")
        c_arr = pick("arrival_date","geliş tarihi","gelis_tarihi","giris_tarihi","arrival_date")
        c_dep = pick("departure_date","gidiş tarihi","gidis_tarihi","cikis_tarihi","departure_date")
        c_ad = pick("adults","yetişkin","yetiskin","adults")
        c_ch = pick("children","çocuk","cocuk","children")
        c_room = pick("room_type","oda_tipi","room_type")
        c_channel = pick("channel","kanal","channel")
        c_company = pick("company","şirket","company")
        c_nat = pick("nationality","milliyet","nationality")

        for col in [c_arr,c_dep,c_ad,c_ch,c_room,c_channel,c_company,c_nat]:
            if col not in df.columns: df[col] = np.nan

        df["arrival_date"] = pd.to_datetime(df[c_arr], errors="coerce")
        df["departure_date"] = pd.to_datetime(df[c_dep], errors="coerce")
        df["adults"] = pd.to_numeric(df[c_ad], errors="coerce").fillna(1).astype(int)
        df["children"] = pd.to_numeric(df[c_ch], errors="coerce").fillna(0).astype(int)
        df["group_size"] = df["adults"] + df["children"]
        df["room_type"] = df[c_room].astype(str)
        df["channel"] = df[c_channel].astype(str)
        df["company"] = df[c_company].astype(str)
        df["nationality"] = df[c_nat].astype(str)
        df["length_of_stay"] = (df["departure_date"] - df["arrival_date"]).dt.days.clip(lower=0)
        df["arrival_weekday"] = df["arrival_date"].dt.weekday
        df["is_weekend_arrival"] = df["arrival_weekday"].isin([4,5,6])

        df["archetype"] = df.apply(self._rule_based, axis=1)
        df = self._detect_tour_groups(df)

        if self.use_unsupervised and KMeans is not None:
            mask = df["archetype"] == Archetype.Other.value
            feat = ["length_of_stay","group_size","arrival_weekday","adults","children"]
            X = df.loc[mask, feat].fillna(0).to_numpy()
            if len(X) >= self.n_clusters:
                km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
                labels = km.fit_predict(X)
                # naive mapping
                cmap = {}
                for i in range(self.n_clusters):
                    gi = X[labels == i]
                    if len(gi)==0: cmap[i]=Archetype.Other.value; continue
                    avg_group = gi[:, feat.index("group_size")].mean()
                    avg_los = gi[:, feat.index("length_of_stay")].mean()
                    if avg_group>=3: cmap[i]=Archetype.FamilyWithKids.value
                    elif avg_los<=2: cmap[i]=Archetype.SoloBusiness.value
                    else: cmap[i]=Archetype.LeisureCouple.value
                df.loc[mask,"archetype"] = [cmap[l] for l in labels]

        if c_res and c_res in df.columns and not df[c_res].isna().all():
            df["reservation_id"] = df[c_res].astype(str)
        else:
            df["reservation_id"] = [f"SYN-{i:06d}" for i in range(len(df))]

        keep = ["reservation_id","arrival_date","departure_date","adults","children","group_size",
                "room_type","channel","company","nationality","length_of_stay",
                "arrival_weekday","is_weekend_arrival","archetype"]
        return df[keep]

    @staticmethod
    def _rule_based(r: pd.Series) -> str:
        adults, children = int(r.get("adults",1)), int(r.get("children",0))
        group = adults + children
        los = int(r.get("length_of_stay",1))
        weekday = int(r.get("arrival_weekday",0))
        is_weekend = weekday in [4,5,6]
        channel = str(r.get("channel","")).lower()
        company = str(r.get("company","")).lower()
        room_type = str(r.get("room_type","")).lower()
        if adults==1 and children==0 and los<=3 and weekday in range(0,5) and ("corp" in channel or company not in ["","nan"]):
            return Archetype.SoloBusiness.value
        if adults==2 and children==0 and (is_weekend or los>=2) and any(k in channel for k in ["ota","direct",""]):
            return Archetype.LeisureCouple.value
        if children>=1 or group>=3 or any(k in room_type for k in ["suite","family"]):
            return Archetype.FamilyWithKids.value
        return Archetype.Other.value

    @staticmethod
    def _detect_tour_groups(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        key = df["company"].replace({"nan":"", "None":""}) + "|" + df["channel"].fillna("")
        df["__k"] = key
        grp = df.groupby(["arrival_date","__k"]).reservation_id.transform("count")
        df.loc[grp >= 10, "archetype"] = Archetype.TourGroup.value
        df.drop(columns=["__k"], inplace=True)
        return df
