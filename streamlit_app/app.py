import pandas as pd
import streamlit as st
from hospops_forecast.segmentation.segmenter import Segmenter
from hospops_forecast.models.labor import LaborForecaster
from hospops_forecast.models.fnb import FNBConsumptionForecaster
from hospops_forecast.models.service import ServiceLoadForecaster
from hospops_forecast.models.departments import DepartmentForecaster

st.set_page_config(page_title="HospOps-Forecast Demo", layout="wide")
st.title("🏨 HospOps-Forecast — v0.3.0")

with st.sidebar:
    st.header("Upload Reservations CSV")
    uploaded = st.file_uploader("CSV", type=["csv"])
    use_unsup = st.checkbox("Use optional KMeans", value=False)
    st.header("Options")
    meal = st.selectbox("Meal for F&B", ["breakfast","lunch","dinner"])
    target_wait = st.number_input("SLA target wait (min, service)", min_value=0.0, value=5.0, step=0.5)
    dept = st.selectbox("Department", ["spa","concierge","valet","engineering"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Raw Reservations")
    st.dataframe(df.head(50))

    seg = Segmenter(use_unsupervised=use_unsup)
    enriched = seg.enrich(df)

    st.subheader("Enriched with Archetypes")
    st.dataframe(enriched.head(50))

    lf = LaborForecaster()
    hk = lf.predict(enriched)
    st.subheader("Housekeeping — Daily Forecast")
    st.dataframe(hk)

    fb = FNBConsumptionForecaster()
    fnb = fb.predict(enriched, meal=meal)
    st.subheader(f"F&B — {meal.title()} Consumption Forecast")
    st.dataframe(fnb)

    svc = ServiceLoadForecaster()
    rec = svc.predict(enriched, area="reception", target_wait_min=target_wait)
    st.subheader("Reception — Hourly Load (SLA-based staffing)")
    st.dataframe(rec)

    dfore = DepartmentForecaster()
    dtab = dfore.predict(enriched, dept=dept)
    st.subheader(f"Department — {dept.title()} Workload")
    st.dataframe(dtab)

else:
    st.info("Upload a CSV to begin. See examples under `examples/data/`.")
