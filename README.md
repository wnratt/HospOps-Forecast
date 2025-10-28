# HospOps-Forecast — Operational Demand & Workforce Forecasting for Hospitality

**HospOps-Forecast** forecasts **operational** load (labor, F&B, service points) by
understanding **who** the guests are — not just **how many**. It assigns **behavioral archetypes**
to reservations and converts them into department-level forecasts.

> 75% occupancy of **Solo Business** travelers is not the same as 75% of **Families with kids**.

---

## 🚀 What's new in v0.3.0

- **Departments**: Spa / Concierge / Valet / Engineering forecasters
- **Advanced F&B**: multi-meal (breakfast/lunch/dinner) menus via **item registry**
- **Airline mode**: Gate / Boarding / Lounge hourly load & staffing
- **SLA staffing**: compute staff to hit **target wait** with an Erlang C queue model
- **Learning**: regression-based learner from **actual logs** (HK man-hours)
- **REST API** (FastAPI): `uvicorn hospops_forecast.api.app:app`
- **Data Quality**: lightweight checks + optional **Great Expectations**

---

## 🧱 Install

> Python 3.9–3.12

```bash
pip install -U pip
pip install -e ".[dev]"        # dev (lint/tests)
# or
pip install -e .               # normal use

# Optional extras
pip install -e ".[app]"        # Streamlit demo
pip install -e ".[api]"        # FastAPI server
pip install -e ".[dq]"         # Great Expectations integration
pip install -e ".[timeseries]" # Prophet extras (optional)
```

---

## 🗂️ Data

**Reservations CSV** (as before): `reservation_id, arrival_date, departure_date (exclusive), adults, children, room_type, channel, company, nationality`

**Flights CSV (Airline mode)** minimal:
```
flight_id, date, gate_time, pax_count, mix_business_share
TK123, 2025-11-03, 15:00, 180, 0.35
```
> You can extend with `bank_id`, `destination_region`, etc.

---

## ⚡ CLI Quickstart

```bash
# Segment & enrich
hospops-forecast segment \
  --input examples/data/sample_reservations.csv \
  --output out/enriched.csv

# Housekeeping (utilization-based)
hospops-forecast forecast labor \
  --enriched out/enriched.csv \
  --output out/hk.csv

# F&B (choose meal)
hospops-forecast forecast fnb \
  --enriched out/enriched.csv \
  --meal breakfast \
  --output out/fnb_breakfast.csv

# Service load (SLA: 5 min target wait @ reception)
hospops-forecast forecast service \
  --enriched out/enriched.csv \
  --area reception \
  --target-wait-min 5 \
  --output out/reception_sla.csv

# Departments (e.g., Spa)
hospops-forecast forecast dept \
  --enriched out/enriched.csv \
  --dept spa \
  --output out/spa_workload.csv

# Airline mode (boarding)
hospops-forecast forecast airline \
  --flights examples/data/sample_flights.csv \
  --area boarding \
  --output out/airline_boarding.csv

# Learning from actuals (HK regression-assisted tuning)
hospops-forecast learn hk \
  --enriched out/enriched.csv \
  --actual examples/data/sample_hk_actual.csv \
  --output out/learned.yaml

# Data Quality
hospops-forecast dq check \
  --input examples/data/sample_reservations.csv
```

---

## 📐 Configuration

- See `hospops_forecast/config/base.yaml` for defaults
- Validated by Pydantic (`AppConfig`) — bad YAML fails fast
- Override per hotel/airline via `--config my.yaml`

New sections:
- `fnb_meals`: meals and item registry
- `departments`: spa/concierge/valet/engineering drivers
- `service_sla`: defaults for SLA targets
- `airline`: distributions & capacities

---

## 🖥️ API (FastAPI)

```bash
uvicorn hospops_forecast.api.app:app --reload
```
Endpoints:
- `POST /segment`
- `POST /forecast/labor`
- `POST /forecast/fnb?meal=breakfast`
- `POST /forecast/service?area=reception&target_wait_min=5`
- `POST /forecast/dept?dept=spa`
- `POST /forecast/airline?area=boarding`
- `POST /calibrate/labor`
- `POST /dq/check`

---

## 🧪 Tests & CI

- `pytest`, `ruff`, `black`, `mypy` via GitHub Actions.
- See `.github/workflows/ci.yml`

---

## 📦 Version & License
- Version: `0.3.0`
- License: MIT
