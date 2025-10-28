# Architecture (v0.3.0)

- `segmentation/` → archetypes & enrichment
- `models/`
  - `labor.py` → housekeeping
  - `fnb.py` → multi-meal F&B forecast
  - `service.py` → service load + SLA (Erlang C)
  - `departments.py` → generic dept forecaster (Spa/Concierge/Valet/Engineering)
  - `airline.py` → hourly staffing for gate/boarding/lounge from flights CSV
  - `learning.py` → regression-assisted HK tuning
- `calibration/` → labor calibrator (LS)
- `dq/` → data quality checks (light + GE optional)
- `api/` → FastAPI app
- `config/` → defaults + schema + loader
