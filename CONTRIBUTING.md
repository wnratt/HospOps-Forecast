# Contributing

- Dev setup:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```
- Run: `ruff check .`, `black .`, `mypy hospops_forecast`, `pytest -q`
