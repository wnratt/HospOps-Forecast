from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from .schema import AppConfig

def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        p = Path(__file__).parent / "base.yaml"
    else:
        p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = AppConfig(**data)  # validate
    return cfg.model_dump()
