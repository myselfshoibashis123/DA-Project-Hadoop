"""Configuration loading and project path helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the YAML configuration file and return it as a dictionary."""
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {cfg_path} is empty or invalid")
    cfg["_config_path"] = str(cfg_path)
    return cfg


def merged(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``cfg`` with ``overrides`` recursively applied."""
    out = copy.deepcopy(cfg)

    def _apply(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _apply(dst[key], value)
            else:
                dst[key] = value

    _apply(out, overrides)
    return out
