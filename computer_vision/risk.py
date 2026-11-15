"""Rule-based risk scoring and Low / Medium / High classification.

The score is a transparent weighted sum of the conditions observed for one worker in one
frame (weights in config.yaml -> risk.weights). It is a heuristic designed for an
academic prototype, NOT a validated occupational-safety risk model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

RISK_LEVELS = ["None", "Low", "Medium", "High"]


def risk_score(conditions: Iterable[str], cfg: Dict[str, Any]) -> int:
    """Sum the configured weight of every condition (unknown conditions weigh 1)."""
    weights = cfg["risk"]["weights"]
    return int(sum(int(weights.get(c, 1)) for c in conditions))


def classify(score: int, cfg: Dict[str, Any]) -> str:
    th = cfg["risk"]["thresholds"]
    if score >= th["high"]:
        return "High"
    if score >= th["medium"]:
        return "Medium"
    if score >= 1:
        return "Low"
    return "None"


def assess(conditions: Iterable[str], cfg: Dict[str, Any]) -> Tuple[int, str]:
    conds = list(conditions)
    score = risk_score(conds, cfg)
    return score, classify(score, cfg)
