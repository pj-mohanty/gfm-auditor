from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denominator)


def median_control_margin(source: np.ndarray, candidate: np.ndarray, controls: list[np.ndarray]) -> float:
    candidate_distance = cosine_distance(source, candidate)
    control_distances = [cosine_distance(source, control) for control in controls]
    return float(np.median(control_distances) - candidate_distance)


def confirmed_severity(confirmation_margin: float, delta: float) -> float:
    return max(0.0, -confirmation_margin - delta)


def trapezoidal_audc(checkpoints: list[int], severities: list[float], normalize: bool = True) -> float:
    if len(checkpoints) != len(severities) or len(checkpoints) < 2:
        raise ValueError("AUDC requires matching checkpoint and severity arrays")
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    area = float(trapezoid(severities, checkpoints))
    return area / (checkpoints[-1] - checkpoints[0]) if normalize else area
