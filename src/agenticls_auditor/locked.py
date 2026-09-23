from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml


def _require_hash(
    value: Any,
    *,
    field: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
    ):
        raise ValueError(
            f"{field} must be a 64-character SHA256"
        )

    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{field} must be hexadecimal"
        ) from error

    return value


def load_locked_config(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "locked configuration must be a mapping"
        )

    if config.get("frozen") is not True:
        raise ValueError(
            "locked configuration is not frozen"
        )

    if config.get("status") != "locked_primary":
        raise ValueError(
            "locked configuration has invalid status"
        )

    calibration = config.get("calibration")

    if not isinstance(calibration, dict):
        raise ValueError(
            "calibration block is required"
        )

    delta = calibration.get("delta")

    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not isfinite(float(delta))
        or float(delta) < 0
    ):
        raise ValueError(
            "calibration delta must be a finite "
            "non-negative number"
        )

    quantile = calibration.get("quantile")

    if (
        isinstance(quantile, bool)
        or not isinstance(
            quantile,
            (int, float),
        )
        or float(quantile) != 0.95
    ):
        raise ValueError(
            "calibration quantile must equal 0.95"
        )

    trajectory = config.get("trajectory")

    if not isinstance(trajectory, dict):
        raise ValueError(
            "trajectory block is required"
        )

    checkpoints = tuple(
        trajectory.get("checkpoints", ())
    )

    if checkpoints != (5, 10, 20, 40):
        raise ValueError(
            "locked checkpoints must be "
            "5, 10, 20, and 40"
        )

    seeds = tuple(
        trajectory.get("seeds", ())
    )

    if seeds != (11, 22, 33, 44, 55):
        raise ValueError(
            "locked seeds do not match "
            "the frozen seed schedule"
        )

    policies = tuple(
        trajectory.get("policies", ())
    )

    if policies != (
        "random",
        "fixed_multistage",
        "auditor",
    ):
        raise ValueError(
            "locked policies do not match "
            "the frozen policy set"
        )

    locked_split = config.get(
        "locked_split"
    )

    if not isinstance(locked_split, dict):
        raise ValueError(
            "locked_split block is required"
        )

    if int(
        locked_split.get("gene_count", -1)
    ) != 50:
        raise ValueError(
            "locked split must contain 50 genes"
        )

    if int(
        locked_split.get(
            "candidates_per_gene",
            -1,
        )
    ) != 40:
        raise ValueError(
            "locked split must contain "
            "40 candidates per gene"
        )

    _require_hash(
        locked_split.get("source_sha256"),
        field="locked_split.source_sha256",
    )

    _require_hash(
        locked_split.get("manifest_sha256"),
        field="locked_split.manifest_sha256",
    )

    _require_hash(
        calibration.get(
            "development_source_sha256"
        ),
        field=(
            "calibration."
            "development_source_sha256"
        ),
    )

    _require_hash(
        calibration.get(
            "development_manifest_sha256"
        ),
        field=(
            "calibration."
            "development_manifest_sha256"
        ),
    )

    return config
