from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointResult:
    gene_id: str
    model: str
    policy: str
    seed: int
    depth: int
    checkpoint: int
    candidate_sequence: str
    discovery_margin: float
    confirmation_margin: float
    confirmed_severity: float
    candidate_evaluations: int
    validation_actions: int
    raw_model_calls: int
    commit: str = "unknown"


def write_jsonl(path: str | Path, rows: list[CheckpointResult]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")

