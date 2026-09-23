from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class ConfirmationBankReference:
    """Non-sensitive reference to a frozen hidden-control bank."""

    bank_id: str
    control_count: int

    def __post_init__(self) -> None:
        if not self.bank_id.strip():
            raise ValueError(
                "confirmation bank ID cannot be empty"
            )
        if self.control_count < 1:
            raise ValueError(
                "confirmation control count must be positive"
            )


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
    delta: float = 0.0
    confirmation_bank_id: str = "unknown"
    confirmation_control_count: int = 0
    commit: str = "unknown"
    schema_version: int = 1


@dataclass(frozen=True)
class TrajectorySummary:
    gene_id: str
    model: str
    policy: str
    seed: int
    depth: int
    normalized_confirmed_audc: float
    checkpoint_count: int
    commit: str = "unknown"
    schema_version: int = 1


ResultRow: TypeAlias = (
    CheckpointResult | TrajectorySummary
)


def write_jsonl(
    path: str | Path,
    rows: list[ResultRow],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    asdict(row),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
