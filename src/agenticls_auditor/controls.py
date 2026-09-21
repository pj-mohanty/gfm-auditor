from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import floor


@dataclass(frozen=True)
class ControlBanks:
    discovery: tuple[str, ...]
    validation: tuple[str, ...]
    confirmation: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return len(self.discovery) >= 4 and len(self.validation) >= 4 and len(self.confirmation) >= 8


def partition_controls(
    controls: list[str], *, source_id: str, candidate_sequence: str, assignment_seed: int
) -> ControlBanks:
    unique = sorted(set(controls))
    candidate_hash = sha256(candidate_sequence.encode()).hexdigest()

    def key(control: str) -> str:
        payload = f"{assignment_seed}|{source_id}|{candidate_hash}|{control}"
        return sha256(payload.encode()).hexdigest()

    ordered = sorted(unique, key=key)
    n = len(ordered)
    n_discovery = floor(0.25 * n)
    n_validation = floor(0.25 * n)
    return ControlBanks(
        discovery=tuple(ordered[:n_discovery]),
        validation=tuple(ordered[n_discovery : n_discovery + n_validation]),
        confirmation=tuple(ordered[n_discovery + n_validation :]),
    )

