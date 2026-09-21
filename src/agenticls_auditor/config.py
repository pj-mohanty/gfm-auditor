from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def max_candidates(self) -> int:
        return int(self.raw["trajectory"]["max_candidates"])

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["trajectory"]["checkpoints"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["seeds"]["values"])

    @property
    def primary_depth(self) -> int:
        return int(self.raw["depths"]["primary"])

    def validate(self) -> None:
        if self.raw["seeds"]["count"] != len(self.seeds):
            raise ValueError("seeds.count does not match seeds.values")
        if self.checkpoints[-1] != self.max_candidates:
            raise ValueError("last checkpoint must equal max_candidates")
        if tuple(sorted(self.checkpoints)) != self.checkpoints:
            raise ValueError("checkpoints must be strictly ordered")

        phases = self.raw["fixed_multistage"]
        expected_start = 1
        total = 0
        for name in ("random_phase", "greedy_phase", "beam_phase"):
            phase = phases[name]
            count = int(phase["end"]) - int(phase["start"]) + 1
            if int(phase["start"]) != expected_start or count != int(phase["count"]):
                raise ValueError(f"invalid inclusive range for {name}")
            expected_start = int(phase["end"]) + 1
            total += count
        if total != self.max_candidates:
            raise ValueError("fixed phases do not cover the full trajectory")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = ExperimentConfig(yaml.safe_load(handle))
    cfg.validate()
    return cfg

