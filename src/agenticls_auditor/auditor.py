from __future__ import annotations

from dataclasses import dataclass

from .mutations import Candidate
from .search.base import FixedMultistagePolicy, Observation


@dataclass
class AuditorState:
    last_validation_query: int = 0
    stagnation: int = 0
    best_margin: float = float("inf")


class ClosedLoopAuditor(FixedMultistagePolicy):
    name = "auditor"

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.state = AuditorState()

    def choose(self, candidates: list[Candidate], history: list[Observation], query_index: int) -> Candidate:
        if history:
            current = min(obs.discovery_margin for obs in history)
            self.state.stagnation = self.state.stagnation + 1 if current >= self.state.best_margin else 0
            self.state.best_margin = min(self.state.best_margin, current)
        # Minimal adaptive switch: restart randomly after three stagnant steps;
        # otherwise exploit. The production auditor should expose every action
        # and stopping decision in its audit trace.
        if self.state.stagnation >= 3:
            self.state.stagnation = 0
            return super(FixedMultistagePolicy, self).choose(candidates, history, query_index)
        return super().choose(candidates, history, query_index)

    def should_validate(self, query_index: int, remaining_allowance: int) -> bool:
        if remaining_allowance <= 0:
            return False
        return self.state.stagnation >= 2 or query_index in {5, 10, 20, 40}

