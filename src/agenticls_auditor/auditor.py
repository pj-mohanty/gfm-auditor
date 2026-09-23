from __future__ import annotations

from dataclasses import dataclass, field

from .mutations import Candidate
from .search.base import (
    FixedMultistagePolicy,
    Observation,
    RandomPolicy,
)


@dataclass
class AuditorState:
    last_validation_query: int = 0
    stagnation: int = 0
    best_margin: float = float("inf")
    last_action: str = "uninitialized"
    decisions: list[dict] = field(default_factory=list)


class ClosedLoopAuditor(FixedMultistagePolicy):
    name = "auditor"

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.state = AuditorState()

    def choose(
        self,
        candidates: list[Candidate],
        history: list[Observation],
        query_index: int,
    ) -> Candidate:
        if history:
            current_best = min(
                observation.discovery_margin
                for observation in history
            )
            if current_best < self.state.best_margin:
                self.state.stagnation = 0
                self.state.best_margin = current_best
            else:
                self.state.stagnation += 1

        if self.state.stagnation >= 3:
            candidate = RandomPolicy.choose(
                self,
                candidates,
                history,
                query_index,
            )
            action = "random_restart"
            self.state.stagnation = 0
        else:
            candidate = FixedMultistagePolicy.choose(
                self,
                candidates,
                history,
                query_index,
            )
            action = FixedMultistagePolicy.action_name(
                self,
                query_index,
            )

        self.state.last_action = action
        self.state.decisions.append(
            {
                "query_index": query_index,
                "action": action,
                "candidate_sequence": candidate.sequence,
                "best_margin_before_query": self.state.best_margin,
                "stagnation": self.state.stagnation,
            }
        )
        return candidate

    def action_name(self, query_index: int) -> str:
        return self.state.last_action

    def should_validate(
        self,
        query_index: int,
        remaining_allowance: int,
    ) -> bool:
        if remaining_allowance <= 0:
            return False

        should_validate = (
            self.state.stagnation >= 2
            or query_index in {5, 10, 20, 40}
        )
        if should_validate:
            self.state.last_validation_query = query_index
        return should_validate
