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
    pending_validation_mode: str | None = None
    validation_feedback: list[dict] = field(default_factory=list)
    validation_margins: dict[str, float] = field(default_factory=dict)


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

        if self.state.pending_validation_mode == "explore":
            candidate = RandomPolicy.choose(
                self, candidates, history, query_index,
            )
            action = "validation_restart"
            self.state.pending_validation_mode = None
            self.state.stagnation = 0
        elif self.state.pending_validation_mode == "exploit":
            candidate = FixedMultistagePolicy.choose(
                self, candidates, history, query_index,
            )
            action = FixedMultistagePolicy.action_name(
                self, query_index,
            )
            self.state.pending_validation_mode = None
            self.state.stagnation = 0
        elif self.state.stagnation >= 3:
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

    def observe_validation(
        self,
        candidate: Candidate,
        discovery_margin: float,
        validation_margin: float,
    ) -> None:
        """Explore another position group after validation weakens a claim.

        Lower margins represent stronger violations. This rule is fixed
        before the prospective run; confirmation scores never reach it.
        """
        disagrees = validation_margin > discovery_margin
        self.state.pending_validation_mode = (
            "explore" if disagrees else "exploit"
        )
        self.state.validation_feedback.append(
            {
                "candidate_sequence": candidate.sequence,
                "discovery_margin": discovery_margin,
                "validation_margin": validation_margin,
                "restart_next_query": disagrees,
            }
        )
        self.state.validation_margins[candidate.sequence] = validation_margin

    def nominate_incumbent(
        self,
        history: list[Observation],
    ) -> Observation:
        """Select the strongest validated evaluated candidate, if any."""
        validated = [
            observation for observation in history
            if observation.candidate.sequence in self.state.validation_margins
        ]
        if validated:
            return min(
                validated,
                key=lambda observation: (
                    self.state.validation_margins[
                        observation.candidate.sequence
                    ],
                    observation.discovery_margin,
                    observation.candidate.sequence,
                ),
            )
        return min(
            history,
            key=lambda observation: (
                observation.discovery_margin,
                observation.candidate.sequence,
            ),
        )

    def choose_validation_target(
        self,
        history: list[Observation],
    ) -> Observation:
        """Validate the best discovery candidate not already validated."""
        remaining = [
            observation for observation in history
            if observation.candidate.sequence not in self.state.validation_margins
        ]
        pool = remaining if remaining else history
        return min(
            pool,
            key=lambda observation: (
                observation.discovery_margin,
                observation.candidate.sequence,
            ),
        )

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
