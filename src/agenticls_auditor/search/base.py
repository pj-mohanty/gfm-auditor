from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import random

from ..mutations import Candidate


@dataclass(frozen=True)
class Observation:
    candidate: Candidate
    discovery_margin: float


class Policy(ABC):
    name: str

    @abstractmethod
    def choose(
        self,
        candidates: list[Candidate],
        history: list[Observation],
        query_index: int,
    ) -> Candidate:
        pass

    def action_name(self, query_index: int) -> str:
        return self.name


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choose(
        self,
        candidates: list[Candidate],
        history: list[Observation],
        query_index: int,
    ) -> Candidate:
        seen = {observation.candidate.sequence for observation in history}
        remaining = [
            candidate
            for candidate in candidates
            if candidate.sequence not in seen
        ]
        if not remaining:
            raise RuntimeError("candidate space exhausted")

        # Balance selection across codon-position groups before sampling an
        # alternative within the selected group.
        position_counts = Counter(
            observation.candidate.codon_positions
            for observation in history
        )
        remaining_groups = {
            candidate.codon_positions
            for candidate in remaining
        }
        minimum_count = min(
            position_counts[group]
            for group in remaining_groups
        )
        least_used_groups = sorted(
            group
            for group in remaining_groups
            if position_counts[group] == minimum_count
        )
        selected_group = self.rng.choice(least_used_groups)
        group_candidates = [
            candidate
            for candidate in remaining
            if candidate.codon_positions == selected_group
        ]
        return self.rng.choice(group_candidates)

    def action_name(self, query_index: int) -> str:
        return "position_balanced_random"


class FixedMultistagePolicy(RandomPolicy):
    name = "fixed_multistage"

    def choose(
        self,
        candidates: list[Candidate],
        history: list[Observation],
        query_index: int,
    ) -> Candidate:
        if query_index <= 16 or not history:
            return super().choose(candidates, history, query_index)

        seen = {observation.candidate.sequence for observation in history}
        remaining = [
            candidate
            for candidate in candidates
            if candidate.sequence not in seen
        ]
        if not remaining:
            raise RuntimeError("candidate space exhausted")

        ranked_history = sorted(
            history,
            key=lambda observation: (
                observation.discovery_margin,
                observation.candidate.sequence,
            ),
        )

        if query_index <= 32:
            incumbent = ranked_history[0].candidate
            return min(
                remaining,
                key=lambda candidate: (
                    _hamming(candidate.sequence, incumbent.sequence),
                    candidate.sequence,
                ),
            )

        beam = [
            observation.candidate
            for observation in ranked_history[: min(4, len(ranked_history))]
        ]
        return min(
            remaining,
            key=lambda candidate: (
                min(
                    _hamming(candidate.sequence, beam_member.sequence)
                    for beam_member in beam
                ),
                candidate.sequence,
            ),
        )

    def action_name(self, query_index: int) -> str:
        if query_index <= 16:
            return "random"
        if query_index <= 32:
            return "greedy"
        return "beam"


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(left != right for left, right in zip(a, b))
