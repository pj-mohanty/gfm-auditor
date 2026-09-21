from __future__ import annotations

from abc import ABC, abstractmethod
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
    def choose(self, candidates: list[Candidate], history: list[Observation], query_index: int) -> Candidate:
        pass


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choose(self, candidates: list[Candidate], history: list[Observation], query_index: int) -> Candidate:
        seen = {obs.candidate.sequence for obs in history}
        remaining = [candidate for candidate in candidates if candidate.sequence not in seen]
        if not remaining:
            raise RuntimeError("candidate space exhausted")
        # Candidate lists are generated position-first, so selecting positions
        # uniformly before alternatives is implemented by the real generator.
        return self.rng.choice(remaining)


class FixedMultistagePolicy(RandomPolicy):
    name = "fixed_multistage"

    def choose(self, candidates: list[Candidate], history: list[Observation], query_index: int) -> Candidate:
        if query_index <= 16 or not history:
            return super().choose(candidates, history, query_index)
        # Scaffold behavior: exploitation picks an unseen candidate nearest in
        # lexical edit representation to the current best. Replace with true
        # greedy/beam neighborhood expansion in the experiment implementation.
        best = min(history, key=lambda obs: obs.discovery_margin).candidate
        seen = {obs.candidate.sequence for obs in history}
        remaining = [candidate for candidate in candidates if candidate.sequence not in seen]
        return min(remaining, key=lambda candidate: _hamming(candidate.sequence, best.sequence))


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

