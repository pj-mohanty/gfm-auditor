from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryAccountant:
    max_candidates: int
    max_validations: int
    candidate_evaluations: int = 0
    validation_actions: int = 0
    raw_model_calls: int = 0
    cache_hits: int = 0
    events: list[dict] = field(default_factory=list)

    def candidate(self, *, raw_calls: int = 1, cache_hits: int = 0) -> None:
        if self.candidate_evaluations >= self.max_candidates:
            raise RuntimeError("candidate budget exhausted")
        self.candidate_evaluations += 1
        self._record("candidate", raw_calls, cache_hits)

    def validation(self, *, raw_calls: int = 0, cache_hits: int = 0) -> None:
        if self.validation_actions >= self.max_validations:
            raise RuntimeError("validation allowance exhausted")
        self.validation_actions += 1
        self._record("online_validation", raw_calls, cache_hits)

    def confirmation(self, *, raw_calls: int = 0, cache_hits: int = 0) -> None:
        self._record("final_confirmation", raw_calls, cache_hits)

    def _record(self, event: str, raw_calls: int, cache_hits: int) -> None:
        self.raw_model_calls += raw_calls
        self.cache_hits += cache_hits
        self.events.append({
            "event": event,
            "candidate_evaluations": self.candidate_evaluations,
            "validation_actions": self.validation_actions,
            "raw_model_calls": raw_calls,
            "cache_hits": cache_hits,
        })

