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

    def candidate(
        self,
        *,
        query_index: int | None = None,
        raw_calls: int = 1,
        cache_hits: int = 0,
    ) -> None:
        if self.candidate_evaluations >= self.max_candidates:
            raise RuntimeError("candidate budget exhausted")

        expected_query = self.candidate_evaluations + 1
        if query_index is not None and query_index != expected_query:
            raise ValueError(
                f"expected candidate query {expected_query}, got {query_index}"
            )

        self.candidate_evaluations += 1
        self._record(
            "candidate",
            raw_calls,
            cache_hits,
            query_index=expected_query,
        )

    def validation(
        self,
        *,
        query_index: int | None = None,
        cumulative_limit: int | None = None,
        raw_calls: int = 0,
        cache_hits: int = 0,
    ) -> None:
        if self.validation_actions >= self.max_validations:
            raise RuntimeError("validation allowance exhausted")

        if query_index is not None:
            if query_index < 1 or query_index > self.candidate_evaluations:
                raise ValueError(
                    "validation query must refer to an evaluated candidate"
                )

        if (
            cumulative_limit is not None
            and self.validation_actions >= cumulative_limit
        ):
            raise RuntimeError(
                "cumulative validation allowance exhausted at this query"
            )

        self.validation_actions += 1
        self._record(
            "online_validation",
            raw_calls,
            cache_hits,
            query_index=query_index,
        )

    def confirmation(
        self,
        *,
        query_index: int | None = None,
        raw_calls: int = 0,
        cache_hits: int = 0,
    ) -> None:
        self._record(
            "final_confirmation",
            raw_calls,
            cache_hits,
            query_index=query_index,
        )

    def _record(
        self,
        event: str,
        raw_calls: int,
        cache_hits: int,
        *,
        query_index: int | None,
    ) -> None:
        if raw_calls < 0 or cache_hits < 0:
            raise ValueError("call counts cannot be negative")

        self.raw_model_calls += raw_calls
        self.cache_hits += cache_hits
        self.events.append(
            {
                "event": event,
                "query_index": query_index,
                "candidate_evaluations": self.candidate_evaluations,
                "validation_actions": self.validation_actions,
                "raw_model_calls": raw_calls,
                "cache_hits": cache_hits,
            }
        )
