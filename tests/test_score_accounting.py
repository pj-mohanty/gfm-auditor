from __future__ import annotations

import pytest

from agenticls_auditor.mutations import Candidate
from agenticls_auditor.search.base import (
    Observation,
    Policy,
)
from agenticls_auditor.search.trajectory import (
    ScoreResult,
    run_trajectory,
)


class OrderedPolicy(Policy):
    name = "ordered"

    def choose(
        self,
        candidates: list[Candidate],
        history: list[Observation],
        query_index: int,
    ) -> Candidate:
        return candidates[query_index - 1]


def make_candidates() -> list[Candidate]:
    return [
        Candidate(
            source_id="gene",
            sequence=f"candidate-{index:02d}",
            codon_positions=(index, index + 1),
            depth=2,
        )
        for index in range(1, 41)
    ]


def candidate_number(candidate: Candidate) -> int:
    return int(candidate.sequence.rsplit("-", 1)[1])


def test_score_result_reports_actual_calls_and_cache_hits():
    trajectory = run_trajectory(
        policy=OrderedPolicy(),
        candidates=make_candidates(),
        discovery_score=lambda candidate: ScoreResult(
            value=-candidate_number(candidate),
            raw_model_calls=2,
            cache_hits=3,
        ),
        validation_score=lambda candidate: ScoreResult(
            value=-candidate_number(candidate),
            raw_model_calls=4,
            cache_hits=5,
        ),
        confirmation_score=lambda candidate: ScoreResult(
            value=-candidate_number(candidate),
            raw_model_calls=6,
            cache_hits=7,
        ),
    )

    # 40 discovery scores, 10 validation actions,
    # and 4 checkpoint confirmations.
    assert trajectory.accountant.raw_model_calls == (
        40 * 2 + 10 * 4 + 4 * 6
    )
    assert trajectory.accountant.cache_hits == (
        40 * 3 + 10 * 5 + 4 * 7
    )

    assert (
        trajectory.checkpoints[-1].raw_model_calls
        == trajectory.accountant.raw_model_calls
    )


def test_plain_float_scores_remain_backward_compatible():
    trajectory = run_trajectory(
        policy=OrderedPolicy(),
        candidates=make_candidates(),
        discovery_score=lambda candidate: (
            -candidate_number(candidate)
        ),
        validation_score=lambda candidate: (
            -candidate_number(candidate)
        ),
        confirmation_score=lambda candidate: (
            -candidate_number(candidate)
        ),
    )

    assert trajectory.accountant.raw_model_calls == 54
    assert trajectory.accountant.cache_hits == 0


@pytest.mark.parametrize(
    ("raw_calls", "cache_hits"),
    [(-1, 0), (0, -1)],
)
def test_score_result_rejects_negative_counts(
    raw_calls: int,
    cache_hits: int,
):
    with pytest.raises(ValueError):
        ScoreResult(
            value=0.0,
            raw_model_calls=raw_calls,
            cache_hits=cache_hits,
        )
