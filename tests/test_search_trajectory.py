from agenticls_auditor.mutations import Candidate
from agenticls_auditor.search.base import Observation, Policy
from agenticls_auditor.search.trajectory import (
    run_trajectory,
    validation_limit_at_query,
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


def make_candidates(count: int = 40) -> list[Candidate]:
    return [
        Candidate(
            source_id="synthetic_gene",
            sequence=f"candidate-{index:02d}",
            codon_positions=(index, index + 1),
            depth=2,
        )
        for index in range(1, count + 1)
    ]


def candidate_number(candidate: Candidate) -> int:
    return int(candidate.sequence.rsplit("-", 1)[1])


def test_validation_schedule_uses_checkpoint_cumulative_limits():
    allowances = {5: 2, 10: 3, 20: 5, 40: 10}

    assert validation_limit_at_query(1, allowances) == 2
    assert validation_limit_at_query(5, allowances) == 2
    assert validation_limit_at_query(6, allowances) == 3
    assert validation_limit_at_query(20, allowances) == 5
    assert validation_limit_at_query(40, allowances) == 10


def test_forty_queries_create_four_frozen_checkpoints():
    candidates = make_candidates()

    result = run_trajectory(
        policy=OrderedPolicy(),
        candidates=candidates,
        discovery_score=lambda candidate: -candidate_number(candidate),
        validation_score=lambda candidate: -candidate_number(candidate) - 0.1,
        confirmation_score=lambda candidate: 1000.0 + candidate_number(candidate),
    )

    assert len(result.events) == 40
    assert [row.checkpoint for row in result.checkpoints] == [5, 10, 20, 40]
    assert [row.candidate_evaluations for row in result.checkpoints] == [
        5,
        10,
        20,
        40,
    ]
    assert [row.incumbent_sequence for row in result.checkpoints] == [
        "candidate-05",
        "candidate-10",
        "candidate-20",
        "candidate-40",
    ]


def test_hidden_confirmation_cannot_change_the_incumbent():
    candidates = make_candidates()

    result = run_trajectory(
        policy=OrderedPolicy(),
        candidates=candidates,
        discovery_score=lambda candidate: -candidate_number(candidate),
        validation_score=lambda candidate: 0.0,
        # Deliberately assigns worse confirmation values to later candidates.
        confirmation_score=lambda candidate: candidate_number(candidate),
    )

    assert result.checkpoints[0].incumbent_sequence == "candidate-05"
    assert result.checkpoints[1].incumbent_sequence == "candidate-10"
    assert result.events[5].candidate_sequence == "candidate-06"
    assert result.events[10].candidate_sequence == "candidate-11"


def test_cumulative_validation_budgets_are_never_exceeded():
    candidates = make_candidates()

    result = run_trajectory(
        policy=OrderedPolicy(),
        candidates=candidates,
        discovery_score=lambda candidate: -candidate_number(candidate),
        validation_score=lambda candidate: -candidate_number(candidate),
        confirmation_score=lambda candidate: -candidate_number(candidate),
        validation_decision=lambda query, history, accountant: True,
    )

    checkpoint_counts = {
        row.checkpoint: row.validation_actions
        for row in result.checkpoints
    }

    assert checkpoint_counts == {5: 2, 10: 3, 20: 5, 40: 10}
    assert result.accountant.candidate_evaluations == 40
    assert result.accountant.validation_actions == 10


def test_fixed_validation_schedule_exhausts_each_allowance():
    from agenticls_auditor.search.trajectory import (
        scheduled_validation_queries,
    )

    schedule = scheduled_validation_queries(
        (5, 10, 20, 40),
        {5: 2, 10: 3, 20: 5, 40: 10},
    )

    assert schedule == frozenset(
        {3, 5, 10, 15, 20, 24, 28, 32, 36, 40}
    )
