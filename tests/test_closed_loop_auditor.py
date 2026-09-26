from agenticls_auditor.auditor import ClosedLoopAuditor
from agenticls_auditor.mutations import Candidate
from agenticls_auditor.search.trajectory import run_trajectory


def make_candidates(count: int = 60) -> list[Candidate]:
    return [
        Candidate(
            source_id="synthetic_gene",
            sequence=f"candidate-{index:03d}",
            codon_positions=(index % 20, (index % 20) + 1),
            depth=2,
        )
        for index in range(count)
    ]


def test_auditor_records_every_search_decision():
    auditor = ClosedLoopAuditor(seed=44)

    result = run_trajectory(
        policy=auditor,
        candidates=make_candidates(),
        discovery_score=lambda candidate: 0.0,
        validation_score=lambda candidate: 0.0,
        confirmation_score=lambda candidate: 0.0,
    )

    assert len(auditor.state.decisions) == 40
    assert len(result.events) == 40
    assert any(
        event.policy_action == "random_restart"
        for event in result.events
    )
    assert all(
        event.policy_action
        in {"random", "greedy", "beam", "random_restart", "validation_restart"}
        for event in result.events
    )


def test_auditor_respects_every_cumulative_validation_limit():
    auditor = ClosedLoopAuditor(seed=55)

    result = run_trajectory(
        policy=auditor,
        candidates=make_candidates(),
        discovery_score=lambda candidate: 0.0,
        validation_score=lambda candidate: -0.1,
        confirmation_score=lambda candidate: -0.2,
    )

    limits = {5: 2, 10: 3, 20: 5, 40: 10}
    counts = {
        snapshot.checkpoint: snapshot.validation_actions
        for snapshot in result.checkpoints
    }

    assert all(
        counts[checkpoint] <= limit
        for checkpoint, limit in limits.items()
    )
    assert result.accountant.validation_actions > 0
    assert result.accountant.validation_actions <= 10


def test_confirmation_events_do_not_enter_auditor_decisions():
    auditor = ClosedLoopAuditor(seed=11)

    result = run_trajectory(
        policy=auditor,
        candidates=make_candidates(),
        discovery_score=lambda candidate: 0.0,
        validation_score=lambda candidate: 0.0,
        confirmation_score=lambda candidate: -9999.0,
    )

    assert len(auditor.state.decisions) == 40
    assert all(
        "confirmation_margin" not in decision
        for decision in auditor.state.decisions
    )
    assert len(result.checkpoints) == 4


def test_validation_feedback_changes_next_search_choice():
    def run_with_validation(value: float):
        auditor = ClosedLoopAuditor(seed=11)
        result = run_trajectory(
            policy=auditor,
            candidates=make_candidates(80),
            discovery_score=lambda candidate: (
                -int(candidate.sequence.rsplit("-", 1)[1]) / 1000
            ),
            validation_score=lambda candidate: value,
            confirmation_score=lambda candidate: 0.0,
            validation_decision=lambda query, history, accountant: (
                query == 16
            ),
        )
        return auditor, result

    rejected, rejected_run = run_with_validation(1.0)
    supported, supported_run = run_with_validation(-1.0)

    assert rejected.state.validation_feedback[0][
        "restart_next_query"
    ] is True
    assert supported.state.validation_feedback[0][
        "restart_next_query"
    ] is False
    assert rejected_run.events[16].policy_action == "validation_restart"
    assert supported_run.events[16].policy_action == "greedy"
    assert (
        rejected_run.events[16].candidate_sequence
        != supported_run.events[16].candidate_sequence
    )
