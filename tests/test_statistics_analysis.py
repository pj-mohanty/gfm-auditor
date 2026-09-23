from dataclasses import replace
import json

import pytest

from agenticls_auditor.mutations import Candidate
from agenticls_auditor.results import (
    ConfirmationBankReference,
    write_jsonl,
)
from agenticls_auditor.search.base import (
    Observation,
    Policy,
)
from agenticls_auditor.search.trajectory import (
    run_trajectory,
)
from agenticls_auditor.statistics import (
    checkpoint_results_from_trajectory,
    confirmed_naudc,
    summarize_trajectory,
    validate_checkpoint_results,
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
            source_id="synthetic_gene",
            sequence=f"candidate-{index:02d}",
            codon_positions=(index, index + 1),
            depth=2,
        )
        for index in range(1, 41)
    ]


def candidate_number(
    candidate: Candidate,
) -> int:
    return int(
        candidate.sequence.rsplit("-", 1)[1]
    )


def make_records():
    trajectory = run_trajectory(
        policy=OrderedPolicy(),
        candidates=make_candidates(),
        discovery_score=lambda candidate: (
            -candidate_number(candidate)
        ),
        validation_score=lambda candidate: (
            -candidate_number(candidate) - 0.1
        ),
        confirmation_score=lambda candidate: (
            -0.02 * candidate_number(candidate)
        ),
    )

    records = checkpoint_results_from_trajectory(
        trajectory=trajectory,
        gene_id="synthetic_gene",
        model="synthetic_model",
        policy="ordered",
        seed=11,
        depth=2,
        delta=0.01,
        confirmation_lookup=lambda sequence: (
            ConfirmationBankReference(
                bank_id=f"hidden-{sequence}",
                control_count=8,
            )
        ),
        commit="synthetic",
    )

    return trajectory, records


def test_trajectory_becomes_four_checkpoint_records():
    trajectory, records = make_records()

    assert len(trajectory.events) == 40
    assert len(records) == 4
    assert [
        row.checkpoint
        for row in records
    ] == [5, 10, 20, 40]
    assert [
        row.candidate_sequence
        for row in records
    ] == [
        "candidate-05",
        "candidate-10",
        "candidate-20",
        "candidate-40",
    ]


def test_confirmation_severity_is_calibrated():
    _, records = make_records()

    assert [
        row.confirmed_severity
        for row in records
    ] == pytest.approx(
        [0.09, 0.19, 0.39, 0.79]
    )


def test_synthetic_trajectory_has_valid_naudc():
    _, records = make_records()

    assert confirmed_naudc(records) == pytest.approx(
        0.44
    )

    summary = summarize_trajectory(records)

    assert summary.checkpoint_count == 4
    assert (
        summary.normalized_confirmed_audc
        == pytest.approx(0.44)
    )


def test_validator_rejects_incorrect_severity():
    _, records = make_records()

    invalid = list(records)
    invalid[2] = replace(
        invalid[2],
        confirmed_severity=99.0,
    )

    with pytest.raises(
        ValueError,
        match="severity is inconsistent",
    ):
        validate_checkpoint_results(invalid)


def test_checkpoint_and_summary_jsonl(tmp_path):
    _, records = make_records()
    summary = summarize_trajectory(records)

    checkpoint_path = (
        tmp_path / "checkpoints.jsonl"
    )
    summary_path = tmp_path / "summary.jsonl"

    write_jsonl(checkpoint_path, records)
    write_jsonl(summary_path, [summary])

    checkpoint_rows = [
        json.loads(line)
        for line in checkpoint_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    summary_rows = [
        json.loads(line)
        for line in summary_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(checkpoint_rows) == 4
    assert len(summary_rows) == 1
    assert (
        summary_rows[0][
            "normalized_confirmed_audc"
        ]
        == pytest.approx(0.44)
    )
