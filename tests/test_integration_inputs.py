from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from agenticls_auditor.integration import (
    ControlBankScorer,
    FrozenCandidateRecord,
    FrozenGeneInputs,
)
from agenticls_auditor.mutations import Candidate


class FakeInformativeAdapter:
    name = "fake-informative"

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def embed_with_info(self, sequence: str):
        cache_hit = sequence in self.seen
        self.seen.add(sequence)

        digest = sha256(sequence.encode()).digest()
        embedding = np.frombuffer(
            digest[:8],
            dtype=np.uint8,
        ).astype(np.float64)

        return FakeCall(
            embedding=embedding,
            cache_hit=cache_hit,
            raw_model_calls=0 if cache_hit else 1,
        )


class FakeCall:
    def __init__(
        self,
        *,
        embedding: np.ndarray,
        cache_hit: bool,
        raw_model_calls: int,
    ) -> None:
        self.embedding = embedding
        self.cache_hit = cache_hit
        self.raw_model_calls = raw_model_calls


def make_gene() -> FrozenGeneInputs:
    candidate = Candidate(
        source_id="gene-1",
        sequence="candidate",
        codon_positions=(1, 2),
        depth=2,
    )

    record = FrozenCandidateRecord(
        candidate=candidate,
        query_index=1,
        candidate_hash=sha256(
            candidate.sequence.encode()
        ).hexdigest(),
        control_manifest_hash="manifest-hash",
        discovery_controls=("control-a", "control-b"),
        validation_controls=("control-c", "control-d"),
        confirmation_controls=(
            "control-e",
            "control-f",
        ),
    )

    return FrozenGeneInputs(
        sequence_id="transcript-1",
        gene_id="gene-1",
        gene_symbol="GENE1",
        source_sequence="source",
        records=(record,),
    )


def test_policy_view_contains_candidates_not_controls():
    gene = make_gene()

    candidates = gene.candidates

    assert len(candidates) == 1
    assert isinstance(candidates[0], Candidate)
    assert not hasattr(
        candidates[0],
        "confirmation_controls",
    )


def test_control_banks_remain_disjoint():
    record = make_gene().records[0]

    assert not (
        set(record.discovery_controls)
        & set(record.validation_controls)
    )
    assert not (
        set(record.discovery_controls)
        & set(record.confirmation_controls)
    )
    assert not (
        set(record.validation_controls)
        & set(record.confirmation_controls)
    )


def test_scorer_reports_real_calls_and_cache_hits():
    gene = make_gene()
    adapter = FakeInformativeAdapter()
    scorer = ControlBankScorer(
        gene=gene,
        adapter=adapter,
    )

    candidate = gene.candidates[0]

    first = scorer.discovery(candidate)

    # Source, candidate, and two discovery controls.
    assert first.raw_model_calls == 4
    assert first.cache_hits == 0
    assert np.isfinite(first.value)

    second = scorer.discovery(candidate)

    assert second.raw_model_calls == 0
    assert second.cache_hits == 4
    assert second.value == first.value


def test_confirmation_reference_hides_sequences():
    gene = make_gene()
    scorer = ControlBankScorer(
        gene=gene,
        adapter=FakeInformativeAdapter(),
    )

    reference = scorer.confirmation_reference(
        "candidate"
    )

    assert reference.control_count == 2
    assert reference.bank_id == (
        "manifest-hash:confirmation"
    )
    assert "control-e" not in reference.bank_id



@pytest.mark.parametrize("candidate_count", [40, 80])
def test_integrated_runner_produces_four_checkpoints(candidate_count):
    from agenticls_auditor.integration import (
        run_integrated_trajectory,
    )

    records = []

    for index in range(1, candidate_count + 1):
        candidate = Candidate(
            source_id="gene-1",
            sequence=f"candidate-{index:02d}",
            codon_positions=(index, index + 1),
            depth=2,
        )

        records.append(
            FrozenCandidateRecord(
                candidate=candidate,
                query_index=index,
                candidate_hash=sha256(
                    candidate.sequence.encode()
                ).hexdigest(),
                control_manifest_hash=(
                    f"manifest-{index:02d}"
                ),
                discovery_controls=(
                    f"discovery-a-{index}",
                    f"discovery-b-{index}",
                ),
                validation_controls=(
                    f"validation-a-{index}",
                    f"validation-b-{index}",
                ),
                confirmation_controls=(
                    f"confirmation-a-{index}",
                    f"confirmation-b-{index}",
                ),
            )
        )

    gene = FrozenGeneInputs(
        sequence_id="transcript-1",
        gene_id="gene-1",
        gene_symbol="GENE1",
        source_sequence="source",
        records=tuple(records),
    )

    result = run_integrated_trajectory(
        gene=gene,
        adapter=FakeInformativeAdapter(),
        policy_name="random",
        seed=11,
        delta=0.0,
        commit="test",
    )

    assert (
        result.trajectory.accountant
        .candidate_evaluations
        == 40
    )
    assert (
        result.trajectory.accountant
        .validation_actions
        == 10
    )
    assert len(result.checkpoint_records) == 4
    assert len({event.candidate_sequence for event in result.trajectory.events}) == 40
    assert len(gene.candidates) == candidate_count
    assert [
        row.checkpoint
        for row in result.checkpoint_records
    ] == [5, 10, 20, 40]
    assert result.summary.checkpoint_count == 4
    assert np.isfinite(
        result.summary.normalized_confirmed_audc
    )
