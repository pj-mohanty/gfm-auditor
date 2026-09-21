from agenticls_auditor.controls import (
    enumerate_matched_nonsynonymous_controls,
    hamming_distance,
    partition_controls,
)
from agenticls_auditor.data import CodingSequence, translate
from agenticls_auditor.mutations import Candidate


def example():
    source = CodingSequence(
        gene_id="example_gene",
        sequence="ATGGCTGGTTTTTAA",
    )
    candidate = Candidate(
        source_id="example_gene",
        sequence="ATGGCCGGCTTTTAA",
        codon_positions=(1, 2),
        depth=2,
    )
    return source, candidate


def test_matched_controls_preserve_edit_budget_and_change_protein():
    source, candidate = example()
    controls = enumerate_matched_nonsynonymous_controls(source, candidate)

    assert len(controls) >= 16
    assert len(controls) == len(set(controls))

    candidate_distance = hamming_distance(
        source.sequence,
        candidate.sequence,
    )

    for control in controls:
        assert hamming_distance(source.sequence, control) == candidate_distance
        assert translate(control) != translate(source.sequence)
        assert "*" not in translate(control)[:-1]


def test_control_partition_is_deterministic_and_disjoint():
    source, candidate = example()
    controls = enumerate_matched_nonsynonymous_controls(source, candidate)

    first = partition_controls(
        controls,
        source_id=source.gene_id,
        candidate_sequence=candidate.sequence,
        assignment_seed=42,
    )
    second = partition_controls(
        list(reversed(controls)),
        source_id=source.gene_id,
        candidate_sequence=candidate.sequence,
        assignment_seed=42,
    )

    assert first == second
    assert first.manifest_hash == second.manifest_hash
    assert first.eligible

    discovery = set(first.discovery)
    validation = set(first.validation)
    confirmation = set(first.confirmation)

    assert discovery.isdisjoint(validation)
    assert discovery.isdisjoint(confirmation)
    assert validation.isdisjoint(confirmation)
    assert discovery | validation | confirmation == set(controls)
