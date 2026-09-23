from dataclasses import replace

import pytest

from agenticls_auditor.inference import (
    paired_gene_clustered_bootstrap,
    within_gene_seed_means,
)
from agenticls_auditor.results import (
    TrajectorySummary,
)


SEEDS = (11, 22)


def make_summaries():
    policy_values = {
        "gene-1": {
            "auditor": (0.5, 0.7),
            "random": (0.2, 0.4),
        },
        "gene-2": {
            "auditor": (0.8, 1.0),
            "random": (0.4, 0.6),
        },
        "gene-3": {
            "auditor": (0.3, 0.5),
            "random": (0.2, 0.2),
        },
    }

    rows = []

    for gene_id, policies in policy_values.items():
        for policy, values in policies.items():
            for seed, value in zip(SEEDS, values):
                rows.append(
                    TrajectorySummary(
                        gene_id=gene_id,
                        model="synthetic-model",
                        policy=policy,
                        seed=seed,
                        depth=2,
                        normalized_confirmed_audc=value,
                        checkpoint_count=4,
                        commit="synthetic",
                    )
                )

    return rows


def test_seed_aggregation_occurs_within_gene():
    means = within_gene_seed_means(
        make_summaries(),
        model="synthetic-model",
        depth=2,
        policies=("auditor", "random"),
        expected_seeds=SEEDS,
    )

    indexed = {
        (row.gene_id, row.policy): row
        for row in means
    }

    assert len(means) == 6
    assert indexed[
        ("gene-1", "auditor")
    ].mean_normalized_confirmed_audc == pytest.approx(
        0.6
    )
    assert indexed[
        ("gene-1", "random")
    ].mean_normalized_confirmed_audc == pytest.approx(
        0.3
    )
    assert all(row.seed_count == 2 for row in means)


def test_paired_bootstrap_uses_genes_as_units():
    result = paired_gene_clustered_bootstrap(
        make_summaries(),
        model="synthetic-model",
        depth=2,
        policy_a="auditor",
        policy_b="random",
        expected_seeds=SEEDS,
        bootstrap_replicates=2_000,
        bootstrap_seed=123,
    )

    # Gene-level differences are 0.3, 0.4, and 0.2.
    assert result.gene_count == 3
    assert result.seeds_per_gene_policy == 2
    assert (
        result.mean_difference_a_minus_b
        == pytest.approx(0.3)
    )
    assert (
        result.confidence_interval_low
        <= result.mean_difference_a_minus_b
        <= result.confidence_interval_high
    )


def test_bootstrap_is_deterministic_for_fixed_seed():
    arguments = dict(
        summaries=make_summaries(),
        model="synthetic-model",
        depth=2,
        policy_a="auditor",
        policy_b="random",
        expected_seeds=SEEDS,
        bootstrap_replicates=1_000,
        bootstrap_seed=77,
    )

    first = paired_gene_clustered_bootstrap(
        **arguments
    )
    second = paired_gene_clustered_bootstrap(
        **arguments
    )

    assert first == second


def test_incomplete_seed_cell_is_rejected():
    rows = make_summaries()

    rows = [
        row
        for row in rows
        if not (
            row.gene_id == "gene-2"
            and row.policy == "auditor"
            and row.seed == 22
        )
    ]

    with pytest.raises(
        ValueError,
        match="complete expected seed set",
    ):
        within_gene_seed_means(
            rows,
            model="synthetic-model",
            depth=2,
            policies=("auditor", "random"),
            expected_seeds=SEEDS,
        )


def test_unpaired_gene_sets_are_rejected():
    rows = make_summaries()

    rows = [
        row
        for row in rows
        if not (
            row.gene_id == "gene-3"
            and row.policy == "random"
        )
    ]

    with pytest.raises(
        ValueError,
        match="identical paired gene sets",
    ):
        within_gene_seed_means(
            rows,
            model="synthetic-model",
            depth=2,
            policies=("auditor", "random"),
            expected_seeds=SEEDS,
        )


def test_duplicate_seed_is_rejected():
    rows = make_summaries()
    rows.append(replace(rows[0]))

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        within_gene_seed_means(
            rows,
            model="synthetic-model",
            depth=2,
            policies=("auditor", "random"),
            expected_seeds=SEEDS,
        )
