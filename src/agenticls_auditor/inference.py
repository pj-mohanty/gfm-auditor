from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from .results import TrajectorySummary


@dataclass(frozen=True)
class GenePolicyMean:
    gene_id: str
    model: str
    policy: str
    depth: int
    seed_count: int
    mean_normalized_confirmed_audc: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    model: str
    depth: int
    policy_a: str
    policy_b: str
    gene_count: int
    seeds_per_gene_policy: int
    mean_difference_a_minus_b: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_replicates: int
    bootstrap_seed: int


def within_gene_seed_means(
    summaries: Sequence[TrajectorySummary],
    *,
    model: str,
    depth: int,
    policies: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[GenePolicyMean]:
    """Average trajectory endpoints across seeds within gene."""

    policy_order = tuple(policies)
    seed_set = frozenset(int(seed) for seed in expected_seeds)

    if not policy_order:
        raise ValueError("at least one policy is required")

    if len(set(policy_order)) != len(policy_order):
        raise ValueError("policies must be unique")

    if not seed_set:
        raise ValueError("expected seeds cannot be empty")

    selected = [
        row
        for row in summaries
        if row.model == model
        and row.depth == depth
        and row.policy in policy_order
    ]

    if not selected:
        raise ValueError(
            "no trajectory summaries match the requested analysis"
        )

    grouped: dict[
        tuple[str, str],
        dict[int, float],
    ] = defaultdict(dict)

    for row in selected:
        if not math.isfinite(
            row.normalized_confirmed_audc
        ):
            raise ValueError(
                "trajectory summary contains a non-finite endpoint"
            )

        if row.normalized_confirmed_audc < 0:
            raise ValueError(
                "trajectory endpoint cannot be negative"
            )

        key = (row.gene_id, row.policy)

        if row.seed in grouped[key]:
            raise ValueError(
                "duplicate gene-policy-seed trajectory summary"
            )

        grouped[key][row.seed] = (
            row.normalized_confirmed_audc
        )

    genes_by_policy = {
        policy: {
            gene
            for gene, observed_policy in grouped
            if observed_policy == policy
        }
        for policy in policy_order
    }

    reference_genes = genes_by_policy[
        policy_order[0]
    ]

    if not reference_genes:
        raise ValueError(
            "no genes are available for the requested policies"
        )

    for policy in policy_order[1:]:
        if genes_by_policy[policy] != reference_genes:
            raise ValueError(
                "policies do not have identical paired gene sets"
            )

    results: list[GenePolicyMean] = []

    for gene_id in sorted(reference_genes):
        for policy in policy_order:
            values_by_seed = grouped[
                (gene_id, policy)
            ]

            if frozenset(values_by_seed) != seed_set:
                raise ValueError(
                    "gene-policy cell does not contain "
                    "the complete expected seed set"
                )

            values = np.asarray(
                [
                    values_by_seed[seed]
                    for seed in sorted(seed_set)
                ],
                dtype=np.float64,
            )

            results.append(
                GenePolicyMean(
                    gene_id=gene_id,
                    model=model,
                    policy=policy,
                    depth=depth,
                    seed_count=len(values),
                    mean_normalized_confirmed_audc=(
                        float(values.mean())
                    ),
                )
            )

    return results


def paired_gene_clustered_bootstrap(
    summaries: Sequence[TrajectorySummary],
    *,
    model: str,
    depth: int,
    policy_a: str,
    policy_b: str,
    expected_seeds: Sequence[int],
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 2026,
) -> PairedBootstrapResult:
    """Bootstrap paired policy differences over genes."""

    if policy_a == policy_b:
        raise ValueError(
            "paired policies must be different"
        )

    if bootstrap_replicates < 1:
        raise ValueError(
            "bootstrap replicates must be positive"
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence level must lie between zero and one"
        )

    means = within_gene_seed_means(
        summaries,
        model=model,
        depth=depth,
        policies=(policy_a, policy_b),
        expected_seeds=expected_seeds,
    )

    by_policy = {
        policy_a: {},
        policy_b: {},
    }

    for row in means:
        by_policy[row.policy][row.gene_id] = (
            row.mean_normalized_confirmed_audc
        )

    genes = tuple(
        sorted(by_policy[policy_a])
    )

    if len(genes) < 2:
        raise ValueError(
            "paired bootstrap requires at least two genes"
        )

    differences = np.asarray(
        [
            by_policy[policy_a][gene]
            - by_policy[policy_b][gene]
            for gene in genes
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(bootstrap_seed)

    sampled_indices = rng.integers(
        low=0,
        high=len(genes),
        size=(bootstrap_replicates, len(genes)),
    )

    bootstrap_means = differences[
        sampled_indices
    ].mean(axis=1)

    alpha = 1.0 - confidence_level
    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )
    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return PairedBootstrapResult(
        model=model,
        depth=depth,
        policy_a=policy_a,
        policy_b=policy_b,
        gene_count=len(genes),
        seeds_per_gene_policy=len(
            frozenset(expected_seeds)
        ),
        mean_difference_a_minus_b=float(
            differences.mean()
        ),
        confidence_level=confidence_level,
        confidence_interval_low=lower,
        confidence_interval_high=upper,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
