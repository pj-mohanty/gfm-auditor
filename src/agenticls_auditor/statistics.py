from __future__ import annotations

from collections.abc import Callable, Sequence
import math

from .metrics import (
    confirmed_severity,
    trapezoidal_audc,
)
from .results import (
    CheckpointResult,
    ConfirmationBankReference,
    TrajectorySummary,
)
from .search.trajectory import TrajectoryRun


ConfirmationLookup = Callable[
    [str],
    ConfirmationBankReference,
]


def checkpoint_results_from_trajectory(
    *,
    trajectory: TrajectoryRun,
    gene_id: str,
    model: str,
    policy: str,
    seed: int,
    depth: int,
    delta: float,
    confirmation_lookup: ConfirmationLookup,
    expected_checkpoints: Sequence[int] = (
        5,
        10,
        20,
        40,
    ),
    commit: str = "unknown",
) -> list[CheckpointResult]:
    """Convert one trajectory into frozen checkpoint records."""

    expected = tuple(
        int(value)
        for value in expected_checkpoints
    )

    observed = tuple(
        snapshot.checkpoint
        for snapshot in trajectory.checkpoints
    )

    if observed != expected:
        raise ValueError(
            "trajectory checkpoints do not match "
            f"the expected schedule: {observed}"
        )

    events_by_query = {
        event.query_index: event
        for event in trajectory.events
    }

    records: list[CheckpointResult] = []

    for snapshot in trajectory.checkpoints:
        event = events_by_query.get(
            snapshot.checkpoint
        )

        if event is None:
            raise ValueError(
                "checkpoint has no corresponding "
                "trajectory event"
            )

        # This verifies that hidden confirmation did not
        # replace the discovery-selected incumbent.
        if (
            snapshot.incumbent_sequence
            != event.incumbent_sequence
        ):
            raise ValueError(
                "checkpoint incumbent differs from "
                "the frozen trajectory incumbent"
            )

        if not math.isclose(
            snapshot.discovery_margin,
            event.incumbent_discovery_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "checkpoint discovery margin differs "
                "from the frozen trajectory incumbent"
            )

        bank = confirmation_lookup(
            snapshot.incumbent_sequence
        )

        if snapshot.confirmation_margin is None:
            raise ValueError("hidden confirmation is not available")

        severity = confirmed_severity(
            snapshot.confirmation_margin,
            delta,
        )

        records.append(
            CheckpointResult(
                gene_id=gene_id,
                model=model,
                policy=policy,
                seed=int(seed),
                depth=int(depth),
                checkpoint=snapshot.checkpoint,
                candidate_sequence=(
                    snapshot.incumbent_sequence
                ),
                discovery_margin=(
                    snapshot.discovery_margin
                ),
                confirmation_margin=(
                    snapshot.confirmation_margin
                ),
                confirmed_severity=severity,
                candidate_evaluations=(
                    snapshot.candidate_evaluations
                ),
                validation_actions=(
                    snapshot.validation_actions
                ),
                raw_model_calls=(
                    snapshot.raw_model_calls
                ),
                delta=float(delta),
                confirmation_bank_id=bank.bank_id,
                confirmation_control_count=(
                    bank.control_count
                ),
                commit=commit,
            )
        )

    validate_checkpoint_results(
        records,
        expected_checkpoints=expected,
    )

    return records


def validate_checkpoint_results(
    records: Sequence[CheckpointResult],
    *,
    expected_checkpoints: Sequence[int] = (
        5,
        10,
        20,
        40,
    ),
    minimum_confirmation_controls: int = 1,
) -> None:
    """Validate one trajectory's checkpoint records."""

    if not records:
        raise ValueError(
            "checkpoint records cannot be empty"
        )

    expected = tuple(
        int(value)
        for value in expected_checkpoints
    )
    observed = tuple(
        row.checkpoint
        for row in records
    )

    if observed != expected:
        raise ValueError(
            "checkpoint records do not match "
            f"the expected schedule: {observed}"
        )

    identity_fields = (
        "gene_id",
        "model",
        "policy",
        "seed",
        "depth",
        "commit",
    )

    first = records[0]

    for field in identity_fields:
        values = {
            getattr(row, field)
            for row in records
        }

        if len(values) != 1:
            raise ValueError(
                f"inconsistent {field} across checkpoints"
            )

    previous_validations = -1
    previous_raw_calls = -1

    for row in records:
        numeric_values = (
            row.discovery_margin,
            row.confirmation_margin,
            row.confirmed_severity,
            row.delta,
        )

        if not all(
            math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "checkpoint contains a non-finite value"
            )

        if row.delta < 0:
            raise ValueError(
                "severity delta cannot be negative"
            )

        if row.candidate_evaluations != row.checkpoint:
            raise ValueError(
                "candidate evaluations must equal "
                "the checkpoint"
            )

        if row.validation_actions < previous_validations:
            raise ValueError(
                "validation actions must be cumulative"
            )

        if row.raw_model_calls < previous_raw_calls:
            raise ValueError(
                "raw model calls must be cumulative"
            )

        if (
            row.confirmation_control_count
            < minimum_confirmation_controls
        ):
            raise ValueError(
                "insufficient hidden confirmation controls"
            )

        if not row.confirmation_bank_id.strip():
            raise ValueError(
                "confirmation bank ID cannot be empty"
            )

        expected_severity = confirmed_severity(
            row.confirmation_margin,
            row.delta,
        )

        if not math.isclose(
            row.confirmed_severity,
            expected_severity,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "confirmed severity is inconsistent "
                "with margin and delta"
            )

        previous_validations = (
            row.validation_actions
        )
        previous_raw_calls = row.raw_model_calls


def confirmed_naudc(
    records: Sequence[CheckpointResult],
) -> float:
    """Return budget-normalized confirmed AUDC."""

    validate_checkpoint_results(records)

    value = trapezoidal_audc(
        [row.checkpoint for row in records],
        [row.confirmed_severity for row in records],
        normalize=True,
    )

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "normalized confirmed AUDC is invalid"
        )

    return float(value)


def summarize_trajectory(
    records: Sequence[CheckpointResult],
) -> TrajectorySummary:
    validate_checkpoint_results(records)

    first = records[0]

    return TrajectorySummary(
        gene_id=first.gene_id,
        model=first.model,
        policy=first.policy,
        seed=first.seed,
        depth=first.depth,
        normalized_confirmed_audc=(
            confirmed_naudc(records)
        ),
        checkpoint_count=len(records),
        commit=first.commit,
    )
