from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..mutations import Candidate
from ..query_budget import QueryAccountant
from .base import Observation, Policy


ScoreFunction = Callable[[Candidate], float]
ValidationDecision = Callable[
    [int, Sequence[Observation], QueryAccountant],
    bool,
]


@dataclass(frozen=True)
class TrajectoryEvent:
    query_index: int
    policy_action: str
    candidate_sequence: str
    discovery_margin: float
    incumbent_sequence: str
    incumbent_discovery_margin: float
    validation_performed: bool
    validation_margin: float | None
    candidate_evaluations: int
    validation_actions: int


@dataclass(frozen=True)
class CheckpointSnapshot:
    checkpoint: int
    incumbent_sequence: str
    discovery_margin: float
    confirmation_margin: float
    candidate_evaluations: int
    validation_actions: int
    raw_model_calls: int


@dataclass
class TrajectoryRun:
    events: list[TrajectoryEvent]
    checkpoints: list[CheckpointSnapshot]
    accountant: QueryAccountant


def validation_limit_at_query(
    query_index: int,
    cumulative_allowances: Mapping[int, int],
) -> int:
    """Return the cumulative allowance available by the next checkpoint."""
    for checkpoint in sorted(cumulative_allowances):
        if query_index <= checkpoint:
            return int(cumulative_allowances[checkpoint])
    raise ValueError(
        f"query {query_index} is beyond the validation allowance schedule"
    )


def scheduled_validation_queries(
    checkpoints: Sequence[int],
    cumulative_allowances: Mapping[int, int],
) -> frozenset[int]:
    """Create a deterministic schedule that exhausts each allowance."""
    scheduled: set[int] = set()
    previous_checkpoint = 0
    previous_allowance = 0

    for checkpoint in checkpoints:
        cumulative_limit = int(cumulative_allowances[checkpoint])
        new_actions = cumulative_limit - previous_allowance
        interval_width = checkpoint - previous_checkpoint

        if new_actions < 0:
            raise ValueError("validation allowances must be cumulative")
        if new_actions > interval_width:
            raise ValueError(
                "validation allowance exceeds available queries"
            )

        for action_index in range(1, new_actions + 1):
            offset = (
                action_index * interval_width + new_actions - 1
            ) // new_actions
            scheduled.add(previous_checkpoint + offset)

        previous_checkpoint = checkpoint
        previous_allowance = cumulative_limit

    return frozenset(scheduled)


def run_trajectory(
    *,
    policy: Policy,
    candidates: list[Candidate],
    discovery_score: ScoreFunction,
    validation_score: ScoreFunction,
    confirmation_score: ScoreFunction,
    max_candidates: int = 40,
    checkpoints: Sequence[int] = (5, 10, 20, 40),
    validation_allowances: Mapping[int, int] | None = None,
    validation_decision: ValidationDecision | None = None,
) -> TrajectoryRun:
    if len(candidates) < max_candidates:
        raise ValueError("candidate space is smaller than the query budget")

    checkpoints = tuple(int(value) for value in checkpoints)
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and strictly increasing")
    if checkpoints[-1] != max_candidates:
        raise ValueError("last checkpoint must equal max_candidates")

    allowances = dict(
        validation_allowances
        or {5: 2, 10: 3, 20: 5, 40: 10}
    )
    if tuple(sorted(allowances)) != checkpoints:
        raise ValueError(
            "validation allowances must be defined for every checkpoint"
        )

    allowance_values = tuple(allowances[point] for point in checkpoints)
    if any(value < 0 for value in allowance_values):
        raise ValueError("validation allowances cannot be negative")
    if tuple(sorted(allowance_values)) != allowance_values:
        raise ValueError("validation allowances must be cumulative")

    accountant = QueryAccountant(
        max_candidates=max_candidates,
        max_validations=allowance_values[-1],
    )
    history: list[Observation] = []
    events: list[TrajectoryEvent] = []
    snapshots: list[CheckpointSnapshot] = []

    for query_index in range(1, max_candidates + 1):
        candidate = policy.choose(candidates, history, query_index)
        if any(
            observation.candidate.sequence == candidate.sequence
            for observation in history
        ):
            raise RuntimeError("policy selected an already evaluated candidate")

        discovery_margin = float(discovery_score(candidate))
        accountant.candidate(query_index=query_index, raw_calls=1)
        history.append(Observation(candidate, discovery_margin))

        incumbent = min(
            history,
            key=lambda observation: observation.discovery_margin,
        )

        cumulative_limit = validation_limit_at_query(
            query_index,
            allowances,
        )
        remaining_allowance = cumulative_limit - accountant.validation_actions

        if validation_decision is not None:
            wants_validation = validation_decision(
                query_index,
                tuple(history),
                accountant,
            )
        else:
            policy_decision = getattr(policy, "should_validate", None)
            if policy_decision is not None:
                wants_validation = bool(
                    policy_decision(query_index, remaining_allowance)
                )
            else:
                baseline_schedule = scheduled_validation_queries(
                    checkpoints,
                    allowances,
                )
                wants_validation = query_index in baseline_schedule

        validation_performed = False
        validation_margin: float | None = None

        if wants_validation and remaining_allowance > 0:
            validation_margin = float(
                validation_score(incumbent.candidate)
            )
            accountant.validation(
                query_index=query_index,
                cumulative_limit=cumulative_limit,
                raw_calls=1,
            )
            validation_performed = True

        events.append(
            TrajectoryEvent(
                query_index=query_index,
                policy_action=policy.action_name(query_index),
                candidate_sequence=candidate.sequence,
                discovery_margin=discovery_margin,
                incumbent_sequence=incumbent.candidate.sequence,
                incumbent_discovery_margin=incumbent.discovery_margin,
                validation_performed=validation_performed,
                validation_margin=validation_margin,
                candidate_evaluations=accountant.candidate_evaluations,
                validation_actions=accountant.validation_actions,
            )
        )

        if query_index in checkpoints:
            # The incumbent is frozen before hidden confirmation is evaluated.
            frozen_sequence = incumbent.candidate.sequence
            frozen_discovery_margin = incumbent.discovery_margin
            hidden_margin = float(
                confirmation_score(incumbent.candidate)
            )
            accountant.confirmation(
                query_index=query_index,
                raw_calls=1,
            )

            snapshots.append(
                CheckpointSnapshot(
                    checkpoint=query_index,
                    incumbent_sequence=frozen_sequence,
                    discovery_margin=frozen_discovery_margin,
                    confirmation_margin=hidden_margin,
                    candidate_evaluations=accountant.candidate_evaluations,
                    validation_actions=accountant.validation_actions,
                    raw_model_calls=accountant.raw_model_calls,
                )
            )

    return TrajectoryRun(
        events=events,
        checkpoints=snapshots,
        accountant=accountant,
    )
