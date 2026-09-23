from agenticls_auditor.mutations import Candidate
from agenticls_auditor.search.base import (
    FixedMultistagePolicy,
    Observation,
    RandomPolicy,
)


def make_grouped_candidates() -> list[Candidate]:
    candidates = []
    for group in range(20):
        positions = (group, group + 1)
        for alternative in range(3):
            candidates.append(
                Candidate(
                    source_id="synthetic_gene",
                    sequence=f"candidate-{group:02d}-{alternative}",
                    codon_positions=positions,
                    depth=2,
                )
            )
    return candidates


def test_random_policy_is_reproducible_and_without_replacement():
    candidates = make_grouped_candidates()
    first_policy = RandomPolicy(seed=11)
    second_policy = RandomPolicy(seed=11)

    first_history = []
    second_history = []

    first_sequences = []
    second_sequences = []

    for query_index in range(1, 41):
        first = first_policy.choose(
            candidates,
            first_history,
            query_index,
        )
        second = second_policy.choose(
            candidates,
            second_history,
            query_index,
        )

        first_sequences.append(first.sequence)
        second_sequences.append(second.sequence)

        first_history.append(Observation(first, float(query_index)))
        second_history.append(Observation(second, float(query_index)))

    assert first_sequences == second_sequences
    assert len(set(first_sequences)) == 40


def test_random_policy_balances_position_groups():
    candidates = make_grouped_candidates()
    policy = RandomPolicy(seed=22)
    history = []

    for query_index in range(1, 21):
        candidate = policy.choose(candidates, history, query_index)
        history.append(Observation(candidate, float(query_index)))

    groups = [
        observation.candidate.codon_positions
        for observation in history
    ]
    assert len(set(groups)) == 20


def test_fixed_multistage_phase_boundaries():
    policy = FixedMultistagePolicy(seed=33)

    assert policy.action_name(1) == "random"
    assert policy.action_name(16) == "random"
    assert policy.action_name(17) == "greedy"
    assert policy.action_name(32) == "greedy"
    assert policy.action_name(33) == "beam"
    assert policy.action_name(40) == "beam"
