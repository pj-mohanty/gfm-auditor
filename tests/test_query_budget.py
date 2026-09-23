import pytest

from agenticls_auditor.query_budget import QueryAccountant


def test_candidate_budget_and_query_order_are_enforced():
    accountant = QueryAccountant(max_candidates=2, max_validations=1)

    accountant.candidate(query_index=1)
    accountant.candidate(query_index=2)

    assert accountant.candidate_evaluations == 2

    with pytest.raises(RuntimeError, match="candidate budget exhausted"):
        accountant.candidate(query_index=3)


def test_candidate_queries_must_be_sequential():
    accountant = QueryAccountant(max_candidates=2, max_validations=1)

    with pytest.raises(ValueError, match="expected candidate query 1"):
        accountant.candidate(query_index=2)


def test_cumulative_validation_allowance_is_enforced():
    accountant = QueryAccountant(max_candidates=40, max_validations=10)

    for query_index in range(1, 6):
        accountant.candidate(query_index=query_index)

    accountant.validation(query_index=3, cumulative_limit=2)
    accountant.validation(query_index=5, cumulative_limit=2)

    with pytest.raises(
        RuntimeError,
        match="cumulative validation allowance exhausted",
    ):
        accountant.validation(query_index=5, cumulative_limit=2)

    assert accountant.validation_actions == 2


def test_validation_requires_an_evaluated_candidate():
    accountant = QueryAccountant(max_candidates=40, max_validations=10)
    accountant.candidate(query_index=1)

    with pytest.raises(
        ValueError,
        match="validation query must refer to an evaluated candidate",
    ):
        accountant.validation(query_index=2, cumulative_limit=2)


def test_accounting_events_keep_logical_and_raw_costs_separate():
    accountant = QueryAccountant(max_candidates=40, max_validations=10)

    accountant.candidate(query_index=1, raw_calls=1)
    accountant.validation(
        query_index=1,
        cumulative_limit=2,
        raw_calls=0,
        cache_hits=1,
    )
    accountant.confirmation(query_index=1, cache_hits=1)

    assert accountant.candidate_evaluations == 1
    assert accountant.validation_actions == 1
    assert accountant.raw_model_calls == 1
    assert accountant.cache_hits == 2
    assert [event["event"] for event in accountant.events] == [
        "candidate",
        "online_validation",
        "final_confirmation",
    ]
