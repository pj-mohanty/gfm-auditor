import json

import pytest

from agenticls_auditor.locked_run import _atomic_json
from agenticls_auditor.mutations import Candidate
from agenticls_auditor.search.base import Policy
from agenticls_auditor.search.trajectory import run_trajectory


class OrderedPolicy(Policy):
    name = "ordered"

    def choose(self, candidates, history, query_index):
        return candidates[query_index - 1]


def test_deferred_search_can_run_without_confirmation_access():
    candidates = [
        Candidate("gene", f"candidate-{i}", (i, i + 1), 2)
        for i in range(40)
    ]
    result = run_trajectory(
        policy=OrderedPolicy(), candidates=candidates,
        discovery_score=lambda candidate: 0.0,
        validation_score=lambda candidate: 0.0,
        confirmation_score=None,
    )
    assert len(result.events) == 40
    assert len(result.checkpoints) == 4
    assert all(row.confirmation_margin is None for row in result.checkpoints)
    assert not any(row["event"] == "final_confirmation" for row in result.accountant.events)


def test_atomic_resume_discards_only_incomplete_write(tmp_path):
    target = tmp_path / "search.json"
    partial = tmp_path / "search.json.partial"
    partial.write_text("incomplete", encoding="utf-8")
    _atomic_json(target, {"completed": True})
    assert json.loads(target.read_text()) == {"completed": True}
    assert not partial.exists()
    with pytest.raises(FileExistsError):
        _atomic_json(target, {"completed": False})
    assert json.loads(target.read_text()) == {"completed": True}
