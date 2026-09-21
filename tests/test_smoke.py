from pathlib import Path

from agenticls_auditor.config import load_config
from agenticls_auditor.controls import partition_controls
from agenticls_auditor.smoke import run_smoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.yaml"


def test_config_is_consistent():
    cfg = load_config(CONFIG)
    assert cfg.max_candidates == 40
    assert cfg.checkpoints == (5, 10, 20, 40)
    assert len(cfg.seeds) == 5


def test_control_partition_is_disjoint_and_deterministic():
    controls = [f"control-{i}" for i in range(20)]
    first = partition_controls(controls, source_id="g1", candidate_sequence="ATGGCTTAA", assignment_seed=42)
    second = partition_controls(list(reversed(controls)), source_id="g1", candidate_sequence="ATGGCTTAA", assignment_seed=42)
    assert first == second
    assert first.eligible
    assert not (set(first.discovery) & set(first.validation))
    assert not (set(first.discovery) & set(first.confirmation))
    assert not (set(first.validation) & set(first.confirmation))


def test_end_to_end_smoke():
    result = run_smoke(CONFIG)
    assert set(result) == {"random", "fixed_multistage", "auditor"}
    assert all(len(values) == 5 for values in result.values())
    assert all(value >= 0 for values in result.values() for value in values)

