# AgenticLS Closed-Loop Auditor

Colab-compatible research scaffold for evaluating whether adaptive orchestration
discovers independently confirmed, task-relative translation-preserving
representation-ordering violations more efficiently than fixed search policies.

## Design invariants

- Primary depth: `k=2`; `k=3` is secondary and `k=1` is diagnostic.
- One cumulative 40-candidate trajectory, checkpointed at 5, 10, 20, and 40.
- Discovery, online-validation, and hidden-confirmation controls are disjoint.
- Policies receive identical candidate-evaluation and validation allowances.
- Raw uncached model calls are logged separately from logical budgets.
- The source gene is the independent statistical unit.

## Layout

```text
config/experiment.yaml        Frozen experiment configuration
src/agenticls_auditor/        Shared implementation
tests/test_smoke.py            CPU-only end-to-end smoke test
notebooks/00_smoke_test.ipynb  Colab launcher
```

## Local or Colab smoke test

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m agenticls_auditor.smoke --config config/experiment.yaml
```

The smoke test uses a deterministic mock embedding model and five synthetic
coding sequences. Real model adapters should subclass `BaseModelAdapter`.

## Workstream ownership

- Data and controls: mutation/control universes and frozen bank manifests.
- Search and auditor: policies, validation actions, and budget accounting.
- Model adapters: real model integration, caching, and clean-runtime verification.
- Statistics and analysis: delta calibration, power, AUDC, result validation, and writing.