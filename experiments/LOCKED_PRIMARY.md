# Locked Primary Experiment Protocol

This document governs execution of the frozen 50-gene Omni-DNA-20M primary
experiment. The authoritative machine-readable specification is:

```text
config/locked_primary_omnidna20m_k2.yaml
```

## Before execution

The run must start from a fresh Colab GPU runtime and the current `main` branch.
Before model loading, verify all of the following:

- The locked configuration passes `load_locked_config`.
- `frozen` is exactly `true`.
- The calibrated delta is `0.0004979782302045738`.
- The source and manifest hashes match the configuration.
- The bundle contains 50 genes and 40 candidates per gene.
- Development and locked gene IDs are disjoint.
- The model revision and weights hash match the configuration.
- The runtime uses the pinned model dependencies and a CUDA device.

If any check fails, stop. Do not modify the locked files to make the check pass.

## Frozen trajectory matrix

The complete run matrix contains:

```text
50 genes x 3 policies x 5 seeds = 750 trajectories
750 trajectories x 4 checkpoints = 3,000 checkpoint incumbents
```

Each trajectory has 40 candidate evaluations. The policies are:

- `random`
- `fixed_multistage`
- `auditor`

The seeds are 11, 22, 33, 44, and 55.

## Resumable execution

Write every completed trajectory immediately to persistent Drive storage. Keep
an append-only completion ledger keyed by:

```text
gene_id | model | policy | seed | depth
```

On restart:

1. Reload and validate the frozen configuration and input hashes.
2. Validate every existing completed trajectory before trusting the ledger.
3. Skip only keys whose complete event and checkpoint records are present.
4. Resume from the first missing key in the deterministic run order.

Never overwrite a completed trajectory with a rerun unless the entire locked
experiment is formally invalidated and restarted under a new version.

## Information separation

During search, policies may receive only candidate identity, discovery scores,
and budgeted online-validation feedback. Policies must never access hidden
confirmation sequences, embeddings, margins, or summaries.

At each checkpoint, freeze the incumbent candidate hash before confirmation is
computed. Confirmation must not change which candidate was selected.

For the strongest audit trail, store pre-confirmation checkpoint records first.
Run the hidden-confirmation join only after all 750 trajectories have passed
completeness and budget checks.

## Required accounting

Every trajectory must record:

- candidate evaluations
- validation actions
- raw uncached model calls
- cache hits
- policy actions
- discovery margins
- online-validation margins when requested
- frozen incumbent hashes at each checkpoint
- configuration hash
- control-manifest hash
- source and model identifiers
- repository commit

Candidate-budget comparisons and raw-call comparisons are distinct. Report
both if caching or control-bank size causes them to diverge.

## Completion validation

Before confirmation, require:

- Exactly 750 unique trajectory keys
- Exactly 40 events per trajectory
- Query indices exactly 1 through 40
- No repeated candidate within a trajectory
- Exactly four frozen checkpoints at 5, 10, 20, and 40
- Candidate budget exactly 40
- Validation use never exceeding cumulative allowances
- No hidden-confirmation fields in search events
- Matching configuration, source, and manifest hashes everywhere

Any incomplete or malformed trajectory blocks analysis.

## Hidden confirmation and endpoint

After trajectories are frozen, apply the hidden-confirmation bank to each
checkpoint incumbent. Confirmed severity is:

```text
S_g(q) = max(0, -M_confirm,g(q) - delta)
```

with frozen `delta = 0.0004979782302045738`.

Compute normalized trapezoidal area across checkpoints 5, 10, 20, and 40.
Average seeds within each gene and policy. The independent statistical unit is
the gene.

The primary contrast is:

```text
mean_seed(nAUDC auditor) - mean_seed(nAUDC fixed_multistage)
```

Use 10,000 paired gene-clustered bootstrap replicates, bootstrap seed 2026,
and a 95% confidence interval.

## No-peeking rules

Until all locked trajectories are complete and frozen:

- Do not calculate policy-level confirmed nAUDC.
- Do not inspect confirmation outcomes by policy.
- Do not change the auditor, search stages, budgets, delta, seeds, or controls.
- Do not remove genes based on model outcomes.
- Do not rerun selectively because a policy result appears unfavorable.

Operational diagnostics such as completion counts, elapsed time, memory,
raw-call counts, and cache hits may be monitored.

## Allowed failure handling

Infrastructure failures may be retried without changing scientific settings.
Record the failure and retry status. Examples include runtime disconnection,
temporary download failure, or a corrupted partial write.

Scientific or schema failures require stopping and documenting the issue before
any protocol change. A changed protocol receives a new configuration version
and cannot silently replace the existing locked experiment.

## After the primary analysis

Interpret the locked result without changing the primary endpoint:

- Positive auditor effect: adaptive orchestration improved confirmed discovery.
- Near-zero effect: fixed search matched the auditor under the tested budgets.
- Negative effect: adaptivity reduced confirmed discovery efficiency.
- Weak confirmation: search optimized unstable discovery extremes.

Secondary models, mutation depths, representation settings, and ablations are
extensions. They must not replace or redefine the locked primary result.
