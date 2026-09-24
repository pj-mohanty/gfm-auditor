# Experiment Program

This directory documents the experimental program for the closed-loop genomic
foundation-model auditor. It records what has been completed, which evidence is
development-only, what has been frozen, and what remains to be run.

## Research question

Under identical candidate-evaluation and online-validation budgets, does a
closed-loop rule-based auditor discover stronger independently confirmed
task-relative ordering violations more efficiently than a fixed multistage
search policy?

A task-relative ordering violation occurs when a translation-preserving
candidate lies farther from its source than edit-matched protein-changing
controls under a use case that expects protein-level similarity.

## Current milestone

The implementation and development phases are complete. The primary locked
experiment has been configured but has not yet been run.

| Component | Status | Evidence |
| --- | --- | --- |
| Complete CDS validation | Complete | Unit tests and five-gene development bundle |
| Synonymous candidates and matched controls | Complete | Deterministic disjoint control manifests |
| Search policies and auditor | Complete | Synthetic trajectories and integration tests |
| Candidate and validation budgets | Complete | Query-accounting tests |
| Omni-DNA-20M adapter | Complete | Real-model T4 validation and benchmark |
| Persistent embedding cache | Complete | Cache tests and real-model reuse |
| Checkpoint statistics and paired inference | Complete | Synthetic statistical tests |
| End-to-end integration | Complete | Fresh-clone tests and five-gene real-model run |
| Development delta calibration | Complete | Frozen numeric threshold and calibration records |
| Locked 50-gene split | Complete | Frozen source and control-manifest hashes |
| Locked primary inference | Not started | Run only from the frozen configuration |
| Hidden-confirmation analysis | Not started | Run after all locked trajectories are frozen |
| Paper figures and final results | Not started | Derived only from completed locked outputs |

## Experimental design

### Primary model and representation

- Model: `zehui127/Omni-DNA-20M`
- Revision: `3b64e6a5ed6c8f72bad76823ce728b3045243026`
- Weights SHA256: `f2927d1e3febd7f2309151fd1f6b47fc1e44500e6af6adad5342dcfc88b4680a`
- Representation: final hidden state
- Pooling: mean over biological tokens only
- Long-sequence aggregation: global biological-token-weighted mean
- Distance: cosine distance
- Maximum tokens per window: 250

### Search trajectory

- Primary mutation depth: `k=2`
- Candidate evaluations per trajectory: 40
- Frozen checkpoints: 5, 10, 20, and 40
- Seeds: 11, 22, 33, 44, and 55
- Policies: position-balanced random, fixed multistage, and closed-loop auditor
- Cumulative validation allowances: 2, 3, 5, and 10 at the four checkpoints

The fixed multistage policy uses position-balanced random search for queries
1–16, greedy search for queries 17–32, and beam search for queries 33–40.

### Control separation

Every candidate has three disjoint control banks:

1. Discovery controls are visible to search and define the optimization score.
2. Online-validation controls are accessible only through budgeted validation
   actions.
3. Hidden-confirmation controls are unavailable to policies and are joined only
   after checkpoint incumbents have been frozen.

Controls are edit matched, protein changing, non-stop, and deterministically
partitioned with assignment seed 42. Eligibility requires at least 4 discovery,
4 validation, and 8 confirmation controls.

## Development results

The real-model development experiment used five genes, three policies, five
seeds, and four checkpoints:

- 75 trajectories
- 300 checkpoint records
- 31 unique frozen checkpoint incumbents

These outputs are development-only and must not be reported as locked evidence.

### Delta calibration

The severity threshold was calibrated only on unique frozen development
checkpoint incumbents:

```text
delta = Q_0.95(abs(M_B1 - M_B2) / 2)
delta = 0.0004979782302045738
```

The two confirmation sub-banks were balanced, disjoint, and deterministically
ordered by SHA256. Calibration used 31 records. After applying the threshold,
216 of 300 checkpoint severities and 60 of 75 trajectory nAUDCs were nonzero.

### Development-only comparison

The auditor was modestly better than random but approximately tied with fixed
multistage. The primary development contrast was:

```text
auditor minus fixed multistage = -0.000003749983651297323
95% bootstrap CI = [-0.00003305094582693883, 0.000021800994873046854]
```

This result motivated a 50-gene locked experiment designed to estimate the
paired effect precisely rather than to tune the auditor further.

## Locked primary experiment

The locked experiment contains 50 untouched genes and 2,000 eligible
candidates. Its configuration is versioned at:

```text
config/locked_primary_omnidna20m_k2.yaml
```

The configuration is frozen and validated by
`agenticls_auditor.locked.load_locked_config`. Locked inference must refuse to
run if the configuration is unfrozen, the calibrated delta is missing or
invalid, the seed or checkpoint schedules change, or the source/manifest hashes
do not match.

See [LOCKED_PRIMARY.md](LOCKED_PRIMARY.md) for execution rules and
[ARTIFACTS.md](ARTIFACTS.md) for paths and hashes.

## Repository verification

At the time the lock configuration was merged:

- Main merge commit: `cedf302`
- Tests: 62 passed
- Smoke run: random, fixed multistage, and auditor each completed five examples
- Blocked-term scan: no matches

## Remaining order of work

1. Run all 750 locked trajectories with immediate resumable writes.
2. Validate trajectory completeness, budgets, hashes, and checkpoint freezing.
3. Join hidden confirmation only after trajectories are complete.
4. Compute calibrated severity and per-trajectory normalized AUDC.
5. Average seeds within each gene and policy.
6. Run the paired gene-clustered bootstrap for auditor minus fixed multistage.
7. Produce confirmed-discovery curves, paired per-gene effects, and audit tables.
8. Add secondary models or ablations only if the primary analysis is complete.
