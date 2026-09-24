# Experiment Artifact Registry

Large data, embeddings, and result files are stored outside Git. This registry
records their expected Google Drive paths and immutable hashes.

The Drive root is:

```text
/content/drive/MyDrive/AgenticLS
```

## Development input bundle

| Artifact | Relative path | SHA256 |
| --- | --- | --- |
| Five-gene sources | `data/development_probe_5_sources.csv` | `05a0ae4fa0ec6f5ee3f7577793b0e6539f4b1689cdf9e1788e1c418641a64b8b` |
| Five-gene control manifest | `control_manifests/development_probe_5_control_manifest.jsonl` | `81ddbf08bcc48baadbce9699a0653c087ab450f17b4a6f9d1b76df9a123926d6` |
| Five-gene metadata | `control_manifests/development_probe_5_metadata.json` | Recorded internally |

The bundle contains five genes, 40 candidates per gene, mutation depth two,
and 200 eligible candidates.

## Development results

Initial integrated development results:

```text
development_results/first_integration_omnidna20m_k2/
```

Expected files:

- `checkpoint_results.jsonl` — 300 checkpoint records
- `trajectory_summaries.jsonl` — 75 trajectories
- `within_gene_seed_means.jsonl` — 15 gene-policy means
- `paired_bootstrap_contrasts.json`
- `run_metadata.json`

Calibrated development results:

```text
development_results/first_integration_omnidna20m_k2_calibrated/
```

## Delta calibration

Directory:

```text
development_results/delta_calibration_omnidna20m_k2/
```

| Artifact | Value |
| --- | --- |
| Calibration records | `delta_calibration_records.jsonl` |
| Calibration metadata | `delta_calibration_metadata.json` |
| Metadata SHA256 | `9c0e82e320d168eb9685abcb80dd0065b44ff4d7071720950a752ecd56ee568d` |
| Frozen delta | `0.0004979782302045738` |
| Quantile | `0.95` |
| Calibration records | 31 |

## Power analysis

```text
development_results/power_analysis_omnidna20m_k2/power_analysis.json
```

The power calculation is development-only and based on five genes. It supports
a 50-gene locked set for useful precision, but its numerical power estimates
must be interpreted cautiously because the development variance estimate is
small and unstable.

## Locked input bundle

| Artifact | Relative path | SHA256 |
| --- | --- | --- |
| Locked sources | `data/locked_primary_50_sources.csv` | `9b7152e47dcd82f4f71010194b6d586a7f5c648be38a2df156010d7cc4643341` |
| Locked control manifest | `control_manifests/locked_primary_50_control_manifest.jsonl` | `3c2388efcf3b99ac4da3be7c140d468efb7abda8e70e712e094ec4604325780b` |
| Locked metadata | `control_manifests/locked_primary_50_metadata.json` | Recorded internally |

Locked-bundle properties:

- 50 genes
- 40 candidates per gene
- 2,000 candidates total
- 2,000 eligible candidates
- Minimum discovery controls: 5
- Minimum validation controls: 5
- Minimum confirmation controls: 10
- Development/locked overlap: zero

## Embedding cache

```text
embeddings/omnidna_20m/
```

Cache identity includes the model checkpoint, representation configuration,
and sequence. Cache hits do not count as raw model calls, but both quantities
must be retained in result metadata.

## Locked outputs

The locked runner should write to a new directory such as:

```text
locked_results/omnidna20m_k2_primary/
```

Expected outputs after completion:

- trajectory events
- checkpoint incumbents without confirmation fields
- completion ledger
- raw-call and cache-hit accounting
- run metadata and hashes
- hidden-confirmation records created only after trajectory completion
- calibrated checkpoint results
- trajectory summaries
- within-gene seed means
- paired bootstrap contrasts

Do not overwrite development outputs or reuse their result directories.
