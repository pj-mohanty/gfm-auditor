# Reproducing the three-model $k=2$ paper

This guide distinguishes **re-analysis of saved results** (CPU, minutes) from
**rerunning model inference** (GPU, roughly hours per model). The paper covers
Omni-DNA-20M, Omni-DNA-116M, and NucEL-93M on the same 50 locked genes.
Each model has 50 genes × 3 policies × 5 seeds = 750 trajectories and 3,000
checkpoints. The prospective five-gene, 80-candidate feedback study is separate
development work, not an additional held-out result.

## 1. Installation and tests

Use Python 3.10 or newer. For CPU analysis alone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

The commands below use Unix-style paths. In PowerShell use
`.venv\Scripts\Activate.ps1` and replace `/` in local paths as needed.

## 2. Reproduce the reported estimates without a GPU

Obtain the three completed analysis bundle ZIP files and run:

```bash
python scripts/reproduce_paper_analysis.py \
  --20m /path/to/omnidna20m_k2_analysis_bundle.zip \
  --116m /path/to/omnidna116m_k2_analysis_bundle.zip \
  --nucel /path/to/nucel93m_k2_analysis_bundle.zip \
  --output paper_analysis.json
```

The script checks that each bundle has exactly 750 unique trajectories,
50 paired genes, all policy/seed cells, four valid hidden-confirmation
checkpoints per run, the frozen input hashes, model-specific calibration
tolerance, and nAUDC recomputed from those checkpoints. It then calls the
project's paired gene-bootstrap function (10,000 replicates; seed 2026) and
reports policy means, checkpoint means, auditor/fixed incumbent agreement, and
both auditor-minus-baseline intervals.

Expected auditor-minus-fixed effects (95% gene-bootstrap CIs):

| Model | Effect | CI |
| --- | ---: | ---: |
| Omni-DNA-20M | +2.9397136828341067e-06 | [-1.5497838739639615e-05, 2.6543625023641617e-05] |
| Omni-DNA-116M | +1.75544193813e-07 | [-9.76550153324e-07, 1.3635124479e-06] |
| NucEL-93M | -8.957603316523943e-09 | [-1.3963494673792519e-07, 8.956294004942887e-08] |

The 20M run was the initial locked experiment. The later 116M and NucEL
runs used the original locked candidate/control inputs and policy, with
model-specific calibration on the same 31 development incumbents. Distinct
embeddings and tolerances make cross-model absolute nAUDC values descriptive;
the paired within-model policy differences are the estimands.

## 3. Rerun model inference from frozen inputs

The frozen source CSV, control manifest JSONL, and cohort metadata JSON must
be supplied separately. They are not reconstructed from result summaries.
The expected filenames and SHA256s are:

| Input | Filename | SHA256 |
| --- | --- | --- |
| Held-out sources | `locked_primary_50_sources.csv` | `9b7152e47dcd82f4f71010194b6d586a7f5c648be38a2df156010d7cc4643341` |
| Candidate/control banks | `locked_primary_50_control_manifest.jsonl` | `3c2388efcf3b99ac4da3be7c140d468efb7abda8e70e712e094ec4604325780b` |
| Cohort metadata | `locked_primary_50_metadata.json` | Contains both preceding hashes and the 50 × 40 dimensions |

The loader rejects changed hashes, wrong dimensions, duplicate candidates,
bank overlap, and control banks smaller than the frozen minima. These files
come from the experiment's saved `data/` and `control_manifests/` directories.
The original 20M frozen configuration is
`config/locked_primary_omnidna20m_k2.yaml`. Tolerances are pinned in the
runner: 0.0004979782302045738 (20M), 2.1845102310180664e-05 (116M),
and 5.190535187188372e-06 (NucEL). Calibration metadata and split-bank
records should accompany a release of the frozen inputs.

First validate inputs without loading a model:

```bash
python scripts/run_paper_k2.py --model 20m \
  --sources /path/to/locked_primary_50_sources.csv \
  --manifest /path/to/locked_primary_50_control_manifest.jsonl \
  --metadata /path/to/locked_primary_50_metadata.json \
  --output /path/to/output/20m --cache /path/to/cache/20m --preflight-only
```

Then use a CUDA runtime, installing the relevant model extras in **separate
environments**: `python -m pip install -e '.[omnidna]'` for either Omni-DNA
checkpoint and `python -m pip install -e '.[nucel]'` for NucEL. The extras pin
different Transformers versions and are not intended to be combined.
Run the same command without `--preflight-only`, choosing `--model 20m`,
`--model 116m`, or `--model nucel` and separate output/cache directories.
Weights are downloaded at a pinned model revision and checked against the
adapter's SHA256. Public checkpoint access and a GPU are required.

The runner writes one JSON trajectory at a time and resumes by validating
completed files against the input/model identity. It does not overwrite a
completed trajectory. The run commit in newly produced files identifies the
current reproduction checkout and will differ from the historical commits.
The historical outputs were produced on Colab T4 GPUs; different supported
library and accelerator versions can cause small floating-point differences.

## 4. Review-anonymous release

Create a **new repository from a source export**, without the original `.git`
directory or commit history. Review every `README`, notebook output, URL,
Git config, result JSON, and archive metadata for names, emails, original
repository URLs, and commit hashes linking to the public repository. The
historical analysis bundles contain original commit identifiers in each
trajectory; do not publish them as-is in a blind-review repository. If sharing
redacted copies, state exactly which provenance fields were redacted and keep
the numerical checkpoint/summary fields unchanged. Never replace the input
hashes or quantitative results with fabricated values. Keep an unredacted
research archive for after review.

This code and the three result ZIPs support CPU reproduction of the analysis.
A full model rerun additionally requires distribution of the three frozen
input files above. Do not claim full inference reproducibility from the
result ZIPs alone.

To create an author-blind source and results ZIP locally (without pushing
anything to the identifiable GitHub repository), run:

```bash
python scripts/build_blind_review_archive.py \
  --20m /path/to/omnidna20m_k2_analysis_bundle.zip \
  --116m /path/to/omnidna116m_k2_analysis_bundle.zip \
  --nucel /path/to/nucel93m_k2_analysis_bundle.zip \
  --output AgenticLS_blind_review_repro.zip
```

The builder re-analyzes both original and redacted archives and fails if a
quantitative output changes. The result contains source and sanitized saved
trajectories for the **analysis**; the original frozen inputs remain a separate
requirement for a fresh model run. Review the exported ZIP yourself before
publishing it in an anonymous repository.
