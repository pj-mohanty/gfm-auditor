"""Validate the three archived locked runs and reproduce the paper statistics.

Usage: python scripts/reproduce_paper_analysis.py --20m 20m.zip --116m 116m.zip --nucel nucel.zip
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from agenticls_auditor.inference import paired_gene_clustered_bootstrap
from agenticls_auditor.results import CheckpointResult, TrajectorySummary
from agenticls_auditor.statistics import confirmed_naudc, validate_checkpoint_results


SEEDS = (11, 22, 33, 44, 55)
POLICIES = ("random", "fixed_multistage", "auditor")
CHECKPOINTS = (5, 10, 20, 40)
INPUT_HASHES = (
    "9b7152e47dcd82f4f71010194b6d586a7f5c648be38a2df156010d7cc4643341",
    "3c2388efcf3b99ac4da3be7c140d468efb7abda8e70e712e094ec4604325780b",
)
MODELS = {
    "20m": ("omnidna-20m", 0.0004979782302045738),
    "116m": ("omnidna-116m", 2.1845102310180664e-05),
    "nucel": ("nucel-93m", 5.190535187188372e-06),
}


def analyze(path: Path, name: str) -> dict:
    model, delta = MODELS[name]
    summaries: list[TrajectorySummary] = []
    severities: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    incumbents: dict[tuple[str, int, int], str] = {}
    seen: set[tuple[str, str, int]] = set()
    run_identity: dict | None = None

    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.startswith("results/") and n.endswith(".json")]
        if len(names) != 750 or len(names) != len(set(names)):
            raise ValueError(f"{path}: expected exactly 750 distinct result members")
        for filename in sorted(names):
            payload = json.loads(archive.read(filename))
            identity = payload["run_identity"]
            if run_identity is None:
                run_identity = identity
            elif run_identity != identity:
                raise ValueError(f"{filename}: mixed run identities")
            if (identity["source_sha256"], identity["manifest_sha256"]) != INPUT_HASHES:
                raise ValueError(f"{filename}: frozen input hashes mismatch")
            if identity["delta"] != delta:
                raise ValueError(f"{filename}: calibration delta mismatch")

            summary = TrajectorySummary(**payload["summary"])
            key = (summary.gene_id, summary.policy, summary.seed)
            if key in seen or summary.model != model or summary.depth != 2 or summary.policy not in POLICIES or summary.seed not in SEEDS:
                raise ValueError(f"{filename}: unexpected or duplicate trajectory")
            if (payload["gene_id"], payload["policy"], payload["seed"]) != key:
                raise ValueError(f"{filename}: outer identity mismatch")
            seen.add(key)

            records = [CheckpointResult(**row) for row in payload["checkpoints"]]
            validate_checkpoint_results(records, minimum_confirmation_controls=8)
            if any((r.gene_id, r.policy, r.seed, r.model, r.delta) !=
                   (summary.gene_id, summary.policy, summary.seed, model, delta) for r in records):
                raise ValueError(f"{filename}: checkpoint identity mismatch")
            if abs(confirmed_naudc(records) - summary.normalized_confirmed_audc) > 1e-12:
                raise ValueError(f"{filename}: nAUDC does not recompute")
            if summary.checkpoint_count != 4:
                raise ValueError(f"{filename}: wrong checkpoint count")
            for record in records:
                severities[summary.policy][record.checkpoint].append(record.confirmed_severity)
                if summary.policy in ("auditor", "fixed_multistage"):
                    incumbents[(summary.gene_id, summary.seed, record.checkpoint, summary.policy)] = record.candidate_sequence
            summaries.append(summary)

    genes = {s.gene_id for s in summaries}
    if len(genes) != 50 or seen != {(g, p, s) for g in genes for p in POLICIES for s in SEEDS}:
        raise ValueError(f"{path}: incomplete paired 50-gene policy/seed design")
    agreement = {
        str(q): sum(incumbents[g, seed, q, "auditor"] == incumbents[g, seed, q, "fixed_multistage"]
                    for g in genes for seed in SEEDS)
        for q in CHECKPOINTS
    }
    comparisons = {
        baseline: asdict(paired_gene_clustered_bootstrap(
            summaries, model=model, depth=2, policy_a="auditor", policy_b=baseline,
            expected_seeds=SEEDS, bootstrap_replicates=10_000, bootstrap_seed=2026,
        ))
        for baseline in ("fixed_multistage", "random")
    }
    return {
        "model": model, "genes": 50, "trajectories": len(summaries),
        "checkpoints": 4 * len(summaries), "delta": delta,
        "source_sha256": INPUT_HASHES[0], "manifest_sha256": INPUT_HASHES[1],
        "mean_naudc": {p: mean(s.normalized_confirmed_audc for s in summaries if s.policy == p) for p in POLICIES},
        "mean_severity_by_checkpoint": {p: {str(q): mean(severities[p][q]) for q in CHECKPOINTS} for p in POLICIES},
        "auditor_fixed_same_incumbent_out_of_250": agreement,
        "paired_gene_bootstrap": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODELS:
        parser.add_argument(f"--{name}", required=True, type=Path, help="analysis bundle ZIP")
    parser.add_argument("--output", type=Path, help="write computed results as JSON")
    args = parser.parse_args()
    report = {name: analyze(getattr(args, name), name) for name in MODELS}
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
