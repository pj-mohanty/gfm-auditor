"""Rerun a frozen 50-gene, 40-query locked model evaluation.

Each model is run in its own environment. This script requires the original
source CSV, control manifest JSONL, and metadata JSON, with pinned input hashes.
It does not regenerate or silently change candidate/control assignments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from agenticls_auditor.integration import file_sha256, load_frozen_development_bundle, run_integrated_trajectory
from agenticls_auditor.models import OmniDNA20MAdapter, OmniDNA116MAdapter, NucEL93MAdapter
from agenticls_auditor.statistics import confirmed_naudc


MODELS = {
    "20m": (OmniDNA20MAdapter, 0.0004979782302045738),
    "116m": (OmniDNA116MAdapter, 2.1845102310180664e-05),
    "nucel": (NucEL93MAdapter, 5.190535187188372e-06),
}
SOURCE_SHA256 = "9b7152e47dcd82f4f71010194b6d586a7f5c648be38a2df156010d7cc4643341"
MANIFEST_SHA256 = "3c2388efcf3b99ac4da3be7c140d468efb7abda8e70e712e094ec4604325780b"
POLICIES = ("random", "fixed_multistage", "auditor")
SEEDS = (11, 22, 33, 44, 55)


def check_saved(path: Path, *, gene: str, policy: str, seed: int, model: str, identity: dict) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    summary = saved["summary"]
    if saved["run_identity"] != identity or (saved["gene_id"], saved["policy"], saved["seed"]) != (gene, policy, seed):
        raise ValueError(f"completed run has a different identity: {path}")
    if (summary["model"], summary["depth"], summary["checkpoint_count"]) != (model, 2, 4):
        raise ValueError(f"completed run has wrong summary: {path}")
    from agenticls_auditor.results import CheckpointResult
    rows = [CheckpointResult(**record) for record in saved["checkpoints"]]
    if any((r.gene_id, r.policy, r.seed) != (gene, policy, seed) for r in rows):
        raise ValueError(f"completed run has wrong checkpoints: {path}")
    if abs(confirmed_naudc(rows) - summary["normalized_confirmed_audc"]) > 1e-12:
        raise ValueError(f"completed run has inconsistent nAUDC: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true", help="validate inputs without loading a model")
    args = parser.parse_args()
    adapter_class, delta = MODELS[args.model]
    if file_sha256(args.sources) != SOURCE_SHA256 or file_sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("frozen source or control manifest SHA256 mismatch")
    bundle = load_frozen_development_bundle(
        sources_path=args.sources, manifest_path=args.manifest, metadata_path=args.metadata,
    )
    if len(bundle.genes) != 50 or bundle.depth != 2 or bundle.candidates_per_gene != 40:
        raise ValueError("frozen cohort is not the 50-gene, k=2, 40-candidate design")
    print(f"Verified 50 genes and 2,000 candidates; model={adapter_class.name}; delta={delta}", flush=True)
    if args.preflight_only:
        return

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime required for --device cuda")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    identity = {
        "commit": commit, "source_sha256": SOURCE_SHA256, "manifest_sha256": MANIFEST_SHA256,
        "delta": delta, "model_revision": adapter_class.revision,
        "weights_sha256": adapter_class.weights_sha256,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    adapter = adapter_class(cache_dir=args.cache, verify_checkpoint=True, device=args.device)
    complete = 0
    for gene in bundle.genes:
        for policy in POLICIES:
            for seed in SEEDS:
                target = args.output / f"{gene.gene_id}__{policy}__{seed}.json"
                if target.exists():
                    check_saved(target, gene=gene.gene_id, policy=policy, seed=seed,
                                model=adapter.name, identity=identity)
                else:
                    run = run_integrated_trajectory(
                        gene=gene, adapter=adapter, policy_name=policy, seed=seed,
                        delta=delta, commit=commit,
                    )
                    payload = {
                        "gene_id": gene.gene_id, "policy": policy, "seed": seed,
                        "run_identity": identity,
                        "summary": asdict(run.summary),
                        "checkpoints": [asdict(row) for row in run.checkpoint_records],
                        "events": [asdict(event) for event in run.trajectory.events],
                    }
                    temporary = target.with_suffix(".json.partial")
                    temporary.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
                    temporary.replace(target)
                complete += 1
                if complete % 15 == 0:
                    print(f"{complete}/750 verified", flush=True)


if __name__ == "__main__":
    main()
