"""Two-phase, resumable locked Omni-DNA-20M experiment launcher."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from .integration import (
    ControlBankScorer,
    file_sha256,
    load_frozen_development_bundle,
    make_policy,
)
from .locked import load_locked_config
from .models.omnidna import OmniDNA20MAdapter
from .search.trajectory import run_trajectory


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists():
        raise FileExistsError(f"completed output already exists: {path}")
    # A partial write has no standing as a completed trajectory.
    temporary.unlink(missing_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _gene_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["gene_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source gene IDs")
    return set(ids)


def _preflight(args):
    config_path = Path(args.config)
    config = load_locked_config(config_path)
    drive = Path(args.drive)
    locked = config["locked_split"]
    source = drive / "data" / locked["source_filename"]
    manifest = drive / "control_manifests" / locked["manifest_filename"]
    metadata = drive / "control_manifests" / "locked_primary_50_metadata.json"
    development = drive / "data" / "development_probe_5_sources.csv"

    for path in (source, manifest, metadata, development):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(source) != locked["source_sha256"]:
        raise ValueError("locked source hash mismatch")
    if file_sha256(manifest) != locked["manifest_sha256"]:
        raise ValueError("locked manifest hash mismatch")
    if file_sha256(development) != config["calibration"]["development_source_sha256"]:
        raise ValueError("development source hash mismatch")
    if _gene_ids(source) & _gene_ids(development):
        raise ValueError("development and locked genes overlap")

    bundle = load_frozen_development_bundle(
        sources_path=source, manifest_path=manifest, metadata_path=metadata,
    )
    if len(bundle.genes) != 50 or bundle.depth != 2 or bundle.candidates_per_gene != 40:
        raise ValueError("locked bundle dimensions mismatch")
    if config["model"]["revision"] != OmniDNA20MAdapter.revision:
        raise ValueError("model revision mismatch")
    if config["model"]["weights_sha256"] != OmniDNA20MAdapter.weights_sha256:
        raise ValueError("model weights hash mismatch")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("repository must be clean before a locked run")
    config_hash = sha256(config_path.read_bytes()).hexdigest()
    identity = {
        "commit": commit,
        "config_sha256": config_hash,
        "source_sha256": bundle.source_sha256,
        "manifest_sha256": bundle.manifest_sha256,
        "model": OmniDNA20MAdapter.name,
        "depth": 2,
    }
    return config, bundle, identity, drive


def _runs(config, bundle):
    for gene in bundle.genes:
        for policy in config["trajectory"]["policies"]:
            for seed in config["trajectory"]["seeds"]:
                yield gene, policy, seed


def _path(output: Path, gene_id: str, policy: str, seed: int) -> Path:
    return output / f"{gene_id}__{policy}__{seed}.json"


def _check_search(path, gene, policy, seed, identity):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("identity") != identity:
        raise ValueError(f"run identity mismatch: {path}")
    if (record.get("gene_id"), record.get("policy"), record.get("seed")) != (
        gene.gene_id, policy, seed,
    ):
        raise ValueError(f"run key mismatch: {path}")
    events, checkpoints = record["events"], record["checkpoints"]
    if [event["query_index"] for event in events] != list(range(1, 41)):
        raise ValueError(f"invalid event sequence: {path}")
    if len({event["candidate_sequence"] for event in events}) != 40:
        raise ValueError(f"duplicate candidate: {path}")
    if {event["candidate_sequence"] for event in events} != {
        candidate.sequence for candidate in gene.candidates
    }:
        raise ValueError(f"candidate universe mismatch: {path}")
    if any("confirmation_margin" in event for event in events):
        raise ValueError(f"confirmation leaked into event: {path}")
    best = None
    for event in events:
        if event["candidate_evaluations"] != event["query_index"]:
            raise ValueError(f"candidate count mismatch: {path}")
        q = event["query_index"]
        if event["validation_actions"] > next(
            config_allowance(point) for point in (5, 10, 20, 40) if q <= point
        ):
            raise ValueError(f"validation budget exceeded: {path}")
        if event["validation_performed"] != (event["validation_margin"] is not None):
            raise ValueError(f"validation record mismatch: {path}")
        if best is None or event["discovery_margin"] < best[1]:
            best = (event["candidate_sequence"], event["discovery_margin"])
        if (event["incumbent_sequence"], event["incumbent_discovery_margin"]) != best:
            raise ValueError(f"incumbent selection mismatch: {path}")
    if [checkpoint["checkpoint"] for checkpoint in checkpoints] != [5, 10, 20, 40]:
        raise ValueError(f"invalid checkpoints: {path}")
    if any("confirmation_margin" in checkpoint for checkpoint in checkpoints):
        raise ValueError(f"confirmation leaked into search: {path}")
    for checkpoint in checkpoints:
        event = events[checkpoint["checkpoint"] - 1]
        if checkpoint["incumbent_sequence"] != event["incumbent_sequence"]:
            raise ValueError(f"checkpoint incumbent mismatch: {path}")
        if checkpoint["validation_actions"] > config_allowance(checkpoint["checkpoint"]):
            raise ValueError(f"validation budget exceeded: {path}")
        if checkpoint["candidate_evaluations"] != checkpoint["checkpoint"]:
            raise ValueError(f"candidate budget mismatch: {path}")
    accounting = record["accounting"]
    if accounting["candidate_evaluations"] != 40:
        raise ValueError(f"incomplete candidate accounting: {path}")
    if accounting["validation_actions"] != events[-1]["validation_actions"]:
        raise ValueError(f"validation accounting mismatch: {path}")
    if any(row["event"] == "final_confirmation" for row in accounting["events"]):
        raise ValueError(f"confirmation call in search: {path}")
    if accounting["raw_model_calls"] != sum(row["raw_model_calls"] for row in accounting["events"]):
        raise ValueError(f"raw-call accounting mismatch: {path}")
    if accounting["cache_hits"] != sum(row["cache_hits"] for row in accounting["events"]):
        raise ValueError(f"cache accounting mismatch: {path}")
    return record


def config_allowance(checkpoint: int) -> int:
    return {5: 2, 10: 3, 20: 5, 40: 10}[checkpoint]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("search", "confirm"))
    parser.add_argument("--drive", required=True)
    parser.add_argument(
        "--config", default="config/locked_primary_omnidna20m_k2.yaml",
    )
    args = parser.parse_args()
    config, bundle, identity, drive = _preflight(args)
    root = drive / "locked_results" / f"omnidna20m_k2_{identity['commit'][:12]}"
    search_dir = root / "search"
    confirmation_dir = root / "confirmation"
    runs = list(_runs(config, bundle))

    # A complete search set must be checked before the first confirmation call.
    if args.phase == "confirm":
        expected = {_path(search_dir, gene.gene_id, policy, seed).name for gene, policy, seed in runs}
        found = {path.name for path in search_dir.glob("*.json")}
        if found != expected:
            raise ValueError("search directory has missing or unexpected run files")
        for gene, policy, seed in runs:
            path = _path(search_dir, gene.gene_id, policy, seed)
            if not path.is_file():
                raise ValueError(f"search trajectory missing: {path}")
            _check_search(path, gene, policy, seed, identity)
        print("All 750 search trajectories verified; starting confirmation.")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    adapter = OmniDNA20MAdapter(
        cache_dir=drive / "embeddings" / "omnidna20m_locked",
        verify_checkpoint=True,
        device="cuda",
    )

    for index, (gene, policy, seed) in enumerate(runs, 1):
        search_path = _path(search_dir, gene.gene_id, policy, seed)
        confirm_path = _path(confirmation_dir, gene.gene_id, policy, seed)
        if args.phase == "search":
            if search_path.exists():
                _check_search(search_path, gene, policy, seed, identity)
                continue
            scorer = ControlBankScorer(gene=gene, adapter=adapter)
            trajectory = run_trajectory(
                policy=make_policy(policy, seed=seed),
                candidates=gene.candidates,
                discovery_score=scorer.discovery,
                validation_score=scorer.validation,
                confirmation_score=None,
                max_candidates=40,
                checkpoints=(5, 10, 20, 40),
                validation_allowances={5: 2, 10: 3, 20: 5, 40: 10},
            )
            payload = {
                "identity": identity, "gene_id": gene.gene_id,
                "policy": policy, "seed": seed,
                "events": [asdict(event) for event in trajectory.events],
                "checkpoints": [
                    {key: value for key, value in asdict(row).items()
                     if key != "confirmation_margin"}
                    for row in trajectory.checkpoints
                ],
                "accounting": {
                    "candidate_evaluations": trajectory.accountant.candidate_evaluations,
                    "validation_actions": trajectory.accountant.validation_actions,
                    "raw_model_calls": trajectory.accountant.raw_model_calls,
                    "cache_hits": trajectory.accountant.cache_hits,
                    "events": trajectory.accountant.events,
                },
            }
            _atomic_json(search_path, payload)
            _check_search(search_path, gene, policy, seed, identity)
        else:
            if confirm_path.exists():
                existing = json.loads(confirm_path.read_text(encoding="utf-8"))
                if (
                    existing.get("identity") != identity
                    or (existing.get("gene_id"), existing.get("policy"), existing.get("seed"))
                    != (gene.gene_id, policy, seed)
                    or [row.get("checkpoint") for row in existing.get("results", [])]
                    != [5, 10, 20, 40]
                    or any(
                        row.get("candidate_hash") != sha256(
                            checkpoint["incumbent_sequence"].encode()
                        ).hexdigest()
                        for row, checkpoint in zip(
                            existing["results"],
                            _check_search(search_path, gene, policy, seed, identity)["checkpoints"],
                        )
                    )
                ):
                    raise ValueError(f"invalid existing confirmation: {confirm_path}")
                continue
            record = _check_search(search_path, gene, policy, seed, identity)
            scorer = ControlBankScorer(gene=gene, adapter=adapter)
            results = []
            for checkpoint in record["checkpoints"]:
                candidate = gene.record_for_sequence(checkpoint["incumbent_sequence"]).candidate
                score = scorer.confirmation(candidate)
                results.append({
                    "checkpoint": checkpoint["checkpoint"],
                    "candidate_hash": sha256(candidate.sequence.encode()).hexdigest(),
                    "confirmation_margin": score.value,
                    "raw_model_calls": score.raw_model_calls,
                    "cache_hits": score.cache_hits,
                })
            _atomic_json(confirm_path, {
                "identity": identity, "gene_id": gene.gene_id,
                "policy": policy, "seed": seed, "results": results,
            })
        print(f"{args.phase}: {index}/750 complete", flush=True)


if __name__ == "__main__":
    main()
