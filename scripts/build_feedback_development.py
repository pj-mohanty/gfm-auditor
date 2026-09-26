"""Create a separate 80-candidate development bundle for feedback search.

Run: python scripts/build_feedback_development.py /content/drive/MyDrive/AgenticLS
This script never writes to the original development or locked input paths.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess

from agenticls_auditor.controls import (
    enumerate_matched_nonsynonymous_controls,
    partition_controls,
)
from agenticls_auditor.data import CodingSequence, translate
from agenticls_auditor.integration import (
    file_sha256,
    load_frozen_development_bundle,
)
from agenticls_auditor.mutations import Candidate, synonymous_alternatives


def build(project: Path) -> None:
    old_sources = project / "data/development_probe_5_sources.csv"
    old_manifest = project / "control_manifests/development_probe_5_control_manifest.jsonl"
    old_metadata = project / "control_manifests/development_probe_5_metadata.json"
    original = load_frozen_development_bundle(
        sources_path=old_sources,
        manifest_path=old_manifest,
        metadata_path=old_metadata,
    )
    source_path = project / "data/feedback_development_5_sources.csv"
    manifest_path = project / "control_manifests/feedback_development_5x80_manifest.jsonl"
    metadata_path = project / "control_manifests/feedback_development_5x80_metadata.json"
    for path in (source_path, manifest_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    with old_sources.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
        columns = list(source_rows[0])
    assert len(source_rows) == 5
    source_by_id = {row["gene_id"]: row for row in source_rows}
    assert len(source_by_id) == 5
    source_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for gene_index, gene in enumerate(original.genes):
        source = CodingSequence(gene.gene_id, gene.source_sequence)
        protein = translate(source.sequence)
        codons = list(source.codons)
        positions = [
            i for i in range(1, len(codons) - 1)
            if synonymous_alternatives(codons[i])
        ]
        rng = random.Random(142 + gene_index)
        seen = set()
        attempts = 0
        while len(seen) < 80:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError(f"Insufficient eligible candidates for {gene.gene_id}")
            chosen = tuple(sorted(rng.sample(positions, 2)))
            changed = codons.copy()
            for position in chosen:
                changed[position] = rng.choice(
                    synonymous_alternatives(codons[position])
                )
            sequence = "".join(changed)
            if sequence in seen:
                continue
            assert translate(sequence) == protein
            candidate = Candidate(gene.gene_id, sequence, chosen, 2)
            controls = enumerate_matched_nonsynonymous_controls(
                source, candidate
            )
            banks = partition_controls(
                controls,
                source_id=gene.gene_id,
                candidate_sequence=sequence,
                assignment_seed=42,
            )
            if not banks.eligible:
                continue
            seen.add(sequence)
            manifest_rows.append({
                "schema_version": 1,
                "assignment_seed": 42,
                "sequence_id": gene.sequence_id,
                "gene_id": gene.gene_id,
                "gene_symbol": gene.gene_symbol,
                "query_index": len(seen),
                "depth": 2,
                "candidate_hash": sha256(sequence.encode()).hexdigest(),
                "candidate_sequence": sequence,
                "codon_positions": list(chosen),
                "discovery_controls": list(banks.discovery),
                "validation_controls": list(banks.validation),
                "confirmation_controls": list(banks.confirmation),
                "control_manifest_hash": banks.manifest_hash,
            })
        print(gene.gene_id, "80 eligible candidates", attempts, "draws", flush=True)

    assert len(manifest_rows) == 400
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(source_rows)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "schema_version": 1,
        "status": "prospective feedback development only",
        "genes": 5,
        "candidate_count": 400,
        "candidates_per_gene": 80,
        "depth": 2,
        "assignment_seed": 42,
        "candidate_seeds": [142, 143, 144, 145, 146],
        "all_candidates_eligible": True,
        "minimum_required_banks": {
            "discovery": 4, "validation": 4, "confirmation": 8,
        },
        "source_file_sha256": file_sha256(source_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "original_source_sha256": original.source_sha256,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    loaded = load_frozen_development_bundle(
        sources_path=source_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    assert len(loaded.genes) == 5
    assert all(len(gene.records) == 80 for gene in loaded.genes)
    print("Validated 5 genes and 400 eligible candidates")
    print("Source SHA256:", loaded.source_sha256)
    print("Manifest SHA256:", loaded.manifest_sha256)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    build(parser.parse_args().project_root)
