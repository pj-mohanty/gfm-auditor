from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .metrics import median_control_margin
from .mutations import Candidate
from .results import ConfirmationBankReference
from .search.trajectory import ScoreResult


class EmbeddingCallLike(Protocol):
    embedding: np.ndarray
    cache_hit: bool
    raw_model_calls: int


class InformativeAdapter(Protocol):
    name: str

    def embed_with_info(
        self,
        sequence: str,
    ) -> EmbeddingCallLike:
        ...


@dataclass(frozen=True)
class FrozenCandidateRecord:
    candidate: Candidate
    query_index: int
    candidate_hash: str
    control_manifest_hash: str
    discovery_controls: tuple[str, ...]
    validation_controls: tuple[str, ...]
    confirmation_controls: tuple[str, ...]

    def controls_for(
        self,
        bank: str,
    ) -> tuple[str, ...]:
        if bank == "discovery":
            return self.discovery_controls
        if bank == "validation":
            return self.validation_controls
        if bank == "confirmation":
            return self.confirmation_controls
        raise ValueError(f"unknown control bank: {bank}")

    def confirmation_reference(
        self,
    ) -> ConfirmationBankReference:
        return ConfirmationBankReference(
            bank_id=(
                f"{self.control_manifest_hash}:confirmation"
            ),
            control_count=len(self.confirmation_controls),
        )


@dataclass(frozen=True)
class FrozenGeneInputs:
    sequence_id: str
    gene_id: str
    gene_symbol: str
    source_sequence: str
    records: tuple[FrozenCandidateRecord, ...]

    @property
    def candidates(self) -> list[Candidate]:
        # Policies receive candidates only. They never receive controls.
        return [record.candidate for record in self.records]

    def record_for_sequence(
        self,
        sequence: str,
    ) -> FrozenCandidateRecord:
        matches = [
            record
            for record in self.records
            if record.candidate.sequence == sequence
        ]

        if len(matches) != 1:
            raise KeyError(
                "candidate sequence does not resolve to "
                "exactly one frozen record"
            )

        return matches[0]


@dataclass(frozen=True)
class FrozenDevelopmentBundle:
    genes: tuple[FrozenGeneInputs, ...]
    source_sha256: str
    manifest_sha256: str
    depth: int
    candidates_per_gene: int


def file_sha256(path: str | Path) -> str:
    digest = sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_frozen_development_bundle(
    *,
    sources_path: str | Path,
    manifest_path: str | Path,
    metadata_path: str | Path,
) -> FrozenDevelopmentBundle:
    sources_path = Path(sources_path)
    manifest_path = Path(manifest_path)
    metadata_path = Path(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    observed_source_hash = file_sha256(sources_path)
    observed_manifest_hash = file_sha256(manifest_path)

    if (
        observed_source_hash
        != metadata["source_file_sha256"]
    ):
        raise ValueError(
            "frozen source file hash does not match metadata"
        )

    if (
        observed_manifest_hash
        != metadata["manifest_file_sha256"]
    ):
        raise ValueError(
            "frozen manifest file hash does not match metadata"
        )

    with sources_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        source_rows = list(csv.DictReader(handle))

    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(source_rows) != int(metadata["genes"]):
        raise ValueError(
            "source row count does not match metadata"
        )

    if (
        len(manifest_rows)
        != int(metadata["candidate_count"])
    ):
        raise ValueError(
            "manifest row count does not match metadata"
        )

    sources_by_gene = {
        row["gene_id"]: row
        for row in source_rows
    }

    if len(sources_by_gene) != len(source_rows):
        raise ValueError(
            "source genes must be unique"
        )

    rows_by_gene: dict[str, list[dict]] = defaultdict(
        list
    )

    for row in manifest_rows:
        rows_by_gene[row["gene_id"]].append(row)

    if set(rows_by_gene) != set(sources_by_gene):
        raise ValueError(
            "manifest and source files contain different genes"
        )

    expected_per_gene = int(
        metadata["candidates_per_gene"]
    )
    expected_depth = int(metadata["depth"])

    minimum_banks = metadata[
        "minimum_required_banks"
    ]

    genes: list[FrozenGeneInputs] = []

    for gene_id in sorted(sources_by_gene):
        source = sources_by_gene[gene_id]
        gene_rows = sorted(
            rows_by_gene[gene_id],
            key=lambda row: int(row["query_index"]),
        )

        if len(gene_rows) != expected_per_gene:
            raise ValueError(
                "gene does not contain the expected "
                "number of candidates"
            )

        observed_queries = [
            int(row["query_index"])
            for row in gene_rows
        ]

        if observed_queries != list(
            range(1, expected_per_gene + 1)
        ):
            raise ValueError(
                "candidate query indices are not complete"
            )

        records: list[FrozenCandidateRecord] = []
        seen_sequences: set[str] = set()

        for row in gene_rows:
            candidate_sequence = row[
                "candidate_sequence"
            ]

            if candidate_sequence in seen_sequences:
                raise ValueError(
                    "candidate sequences must be unique "
                    "within a gene"
                )

            seen_sequences.add(candidate_sequence)

            expected_candidate_hash = sha256(
                candidate_sequence.encode()
            ).hexdigest()

            if row["candidate_hash"] != expected_candidate_hash:
                raise ValueError(
                    "candidate hash does not match sequence"
                )

            if int(row["depth"]) != expected_depth:
                raise ValueError(
                    "candidate depth does not match metadata"
                )

            banks = {
                "discovery": tuple(
                    row["discovery_controls"]
                ),
                "validation": tuple(
                    row["validation_controls"]
                ),
                "confirmation": tuple(
                    row["confirmation_controls"]
                ),
            }

            for bank_name, controls in banks.items():
                if len(controls) != len(set(controls)):
                    raise ValueError(
                        f"{bank_name} controls contain duplicates"
                    )

                if len(controls) < int(
                    minimum_banks[bank_name]
                ):
                    raise ValueError(
                        f"{bank_name} control bank is too small"
                    )

            if (
                set(banks["discovery"])
                & set(banks["validation"])
            ):
                raise ValueError(
                    "discovery and validation controls overlap"
                )

            if (
                set(banks["discovery"])
                & set(banks["confirmation"])
            ):
                raise ValueError(
                    "discovery and confirmation controls overlap"
                )

            if (
                set(banks["validation"])
                & set(banks["confirmation"])
            ):
                raise ValueError(
                    "validation and confirmation controls overlap"
                )

            records.append(
                FrozenCandidateRecord(
                    candidate=Candidate(
                        source_id=gene_id,
                        sequence=candidate_sequence,
                        codon_positions=tuple(
                            int(position)
                            for position in row[
                                "codon_positions"
                            ]
                        ),
                        depth=int(row["depth"]),
                    ),
                    query_index=int(row["query_index"]),
                    candidate_hash=row["candidate_hash"],
                    control_manifest_hash=row[
                        "control_manifest_hash"
                    ],
                    discovery_controls=banks[
                        "discovery"
                    ],
                    validation_controls=banks[
                        "validation"
                    ],
                    confirmation_controls=banks[
                        "confirmation"
                    ],
                )
            )

        genes.append(
            FrozenGeneInputs(
                sequence_id=source["sequence_id"],
                gene_id=gene_id,
                gene_symbol=source["gene_symbol"],
                source_sequence=source[
                    "original_sequence"
                ],
                records=tuple(records),
            )
        )

    return FrozenDevelopmentBundle(
        genes=tuple(genes),
        source_sha256=observed_source_hash,
        manifest_sha256=observed_manifest_hash,
        depth=expected_depth,
        candidates_per_gene=expected_per_gene,
    )


class ControlBankScorer:
    def __init__(
        self,
        *,
        gene: FrozenGeneInputs,
        adapter: InformativeAdapter,
    ) -> None:
        self.gene = gene
        self.adapter = adapter

    def score(
        self,
        candidate: Candidate,
        *,
        bank: str,
    ) -> ScoreResult:
        record = self.gene.record_for_sequence(
            candidate.sequence
        )
        controls = record.controls_for(bank)

        sequences = (
            self.gene.source_sequence,
            candidate.sequence,
            *controls,
        )

        calls = [
            self.adapter.embed_with_info(sequence)
            for sequence in sequences
        ]

        source_embedding = calls[0].embedding
        candidate_embedding = calls[1].embedding
        control_embeddings = [
            call.embedding
            for call in calls[2:]
        ]

        margin = median_control_margin(
            source_embedding,
            candidate_embedding,
            control_embeddings,
        )

        return ScoreResult(
            value=margin,
            raw_model_calls=sum(
                int(call.raw_model_calls)
                for call in calls
            ),
            cache_hits=sum(
                int(call.cache_hit)
                for call in calls
            ),
        )

    def discovery(
        self,
        candidate: Candidate,
    ) -> ScoreResult:
        return self.score(
            candidate,
            bank="discovery",
        )

    def validation(
        self,
        candidate: Candidate,
    ) -> ScoreResult:
        return self.score(
            candidate,
            bank="validation",
        )

    def confirmation(
        self,
        candidate: Candidate,
    ) -> ScoreResult:
        return self.score(
            candidate,
            bank="confirmation",
        )

    def confirmation_reference(
        self,
        candidate_sequence: str,
    ) -> ConfirmationBankReference:
        record = self.gene.record_for_sequence(
            candidate_sequence
        )
        return record.confirmation_reference()



@dataclass(frozen=True)
class IntegratedTrajectory:
    trajectory: object
    checkpoint_records: tuple[object, ...]
    summary: object


def make_policy(
    policy_name: str,
    *,
    seed: int,
):
    from .auditor import ClosedLoopAuditor
    from .search.base import (
        FixedMultistagePolicy,
        RandomPolicy,
    )

    factories = {
        "random": RandomPolicy,
        "fixed_multistage": FixedMultistagePolicy,
        "auditor": ClosedLoopAuditor,
    }

    if policy_name not in factories:
        raise ValueError(
            f"unknown policy: {policy_name}"
        )

    return factories[policy_name](seed)


def run_integrated_trajectory(
    *,
    gene: FrozenGeneInputs,
    adapter: InformativeAdapter,
    policy_name: str,
    seed: int,
    delta: float,
    commit: str,
) -> IntegratedTrajectory:
    from .search.trajectory import run_trajectory
    from .statistics import (
        checkpoint_results_from_trajectory,
        summarize_trajectory,
    )

    if len(gene.candidates) != 40:
        raise ValueError(
            "integration trajectory requires "
            "exactly 40 frozen candidates"
        )

    policy = make_policy(
        policy_name,
        seed=seed,
    )

    scorer = ControlBankScorer(
        gene=gene,
        adapter=adapter,
    )

    trajectory = run_trajectory(
        policy=policy,
        candidates=gene.candidates,
        discovery_score=scorer.discovery,
        validation_score=scorer.validation,
        confirmation_score=scorer.confirmation,
        max_candidates=40,
        checkpoints=(5, 10, 20, 40),
        validation_allowances={
            5: 2,
            10: 3,
            20: 5,
            40: 10,
        },
    )

    records = checkpoint_results_from_trajectory(
        trajectory=trajectory,
        gene_id=gene.gene_id,
        model=adapter.name,
        policy=policy_name,
        seed=seed,
        depth=2,
        delta=delta,
        confirmation_lookup=(
            scorer.confirmation_reference
        ),
        expected_checkpoints=(5, 10, 20, 40),
        commit=commit,
    )

    summary = summarize_trajectory(records)

    return IntegratedTrajectory(
        trajectory=trajectory,
        checkpoint_records=tuple(records),
        summary=summary,
    )
