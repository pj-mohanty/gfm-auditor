from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import floor
import json

from .data import CODON_TABLE, CodingSequence, translate
from .mutations import Candidate


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(a != b for a, b in zip(left, right))


def matched_nonsynonymous_codons(
    source_codon: str,
    nucleotide_distance: int,
) -> tuple[str, ...]:
    """Return non-stop nonsynonymous codons at an exact nucleotide distance."""
    if source_codon not in CODON_TABLE:
        raise ValueError(f"unknown source codon: {source_codon}")
    if nucleotide_distance not in (1, 2, 3):
        raise ValueError("codon nucleotide distance must be 1, 2, or 3")

    source_aa = CODON_TABLE[source_codon]

    return tuple(
        sorted(
            codon
            for codon, amino_acid in CODON_TABLE.items()
            if amino_acid not in (source_aa, "*")
            and hamming_distance(source_codon, codon)
            == nucleotide_distance
        )
    )


def enumerate_matched_nonsynonymous_controls(
    source: CodingSequence,
    candidate: Candidate,
) -> list[str]:
    """Enumerate all unique controls matched to a synonymous candidate.

    Each control edits the same codon positions as the candidate, matches the
    candidate's per-codon nucleotide distances, changes the encoded amino acid,
    and does not introduce a stop codon.
    """
    if candidate.source_id != source.gene_id:
        raise ValueError("candidate source_id does not match source gene_id")
    if candidate.depth != len(candidate.codon_positions):
        raise ValueError("candidate depth does not match codon_positions")
    if len(candidate.sequence) != len(source.sequence):
        raise ValueError("candidate and source lengths differ")
    if translate(candidate.sequence) != translate(source.sequence):
        raise ValueError("candidate is not translation-preserving")

    source_codons = list(source.codons)
    candidate_codons = [
        candidate.sequence[i : i + 3]
        for i in range(0, len(candidate.sequence), 3)
    ]

    actual_positions = tuple(
        i
        for i, (source_codon, candidate_codon) in enumerate(
            zip(source_codons, candidate_codons)
        )
        if source_codon != candidate_codon
    )

    if actual_positions != tuple(candidate.codon_positions):
        raise ValueError(
            "candidate codon_positions do not match the changed codons"
        )

    choices = []

    for position in candidate.codon_positions:
        # Start and terminal stop codons are protected.
        if position == 0 or position == len(source_codons) - 1:
            raise ValueError("candidate edits a protected start/stop position")

        source_codon = source_codons[position]
        candidate_codon = candidate_codons[position]
        distance = hamming_distance(source_codon, candidate_codon)

        alternatives = matched_nonsynonymous_codons(
            source_codon,
            distance,
        )

        if not alternatives:
            return []

        choices.append(alternatives)

    controls = set()

    for replacements in product(*choices):
        changed = source_codons.copy()

        for position, replacement in zip(
            candidate.codon_positions,
            replacements,
        ):
            changed[position] = replacement

        control = "".join(changed)

        if hamming_distance(control, source.sequence) != hamming_distance(
            candidate.sequence,
            source.sequence,
        ):
            raise AssertionError("control failed edit-distance matching")

        if translate(control) == translate(source.sequence):
            raise AssertionError("control remained translation-preserving")

        controls.add(control)

    return sorted(controls)


@dataclass(frozen=True)
class ControlBanks:
    discovery: tuple[str, ...]
    validation: tuple[str, ...]
    confirmation: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return (
            len(self.discovery) >= 4
            and len(self.validation) >= 4
            and len(self.confirmation) >= 8
        )

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            {
                "discovery": self.discovery,
                "validation": self.validation,
                "confirmation": self.confirmation,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode()).hexdigest()


def partition_controls(
    controls: list[str],
    *,
    source_id: str,
    candidate_sequence: str,
    assignment_seed: int,
) -> ControlBanks:
    unique = sorted(set(controls))
    candidate_hash = sha256(candidate_sequence.encode()).hexdigest()

    def key(control: str) -> str:
        payload = (
            f"{assignment_seed}|{source_id}|"
            f"{candidate_hash}|{control}"
        )
        return sha256(payload.encode()).hexdigest()

    ordered = sorted(unique, key=key)
    n = len(ordered)
    n_discovery = floor(0.25 * n)
    n_validation = floor(0.25 * n)

    return ControlBanks(
        discovery=tuple(ordered[:n_discovery]),
        validation=tuple(
            ordered[n_discovery : n_discovery + n_validation]
        ),
        confirmation=tuple(
            ordered[n_discovery + n_validation :]
        ),
    )
