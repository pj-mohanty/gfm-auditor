from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from .data import CODON_TABLE, CodingSequence, translate


@dataclass(frozen=True)
class Candidate:
    source_id: str
    sequence: str
    codon_positions: tuple[int, ...]
    depth: int


def synonymous_alternatives(codon: str) -> tuple[str, ...]:
    aa = CODON_TABLE[codon]
    return tuple(sorted(c for c, residue in CODON_TABLE.items() if residue == aa and c != codon))


def enumerate_synonymous_candidates(source: CodingSequence, depth: int) -> list[Candidate]:
    codons = list(source.codons)
    eligible = [i for i in range(1, len(codons) - 1) if synonymous_alternatives(codons[i])]
    candidates: list[Candidate] = []
    for positions in combinations(eligible, depth):
        choices = [synonymous_alternatives(codons[i]) for i in positions]
        for replacements in product(*choices):
            changed = codons.copy()
            for position, replacement in zip(positions, replacements):
                changed[position] = replacement
            sequence = "".join(changed)
            if translate(sequence) != translate(source.sequence):
                raise AssertionError("candidate failed translation preservation")
            candidates.append(Candidate(source.gene_id, sequence, positions, depth))
    return candidates

