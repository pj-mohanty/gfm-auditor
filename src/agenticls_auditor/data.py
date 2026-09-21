from __future__ import annotations

from dataclasses import dataclass


CODON_TABLE = {
    # Compact standard-code subset used by the smoke test. Real data loaders
    # should replace this with a complete, versioned table.
    "ATG": "M", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "TTT": "F", "TTC": "F", "TAA": "*", "TAG": "*", "TGA": "*",
}


@dataclass(frozen=True)
class CodingSequence:
    gene_id: str
    sequence: str

    def __post_init__(self) -> None:
        seq = self.sequence.upper()
        if len(seq) % 3:
            raise ValueError("CDS length must be divisible by three")
        if any(base not in "ACGT" for base in seq):
            raise ValueError("CDS contains non-ACGT symbols")
        object.__setattr__(self, "sequence", seq)

    @property
    def codons(self) -> tuple[str, ...]:
        return tuple(self.sequence[i : i + 3] for i in range(0, len(self.sequence), 3))


def translate(sequence: str) -> str:
    codons = (sequence[i : i + 3] for i in range(0, len(sequence), 3))
    try:
        return "".join(CODON_TABLE[c] for c in codons)
    except KeyError as exc:
        raise ValueError(f"codon table missing {exc.args[0]}") from exc

