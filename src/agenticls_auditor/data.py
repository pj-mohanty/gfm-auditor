from __future__ import annotations

from dataclasses import dataclass


# Standard nuclear genetic code.
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA_TO_CODONS = {
    amino_acid: tuple(
        sorted(
            codon
            for codon, residue in CODON_TABLE.items()
            if residue == amino_acid
        )
    )
    for amino_acid in sorted(set(CODON_TABLE.values()))
}


@dataclass(frozen=True)
class CodingSequence:
    gene_id: str
    sequence: str

    def __post_init__(self) -> None:
        seq = self.sequence.upper()

        if not self.gene_id:
            raise ValueError("gene_id must be non-empty")
        if len(seq) == 0:
            raise ValueError("CDS must be non-empty")
        if len(seq) % 3:
            raise ValueError("CDS length must be divisible by three")
        if any(base not in "ACGT" for base in seq):
            raise ValueError("CDS contains non-ACGT symbols")

        object.__setattr__(self, "sequence", seq)

    @property
    def codons(self) -> tuple[str, ...]:
        return tuple(
            self.sequence[i : i + 3]
            for i in range(0, len(self.sequence), 3)
        )


def translate(sequence: str) -> str:
    seq = sequence.upper()

    if len(seq) == 0:
        raise ValueError("sequence must be non-empty")
    if len(seq) % 3:
        raise ValueError("sequence length must be divisible by three")
    if any(base not in "ACGT" for base in seq):
        raise ValueError("sequence contains non-ACGT symbols")

    return "".join(
        CODON_TABLE[seq[i : i + 3]]
        for i in range(0, len(seq), 3)
    )


def validate_complete_cds(sequence: str) -> tuple[bool, str]:
    """Validate canonical complete nuclear CDS criteria."""
    seq = sequence.upper()

    if not seq:
        return False, "empty_sequence"
    if any(base not in "ACGT" for base in seq):
        return False, "non_acgt"
    if len(seq) % 3:
        return False, "out_of_frame"
    if not seq.startswith("ATG"):
        return False, "noncanonical_start"

    protein = translate(seq)

    if not protein.endswith("*"):
        return False, "missing_terminal_stop"
    if "*" in protein[:-1]:
        return False, "internal_stop"

    return True, "valid"
