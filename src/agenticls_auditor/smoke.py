from __future__ import annotations

import argparse
from pathlib import Path

from .auditor import ClosedLoopAuditor
from .config import load_config
from .data import CodingSequence
from .metrics import confirmed_severity, cosine_distance, trapezoidal_audc
from .models import DeterministicMockAdapter
from .mutations import enumerate_synonymous_candidates
from .query_budget import QueryAccountant
from .search import FixedMultistagePolicy, RandomPolicy
from .search.base import Observation


def synthetic_genes(count: int = 5) -> list[CodingSequence]:
    bodies = [
        "GCTGGTGCCGGCGCAGGAGCG",
        "GGTGCTGGCGCCGGAGCAGCG",
        "GCCGGAGCTGGTGCGGCAGGC",
        "GCAGCGGGTGCCGGAGCTGGC",
        "GCGGCCGGCGGAGCTGGTGCA",
    ]
    return [CodingSequence(f"smoke_{i+1}", "ATG" + bodies[i] + "TAA") for i in range(count)]


def run_smoke(config_path: str | Path) -> dict[str, list[float]]:
    cfg = load_config(config_path)
    model = DeterministicMockAdapter()
    output: dict[str, list[float]] = {name: [] for name in ("random", "fixed_multistage", "auditor")}
    for gene in synthetic_genes(cfg.raw["smoke_test"]["development_genes"]):
        candidates = enumerate_synonymous_candidates(gene, cfg.primary_depth)
        if len(candidates) < cfg.max_candidates:
            raise RuntimeError("smoke candidate space is too small")
        source_embedding = model.embed(gene.sequence)
        for seed in cfg.seeds[:1]:
            policies = [RandomPolicy(seed), FixedMultistagePolicy(seed), ClosedLoopAuditor(seed)]
            for policy in policies:
                history: list[Observation] = []
                accountant = QueryAccountant(cfg.max_candidates, 10)
                checkpoint_severity: list[float] = []
                for query_index in range(1, cfg.max_candidates + 1):
                    candidate = policy.choose(candidates, history, query_index)
                    margin = -cosine_distance(source_embedding, model.embed(candidate.sequence))
                    accountant.candidate(raw_calls=1)
                    history.append(Observation(candidate, margin))
                    if isinstance(policy, ClosedLoopAuditor) and policy.should_validate(
                        query_index, accountant.max_validations - accountant.validation_actions
                    ):
                        accountant.validation(cache_hits=1)
                    if query_index in cfg.checkpoints:
                        incumbent = min(history, key=lambda obs: obs.discovery_margin)
                        confirmation_margin = incumbent.discovery_margin * 0.9
                        severity = confirmed_severity(confirmation_margin, delta=0.01)
                        accountant.confirmation(cache_hits=1)
                        checkpoint_severity.append(severity)
                output[policy.name].append(trapezoidal_audc(list(cfg.checkpoints), checkpoint_severity))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    results = run_smoke(args.config)
    print({name: len(values) for name, values in results.items()})


if __name__ == "__main__":
    main()

