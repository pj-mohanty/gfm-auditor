import numpy as np
import pytest

from agenticls_auditor.models import NucEL93MAdapter, OmniDNA116MAdapter


class StubNucEL(NucEL93MAdapter):
    def _embed_uncached(self, sequence):
        return np.ones(512, dtype=np.float32), {
            "bases": len(sequence), "biological_tokens": len(sequence),
            "window_count": 1, "window_token_lengths": [len(sequence)],
        }, 1


def test_distinct_checkpoint_and_cache(tmp_path):
    adapter = StubNucEL(cache_dir=tmp_path)
    first = adapter.embed_with_info("atggcttaa")
    second = adapter.embed_with_info("ATGGCTTAA")
    assert first.embedding.shape == (512,)
    assert first.raw_model_calls == 1
    assert second.cache_hit and second.raw_model_calls == 0
    assert first.metadata["revision"] == NucEL93MAdapter.revision
    assert first.metadata["weights_sha256"] == NucEL93MAdapter.weights_sha256
    assert adapter.cache_namespace != OmniDNA116MAdapter().cache_namespace


def test_invalid_sequence_and_width():
    adapter = StubNucEL()
    with pytest.raises(ValueError):
        adapter.embed("ACNT")
    with pytest.raises(RuntimeError, match="shape"):
        adapter._validate_embedding(np.zeros(256, dtype=np.float32))


def test_pinned_single_nucleotide_vocabulary():
    assert NucEL93MAdapter.nucleotide_ids == {
        "A": 11, "C": 12, "G": 13, "T": 14,
    }
