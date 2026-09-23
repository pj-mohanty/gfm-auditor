import numpy as np
import pytest

from agenticls_auditor.models.omnidna import (
    OmniDNA20MAdapter,
)


class StubOmniDNA20MAdapter(OmniDNA20MAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uncached_calls = 0

    def _embed_uncached(self, sequence):
        self.uncached_calls += 1

        embedding = np.arange(
            self.embedding_width,
            dtype=np.float32,
        )

        metadata = {
            "bases": len(sequence),
            "biological_tokens": 3,
            "window_count": 1,
            "window_token_lengths": [3],
        }

        return embedding, metadata, 1


def test_adapter_conforms_to_embedding_interface():
    adapter = StubOmniDNA20MAdapter()

    embedding = adapter.embed("ATGGCTTAA")

    assert embedding.shape == (256,)
    assert embedding.dtype == np.float32
    assert adapter.uncached_calls == 1


def test_adapter_uses_cache_without_loading_model(
    tmp_path,
):
    adapter = StubOmniDNA20MAdapter(
        cache_dir=tmp_path,
    )

    first = adapter.embed_with_info("ATGGCTTAA")
    second = adapter.embed_with_info("ATGGCTTAA")

    assert first.cache_hit is False
    assert first.raw_model_calls == 1

    assert second.cache_hit is True
    assert second.raw_model_calls == 0

    assert adapter.uncached_calls == 1
    assert np.array_equal(
        first.embedding,
        second.embedding,
    )
    assert first.metadata == second.metadata


def test_adapter_cache_is_case_insensitive(tmp_path):
    adapter = StubOmniDNA20MAdapter(
        cache_dir=tmp_path,
    )

    first = adapter.embed_with_info("atggcttaa")
    second = adapter.embed_with_info("ATGGCTTAA")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert adapter.uncached_calls == 1


@pytest.mark.parametrize(
    "sequence, message",
    [
        ("", "cannot be empty"),
        ("ATGN", "unsupported symbols"),
        ("ATG-U", "unsupported symbols"),
    ],
)
def test_adapter_rejects_invalid_sequences(
    sequence,
    message,
):
    adapter = StubOmniDNA20MAdapter()

    with pytest.raises(ValueError, match=message):
        adapter.embed(sequence)


def test_adapter_records_reproducibility_metadata():
    adapter = StubOmniDNA20MAdapter()

    result = adapter.embed_with_info("ATGGCTTAA")

    assert result.metadata["model_id"] == (
        "zehui127/Omni-DNA-20M"
    )
    assert result.metadata["revision"] == (
        "3b64e6a5ed6c8f72bad76823ce728b3045243026"
    )
    assert result.metadata["pooling"] == (
        "final_hidden_biological_token_mean"
    )
    assert result.metadata["window_aggregation"] == (
        "global_biological_token_mean"
    )
