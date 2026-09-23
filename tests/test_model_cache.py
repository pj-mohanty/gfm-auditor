import numpy as np
import pytest

from agenticls_auditor.models.cache import EmbeddingCache


def test_cache_miss_then_round_trip(tmp_path):
    cache = EmbeddingCache(
        tmp_path,
        namespace="model@revision|pooling-v1",
    )

    sequence = "ATGGCTTAA"
    embedding = np.asarray(
        [0.25, -0.5, 1.5],
        dtype=np.float32,
    )
    metadata = {
        "window_count": 1,
        "biological_tokens": 3,
    }

    assert cache.get(sequence) is None

    written = cache.put(
        sequence,
        embedding,
        metadata,
    )
    loaded = cache.get(sequence)

    assert loaded is not None
    assert written.key == loaded.key
    assert loaded.embedding.dtype == np.float32
    assert np.array_equal(
        loaded.embedding,
        embedding,
    )
    assert loaded.metadata == metadata


def test_cache_normalizes_sequence_case(tmp_path):
    cache = EmbeddingCache(
        tmp_path,
        namespace="test",
    )

    embedding = np.asarray(
        [1.0, 2.0],
        dtype=np.float32,
    )

    cache.put("atggcttaa", embedding, {})

    loaded = cache.get("ATGGCTTAA")

    assert loaded is not None
    assert np.array_equal(
        loaded.embedding,
        embedding,
    )


def test_namespaces_produce_different_keys(tmp_path):
    first = EmbeddingCache(
        tmp_path,
        namespace="revision-a",
    )
    second = EmbeddingCache(
        tmp_path,
        namespace="revision-b",
    )

    assert (
        first.key_for("ATGGCTTAA")
        != second.key_for("ATGGCTTAA")
    )


def test_cache_rejects_non_vector_embedding(tmp_path):
    cache = EmbeddingCache(
        tmp_path,
        namespace="test",
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        cache.put(
            "ATGGCTTAA",
            np.zeros((2, 2), dtype=np.float32),
            {},
        )


def test_cache_rejects_non_finite_embedding(tmp_path):
    cache = EmbeddingCache(
        tmp_path,
        namespace="test",
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        cache.put(
            "ATGGCTTAA",
            np.asarray([1.0, np.nan]),
            {},
        )
