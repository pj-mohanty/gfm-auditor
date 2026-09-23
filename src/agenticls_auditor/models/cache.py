from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CachedEmbedding:
    embedding: np.ndarray
    metadata: dict[str, Any]
    key: str


class EmbeddingCache:
    """Deterministic disk cache for sequence-level embeddings."""

    def __init__(self, root: str | Path, *, namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("cache namespace cannot be empty")

        self.root = Path(root)
        self.namespace = namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, sequence: str) -> str:
        normalized = sequence.upper()

        payload = json.dumps(
            {
                "namespace": self.namespace,
                "sequence_sha256": sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        return sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, sequence: str) -> Path:
        key = self.key_for(sequence)

        return (
            self.root
            / key[:2]
            / key[2:4]
            / f"{key}.npz"
        )

    def get(self, sequence: str) -> CachedEmbedding | None:
        path = self.path_for(sequence)

        if not path.exists():
            return None

        with np.load(path, allow_pickle=False) as archive:
            embedding = np.asarray(
                archive["embedding"],
                dtype=np.float32,
            )
            metadata_json = str(
                archive["metadata_json"].item()
            )
            stored_key = str(archive["key"].item())

        expected_key = self.key_for(sequence)

        if stored_key != expected_key:
            raise RuntimeError(
                f"cache-key mismatch in {path}"
            )

        if embedding.ndim != 1:
            raise RuntimeError(
                f"cached embedding must be one-dimensional: {path}"
            )

        if not np.isfinite(embedding).all():
            raise RuntimeError(
                f"cached embedding contains non-finite values: {path}"
            )

        metadata = json.loads(metadata_json)

        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"cached metadata must be a JSON object: {path}"
            )

        return CachedEmbedding(
            embedding=embedding,
            metadata=metadata,
            key=stored_key,
        )

    def put(
        self,
        sequence: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> CachedEmbedding:
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError(
                "embedding must be one-dimensional"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "embedding contains non-finite values"
            )

        # Validate serializability before creating a temporary file.
        metadata_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        )

        key = self.key_for(sequence)
        path = self.path_for(sequence)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{key}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)

            np.savez_compressed(
                temporary_path,
                embedding=vector,
                metadata_json=np.asarray(metadata_json),
                key=np.asarray(key),
            )

            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

        return CachedEmbedding(
            embedding=vector.copy(),
            metadata=dict(metadata),
            key=key,
        )
