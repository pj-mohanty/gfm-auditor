from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModelAdapter
from .cache import EmbeddingCache


@dataclass(frozen=True)
class EmbeddingCall:
    embedding: np.ndarray
    cache_hit: bool
    raw_model_calls: int
    metadata: dict[str, Any]


class OmniDNA20MAdapter(BaseModelAdapter):
    """Pinned Omni-DNA-20M sequence-embedding adapter."""

    name = "omnidna-20m"

    model_id = "zehui127/Omni-DNA-20M"
    revision = "3b64e6a5ed6c8f72bad76823ce728b3045243026"
    weights_sha256 = (
        "f2927d1e3febd7f2309151fd1f6b47fc"
        "1e44500e6af6adad5342dcfc88b4680a"
    )

    embedding_width = 256
    maximum_model_tokens = 250
    special_tokens_per_window = 2
    maximum_biological_tokens = 248

    pooling_name = "final_hidden_biological_token_mean"
    window_aggregation = "global_biological_token_mean"
    implementation_version = 1

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        verify_checkpoint: bool = True,
    ) -> None:
        self.device_name = device
        self.verify_checkpoint = verify_checkpoint

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

        self.cache = (
            EmbeddingCache(
                cache_dir,
                namespace=self.cache_namespace,
            )
            if cache_dir is not None
            else None
        )

        self.last_call: EmbeddingCall | None = None

    @property
    def cache_namespace(self) -> str:
        return "|".join(
            [
                self.name,
                self.model_id,
                self.revision,
                self.weights_sha256,
                self.pooling_name,
                self.window_aggregation,
                f"implementation-v{self.implementation_version}",
            ]
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_with_info(sequence).embedding

    def embed_with_info(
        self,
        sequence: str,
    ) -> EmbeddingCall:
        normalized = self._validate_sequence(sequence)

        if self.cache is not None:
            cached = self.cache.get(normalized)

            if cached is not None:
                self._validate_embedding(cached.embedding)

                result = EmbeddingCall(
                    embedding=cached.embedding.copy(),
                    cache_hit=True,
                    raw_model_calls=0,
                    metadata=dict(cached.metadata),
                )
                self.last_call = result
                return result

        embedding, metadata, raw_model_calls = (
            self._embed_uncached(normalized)
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )
        self._validate_embedding(embedding)

        metadata = {
            **metadata,
            "model_id": self.model_id,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "pooling": self.pooling_name,
            "window_aggregation": self.window_aggregation,
        }

        if self.cache is not None:
            stored = self.cache.put(
                normalized,
                embedding,
                metadata,
            )
            embedding = stored.embedding

        result = EmbeddingCall(
            embedding=embedding.copy(),
            cache_hit=False,
            raw_model_calls=raw_model_calls,
            metadata=metadata,
        )
        self.last_call = result
        return result

    @staticmethod
    def _validate_sequence(sequence: str) -> str:
        normalized = sequence.upper()

        if not normalized:
            raise ValueError("DNA sequence cannot be empty")

        invalid = sorted(set(normalized) - set("ACGT"))

        if invalid:
            raise ValueError(
                f"DNA sequence contains unsupported symbols: {invalid}"
            )

        return normalized

    def _validate_embedding(
        self,
        embedding: np.ndarray,
    ) -> None:
        if embedding.shape != (self.embedding_width,):
            raise RuntimeError(
                "unexpected embedding shape: "
                f"{embedding.shape}; expected "
                f"({self.embedding_width},)"
            )

        if not np.isfinite(embedding).all():
            raise RuntimeError(
                "embedding contains non-finite values"
            )

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Omni-DNA dependencies are not installed. "
                "Install the project model extras."
            ) from exc

        config = AutoConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
        )

        if int(config.max_sequence_length) != (
            self.maximum_model_tokens
        ):
            raise RuntimeError(
                "unexpected model context length"
            )

        if int(config.d_model) != self.embedding_width:
            raise RuntimeError(
                "unexpected model hidden width"
            )

        weights_path = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors",
                revision=self.revision,
            )
        )

        if self.verify_checkpoint:
            digest = sha256(
                weights_path.read_bytes()
            ).hexdigest()

            if digest != self.weights_sha256:
                raise RuntimeError(
                    "checkpoint SHA256 does not match "
                    "the pinned checkpoint"
                )

        config.init_device = "cpu"

        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )

        checkpoint_state = load_file(
            str(weights_path),
            device="cpu",
        )

        load_result = model.load_state_dict(
            checkpoint_state,
            strict=False,
        )

        if load_result.missing_keys != [
            "word_embeddings.weight"
        ]:
            raise RuntimeError(
                "unexpected missing checkpoint keys: "
                f"{load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "unexpected checkpoint keys: "
                f"{load_result.unexpected_keys}"
            )

        if self.verify_checkpoint:
            loaded_state = model.state_dict()

            mismatches = [
                key
                for key, tensor in checkpoint_state.items()
                if not torch.equal(
                    loaded_state[key],
                    tensor,
                )
            ]

            if mismatches:
                raise RuntimeError(
                    "checkpoint tensors failed exact "
                    f"verification: {mismatches}"
                )

            if not torch.equal(
                loaded_state["word_embeddings.weight"],
                loaded_state[
                    "model.transformer.wte.weight"
                ],
            ):
                raise RuntimeError(
                    "input-embedding alias verification failed"
                )

        if self.device_name is None:
            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device_name = self.device_name

        device = torch.device(device_name)
        model = model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _biological_token_ids(
        self,
        sequence: str,
    ) -> list[int]:
        token_ids = list(
            self._tokenizer(
                sequence,
                add_special_tokens=False,
                truncation=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

        if not token_ids:
            raise RuntimeError(
                "tokenizer produced no biological tokens"
            )

        if self._tokenizer.unk_token_id in token_ids:
            raise RuntimeError(
                "tokenizer produced an unknown token"
            )

        return token_ids

    def _prepare_window(
        self,
        biological_ids: list[int],
    ) -> dict[str, Any]:
        tokenizer = self._tokenizer
        torch = self._torch

        if tokenizer.cls_token_id is None:
            raise RuntimeError(
                "tokenizer has no CLS token"
            )

        if tokenizer.sep_token_id is None:
            raise RuntimeError(
                "tokenizer has no SEP token"
            )

        # The generic fast-tokenizer wrapper does not add these
        # tokens through build_inputs_with_special_tokens().
        input_ids = [
            tokenizer.cls_token_id,
            *biological_ids,
            tokenizer.sep_token_id,
        ]
        special_tokens_mask = [
            1,
            *([0] * len(biological_ids)),
            1,
        ]

        if len(input_ids) > self.maximum_model_tokens:
            raise RuntimeError(
                "prepared window exceeds model context"
            )

        return {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self._device,
            ),
            "attention_mask": torch.ones(
                (1, len(input_ids)),
                dtype=torch.long,
                device=self._device,
            ),
            "special_tokens_mask": torch.tensor(
                [special_tokens_mask],
                dtype=torch.long,
                device=self._device,
            ),
        }

    def _embed_uncached(
        self,
        sequence: str,
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        self._ensure_loaded()

        torch = self._torch
        token_ids = self._biological_token_ids(sequence)

        windows = [
            token_ids[
                start:
                start + self.maximum_biological_tokens
            ]
            for start in range(
                0,
                len(token_ids),
                self.maximum_biological_tokens,
            )
        ]

        hidden_sum = torch.zeros(
            self.embedding_width,
            dtype=torch.float32,
            device=self._device,
        )

        total_biological_tokens = 0
        window_token_lengths: list[int] = []

        with torch.inference_mode():
            for biological_ids in windows:
                prepared = self._prepare_window(
                    biological_ids
                )

                output = self._model(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared[
                        "attention_mask"
                    ],
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

                hidden_states = output.hidden_states[-1]

                valid_mask = (
                    prepared["attention_mask"].bool()
                    & ~prepared[
                        "special_tokens_mask"
                    ].bool()
                )

                biological_hidden = hidden_states[
                    0,
                    valid_mask[0],
                ].float()

                biological_count = int(
                    valid_mask.sum().item()
                )

                if biological_count != len(
                    biological_ids
                ):
                    raise RuntimeError(
                        "biological-token mask is inconsistent"
                    )

                hidden_sum += biological_hidden.sum(
                    dim=0
                )
                total_biological_tokens += (
                    biological_count
                )
                window_token_lengths.append(
                    biological_count
                )

        embedding = (
            hidden_sum / total_biological_tokens
        ).cpu().numpy().astype(
            np.float32,
            copy=False,
        )

        metadata = {
            "bases": len(sequence),
            "biological_tokens": len(token_ids),
            "window_count": len(windows),
            "window_token_lengths": (
                window_token_lengths
            ),
        }

        return embedding, metadata, len(windows)
