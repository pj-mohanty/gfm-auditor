from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .omnidna import OmniDNA20MAdapter


class NucEL93MAdapter(OmniDNA20MAdapter):
    """Pinned NucEL discriminator, mean pooled over nucleotide tokens."""

    name = "nucel-93m"
    model_id = "FreakingPotato/NucEL"
    revision = "0c802982985e572d9dc0659b072a1ceb03c8000f"
    weights_sha256 = "4313456e971ed018bef72c2628eda8229a9fde5877f35b329372b757a13a38de"
    embedding_width = 512
    maximum_biological_tokens = 1022
    # Published k=1 vocabulary at this revision: [CLS]=2, A/C/G/T=11/12/13/14.
    # The author's tokenizer prepends CLS and does not append SEP.
    nucleotide_ids = {"A": 11, "C": 12, "G": 13, "T": 14}
    pooling_name = "last_hidden_nucleotide_mean_excluding_special_tokens"
    window_aggregation = "global_nucleotide_mean"
    implementation_version = 1

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel
            from transformers import __version__ as transformers_version
        except ImportError as exc:
            raise RuntimeError("Install the project nucel model extras") from exc

        version = tuple(int(x) for x in transformers_version.split(".")[:2])
        if not (version >= (4, 48) and version < (5, 4)):
            raise RuntimeError("NucEL requires transformers>=4.48,<5.4")

        weights_path = Path(hf_hub_download(
            repo_id=self.model_id, filename="model.safetensors",
            revision=self.revision,
        ))
        if self.verify_checkpoint and sha256(weights_path.read_bytes()).hexdigest() != self.weights_sha256:
            raise RuntimeError("NucEL checkpoint SHA256 mismatch")

        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, attn_implementation="sdpa",
        ).to(device).eval()
        if int(model.config.hidden_size) != self.embedding_width:
            raise RuntimeError("unexpected NucEL hidden width")
        self._torch = torch
        self._model = model
        self._device = device

    def _embed_uncached(self, sequence: str) -> tuple[np.ndarray, dict[str, Any], int]:
        self._ensure_loaded()
        torch = self._torch
        hidden_sum = torch.zeros(self.embedding_width, dtype=torch.float32, device=self._device)
        window_lengths: list[int] = []
        windows = [sequence[i:i + self.maximum_biological_tokens]
                   for i in range(0, len(sequence), self.maximum_biological_tokens)]

        with torch.inference_mode():
            for window in windows:
                input_ids = torch.tensor(
                    [[2] + [self.nucleotide_ids[base] for base in window]],
                    dtype=torch.long, device=self._device,
                )
                tokens = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                if tokens["input_ids"].shape[1] > 1024:
                    raise RuntimeError("NucEL window exceeds pretraining length")
                output = self._model(**tokens)
                hidden_sum += output.last_hidden_state[0, 1:].float().sum(dim=0)
                window_lengths.append(len(window))

        embedding = (hidden_sum / sum(window_lengths)).cpu().numpy().astype(np.float32)
        metadata = {
            "bases": len(sequence), "biological_tokens": sum(window_lengths),
            "window_count": len(windows), "window_token_lengths": window_lengths,
        }
        return embedding, metadata, len(windows)
