from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

import numpy as np


class BaseModelAdapter(ABC):
    name: str

    @abstractmethod
    def embed(self, sequence: str) -> np.ndarray:
        """Return one documented sequence-level representation."""


class DeterministicMockAdapter(BaseModelAdapter):
    name = "deterministic-mock"

    def __init__(self, width: int = 16) -> None:
        self.width = width

    def embed(self, sequence: str) -> np.ndarray:
        digest = sha256(sequence.encode()).digest()
        values = np.frombuffer(digest, dtype=np.uint8)[: self.width].astype(np.float64)
        return values / 255.0

