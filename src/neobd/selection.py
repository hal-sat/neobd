"""Replaceable segment-selection policies."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence
import numpy as np
from numpy.typing import NDArray


class SegmentSelector(Protocol):
    """Interface implemented by segment-selection strategies."""

    def select(
        self, normalized_rms: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.int64]: ...


class ModalRMSSelector:
    """Select segments near the modal normalized-RMS class at every channel."""

    def __init__(self, allowance: float = 0.2) -> None:
        self.allowance = allowance

    def select(
        self, normalized_rms: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.int64]:
        if not normalized_rms:
            return np.empty(0, dtype=np.int64)
        count = normalized_rms[0].shape[1]
        valid = np.ones(count, dtype=bool)
        edges = 0.05 + np.arange(20) * 0.1
        for receiver in normalized_rms:
            for channel in receiver:
                histogram, _ = np.histogram(channel, bins=edges)
                mode = np.argmax(histogram) * 0.1 + 0.1
                valid &= np.abs(channel - mode) <= self.allowance
        return np.flatnonzero(valid)


def read_valid_segments(path: Path) -> NDArray[np.int64]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.asarray([int(Path(entry).stem) for entry in entries], dtype=np.int64)


def write_valid_segments(path: Path, indices: NDArray[np.int64]) -> None:
    text = "".join(f"{index:06d}.csv\n" for index in indices)
    path.write_text(text, encoding="utf-8")
