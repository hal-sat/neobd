"""Frequency-domain smoothing policies for spectral statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


class SpectralSmoother(Protocol):
    """Apply a smoothing policy along the final frequency axis."""

    def apply(
        self, values: NDArray[np.generic], frequency: NDArray[np.float64]
    ) -> NDArray[np.generic]: ...


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration for a frequency-domain smoothing policy."""

    type: str = "Hann_3point"
    params: tuple[float, ...] = (0.0,)

    @classmethod
    def from_raw(cls, raw: Any) -> "SmoothingConfig":
        if not isinstance(raw, dict):
            raise ValueError("smoothing must be an object")
        raw_type = str(raw.get("type", "")).strip().lower()
        names = {"parzen": "Parzen", "hann_3point": "Hann_3point"}
        if raw_type not in names:
            raise ValueError("smoothing type must be Parzen or Hann_3point")
        params = raw.get("params")
        if not isinstance(params, (list, tuple)) or len(params) != 1:
            raise ValueError("smoothing params must contain exactly one value")
        value = float(params[0])
        if not math.isfinite(value):
            raise ValueError("smoothing parameter must be finite")
        if raw_type == "parzen" and value <= 0:
            raise ValueError("Parzen bandwidth must be positive")
        if raw_type == "hann_3point" and (value < 0 or not value.is_integer()):
            raise ValueError("Hann_3point iterations must be a non-negative integer")
        return cls(names[raw_type], (value,))

    @classmethod
    def hann_3point(cls, iterations: int) -> "SmoothingConfig":
        if iterations < 0:
            raise ValueError("Hann_3point iterations must be non-negative")
        return cls("Hann_3point", (float(iterations),))


@dataclass(frozen=True)
class Hann3PointSmoother:
    """Repeatedly apply the normalized three-point Hann kernel."""

    iterations: int

    def apply(
        self, values: NDArray[np.generic], frequency: NDArray[np.float64]
    ) -> NDArray[np.generic]:
        _validate_axes(values, frequency)
        result = values.copy()
        for _ in range(self.iterations):
            previous = result.copy()
            result[..., 1:-1] = (
                0.25 * previous[..., :-2]
                + 0.5 * previous[..., 1:-1]
                + 0.25 * previous[..., 2:]
            )
        return result


@dataclass(frozen=True)
class ParzenSmoother:
    """Apply a sinc-to-the-fourth Parzen kernel on the frequency axis."""

    bandwidth: float

    def apply(
        self, values: NDArray[np.generic], frequency: NDArray[np.float64]
    ) -> NDArray[np.generic]:
        spacing = _validate_axes(values, frequency)
        count = frequency.size
        offsets = np.arange(-(count - 1), count, dtype=float) * spacing
        # The scale makes bandwidth equal to the width of a rectangular
        # kernel having the same variance.
        scale = np.pi * self.bandwidth / 3.0
        kernel = np.sinc(offsets / scale) ** 4
        shape = (1,) * (values.ndim - 1) + (kernel.size,)
        numerator = fftconvolve(values, kernel.reshape(shape), mode="same", axes=-1)
        denominator = fftconvolve(np.ones(count), kernel, mode="same")
        result = numerator / denominator
        if np.isrealobj(values):
            result = result.real
        return result.astype(values.dtype, copy=False)


def create_smoother(config: SmoothingConfig) -> SpectralSmoother:
    """Create a smoothing policy from validated configuration."""
    if config.type == "Hann_3point":
        return Hann3PointSmoother(int(config.params[0]))
    if config.type == "Parzen":
        return ParzenSmoother(config.params[0])
    raise ValueError(f"Unknown smoothing type: {config.type}")


def _validate_axes(
    values: NDArray[np.generic], frequency: NDArray[np.float64]
) -> float:
    if values.shape[-1] != frequency.size:
        raise ValueError("The frequency axis does not match the spectral data")
    if frequency.size < 2:
        raise ValueError("At least two frequency bins are required")
    differences = np.diff(frequency)
    spacing = float(np.median(differences))
    if spacing <= 0 or not np.allclose(differences, spacing):
        raise ValueError("Smoothing requires a uniformly increasing frequency axis")
    return spacing
