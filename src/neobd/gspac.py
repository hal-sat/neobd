"""Generalized SPAC methods for circular arrays."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import j0, j1

from .config import GSPACArrayConfig
from .io import write_complex, write_real
from .preprocess import SpectralStatistics

RatioModel = Callable[[NDArray[np.float64] | float], NDArray[np.float64] | float]


def _ratio_model(method: str) -> RatioModel:
    tiny = np.finfo(float).tiny
    if method == "cca":
        return lambda x: j0(x) ** 2 / np.maximum(j1(x) ** 2, tiny)
    if method == "h0":
        return lambda x: j0(x) ** 2
    if method == "h1":
        return lambda x: j1(x) ** 2
    if method == "v":
        return lambda x: j0(x) / np.maximum(j1(x) ** 2, tiny)
    raise ValueError(f"Unknown GSPAC method: {method!r}")


def find_first_monotonic_interval(
    model: RatioModel, maximum: float = 10.0
) -> tuple[float, float]:
    """Locate the first monotonic branch by scanning for its first reversal."""
    arguments = np.linspace(1e-4, maximum, 200_001)
    values = np.asarray(model(arguments), dtype=float)
    differences = np.diff(values)
    finite = np.isfinite(differences) & (differences != 0)
    if not np.any(finite):
        raise ValueError("The theoretical spectral-ratio function is not variable")
    first = int(np.flatnonzero(finite)[0])
    direction = np.sign(differences[first])
    opposite = np.flatnonzero(finite & (differences * direction < 0))
    opposite = opposite[opposite > first]
    if opposite.size == 0:
        raise ValueError("Could not find the end of the first monotonic branch")
    end = int(opposite[0])
    return float(arguments[first]), float(arguments[end])


def _invert_ratio(
    model: RatioModel, interval: tuple[float, float], observed: float
) -> float:
    lower, upper = interval
    lower_value, upper_value = float(model(lower)), float(model(upper))
    increasing = upper_value > lower_value
    if increasing:
        if observed <= lower_value:
            return lower
        if observed >= upper_value:
            return upper
    else:
        if observed >= lower_value:
            return lower
        if observed <= upper_value:
            return upper
    return float(brentq(lambda value: float(model(value)) - observed, lower, upper))


def _spectral_ratios(
    statistics: SpectralStatistics,
    config: GSPACArrayConfig,
) -> tuple[dict[str, NDArray[np.complex128]], float]:
    indices = {
        location.name: index for index, location in enumerate(statistics.locations)
    }
    missing = ({config.center} | set(config.ring)) - indices.keys()
    if missing:
        raise ValueError(f"Unknown GSPAC receivers: {sorted(missing)}")
    center = indices[config.center]
    ring = np.asarray([indices[name] for name in config.ring])
    center_location = statistics.locations[center]
    offsets = np.asarray(
        [
            (
                statistics.locations[index].x - center_location.x,
                statistics.locations[index].y - center_location.y,
            )
            for index in ring
        ]
    )
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances <= 0):
        raise ValueError("GSPAC ring receivers must not coincide with the center")
    radius = float(np.mean(distances))
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    weight0 = np.ones(ring.size, dtype=complex) / ring.size
    weight1 = np.exp(-1j * angles) / ring.size
    matrix = statistics.cross[0, ring[:, None], ring[None, :], :]
    g00 = np.einsum("i,ijf,j->f", weight0.conj(), matrix, weight0)
    g11 = np.einsum("i,ijf,j->f", weight1.conj(), matrix, weight1)
    gcc = statistics.cross[0, center, center]
    gc0 = np.einsum("jf,j->f", statistics.cross[0, center, ring], weight0)
    denominator_floor = np.finfo(float).eps * max(
        float(np.max(np.abs(gcc))), float(np.max(np.abs(g11))), 1.0
    )

    def divide(
        numerator: NDArray[np.complex128], denominator: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan + 0j),
            where=np.abs(denominator) > denominator_floor,
        )

    ratios = {
        "cca": divide(g00, g11),
        "h0": divide(g00, gcc),
        "h1": divide(g11, gcc),
        "v": divide(gc0, g11),
    }
    return ratios, radius


def run_gspac(
    statistics: SpectralStatistics, case_dir: Path, arrays: dict[str, GSPACArrayConfig]
) -> None:
    """Run requested generalized SPAC methods and write spectral ratios and velocities."""
    for name, config in arrays.items():
        ratios, radius = _spectral_ratios(statistics, config)
        for method in config.methods:
            model = _ratio_model(method)
            interval = find_first_monotonic_interval(model)
            ratio = ratios[method]
            velocity = np.full(statistics.frequency.shape, np.nan)
            for index, (frequency, observed) in enumerate(
                zip(statistics.frequency, ratio.real)
            ):
                if frequency == 0 or not np.isfinite(observed):
                    continue
                rk = _invert_ratio(model, interval, float(observed))
                velocity[index] = min(2 * np.pi * frequency * radius / rk, 5000.0)
            output = case_dir / "results_neobd" / "gspac" / method
            output.mkdir(parents=True, exist_ok=True)
            write_complex(output / f"spr_{name}.csv", statistics.frequency, ratio)
            write_real(output / f"phv_{name}.csv", statistics.frequency, velocity)
