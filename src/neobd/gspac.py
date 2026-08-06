"""Generalized SPAC methods based on time-domain circle processes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import j0, j1

from .config import GSPACArrayConfig
from .io import write_complex, write_real
from .preprocess import SpectralStatistics, smooth

RatioModel = Callable[[NDArray[np.float64] | float], NDArray[np.float64] | float]


@dataclass(frozen=True)
class CircularStatistics:
    """Spectral statistics calculated from time-domain circle processes."""

    frequency: NDArray[np.float64]
    radius: float
    g00: NDArray[np.complex128]
    g11: NDArray[np.complex128]
    gcc: NDArray[np.complex128] | None
    gc0: NDArray[np.complex128] | None


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


def _positive_dft(values: NDArray[np.generic]) -> NDArray[np.complex128]:
    """Return the positive-exponent DFT used by the preprocessing stage."""
    length = values.shape[-1]
    transformed = np.conj(np.fft.fft(np.conj(values), axis=-1)) / length
    return transformed[..., : length // 2 + 1]


def _fit_circle(
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a circle to ring coordinates without using a center receiver."""
    if coordinates.shape[0] < 3:
        raise ValueError("GSPAC requires at least three ring receivers")
    system = np.column_stack(
        (2 * coordinates[:, 0], 2 * coordinates[:, 1], np.ones(coordinates.shape[0]))
    )
    if np.linalg.matrix_rank(system) < 3:
        raise ValueError("GSPAC ring coordinates cannot define a circle")
    right_hand_side = np.sum(coordinates**2, axis=1)
    solution = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
    center = solution[:2]
    offsets = coordinates - center
    return offsets, np.linalg.norm(offsets, axis=1)


def _circle_processes(
    statistics: SpectralStatistics, config: GSPACArrayConfig
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64] | None, float
]:
    """Calculate zeroth- and first-order processes from selected time histories."""
    if statistics.time_histories is None:
        raise ValueError("GSPAC requires selected time histories from preprocessing")
    indices = {
        location.name: index for index, location in enumerate(statistics.locations)
    }
    needs_center_record = any(method != "cca" for method in config.methods)
    required = set(config.ring)
    if needs_center_record:
        required.add(config.center)
    missing = required - indices.keys()
    if missing:
        raise ValueError(f"Unknown GSPAC receivers: {sorted(missing)}")
    ring = np.asarray([indices[name] for name in config.ring])
    coordinates = np.asarray(
        [
            (statistics.locations[index].x, statistics.locations[index].y)
            for index in ring
        ]
    )
    offsets, distances = _fit_circle(coordinates)
    if np.any(distances <= 0):
        raise ValueError("GSPAC ring radius must be positive")
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    design = np.column_stack(
        (np.ones(ring.size), 2 * np.cos(angles), -2 * np.sin(angles))
    )
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("GSPAC requires at least three independent ring azimuths")
    ring_histories = statistics.time_histories[ring, 0]
    coefficients = np.einsum("kr,rsn->ksn", np.linalg.pinv(design), ring_histories)
    process0 = coefficients[0].astype(np.complex128)
    process1 = coefficients[1] + 1j * coefficients[2]
    center_history = None
    if needs_center_record:
        center = indices[config.center]
        center_history = statistics.time_histories[center, 0]
    return process0, process1, center_history, float(np.mean(distances))


def _circular_statistics(
    statistics: SpectralStatistics,
    config: GSPACArrayConfig,
    smoothing_iterations: int,
) -> CircularStatistics:
    process0, process1, center, radius = _circle_processes(statistics, config)
    window = np.hanning(process0.shape[-1])
    spectrum0 = _positive_dft(process0 * window)
    spectrum1 = _positive_dft(process1 * window)
    scale = (
        8.0
        * process0.shape[-1]
        / (2.0 * (statistics.frequency[-1] if statistics.frequency.size > 1 else 0.5))
    )

    def spectral_density(
        first: NDArray[np.complex128], second: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        return smooth(
            np.mean(np.conj(first) * second, axis=0) * scale,
            smoothing_iterations,
        )

    g00 = spectral_density(spectrum0, spectrum0)
    g11 = spectral_density(spectrum1, spectrum1)
    gcc = None
    gc0 = None
    if center is not None:
        center_spectrum = _positive_dft(center * window)
        gcc = spectral_density(center_spectrum, center_spectrum)
        gc0 = spectral_density(center_spectrum, spectrum0)
    return CircularStatistics(statistics.frequency, radius, g00, g11, gcc, gc0)


def _spectral_ratios(
    circular: CircularStatistics,
) -> dict[str, NDArray[np.complex128]]:
    floor = np.finfo(float).eps * max(
        float(np.max(np.abs(circular.g00))),
        float(np.max(np.abs(circular.g11))),
        1.0,
    )

    def divide(
        numerator: NDArray[np.complex128], denominator: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan + 0j),
            where=np.abs(denominator) > floor,
        )

    ratios = {"cca": divide(circular.g00, circular.g11)}
    if circular.gcc is not None and circular.gc0 is not None:
        ratios.update(
            {
                "h0": divide(circular.g00, circular.gcc),
                "h1": divide(circular.g11, circular.gcc),
                "v": divide(circular.gc0, circular.g11),
            }
        )
    return ratios


def _write_circular_statistics(
    case_dir: Path, name: str, circular: CircularStatistics
) -> None:
    output = case_dir / "results_neobd" / "circular_statistics" / name
    output.mkdir(parents=True, exist_ok=True)
    write_complex(output / "G00.csv", circular.frequency, circular.g00)
    write_complex(output / "G11.csv", circular.frequency, circular.g11)
    if circular.gcc is not None and circular.gc0 is not None:
        write_complex(output / "Gcc.csv", circular.frequency, circular.gcc)
        write_complex(output / "Gc0.csv", circular.frequency, circular.gc0)


def run_gspac(
    statistics: SpectralStatistics,
    case_dir: Path,
    arrays: dict[str, GSPACArrayConfig],
    smoothing_iterations: int = 0,
) -> None:
    """Calculate circular statistics, spectral ratios, and phase velocities."""
    for name, config in arrays.items():
        circular = _circular_statistics(statistics, config, smoothing_iterations)
        _write_circular_statistics(case_dir, name, circular)
        ratios = _spectral_ratios(circular)
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
                velocity[index] = min(
                    2 * np.pi * frequency * circular.radius / rk, 5000.0
                )
            output = case_dir / "results_neobd" / "gspac" / method
            output.mkdir(parents=True, exist_ok=True)
            write_complex(output / f"spr_{name}.csv", statistics.frequency, ratio)
            write_real(output / f"phv_{name}.csv", statistics.frequency, velocity)
