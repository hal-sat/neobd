"""Directional SPAC analysis using parallel configurable optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.special import jv

from .config import DSPACConfig
from .io import write_real
from .optimization import minimize
from .parallel import ordered_parallel_map
from .preprocess import SpectralStatistics


@dataclass(frozen=True)
class DSPACData:
    frequency: NDArray[np.float64]
    radii: NDArray[np.float64]
    angles: NDArray[np.float64]
    coherence: NDArray[np.complex128]


@dataclass(frozen=True)
class DSPACTask:
    index: int
    frequency: float
    radii: NDArray[np.float64]
    angles: NDArray[np.float64]
    observed: NDArray[np.complex128]
    config: DSPACConfig
    frequency_count: int


def _prepare_data(statistics: SpectralStatistics, config: DSPACConfig) -> DSPACData:
    requested = set(config.sites)
    missing = requested - {site.name for site in statistics.locations}
    if missing:
        raise ValueError(f"Unknown DSPAC receivers: {sorted(missing)}")
    indices = [
        i for i, site in enumerate(statistics.locations) if site.name in requested
    ]
    radii, angles, coherence = [], [], []
    offset = np.deg2rad(config.azimuth_offset_degrees)
    for first_position in range(1, len(indices)):
        first = indices[first_position]
        for second_position in range(first_position):
            second = indices[second_position]
            dx = statistics.locations[first].x - statistics.locations[second].x
            dy = statistics.locations[first].y - statistics.locations[second].y
            radius = float(np.hypot(dx, dy))
            if radius <= 0:
                raise ValueError("DSPAC receivers must have distinct coordinates")
            denominator = np.sqrt(
                statistics.power[0, second] * statistics.power[0, first]
            )
            normalized = np.divide(
                statistics.cross[0, second, first],
                denominator,
                out=np.full_like(statistics.cross[0, second, first], np.nan + 0j),
                where=denominator > np.finfo(float).tiny,
            )
            radii.append(radius)
            angles.append(float(np.arctan2(dy, dx) - offset))
            coherence.append(normalized)
    mask = (statistics.frequency > 0) & (statistics.frequency <= config.max_frequency)
    return DSPACData(
        statistics.frequency[mask],
        np.asarray(radii),
        np.asarray(angles),
        np.asarray(coherence)[:, mask],
    )


def _real_misfit(
    population: NDArray[np.float64],
    radii: NDArray[np.float64],
    angles: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> NDArray[np.float64]:
    wave_number = population[:, 0, None]
    prediction = jv(0, wave_number * radii[None, :])
    for order in (1, 2):
        harmonic = 2 * order
        core = population[:, 2 * order - 1, None] * np.cos(
            harmonic * angles
        ) + population[:, 2 * order, None] * np.sin(harmonic * angles)
        prediction += (
            2 * (-1) ** order * jv(harmonic, wave_number * radii[None, :]) * core
        )
    return np.sum((observed[None, :] - prediction) ** 2, axis=1)


def _imaginary_misfit(
    population: NDArray[np.float64],
    wave_number: float,
    radii: NDArray[np.float64],
    angles: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> NDArray[np.float64]:
    prediction = np.zeros((population.shape[0], radii.size))
    for order in (1, 2):
        harmonic = 2 * order - 1
        parameter = 2 * order - 2
        core = population[:, parameter, None] * np.cos(harmonic * angles) + population[
            :, parameter + 1, None
        ] * np.sin(harmonic * angles)
        prediction += (
            2 * (-1) ** order * jv(harmonic, wave_number * radii)[None, :] * core
        )
    return np.sum((observed[None, :] + prediction) ** 2, axis=1)


def _solve_frequency(
    task: DSPACTask,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    valid = np.isfinite(task.observed.real) & np.isfinite(task.observed.imag)
    if np.count_nonzero(valid) < 3:
        return None
    radii, angles = task.radii[valid], task.angles[valid]
    observed = task.observed[valid]
    real_bounds = np.asarray(
        [
            [0.0, np.pi / np.max(task.radii)],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    imaginary_bounds = np.tile(np.asarray([[-1.0, 1.0]]), (4, 1))
    optimizer = task.config.optimizer
    options = dict(
        population_size=optimizer.population,
        iterations=optimizer.iterations,
        mutation=optimizer.mutation,
        crossover=optimizer.crossover,
        inertia_start=optimizer.inertia_start,
        inertia_end=optimizer.inertia_end,
        w4loc=optimizer.w4loc,
        w4glo=optimizer.w4glo,
        tolerance=optimizer.target,
        patience=optimizer.patience,
    )
    real_solution = minimize(
        lambda population: _real_misfit(population, radii, angles, observed.real),
        real_bounds,
        optimizer.method,
        seed=optimizer.seed + task.index,
        **options,
    )
    wave_number = float(real_solution[0])
    velocity = (
        np.inf
        if wave_number <= np.finfo(float).tiny
        else 2 * np.pi * task.frequency / wave_number
    )
    imaginary_solution = minimize(
        lambda population: _imaginary_misfit(
            population, wave_number, radii, angles, observed.imag
        ),
        imaginary_bounds,
        optimizer.method,
        seed=optimizer.seed + task.frequency_count + task.index,
        **options,
    )
    return (
        np.concatenate(([task.frequency, velocity], real_solution[1:])),
        np.concatenate(([task.frequency], imaginary_solution)),
    )


def run_dspac(
    statistics: SpectralStatistics, case_dir: Path, config: DSPACConfig, n_para: int = 1
) -> None:
    """Estimate frequency bins in parallel and write ordered DSPAC results."""
    data = _prepare_data(statistics, config)
    if data.radii.size < 3:
        raise ValueError("DSPAC requires at least three independent receiver pairs")
    tasks = [
        DSPACTask(
            index,
            float(frequency),
            data.radii,
            data.angles,
            data.coherence[:, index],
            config,
            data.frequency.size,
        )
        for index, frequency in enumerate(data.frequency)
    ]
    solved = [
        result
        for result in ordered_parallel_map(_solve_frequency, tasks, n_para)
        if result is not None
    ]
    real_values = np.asarray([result[0] for result in solved]).reshape(-1, 6)
    imaginary_values = np.asarray([result[1] for result in solved]).reshape(-1, 5)
    output = case_dir / "results_neobd" / "dspac"
    output.mkdir(parents=True, exist_ok=True)
    write_real(output / "result_real.csv", *real_values.T)
    write_real(output / "result_imag.csv", *imaginary_values.T)
