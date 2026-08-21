"""FK orchestration with process-parallel frequency estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from .config import FKConfig
from .fk_estimators import create_estimator
from .io import write_real
from .optimization import minimize
from .parallel import ordered_parallel_map
from .preprocess import SpectralStatistics


@dataclass(frozen=True)
class FKTask:
    index: int
    frequency: float
    matrix: NDArray[np.complex128]
    coordinates: NDArray[np.float64]
    sx: NDArray[np.float64]
    sy: NDArray[np.float64]
    angle: NDArray[np.float64]
    method: str
    config: FKConfig


@dataclass(frozen=True)
class FKResult:
    frequency: float
    velocity: float
    phases: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    power: NDArray[np.float64]
    fv_power: NDArray[np.float64]
    optimum: NDArray[np.float64]


def _steering(
    points: NDArray[np.float64], coordinates: NDArray[np.float64], frequency: float
) -> NDArray[np.complex128]:
    return np.exp(-2j * np.pi * frequency * (points @ coordinates.T))


def _circular_grid(
    config: FKConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if config.min_velocity <= 0 or config.max_velocity <= config.min_velocity:
        raise ValueError("FK bounds must satisfy 0 < min < max")
    if config.radial_density < 2 or config.angular_density < 3:
        raise ValueError("FK density must be at least [2, 3]")
    slowness = np.linspace(
        1 / config.max_velocity, 1 / config.min_velocity, config.radial_density
    )
    angle = np.linspace(-np.pi, np.pi, config.angular_density, endpoint=False)
    return slowness[:, None] * np.cos(angle), slowness[:, None] * np.sin(angle), angle


def _polar_points(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    slowness, angle = parameters[:, 0], parameters[:, 1]
    return np.column_stack((slowness * np.cos(angle), slowness * np.sin(angle)))


def _velocity_grid(config: FKConfig) -> NDArray[np.float64]:
    """Return the uniformly sampled phase velocities used by F-V output."""
    return np.linspace(config.min_velocity, config.max_velocity, config.radial_density)


def _solve_frequency(task: FKTask) -> FKResult | None:
    estimator = create_estimator(task.method, task.config.diagonal_loading)
    prepared = estimator.prepare(task.matrix)
    grid = np.column_stack((task.sx.ravel(), task.sy.ravel()))
    power = estimator.power(
        prepared, _steering(grid, task.coordinates, task.frequency)
    ).reshape(task.sx.shape)
    maximum = float(np.max(power))
    if not np.isfinite(maximum) or maximum <= 0:
        return None
    power /= maximum

    velocity_grid = _velocity_grid(task.config)
    fv_slow = 1.0 / velocity_grid[:, None]
    fv_points = np.column_stack(
        (
            (fv_slow * np.cos(task.angle)).ravel(),
            (fv_slow * np.sin(task.angle)).ravel(),
        )
    )
    fv_power = estimator.power(
        prepared, _steering(fv_points, task.coordinates, task.frequency)
    ).reshape(velocity_grid.size, task.angle.size)
    fv_maximum = float(np.max(fv_power))
    if not np.isfinite(fv_maximum) or fv_maximum <= 0:
        return None
    fv_power = np.mean(fv_power / fv_maximum, axis=1)
    profile_maximum = float(np.max(fv_power))
    if not np.isfinite(profile_maximum) or profile_maximum <= 0:
        return None
    fv_power /= profile_maximum

    def objective(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return -estimator.power(
            prepared,
            _steering(_polar_points(parameters), task.coordinates, task.frequency),
        )

    optimizer = task.config.optimizer
    parameters = minimize(
        objective,
        np.asarray(
            [
                [1 / task.config.max_velocity, 1 / task.config.min_velocity],
                [-np.pi, np.pi],
            ]
        ),
        optimizer.method,
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
        seed=optimizer.seed + task.index,
    )
    optimum = _polar_points(parameters[None, :])[0]
    weights = power[int(np.unravel_index(np.argmax(power), power.shape)[0])]
    coefficient = np.sum(
        weights * np.exp(-1j * np.arange(20)[:, None] * task.angle), axis=1
    ) / np.sum(weights)
    return FKResult(
        task.frequency,
        1 / parameters[0],
        np.angle(coefficient),
        np.abs(coefficient),
        power,
        fv_power,
        optimum,
    )


def _write_map(
    path: Path, sx: NDArray[np.float64], sy: NDArray[np.float64], result: FKResult
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            f"# optimum_sx={result.optimum[0]:.15g}, optimum_sy={result.optimum[1]:.15g}\n"
        )
        for radial in range(result.power.shape[0]):
            np.savetxt(
                stream,
                np.column_stack((sx[radial], sy[radial], result.power[radial])),
                delimiter=", ",
                fmt="%.15g",
            )
            closing = 0.5 * (result.power[radial, -1] + result.power[radial, 0])
            stream.write(
                f"{sx[radial, 0]:.15g}, {sy[radial, 0]:.15g}, {closing:.15g}\n\n"
            )


def _run_method(
    statistics: SpectralStatistics,
    output: Path,
    config: FKConfig,
    method: str,
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    angle: NDArray[np.float64],
    n_para: int,
) -> None:
    if config.output_interval <= 0:
        raise ValueError("output_interval must be positive")
    output.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray([(site.x, site.y) for site in statistics.locations])
    tasks = [
        FKTask(
            index,
            float(frequency),
            statistics.cross[0, :, :, index],
            coordinates,
            sx,
            sy,
            angle,
            method,
            config,
        )
        for index, frequency in enumerate(statistics.frequency)
        if index >= 1
        and (index - 1) % config.output_interval == 0
        and frequency <= config.max_frequency
    ]
    results = [
        result
        for result in ordered_parallel_map(_solve_frequency, tasks, n_para)
        if result is not None
    ]
    for result in results:
        filename = f"FK_{result.frequency:08.5f}".replace(".", "p") + "_Hz.csv"
        _write_map(output / filename, sx, sy, result)
    frequency = np.asarray([result.frequency for result in results])
    write_real(
        output / "phv_fk.csv",
        frequency,
        np.asarray([result.velocity for result in results]),
    )
    if results:
        write_real(
            output / "phases.csv",
            frequency,
            *np.asarray([result.phases for result in results]).T,
        )
        write_real(
            output / "amps.csv",
            frequency,
            *np.asarray([result.amplitudes for result in results]).T,
        )
    velocity = _velocity_grid(config)
    write_real(
        output / "fv.csv",
        np.repeat(frequency, velocity.size),
        np.tile(velocity, frequency.size),
        np.asarray([result.fv_power for result in results]).reshape(-1),
    )


def run_fk(
    statistics: SpectralStatistics, case_dir: Path, config: FKConfig, n_para: int = 1
) -> None:
    """Run requested methods with frequency bins distributed across processes."""
    sx, sy, angle = _circular_grid(config)
    aliases = {"capon": "MLM", "bartlett": "BFM", "bmf": "BFM"}
    for method in config.methods:
        _run_method(
            statistics,
            case_dir / "results_neobd" / "fk" / aliases.get(method, method.upper()),
            config,
            method,
            sx,
            sy,
            angle,
            n_para,
        )
