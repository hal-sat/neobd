"""Small, NumPy-vectorized differential evolution implementation."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray


Objective = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def differential_evolution(
    objective: Objective,
    bounds: NDArray[np.float64],
    population_size: int = 80,
    iterations: int = 120,
    mutation: float = 0.5,
    crossover: float = 0.9,
    seed: int = 1213,
) -> NDArray[np.float64]:
    """Minimize a vectorized objective using current-to-best/1/bin DE."""
    if population_size < 4:
        raise ValueError("population_size must be at least four")
    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    population = rng.uniform(lower, upper, size=(population_size, bounds.shape[0]))
    values = np.asarray(objective(population), dtype=float)
    for _ in range(iterations):
        best = population[int(np.argmin(values))]
        r1 = np.empty(population_size, dtype=int)
        r2 = np.empty(population_size, dtype=int)
        for i in range(population_size):
            candidates = np.delete(np.arange(population_size), i)
            r1[i], r2[i] = rng.choice(candidates, 2, replace=False)
        mutant = population + mutation * (
            best - population + population[r1] - population[r2]
        )
        mutant = np.where(mutant < lower, 0.5 * (population + lower), mutant)
        mutant = np.where(mutant > upper, 0.5 * (population + upper), mutant)
        mask = rng.random(population.shape) < crossover
        mask[
            np.arange(population_size),
            rng.integers(0, bounds.shape[0], population_size),
        ] = True
        trial = np.where(mask, mutant, population)
        trial_values = np.asarray(objective(trial), dtype=float)
        improved = trial_values < values
        population[improved] = trial[improved]
        values[improved] = trial_values[improved]
    return population[int(np.argmin(values))].copy()


def particle_swarm(
    objective: Objective,
    bounds: NDArray[np.float64],
    population_size: int = 1000,
    iterations: int = 1000,
    inertia_start: float = 0.9,
    inertia_end: float = 0.4,
    w4loc: float = 1.4,
    w4glo: float = 0.7,
    tolerance: float | None = 1e-10,
    patience: int = 20,
    seed: int = 1,
) -> NDArray[np.float64]:
    """Minimize a vectorized objective with a bounded particle swarm."""
    if population_size < 2:
        raise ValueError("population_size must be at least two")
    if iterations < 1 or patience < 1:
        raise ValueError("iterations and patience must be positive")
    if inertia_start < 0 or inertia_end < 0 or w4loc < 0 or w4glo < 0:
        raise ValueError("PSO coefficients must be non-negative")
    if tolerance is not None and tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    positions = rng.uniform(lower, upper, size=(population_size, bounds.shape[0]))
    velocities = np.zeros_like(positions)
    personal_positions = positions.copy()
    personal_values = np.asarray(objective(positions), dtype=float)
    best_index = int(np.argmin(personal_values))
    best_position = personal_positions[best_index].copy()
    best_value = float(personal_values[best_index])
    stale = 0
    for iteration in range(iterations):
        if tolerance is not None and best_value <= tolerance:
            break
        fraction = iteration / max(iterations - 1, 1)
        inertia = inertia_start + fraction * (inertia_end - inertia_start)
        local_random = rng.random((population_size, 1))
        global_random = rng.random((population_size, 1))
        velocities = (
            inertia * velocities
            + w4loc * local_random * (personal_positions - positions)
            + w4glo * global_random * (best_position - positions)
        )
        positions = positions + velocities
        outside = (positions < lower) | (positions > upper)
        replacements = rng.uniform(lower, upper, size=positions.shape)
        positions = np.where(outside, replacements, positions)
        velocities = np.where(outside, 0.0, velocities)
        values = np.asarray(objective(positions), dtype=float)
        improved = values < personal_values
        personal_positions[improved] = positions[improved]
        personal_values[improved] = values[improved]
        candidate = int(np.argmin(personal_values))
        candidate_value = float(personal_values[candidate])
        threshold = np.finfo(float).eps * max(abs(best_value), 1.0)
        if candidate_value < best_value - threshold:
            best_value = candidate_value
            best_position = personal_positions[candidate].copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_position


def minimize(
    objective: Objective,
    bounds: NDArray[np.float64],
    method: str,
    **options: float | int | None,
) -> NDArray[np.float64]:
    """Dispatch a vectorized objective to a configured optimizer."""
    canonical = method.lower()
    if canonical == "de":
        allowed = {"population_size", "iterations", "mutation", "crossover", "seed"}
        return differential_evolution(
            objective,
            bounds,
            **{key: value for key, value in options.items() if key in allowed},
        )
    if canonical == "pso":
        allowed = {
            "population_size",
            "iterations",
            "inertia_start",
            "inertia_end",
            "w4loc",
            "w4glo",
            "tolerance",
            "patience",
            "seed",
        }
        return particle_swarm(
            objective,
            bounds,
            **{key: value for key, value in options.items() if key in allowed},
        )
    raise ValueError(f"Unknown optimizer method: {method}")
