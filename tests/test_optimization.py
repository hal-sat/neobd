import numpy as np

from neobd.optimization import differential_evolution, particle_swarm


def test_differential_evolution_finds_quadratic_minimum() -> None:
    optimum = differential_evolution(
        lambda points: np.sum((points - np.array([0.25, -0.5])) ** 2, axis=1),
        np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        population_size=30,
        iterations=80,
    )
    np.testing.assert_allclose(optimum, [0.25, -0.5], atol=1e-4)


def test_particle_swarm_finds_quadratic_minimum() -> None:
    optimum = particle_swarm(
        lambda values: np.sum((values - np.array([0.25, -0.5])) ** 2, axis=1),
        np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        population_size=40,
        iterations=200,
        w4loc=1.8,
        w4glo=0.6,
        tolerance=None,
        patience=50,
        seed=3,
    )
    np.testing.assert_allclose(optimum, [0.25, -0.5], atol=1e-4)
