from pathlib import Path

import numpy as np

from neobd.config import FKConfig, OptimizerConfig
from neobd.fk_estimators import BFMEstimator, MLMEstimator
from neobd.fk_runner import _circular_grid, run_fk
from neobd.io import SiteLocation
from neobd.preprocess import SpectralStatistics


def test_mlm_stabilizes_non_hermitian_singular_matrix() -> None:
    matrix = np.array([[1.0, 1.0 + 0.2j], [1.0 - 0.1j, 1.0]], dtype=complex)
    inverse = MLMEstimator(diagonal_loading=0.02).prepare(matrix)
    np.testing.assert_allclose(inverse, inverse.conj().T, atol=1e-12)
    assert np.all(np.isfinite(inverse))


def test_mlm_power_is_invariant_to_cross_spectral_scale() -> None:
    matrix = np.array([[2.0, 0.4 + 0.2j], [0.4 - 0.2j, 1.0]], dtype=complex)
    steering = np.array([[1.0, 1.0], [1.0, 1.0j], [1.0, -1.0]], dtype=complex)
    estimator = MLMEstimator(diagonal_loading=0.02)
    reference = estimator.power(estimator.prepare(matrix), steering)
    for scale in (1e-20, 1e20):
        scaled = estimator.power(estimator.prepare(scale * matrix), steering)
        np.testing.assert_allclose(scaled, reference, rtol=1e-12, atol=0.0)


def test_mlm_rejects_a_zero_power_matrix() -> None:
    import pytest

    with pytest.raises(ValueError, match="no usable power"):
        MLMEstimator().prepare(np.zeros((2, 2), dtype=complex))


def test_bfm_uses_hermitian_cross_spectral_matrix() -> None:
    matrix = np.array([[2.0, 1.0 + 0.2j], [1.0 - 0.1j, 2.0]], dtype=complex)
    prepared = BFMEstimator().prepare(matrix)
    np.testing.assert_allclose(prepared, prepared.conj().T)


def test_circular_grid_uses_phase_velocity_limits() -> None:
    config = FKConfig(
        min_velocity=100, max_velocity=2000, radial_density=8, angular_density=12
    )
    sx, sy, _ = _circular_grid(config)
    np.testing.assert_allclose(np.max(np.abs(sx)), 0.01)
    np.testing.assert_allclose(np.max(np.abs(sy)), 0.01)
    radius = np.hypot(sx, sy)
    np.testing.assert_allclose(radius[0], 1 / 2000)
    np.testing.assert_allclose(radius[-1], 1 / 100)


def test_run_fk_writes_mlm_and_bfm_directories(tmp_path: Path) -> None:
    frequency = np.array([0.0, 1.0, 2.0])
    cross = np.zeros((1, 2, 2, 3), dtype=complex)
    cross[0, :, :, 1:] = np.array([[1.0, 0.5], [0.5, 1.0]])[:, :, None]
    statistics = SpectralStatistics(
        frequency=frequency,
        locations=(SiteLocation("A", 0.0, 0.0), SiteLocation("B", 1.0, 0.0)),
        power=np.ones((1, 2, 3)),
        cross=cross,
        valid_segments=np.array([0]),
    )
    config = FKConfig(
        methods=("mlm", "bfm"),
        output_interval=1,
        max_frequency=2.0,
        radial_density=5,
        angular_density=8,
        optimizer=OptimizerConfig(method="pso", population=8, iterations=3),
    )
    run_fk(statistics, tmp_path, config, n_para=2)
    for method in ("MLM", "BFM"):
        output = tmp_path / "results_neobd" / "fk" / method
        assert (output / "phv_fk.csv").is_file()
        assert len(list(output.glob("FK_*.csv"))) == 2
        fv = np.loadtxt(output / "fv.csv", delimiter=",", ndmin=2)
        assert fv.shape == (10, 3)
        np.testing.assert_allclose(np.unique(fv[:, 1]), np.linspace(100, 3500, 5))
        assert np.all((fv[:, 2] >= 0) & (fv[:, 2] <= 1))
        for frequency in np.unique(fv[:, 0]):
            np.testing.assert_allclose(np.max(fv[fv[:, 0] == frequency, 2]), 1.0)


def test_visualization_saves_only_when_output_is_requested(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from neobd.visualization import PHASE_VELOCITIES, _inner_hole_mask, visualize_fk

    angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    radii = np.linspace(0.001, 0.01, 5)
    sx = (radii[:, None] * np.cos(angles)).ravel()
    sy = (radii[:, None] * np.sin(angles)).ravel()
    power = np.exp(-((sx - 0.002) ** 2 + sy**2) / 1e-5)
    source = tmp_path / "FK_04p44410_Hz.csv"
    with source.open("w", encoding="utf-8") as stream:
        stream.write("# optimum_sx=0.002, optimum_sy=0\n")
        np.savetxt(stream, np.column_stack((sx, sy, power)), delimiter=",")
    mask = _inner_hole_mask(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.0]]),
        np.array([[0, 1, 2], [0, 1, 3]]),
    )
    np.testing.assert_array_equal(mask, [True, False])
    destination = tmp_path / "map.png"
    result = visualize_fk(source, destination, show=False)
    assert result == destination.resolve()
    assert destination.is_file()
    assert PHASE_VELOCITIES == (100.0, 125.0, 250.0, 500.0, 750.0, 1000.0, 1500.0)


def test_fv_visualization_saves_requested_output(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from neobd.fv_visualization import _read_fv, _to_decibels, visualize_fv

    frequency = np.array([1.0, 2.0, 3.0])
    velocity = np.array([100.0, 200.0, 300.0, 400.0])
    values = np.array(
        [
            [f, c, np.exp(-(((c - 100 * f) / 100) ** 2))]
            for f in frequency
            for c in velocity
        ]
    )
    source = tmp_path / "fv.csv"
    np.savetxt(source, values, delimiter=",")
    loaded_frequency, loaded_velocity, power = _read_fv(source)
    np.testing.assert_array_equal(loaded_frequency, frequency)
    np.testing.assert_array_equal(loaded_velocity, velocity)
    assert power.shape == (velocity.size, frequency.size)
    power_db = _to_decibels(power, -20.0)
    np.testing.assert_allclose(np.max(power_db, axis=0), 0.0)
    assert np.min(power_db) == -20.0
    destination = tmp_path / "fv.png"
    result = visualize_fv(
        source, destination, show=False, decibels=True, minimum_db=-20.0
    )
    assert result == destination.resolve()
    assert destination.is_file()
