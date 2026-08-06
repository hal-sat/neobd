import json
from pathlib import Path

import numpy as np
from scipy.special import jv

from neobd.config import AnalysisConfig, DSPACConfig, OptimizerConfig
from neobd.dspac import _imaginary_misfit, _real_misfit, run_dspac
from neobd.io import SiteLocation
from neobd.preprocess import SpectralStatistics


def test_dspac_objectives_match_the_cpp_formulation() -> None:
    radii = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    angles = np.linspace(-1.0, 1.0, radii.size)
    real_parameters = np.array([0.4, 0.2, -0.1, 0.05, 0.3])
    real_observed = jv(0, real_parameters[0] * radii)
    for order in (1, 2):
        harmonic = 2 * order
        core = real_parameters[2 * order - 1] * np.cos(
            harmonic * angles
        ) + real_parameters[2 * order] * np.sin(harmonic * angles)
        real_observed += (
            2 * (-1) ** order * jv(harmonic, real_parameters[0] * radii) * core
        )
    np.testing.assert_allclose(
        _real_misfit(real_parameters[None, :], radii, angles, real_observed),
        0.0,
        atol=1e-14,
    )

    imaginary_parameters = np.array([0.1, -0.2, 0.3, -0.4])
    imaginary_observed = np.zeros(radii.size)
    for order in (1, 2):
        harmonic = 2 * order - 1
        parameter = 2 * order - 2
        core = imaginary_parameters[parameter] * np.cos(
            harmonic * angles
        ) + imaginary_parameters[parameter + 1] * np.sin(harmonic * angles)
        imaginary_observed -= (
            2 * (-1) ** order * jv(harmonic, real_parameters[0] * radii) * core
        )
    np.testing.assert_allclose(
        _imaginary_misfit(
            imaginary_parameters[None, :],
            real_parameters[0],
            radii,
            angles,
            imaginary_observed,
        ),
        0.0,
        atol=1e-14,
    )


def test_dspac_config_reads_pso_parameters(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "DSPAC": {
                    "array": ["A", "B", "C"],
                    "optimizer": {
                        "method": "pso",
                        "population": 12,
                        "iterations": 7,
                        "w4loc": 1.8,
                        "w4glo": 0.6,
                    },
                }
            }
        )
    )
    config = AnalysisConfig.load(path).dspac
    assert config is not None
    assert config.optimizer.method == "pso"
    assert config.optimizer.population == 12
    assert config.optimizer.iterations == 7
    assert config.optimizer.w4loc == 1.8
    assert config.optimizer.w4glo == 0.6


def test_dspac_writes_real_and_imaginary_results(tmp_path: Path) -> None:
    locations = (
        SiteLocation("A", 0.0, 0.0),
        SiteLocation("B", 1.0, 0.0),
        SiteLocation("C", 0.0, 1.0),
    )
    frequency = np.array([0.0, 1.0, 2.0])
    cross = np.zeros((1, 3, 3, 3), dtype=complex)
    cross[0, :, :, 1:] = np.eye(3)[:, :, None]
    statistics = SpectralStatistics(
        frequency,
        locations,
        np.ones((1, 3, 3)),
        cross,
        np.array([0]),
    )
    config = DSPACConfig(
        sites=("A", "B", "C"),
        max_frequency=2.0,
        optimizer=OptimizerConfig(
            method="pso", population=8, iterations=3, target=None
        ),
    )
    run_dspac(statistics, tmp_path, config, n_para=2)
    output = tmp_path / "results_neobd" / "dspac"
    real = np.loadtxt(output / "result_real.csv", delimiter=",", ndmin=2)
    imaginary = np.loadtxt(output / "result_imag.csv", delimiter=",", ndmin=2)
    assert real.shape == (2, 6)
    assert imaginary.shape == (2, 5)
    assert real[0, 0] == imaginary[0, 0] == 1.0
    assert np.all(np.isfinite(real))
    assert np.all(np.isfinite(imaginary))
