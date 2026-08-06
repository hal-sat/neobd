import json
from pathlib import Path

import numpy as np

from neobd.config import AnalysisConfig, GSPACArrayConfig
from neobd.gspac import (
    _circle_processes,
    _circular_statistics,
    _fit_circle,
    _ratio_model,
    _spectral_ratios,
    find_first_monotonic_interval,
    run_gspac,
)
from neobd.io import SiteLocation
from neobd.preprocess import SpectralStatistics


def _statistics(include_center: bool = True) -> SpectralStatistics:
    ring_locations = (
        SiteLocation("R01", 3.0, -1.0),
        SiteLocation("R02", 0.75, -1.0 + 0.75 * np.sqrt(3)),
        SiteLocation("R03", 0.75, -1.0 - 0.75 * np.sqrt(3)),
    )
    locations = (
        (SiteLocation("R04", 1.5, -1.0),) + ring_locations
        if include_center
        else ring_locations
    )
    sample_count = 16
    segment_count = 3
    frequency = np.fft.rfftfreq(sample_count, 0.125)
    time = np.arange(sample_count) * 0.125
    histories = np.empty((len(locations), 1, segment_count, sample_count))
    for site, location in enumerate(locations):
        for segment in range(segment_count):
            histories[site, 0, segment] = np.cos(
                2 * np.pi * time + 0.2 * site
            ) + 0.1 * segment * np.sin(4 * np.pi * time)
    cross = np.zeros((1, len(locations), len(locations), frequency.size), complex)
    power = np.ones((1, len(locations), frequency.size))
    return SpectralStatistics(
        frequency,
        locations,
        power,
        cross,
        np.arange(segment_count),
        histories,
    )


def test_loads_compact_gspac_config(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "GSPAC": {
                    "01p5": {
                        "center": "R04",
                        "ring": ["R01", "R02", "R03"],
                        "methods": ["cca", "v", "h0", "h1"],
                    }
                }
            }
        )
    )
    config = AnalysisConfig.load(path)
    assert config.gspac_arrays["01p5"].center == "R04"
    assert config.gspac_arrays["01p5"].methods == ("cca", "v", "h0", "h1")


def test_cca_config_does_not_require_a_center(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "GSPAC": {
                    "ring_only": {
                        "ring": ["R01", "R02", "R03"],
                        "methods": ["cca"],
                    }
                }
            }
        )
    )
    config = AnalysisConfig.load(path).gspac_arrays["ring_only"]
    assert config.center == ""
    assert config.methods == ("cca",)


def test_finds_expected_first_monotonic_branches() -> None:
    expected = {"cca": 2.4048, "h0": 2.4048, "h1": 1.8412, "v": 3.8317}
    for method, upper in expected.items():
        _, detected = find_first_monotonic_interval(_ratio_model(method))
        np.testing.assert_allclose(detected, upper, atol=1e-3)


def test_circle_is_fitted_from_ring_coordinates() -> None:
    coordinates = np.array(
        [
            [3.0, -1.0],
            [0.75, -1.0 + 0.75 * np.sqrt(3)],
            [0.75, -1.0 - 0.75 * np.sqrt(3)],
        ]
    )
    offsets, radii = _fit_circle(coordinates)
    np.testing.assert_allclose(np.mean(coordinates - offsets, axis=0), [1.5, -1.0])
    np.testing.assert_allclose(radii, 1.5)


def test_cca_does_not_require_center_record_or_coordinate() -> None:
    config = GSPACArrayConfig("MISSING", ("R01", "R02", "R03"), ("cca",))
    process0, process1, center, radius = _circle_processes(
        _statistics(include_center=False), config
    )
    assert center is None
    assert process0.shape == process1.shape == (3, 16)
    np.testing.assert_allclose(radius, 1.5)


def test_circular_statistics_are_smoothed_and_written(tmp_path: Path) -> None:
    config = GSPACArrayConfig("R04", ("R01", "R02", "R03"), ("cca", "v", "h0", "h1"))
    statistics = _statistics()
    unsmoothed = _circular_statistics(statistics, config, 0)
    smoothed = _circular_statistics(statistics, config, 1)
    assert not np.array_equal(unsmoothed.g00, smoothed.g00)
    assert set(_spectral_ratios(smoothed)) == {"cca", "v", "h0", "h1"}

    run_gspac(statistics, tmp_path, {"01p5": config}, smoothing_iterations=1)
    circular_output = tmp_path / "results_neobd" / "circular_statistics" / "01p5"
    for filename in ("G00.csv", "G11.csv", "Gcc.csv", "Gc0.csv"):
        path = circular_output / filename
        assert path.is_file()
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        np.testing.assert_array_equal(data[:, 0], statistics.frequency[1:])
    for method in config.methods:
        output = tmp_path / "results_neobd" / "gspac" / method
        assert (output / "spr_01p5.csv").is_file()
        assert (output / "phv_01p5.csv").is_file()
