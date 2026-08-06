import json
from pathlib import Path

import numpy as np

from neobd.config import AnalysisConfig, GSPACArrayConfig
from neobd.gspac import (
    _ratio_model,
    _spectral_ratios,
    find_first_monotonic_interval,
    run_gspac,
)
from neobd.io import SiteLocation
from neobd.preprocess import SpectralStatistics


def _statistics() -> SpectralStatistics:
    locations = (
        SiteLocation("R04", 0.0, 0.0),
        SiteLocation("R01", 1.0, 0.0),
        SiteLocation("R02", -0.5, np.sqrt(3) / 2),
        SiteLocation("R03", -0.5, -np.sqrt(3) / 2),
    )
    frequency = np.array([0.0, 1.0, 2.0])
    cross = np.zeros((1, 4, 4, 3), dtype=complex)
    cross[0] = np.eye(4)[:, :, None]
    return SpectralStatistics(
        frequency, locations, np.ones((1, 4, 3)), cross, np.array([0])
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


def test_finds_expected_first_monotonic_branches() -> None:
    expected = {"cca": 2.4048, "h0": 2.4048, "h1": 1.8412, "v": 3.8317}
    for method, upper in expected.items():
        _, detected = find_first_monotonic_interval(_ratio_model(method))
        np.testing.assert_allclose(detected, upper, atol=1e-3)


def test_radius_is_automatic_and_outputs_are_separated(tmp_path: Path) -> None:
    config = GSPACArrayConfig("R04", ("R01", "R02", "R03"), ("cca", "v", "h0", "h1"))
    ratios, radius = _spectral_ratios(_statistics(), config)
    np.testing.assert_allclose(radius, 1.0)
    assert set(ratios) == {"cca", "v", "h0", "h1"}
    run_gspac(_statistics(), tmp_path, {"01p5": config})
    for method in config.methods:
        output = tmp_path / "results_neobd" / "gspac" / method
        for prefix in ("spr", "phv"):
            path = output / f"{prefix}_01p5.csv"
            assert path.is_file()
            data = np.loadtxt(path, delimiter=",", ndmin=2)
            np.testing.assert_array_equal(data[:, 0], [1.0, 2.0])
