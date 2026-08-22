from pathlib import Path

import numpy as np

from neobd.io import SiteLocation
from neobd.preprocess import SpectralStatistics, normalize_cross_spectra
from neobd.spac import run_spac


def test_nonrobust_cross_spectra_use_the_first_receiver_power() -> None:
    power = np.array([[[4.0], [9.0]]])
    cross = np.zeros((1, 2, 2, 1), dtype=complex)
    cross[0, 0, 1, 0] = 1.0 + 0.4j
    cross[0, 1, 0, 0] = 1.0 - 0.4j
    normalized = normalize_cross_spectra(cross, power, robust=False)
    np.testing.assert_allclose(normalized[0, 0, 1], (1.0 + 0.4j) / 4.0)
    np.testing.assert_allclose(normalized[0, 1, 0], (1.0 - 0.4j) / 9.0)


def test_spac_uses_the_normalized_cross_spectrum(tmp_path: Path) -> None:
    frequency = np.array([0.0, 1.0])
    power = np.array([[[4.0, 4.0], [9.0, 9.0]]])
    cross = np.zeros((1, 2, 2, 2), dtype=complex)
    cross[0, 0, 1] = 1.0 + 0.4j
    cross[0, 1, 0] = 1.0 - 0.4j
    normalized = normalize_cross_spectra(cross, power, robust=False)
    statistics = SpectralStatistics(
        frequency=frequency,
        locations=(SiteLocation("X01", 0.0, 0.0), SiteLocation("X02", 1.0, 0.0)),
        power=power,
        cross=cross,
        valid_segments=np.array([0]),
        normalized_cross=normalized,
    )
    output = tmp_path / "results_neobd" / "spac"
    output.mkdir(parents=True)
    run_spac(statistics, tmp_path, {"pair": ("X01", "X02")})
    result = np.loadtxt(output / "spr_pair.csv", delimiter=",", ndmin=2)
    np.testing.assert_allclose(result[0], [1.0, 0.25, 0.1])
