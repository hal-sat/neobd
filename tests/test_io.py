from pathlib import Path

import numpy as np

from neobd.io import read_complex_csv, write_complex, write_real


def test_frequency_writers_omit_dc_row(tmp_path: Path) -> None:
    frequency = np.array([0.0, 1.0, 2.0])
    real_path = tmp_path / "real.csv"
    complex_path = tmp_path / "complex.csv"

    write_real(real_path, frequency, np.array([10.0, 20.0, 30.0]))
    write_complex(
        complex_path,
        frequency,
        np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]),
    )

    real_data = np.loadtxt(real_path, delimiter=",", ndmin=2)
    complex_frequency, complex_values = read_complex_csv(complex_path)
    np.testing.assert_array_equal(real_data[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(complex_frequency, [1.0, 2.0])
    np.testing.assert_array_equal(complex_values, [3.0 + 4.0j, 5.0 + 6.0j])
