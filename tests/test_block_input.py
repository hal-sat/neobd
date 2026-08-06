from pathlib import Path

import numpy as np

from neobd.io import SiteLocation
from neobd.preprocess import Preprocessor


def _write_block(path: Path, time: np.ndarray) -> None:
    signal = np.sin(np.arange(time.size, dtype=float))
    np.savetxt(path, np.column_stack((time, signal)), delimiter=",")


def test_block_input_uses_median_of_all_positive_time_steps(tmp_path: Path) -> None:
    site_dir = tmp_path / "R01"
    site_dir.mkdir()
    _write_block(
        site_dir / "000.csv",
        np.array([0.000, 0.016, 0.033, 0.050, 0.066, 0.083, 0.100, 0.116]),
    )
    _write_block(
        site_dir / "001.csv",
        np.array([1.000, 1.017, 1.034, 1.050, 1.067, 1.084, 1.100, 1.117]),
    )
    (tmp_path / "valid_segments.csv").write_text("000000.csv\n", encoding="utf-8")
    for name in ("statistics", "spectra"):
        (tmp_path / "results_neobd" / name).mkdir(parents=True, exist_ok=True)

    statistics = Preprocessor(tmp_path, segment_length=4).run(
        (SiteLocation("R01", 0.0, 0.0),),
        smoothing_iterations=0,
        robust_normalization=True,
    )

    all_steps = np.concatenate(
        [
            np.diff(np.loadtxt(path, delimiter=",")[:, 0])
            for path in sorted(site_dir.glob("*.csv"))
        ]
    )
    expected_dt = np.median(all_steps[all_steps > 0])
