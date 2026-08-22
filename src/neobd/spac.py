"""Spatial autocorrelation analysis."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, jn_zeros

from .io import write_complex, write_real
from .preprocess import SpectralStatistics


def run_spac(
    statistics: SpectralStatistics, case_dir: Path, arrays: dict[str, tuple[str, ...]]
) -> None:
    index = {
        location.name: position
        for position, location in enumerate(statistics.locations)
    }
    coordinate = {
        location.name: (location.x, location.y) for location in statistics.locations
    }
    minimum = float(j0(jn_zeros(1, 1)[0]))
    kr_max = float(jn_zeros(1, 1)[0])
    for name, sites in arrays.items():
        if len(sites) == 0 or len(sites) % 2:
            raise ValueError(f"SPAC array {name!r} must contain receiver pairs")
        coefficients = []
        radii = []
        for first, second in zip(sites[::2], sites[1::2]):
            i, j = index[first], index[second]
            if statistics.normalized_cross is None:
                raise ValueError("SPAC requires normalized cross spectra")
            coefficients.append(statistics.normalized_cross[0, i, j])
            radii.append(
                float(
                    np.hypot(
                        coordinate[first][0] - coordinate[second][0],
                        coordinate[first][1] - coordinate[second][1],
                    )
                )
            )
        coefficient = np.mean(coefficients, axis=0)
        radius = float(np.mean(radii))
        phase_velocity = np.empty(statistics.frequency.size)
        for k, (frequency, value) in enumerate(
            zip(statistics.frequency, coefficient.real)
        ):
            if frequency == 0.0 or value >= 1.0:
                phase_velocity[k] = 5000.0
            elif value <= minimum:
                phase_velocity[k] = min(2 * np.pi * frequency * radius / kr_max, 5000.0)
            else:
                root = brentq(lambda argument: j0(argument) - value, 0.0, kr_max)
                phase_velocity[k] = min(2 * np.pi * frequency * radius / root, 5000.0)
        write_complex(
            case_dir / "results_neobd" / "spac" / f"spr_{name}.csv",
            statistics.frequency,
            coefficient,
        )
        write_real(
            case_dir / "results_neobd" / "spac" / f"phv_{name}.csv",
            statistics.frequency,
            phase_velocity,
        )
