"""Stable input and output adapters for the original CSV layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SiteLocation:
    """A receiver name and Cartesian coordinate."""

    name: str
    x: float
    y: float


def read_coordinates(case_dir: Path) -> tuple[SiteLocation, ...]:
    locations: list[SiteLocation] = []
    with (case_dir / "array_coord.csv").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                raise ValueError(f"Invalid array_coord.csv line {number}")
            locations.append(
                SiteLocation(Path(fields[2]).stem, float(fields[0]), float(fields[1]))
            )
    if not locations:
        raise ValueError("array_coord.csv contains no receivers")
    return tuple(locations)


def collect_site_files(case_dir: Path, site: str) -> tuple[Path, ...]:
    base = case_dir / site
    if base.is_dir():
        files = tuple(sorted(base.glob("*.csv")))
    elif base.is_file():
        files = (base,)
    elif base.with_suffix(".csv").is_file():
        files = (base.with_suffix(".csv"),)
    else:
        files = ()
    if not files:
        raise FileNotFoundError(f"No CSV input found for receiver {site!r}")
    return files


def read_timeseries(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(f"Time-series file is too small: {path}")
    return data[:, 0], data[:, 1:].T


def read_complex_csv(path: Path) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def write_real(path: Path, *columns: NDArray[np.generic]) -> None:
    """Write frequency-domain columns while omitting the DC row."""
    if not columns:
        raise ValueError("At least one output column is required")
    values = np.column_stack(columns)
    if values.size:
        values = values[values[:, 0] != 0.0]
    np.savetxt(path, values, delimiter=", ", fmt="%.15g")


def write_complex(
    path: Path, frequency: NDArray[np.float64], values: NDArray[np.complex128]
) -> None:
    write_real(path, frequency, values.real, values.imag)
