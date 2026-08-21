"""Standalone visualization for FK frequency-velocity data."""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np


def _read_fv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Invalid F-V data: {path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"F-V data contains non-finite values: {path}")
    frequency = np.unique(data[:, 0])
    velocity = np.unique(data[:, 1])
    if frequency.size < 2 or velocity.size < 2:
        raise ValueError(f"F-V data requires at least a 2-by-2 grid: {path}")
    if data.shape[0] != frequency.size * velocity.size:
        raise ValueError(f"F-V data does not form a complete grid: {path}")
    power = np.full((velocity.size, frequency.size), np.nan)
    frequency_index = np.searchsorted(frequency, data[:, 0])
    velocity_index = np.searchsorted(velocity, data[:, 1])
    if (
        np.unique(np.column_stack((velocity_index, frequency_index)), axis=0).shape[0]
        != data.shape[0]
    ):
        raise ValueError(f"F-V data contains duplicate grid points: {path}")
    power[velocity_index, frequency_index] = data[:, 2]
    return frequency, velocity, power


def _to_decibels(power: np.ndarray, minimum_db: float) -> np.ndarray:
    """Convert power to per-frequency decibels with a fixed lower limit."""
    if not math.isfinite(minimum_db) or minimum_db >= 0:
        raise ValueError("minimum_db must be finite and negative")
    maximum = np.max(power, axis=0, keepdims=True)
    relative = np.divide(
        power,
        maximum,
        out=np.zeros_like(power),
        where=maximum > 0,
    )
    floor = 10.0 ** (minimum_db / 10.0)
    return np.clip(10.0 * np.log10(np.maximum(relative, floor)), minimum_db, 0.0)


def visualize_fv(
    path: str | Path,
    output: str | Path | None = None,
    show: bool = True,
    decibels: bool = False,
    minimum_db: float = -30.0,
) -> Path | None:
    """Display an F-V map and optionally save it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "F-V visualization requires the matplotlib optional dependency"
        ) from error
    source = Path(path).expanduser().resolve()
    frequency, velocity, power = _read_fv(source)
    if decibels:
        power = _to_decibels(power, minimum_db)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    color = axis.pcolormesh(
        frequency, velocity, power, shading="gouraud", cmap="viridis"
    )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Phase velocity (m/s)")
    axis.set_title(source.parent.name + " F-V spectrum")
    figure.colorbar(
        color,
        ax=axis,
        label=(
            f"Azimuthal mean normalized power (dB, min {minimum_db:g})"
            if decibels
            else "Azimuthal mean normalized power"
        ),
    )
    destination = None if output is None else Path(output).expanduser().resolve()
    if destination is not None:
        figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return destination
