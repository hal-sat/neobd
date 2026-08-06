"""Standalone visualization for FK CSV maps."""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np

PHASE_VELOCITIES = (100.0, 125.0, 250.0, 500.0, 750.0, 1000.0, 1500.0)


def _read_optimum(header: str) -> tuple[float, float] | None:
    match = re.search(r"optimum_sx=([^,]+),\s*optimum_sy=([^\s]+)", header)
    if match:
        return float(match.group(1)), float(match.group(2))
    fields = header.lstrip("# ").split(",")
    if len(fields) == 2:
        try:
            return float(fields[0]), float(fields[1])
        except ValueError:
            return None
    return None


def _frequency_from_name(path: Path) -> float | None:
    match = re.search(r"FK_([0-9]+p[0-9]+)(?:_Hz)?", path.stem)
    return None if match is None else float(match.group(1).replace("p", "."))


def _inner_hole_mask(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Mask Delaunay triangles spanning the unsampled central disk."""
    radii = np.hypot(points[:, 0], points[:, 1])
    positive = radii[radii > 0]
    if positive.size == 0:
        return np.zeros(triangles.shape[0], dtype=bool)
    inner_radius = float(np.min(positive))
    inner_vertices = np.isclose(radii, inner_radius, rtol=1e-7, atol=0.0)
    return np.all(inner_vertices[triangles], axis=1)


def visualize_fk(
    path: str | Path,
    output: str | Path | None = None,
    show: bool = True,
    decibels: bool = False,
) -> Path | None:
    """Display an FK map and optionally save it when output is specified."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.tri as mtri
    except ImportError as error:
        raise RuntimeError(
            "FK visualization requires the matplotlib optional dependency"
        ) from error
    source = Path(path).expanduser().resolve()
    header = source.read_text(encoding="utf-8").splitlines()[0]
    data = np.loadtxt(source, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Invalid FK map: {source}")
    points, indices = np.unique(data[:, :2], axis=0, return_index=True)
    values = data[indices, 2]
    if points.shape[0] < 3:
        raise ValueError(f"FK map contains too few unique points: {source}")
    if decibels:
        positive = values[values > 0]
        floor = float(np.min(positive)) if positive.size else np.finfo(float).tiny
        values = 10 * np.log10(np.maximum(values, floor) / np.max(values))
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    triangulation.set_mask(_inner_hole_mask(points, triangulation.triangles))
    refined, smooth_values = mtri.UniformTriRefiner(triangulation).refine_field(
        values, subdiv=3
    )
    smooth_values = np.clip(smooth_values, np.min(values), np.max(values))
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    levels = np.linspace(float(np.min(values)), float(np.max(values)), 200)
    contour = axis.tricontourf(refined, smooth_values, levels=levels, cmap="viridis")
    optimum = _read_optimum(header)
    frequency = _frequency_from_name(source)
    title = source.stem
    if optimum is not None:
        axis.plot(*optimum, color="red", marker="x", markersize=9, markeredgewidth=1.5)
        norm = float(np.hypot(*optimum))
        velocity = np.inf if norm == 0 else 1 / norm
        if frequency is not None:
            title = f"{frequency:.2f} Hz, {velocity:.1f} m/s"
    for velocity in PHASE_VELOCITIES:
        axis.add_patch(
            patches.Circle(
                (0, 0),
                1 / velocity,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )
        )
    labels = ", ".join(f"{velocity:g}" for velocity in PHASE_VELOCITIES)
    axis.text(
        0.5,
        -0.14,
        f"White concentric circles correspond to c = [{labels}] m/s.",
        transform=axis.transAxes,
        fontsize=8,
        ha="center",
        va="top",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("Slowness x (s/m)")
    axis.set_ylabel("Slowness y (s/m)")
    axis.set_title(title)
    figure.colorbar(
        contour,
        ax=axis,
        label="Normalized power (dB)" if decibels else "Normalized power",
    )
    destination = None if output is None else Path(output).expanduser().resolve()
    if destination is not None:
        figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return destination
