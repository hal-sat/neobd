"""Configuration models and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GSPACArrayConfig:
    """One circular array used by generalized SPAC methods."""

    center: str
    ring: tuple[str, ...]
    methods: tuple[str, ...]

    @classmethod
    def from_raw(cls, name: str, raw: Any) -> "GSPACArrayConfig":
        center = str(raw.get("center", "")).strip()
        ring = tuple(str(site) for site in raw.get("ring", ()))
        methods = tuple(
            dict.fromkeys(str(method).lower() for method in raw.get("methods", ()))
        )
        if not ring:
            raise ValueError(f"GSPAC array {name!r} requires ring receivers")
        if center and center in ring:
            raise ValueError(f"GSPAC array {name!r} includes its center in the ring")
        unknown = set(methods) - {"cca", "v", "h0", "h1"}
        if unknown:
            raise ValueError(f"Unknown GSPAC methods for {name!r}: {sorted(unknown)}")
        if not methods:
            raise ValueError(f"GSPAC array {name!r} requires at least one method")
        if any(method != "cca" for method in methods) and not center:
            raise ValueError(
                f"GSPAC array {name!r} requires a center receiver for V/H0/H1"
            )
        return cls(center=center, ring=ring, methods=methods)


@dataclass(frozen=True)
class OptimizerConfig:
    """Settings shared by optimization-based analysis methods."""

    method: str = "de"
    population: int = 80
    iterations: int = 120
    target: float | None = None
    patience: int = 20
    mutation: float = 0.5
    crossover: float = 0.9
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    w4loc: float = 1.4
    w4glo: float = 0.7
    seed: int = 1213

    @classmethod
    def from_raw(
        cls, raw: Any, *, population: int, iterations: int, seed: int
    ) -> "OptimizerConfig":
        if raw is None:
            raw = {}
        if isinstance(raw, str):
            raw = {"method": raw}
        if not isinstance(raw, dict):
            raise ValueError("optimizer must be an object or a method name")
        inertia = raw.get("inertia", (0.9, 0.4))
        if not isinstance(inertia, (list, tuple)) or len(inertia) != 2:
            raise ValueError("optimizer inertia must contain [start, end]")
        target = raw.get("target")
        config = cls(
            method=str(raw.get("method", "de")).lower(),
            population=int(raw.get("population", population)),
            iterations=int(raw.get("iterations", iterations)),
            target=None if target is None else float(target),
            patience=int(raw.get("patience", 20)),
            mutation=float(raw.get("mutation", 0.5)),
            crossover=float(raw.get("crossover", 0.9)),
            inertia_start=float(inertia[0]),
            inertia_end=float(inertia[1]),
            w4loc=float(raw.get("w4loc", 1.4)),
            w4glo=float(raw.get("w4glo", 0.7)),
            seed=int(raw.get("seed", seed)),
        )
        if config.method not in {"de", "pso"}:
            raise ValueError("optimizer method must be de or pso")
        if config.population < 4:
            raise ValueError("optimizer population must be at least four")
        if config.iterations < 1 or config.patience < 1:
            raise ValueError("optimizer iterations and patience must be positive")
        if config.target is not None and config.target < 0:
            raise ValueError("optimizer target must be non-negative")
        if config.mutation < 0 or not 0 <= config.crossover <= 1:
            raise ValueError("invalid differential evolution coefficients")
        if (
            min(config.inertia_start, config.inertia_end, config.w4loc, config.w4glo)
            < 0
        ):
            raise ValueError("PSO coefficients must be non-negative")
        return config


@dataclass(frozen=True)
class FKConfig:
    """Parameters controlling FK analysis."""

    methods: tuple[str, ...] = ("mlm",)
    diagonal_loading: float = 0.02
    output_interval: int = 10
    max_frequency: float = 30.0
    radial_density: int = 100
    angular_density: int = 36
    min_velocity: float = 100.0
    max_velocity: float = 3500.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @classmethod
    def from_raw(cls, raw: Any) -> "FKConfig | None":
        if raw in (None, False, "False", "false"):
            return None
        if raw in (True, "True", "true"):
            return cls()
        density = raw.get("density", (100, 36))
        bounds = raw.get("bounds", (100, 3500))
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("FK bounds must contain [min_velocity, max_velocity]")
        min_velocity, max_velocity = (float(value) for value in bounds)
        if not math.isfinite(min_velocity) or not math.isfinite(max_velocity):
            raise ValueError("FK bounds must be finite")
        if min_velocity <= 0 or max_velocity <= min_velocity:
            raise ValueError("FK bounds must satisfy 0 < min < max")
        requested_methods = raw.get("methods", raw.get("method", "mlm"))
        if isinstance(requested_methods, str):
            methods = (requested_methods.lower(),)
        else:
            methods = tuple(str(method).lower() for method in requested_methods)
        methods = tuple(dict.fromkeys(methods))
        if not methods:
            raise ValueError("FK method list must not be empty")
        diagonal_loading = float(raw.get("diagonal_loading", 0.02))
        if diagonal_loading < 0:
            raise ValueError("diagonal_loading must be non-negative")
        return cls(
            methods=methods,
            diagonal_loading=diagonal_loading,
            output_interval=int(raw.get("output_interval", 10)),
            max_frequency=float(raw.get("max_frequency", 30)),
            radial_density=int(density[0]),
            angular_density=int(density[1]),
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            optimizer=OptimizerConfig.from_raw(
                raw.get("optimizer"), population=80, iterations=120, seed=1213
            ),
        )


@dataclass(frozen=True)
class DSPACConfig:
    """Parameters controlling directional SPAC analysis."""

    sites: tuple[str, ...]
    max_frequency: float = 3.0
    azimuth_offset_degrees: float = 0.0
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(population=100, iterations=100, seed=1)
    )

    @classmethod
    def from_raw(cls, raw: Any) -> "DSPACConfig | None":
        if raw is None:
            return None
        sites = tuple(str(site) for site in raw.get("array", ()))
        if len(set(sites)) < 3:
            raise ValueError("DSPAC requires at least three distinct receivers")
        max_frequency = float(raw.get("max_frequency", 3.0))
        if not math.isfinite(max_frequency) or max_frequency <= 0:
            raise ValueError("DSPAC max_frequency must be positive and finite")
        return cls(
            sites=sites,
            max_frequency=max_frequency,
            azimuth_offset_degrees=float(raw.get("azimuth_offset_degrees", 0.0)),
            optimizer=OptimizerConfig.from_raw(
                raw.get("optimizer"), population=100, iterations=100, seed=1
            ),
        )


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete analysis configuration."""

    path: Path
    segment_length: int = 1024
    smoothing_iterations: int = 0
    acceptance_range: float = 0.2
    robust_normalization: bool = True
    spac_arrays: dict[str, tuple[str, ...]] = field(default_factory=dict)
    gspac_arrays: dict[str, GSPACArrayConfig] = field(default_factory=dict)
    fk: FKConfig | None = None
    dspac: DSPACConfig | None = None
    n_para: int = 1

    @property
    def case_dir(self) -> Path:
        return self.path.parent

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        n_para = int(raw.get("n_para", 1))
        if n_para < 1:
            raise ValueError("n_para must be at least one")
        segment_length = int(raw.get("seg_len", 1024))
        if segment_length < 2 or segment_length & (segment_length - 1):
            raise ValueError("seg_len must be a power of two greater than one")
        spac = {
            str(name): tuple(str(site) for site in sites)
            for name, sites in raw.get("SPAC", {}).items()
        }
        gspac = {
            str(name): GSPACArrayConfig.from_raw(str(name), value)
            for name, value in raw.get("GSPAC", {}).items()
        }
        return cls(
            path=config_path,
            segment_length=segment_length,
            smoothing_iterations=int(raw.get("n_smooth", raw.get("n_smoothing", 0))),
            acceptance_range=float(raw.get("acceptance_range", 0.2)),
            robust_normalization=bool(raw.get("robust_normalization", True)),
            spac_arrays=spac,
            gspac_arrays=gspac,
            fk=FKConfig.from_raw(raw.get("FK")),
            dspac=DSPACConfig.from_raw(raw.get("DSPAC")),
            n_para=n_para,
        )
