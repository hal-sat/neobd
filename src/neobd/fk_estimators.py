"""Numerically stable FK spectral estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


def hermitian_part(matrix: ComplexMatrix) -> ComplexMatrix:
    """Remove finite-sample asymmetry from a cross-spectral matrix."""
    return 0.5 * (matrix + matrix.conj().T)


class FKEstimator(ABC):
    """Interface for an FK beam-power estimator."""

    @abstractmethod
    def prepare(self, cross_spectral_matrix: ComplexMatrix) -> ComplexMatrix:
        """Prepare a matrix for repeated steering-vector evaluations."""

    @abstractmethod
    def power(
        self, prepared: ComplexMatrix, steering: ComplexMatrix
    ) -> NDArray[np.float64]:
        """Evaluate beam power for one or more steering vectors."""


class MLMEstimator(FKEstimator):
    """Capon/MLM estimator using loaded Hermitian eigendecomposition."""

    def __init__(self, diagonal_loading: float = 0.02) -> None:
        if diagonal_loading < 0:
            raise ValueError("diagonal_loading must be non-negative")
        self.diagonal_loading = diagonal_loading

    def prepare(self, cross_spectral_matrix: ComplexMatrix) -> ComplexMatrix:
        covariance = hermitian_part(cross_spectral_matrix)
        scale = max(float(np.trace(covariance).real / covariance.shape[0]), 0.0)
        if not np.isfinite(scale) or scale == 0.0:
            scale = float(np.linalg.norm(covariance, ord=2))
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            raise ValueError("Cross-spectral matrix has no usable power")
        normalized = covariance / scale
        loaded = normalized + self.diagonal_loading * np.eye(covariance.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(loaded)
        floor = np.finfo(float).eps * covariance.shape[0]
        eigenvalues = np.maximum(eigenvalues.real, floor)
        return (eigenvectors / eigenvalues) @ eigenvectors.conj().T

    def power(
        self, prepared: ComplexMatrix, steering: ComplexMatrix
    ) -> NDArray[np.float64]:
        denominator = np.einsum(
            "...i,ij,...j->...", steering.conj(), prepared, steering
        ).real
        scale = float(np.max(np.abs(denominator)))
        if not np.isfinite(scale) or scale == 0.0:
            return np.zeros_like(denominator)
        floor = np.finfo(float).eps * scale
        return np.divide(
            1.0,
            denominator,
            out=np.zeros_like(denominator),
            where=np.isfinite(denominator) & (denominator > floor),
        )


class BFMEstimator(FKEstimator):
    """Conventional Bartlett beam-forming estimator."""

    def prepare(self, cross_spectral_matrix: ComplexMatrix) -> ComplexMatrix:
        return hermitian_part(cross_spectral_matrix)

    def power(
        self, prepared: ComplexMatrix, steering: ComplexMatrix
    ) -> NDArray[np.float64]:
        values = np.einsum(
            "...i,ij,...j->...", steering.conj(), prepared, steering
        ).real
        return np.maximum(values / prepared.shape[0] ** 2, 0.0)


def create_estimator(method: str, diagonal_loading: float) -> FKEstimator:
    """Create an estimator while accepting established method aliases."""
    aliases = {"capon": "mlm", "bartlett": "bfm", "bmf": "bfm"}
    canonical = aliases.get(method.lower(), method.lower())
    if canonical == "mlm":
        return MLMEstimator(diagonal_loading)
    if canonical == "bfm":
        return BFMEstimator()
    raise ValueError(f"Unknown FK method: {method!r}")
