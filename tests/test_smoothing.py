import numpy as np
import pytest

from neobd.smoothing import (
    Hann3PointSmoother,
    ParzenSmoother,
    SmoothingConfig,
)


def test_hann_3point_matches_the_legacy_kernel() -> None:
    frequency = np.arange(5.0)
    values = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    result = Hann3PointSmoother(1).apply(values, frequency)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 1.0, 0.0])


def test_parzen_preserves_a_constant_at_frequency_edges() -> None:
    frequency = np.linspace(0.0, 10.0, 101)
    values = np.full((2, frequency.size), 3.0 + 2.0j)
    result = ParzenSmoother(0.3).apply(values, frequency)
    np.testing.assert_allclose(result, values, atol=1e-12)


def test_parzen_bandwidth_matches_rectangular_kernel_variance() -> None:
    bandwidth = 0.3
    frequency = np.linspace(-50.0, 50.0, 20_001)
    values = np.zeros(frequency.size)
    values[frequency.size // 2] = 1.0
    weights = ParzenSmoother(bandwidth).apply(values, frequency)
    weights /= np.sum(weights)
    variance = np.sum(frequency**2 * weights)
    np.testing.assert_allclose(variance, bandwidth**2 / 12.0, rtol=2e-3)


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "Parzen", "params": [0.0]},
        {"type": "Hann_3point", "params": [1.5]},
        {"type": "Gaussian", "params": [1.0]},
        {"type": "Parzen", "params": []},
    ],
)
def test_invalid_smoothing_configuration_is_rejected(raw) -> None:
    with pytest.raises(ValueError):
        SmoothingConfig.from_raw(raw)
