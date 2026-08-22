"""Segmentation and cross-spectral statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence
import numpy as np
from numpy.typing import NDArray

from .io import (
    SiteLocation,
    collect_site_files,
    read_timeseries,
    write_complex,
    write_real,
)
from .selection import (
    ModalRMSSelector,
    SegmentSelector,
    read_valid_segments,
    write_valid_segments,
)
from .smoothing import SmoothingConfig, create_smoother


COMPONENTS = ("UD", "NS", "EW")


@dataclass
class ReceiverSpectra:
    """Segment spectra and metadata for one receiver."""

    location: SiteLocation
    segments: NDArray[np.float64]
    spectra: NDArray[np.complex128]
    normalized_rms: NDArray[np.float64]


@dataclass
class SpectralStatistics:
    """In-memory spectral products shared by SPAC and FK estimators."""

    frequency: NDArray[np.float64]
    locations: tuple[SiteLocation, ...]
    power: NDArray[np.float64]
    cross: NDArray[np.complex128]
    valid_segments: NDArray[np.int64]
    time_histories: NDArray[np.float64] | None = None
    normalized_cross: NDArray[np.complex128] | None = None


def normalize_cross_spectra(
    cross: NDArray[np.complex128],
    power: NDArray[np.float64],
    robust: bool,
) -> NDArray[np.complex128]:
    """Normalize CSDs for elastic-wave or attenuation analysis."""
    if robust:
        denominator = np.sqrt(power[:, :, None, :] * power[:, None, :, :])
    else:
        denominator = power[:, :, None, :]
    return np.divide(
        cross,
        denominator,
        out=np.full_like(cross, np.nan + 0j),
        where=denominator > np.finfo(float).tiny,
    )


class Preprocessor:
    """Read receivers and produce spectra using the original numerical conventions."""

    def __init__(self, case_dir: Path, segment_length: int) -> None:
        self.case_dir = case_dir
        self.segment_length = segment_length

    def _load_receiver(
        self, location: SiteLocation
    ) -> tuple[ReceiverSpectra, NDArray[np.float64]]:
        blocks: list[NDArray[np.float64]] = []
        time_steps: list[NDArray[np.float64]] = []
        for path in collect_site_files(self.case_dir, location.name):
            time, channels = read_timeseries(path)
            differences = np.diff(time)
            time_steps.append(differences[np.isfinite(differences) & (differences > 0)])
            channels = channels - channels.mean(axis=1, keepdims=True)
            blocks.append(channels)
        if not any(steps.size for steps in time_steps):
            raise ValueError(f"No valid sampling interval at {location.name}")
        component_count = blocks[0].shape[0]
        if any(block.shape[0] != component_count for block in blocks):
            raise ValueError(f"Inconsistent component count at {location.name}")
        step = self.segment_length // 2
        segments: list[NDArray[np.float64]] = []
        for block in blocks:
            count = block.shape[1] // step - 1
            segments.extend(
                block[:, index * step : index * step + self.segment_length]
                for index in range(count)
            )
        if not segments:
            raise ValueError(f"No complete segments at {location.name}")
        segment_array = np.stack(segments, axis=1)
        samples = sum(block.shape[1] for block in blocks)
        overall_rms = np.sqrt(
            sum(np.sum(block * block, axis=1) for block in blocks) / samples
        )
        segment_rms = np.std(segment_array, axis=2) / overall_rms[:, None]
        windowed = segment_array * np.hanning(self.segment_length)
        # The C++ implementation uses the positive-exponent DFT.
        spectra = np.conj(np.fft.rfft(windowed, axis=2)) / self.segment_length
        return ReceiverSpectra(
            location, segment_array, spectra, segment_rms
        ), np.concatenate(time_steps)

    def run(
        self,
        locations: tuple[SiteLocation, ...],
        smoothing: SmoothingConfig,
        robust_normalization: bool,
        selector: SegmentSelector | None = None,
        acceptance_range: float = 0.2,
        reporter: Callable[[str], None] | None = None,
    ) -> SpectralStatistics:
        loaded = [self._load_receiver(location) for location in locations]
        receivers = [item[0] for item in loaded]
        dt = float(np.median(np.concatenate([item[1] for item in loaded])))
        shape = receivers[0].spectra.shape
        if any(receiver.spectra.shape != shape for receiver in receivers):
            raise ValueError("Receivers do not have matching segments and components")
        valid_path = self.case_dir / "valid_segments.csv"
        if valid_path.exists():
            valid = read_valid_segments(valid_path)
        else:
            policy = selector or ModalRMSSelector(acceptance_range)
            valid = policy.select([receiver.normalized_rms for receiver in receivers])
            write_valid_segments(valid_path, valid)
        if valid.size == 0:
            raise ValueError("No valid segment was selected")
        if valid.min() < 0 or valid.max() >= shape[1]:
            raise IndexError("valid_segments.csv contains an out-of-range segment")
        if reporter is not None:
            reporter(f"{valid.size}/{shape[1]} segments selected")
        frequency = np.fft.rfftfreq(self.segment_length, dt)
        smoother = create_smoother(smoothing)
        stack = np.stack([receiver.spectra[:, valid, :] for receiver in receivers])
        scale = 8.0 * self.segment_length * dt
        power = smoother.apply(
            np.mean(np.abs(stack) ** 2, axis=2).transpose(1, 0, 2) * scale,
            frequency,
        )
        cross = np.empty(
            (shape[0], len(receivers), len(receivers), frequency.size),
            dtype=np.complex128,
        )
        for component in range(shape[0]):
            for first in range(len(receivers)):
                for second in range(len(receivers)):
                    values = (
                        np.mean(
                            np.conj(stack[first, component]) * stack[second, component],
                            axis=0,
                        )
                        * scale
                    )
                    cross[component, first, second] = smoother.apply(values, frequency)
        normalized_cross = normalize_cross_spectra(cross, power, robust_normalization)
        self._write_outputs(receivers, frequency, power, cross, normalized_cross)
        time_histories = np.stack(
            [receiver.segments[:, valid, :] for receiver in receivers]
        )
        return SpectralStatistics(
            frequency,
            locations,
            power,
            cross,
            valid,
            time_histories,
            normalized_cross,
        )

    def _write_outputs(
        self,
        receivers: Sequence[ReceiverSpectra],
        frequency: NDArray[np.float64],
        power: NDArray[np.float64],
        cross: NDArray[np.complex128],
        normalized_cross: NDArray[np.complex128],
    ) -> None:
        result_dir = self.case_dir / "results_neobd"
        statistics_dir = result_dir / "statistics"
        spectra_dir = result_dir / "spectra"
        for site_index, receiver in enumerate(receivers):
            for component in range(receiver.spectra.shape[0]):
                component_name = COMPONENTS[component]
                target = spectra_dir / f"{receiver.location.name}_{component_name}"
                target.mkdir(parents=True, exist_ok=True)
                for segment, spectrum in enumerate(receiver.spectra[component]):
                    amplitude = (
                        np.abs(spectrum)
                        * 4.0
                        * self.segment_length
                        * (frequency[1] - frequency[0]) ** -1
                        / self.segment_length
                    )
                    write_real(
                        target / f"{segment:06d}.csv",
                        frequency,
                        amplitude,
                        np.angle(spectrum),
                    )
                name = receiver.location.name
                write_real(
                    statistics_dir / f"{component_name}_{name}-{name}.csv",
                    frequency,
                    power[component, site_index],
                    np.zeros_like(frequency),
                )
        for component in range(cross.shape[0]):
            component_name = COMPONENTS[component]
            for first, first_receiver in enumerate(receivers):
                for second, second_receiver in enumerate(receivers):
                    if first == second:
                        continue
                    csd = cross[component, first, second]
                    write_complex(
                        statistics_dir
                        / f"{component_name}_{first_receiver.location.name}-{second_receiver.location.name}.csv",
                        frequency,
                        csd,
                    )
                    write_complex(
                        statistics_dir
                        / f"CCF_{component_name}_{first_receiver.location.name}-{second_receiver.location.name}.csv",
                        frequency,
                        normalized_cross[component, first, second],
                    )
        if cross.shape[0] == 3:
            for site_index, receiver in enumerate(receivers):
                vertical = np.sqrt(power[0, site_index])
                write_real(
                    statistics_dir / f"HVSR_{receiver.location.name}.csv",
                    frequency,
                    np.sqrt(power[1, site_index]) / vertical,
                    np.sqrt(power[2, site_index]) / vertical,
                )
