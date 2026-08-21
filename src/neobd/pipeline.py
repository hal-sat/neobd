"""High-level analysis orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import os
from pathlib import Path
import shutil

from .config import AnalysisConfig
from .dspac import run_dspac
from .fk_runner import run_fk
from .gspac import run_gspac
from .io import read_coordinates
from .preprocess import Preprocessor, SpectralStatistics
from .selection import SegmentSelector
from .spac import run_spac


RESULT_SUBDIRECTORIES = (
    "reports",
    "spectra",
    "statistics",
    "circular_statistics",
    "spac",
    "gspac",
    "fk",
    "dspac",
)


class AnalysisPipeline:
    """Coordinate independent analysis stages and replaceable policies."""

    def __init__(
        self,
        config: AnalysisConfig,
        selector: SegmentSelector | None = None,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.selector = selector
        self.reporter = reporter or (lambda message: None)

    def prepare_output(self, replace: bool = True) -> None:
        result_dir = self.config.case_dir / "results_neobd"
        if replace and result_dir.exists():
            shutil.rmtree(result_dir)
        for name in RESULT_SUBDIRECTORIES:
            (result_dir / name).mkdir(parents=True, exist_ok=True)

    def run(self, replace_results: bool = True) -> SpectralStatistics:
        self.prepare_output(replace_results)
        locations = read_coordinates(self.config.case_dir)
        statistics = Preprocessor(self.config.case_dir, self.config.segment_length).run(
            locations,
            self.config.smoothing,
            self.config.robust_normalization,
            self.selector,
            self.config.acceptance_range,
            self.reporter,
        )
        self.reporter("Preprocess completed")
        if self.config.spac_arrays:
            run_spac(statistics, self.config.case_dir, self.config.spac_arrays)
            self.reporter("SPAC completed")
        if self.config.gspac_arrays:
            run_gspac(
                statistics,
                self.config.case_dir,
                self.config.gspac_arrays,
                self.config.smoothing,
            )
            self.reporter("GSPAC completed")
        if self.config.fk is not None:
            run_fk(statistics, self.config.case_dir, self.config.fk, self.config.n_para)
            self.reporter("FK completed")
        if self.config.dspac is not None:
            run_dspac(
                statistics, self.config.case_dir, self.config.dspac, self.config.n_para
            )
            self.reporter("DSPAC completed")
        return statistics


def run_analysis(
    path: str | Path,
    replace_results: bool = True,
    reporter: Callable[[str], None] | None = None,
    n_para: int | None = None,
) -> SpectralStatistics:
    """Load parameters and execute analyses with an optional process override."""
    config = AnalysisConfig.load(path)
    if n_para is not None:
        if n_para < 0:
            raise ValueError("n_para must be non-negative")
        effective_n_para = (os.cpu_count() or 1) if n_para == 0 else n_para
        config = replace(config, n_para=effective_n_para)
    return AnalysisPipeline(config, reporter=reporter).run(replace_results)
