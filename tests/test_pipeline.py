import json
from pathlib import Path
from unittest.mock import patch

from neobd.pipeline import run_analysis


def _write_params(path: Path, n_para: int | None = None) -> None:
    values = {} if n_para is None else {"n_para": n_para}
    path.write_text(json.dumps(values))


def test_run_analysis_uses_json_parallelism_without_override(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    _write_params(path, 3)
    captured = []

    def capture(self, replace_results=True):
        captured.append(self.config.n_para)

    with patch("neobd.pipeline.AnalysisPipeline.run", capture):
        run_analysis(path)
    assert captured == [3]


def test_run_analysis_cli_parallelism_overrides_json(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    _write_params(path, 3)
    captured = []

    def capture(self, replace_results=True):
        captured.append(self.config.n_para)

    with patch("neobd.pipeline.AnalysisPipeline.run", capture):
        run_analysis(path, n_para=4)
    assert captured == [4]


def test_run_analysis_zero_uses_all_available_cpus(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    _write_params(path, 3)
    captured = []

    def capture(self, replace_results=True):
        captured.append(self.config.n_para)

    with (
        patch("neobd.pipeline.os.cpu_count", return_value=12),
        patch("neobd.pipeline.AnalysisPipeline.run", capture),
    ):
        run_analysis(path, n_para=0)
    assert captured == [12]
