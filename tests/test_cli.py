from unittest.mock import patch

import pytest

from neobd.cli import main


def test_cli_passes_parallel_override() -> None:
    with patch("neobd.cli.run_analysis") as run_analysis:
        assert main(["params.json", "--npara=4"]) == 0
    assert run_analysis.call_args.kwargs["n_para"] == 4


def test_cli_preserves_json_parallelism_when_option_is_omitted() -> None:
    with patch("neobd.cli.run_analysis") as run_analysis:
        assert main(["params.json"]) == 0
    assert run_analysis.call_args.kwargs["n_para"] is None


def test_cli_rejects_negative_parallelism() -> None:
    with patch("neobd.cli.run_analysis"):
        with pytest.raises(SystemExit):
            main(["params.json", "--npara=-1"])
