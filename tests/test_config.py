import json

from neobd.config import AnalysisConfig


def test_loads_fk_bounds(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "seg_len": 2048,
                "FK": {"density": ["200", "36"], "bounds": ["100", "1500"]},
            }
        )
    )
    config = AnalysisConfig.load(path)
    assert config.fk is not None
    assert config.fk.radial_density == 200
    assert config.fk.max_velocity == 1500


def test_loads_parzen_smoothing(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"smoothing": {"type": "Parzen", "params": [0.3]}}))
    config = AnalysisConfig.load(path)
    assert config.smoothing.type == "Parzen"
    assert config.smoothing.params == (0.3,)


def test_legacy_smoothing_uses_hann_3point(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"n_smoothing": 7}))
    config = AnalysisConfig.load(path)
    assert config.smoothing.type == "Hann_3point"
    assert config.smoothing.params == (7.0,)


def test_rejects_ambiguous_smoothing_configuration(tmp_path) -> None:
    import pytest

    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "n_smoothing": 7,
                "smoothing": {"type": "Parzen", "params": [0.3]},
            }
        )
    )
    with pytest.raises(ValueError, match="either smoothing or n_smoothing"):
        AnalysisConfig.load(path)
