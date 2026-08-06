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
