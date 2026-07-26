from pathlib import Path

import yaml


def test_full_catalogue_memorisation_defaults_are_clean_and_unregularized():
    config_path = Path(__file__).parents[1] / "configs" / "fma_large.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["data"]["clean_probability"] == 1.0
    assert all(
        not section["enabled"] and section["probability"] == 0.0
        for section in cfg["data"]["augmentation"].values()
    )
    assert cfg["model"]["decoder"]["dropout"] == 0.0
    assert cfg["train"]["weight_decay"] == 0.0
    assert cfg["train"]["songs_per_batch"] == 8
    assert cfg["train"]["views_per_song"] == 8
    assert cfg["train"]["phase1_exposures"] == 2
