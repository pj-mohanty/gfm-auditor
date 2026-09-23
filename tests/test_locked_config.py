from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from agenticls_auditor.locked import (
    load_locked_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config"
    / "locked_primary_omnidna20m_k2.yaml"
)


def read_config():
    with CONFIG.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def write_config(
    path: Path,
    config: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )


def test_locked_config_is_valid():
    config = load_locked_config(CONFIG)

    assert config["frozen"] is True
    assert (
        config["calibration"]["delta"]
        == pytest.approx(
            0.0004979782302045738
        )
    )
    assert (
        config["locked_split"]["gene_count"]
        == 50
    )


def test_unfrozen_config_is_rejected(
    tmp_path,
):
    config = read_config()
    config["frozen"] = False

    path = tmp_path / "unfrozen.yaml"
    write_config(path, config)

    with pytest.raises(
        ValueError,
        match="not frozen",
    ):
        load_locked_config(path)


@pytest.mark.parametrize(
    "bad_delta",
    [
        None,
        "0.001",
        float("nan"),
        float("inf"),
        -0.1,
        True,
    ],
)
def test_invalid_delta_is_rejected(
    tmp_path,
    bad_delta,
):
    config = read_config()
    config["calibration"]["delta"] = (
        bad_delta
    )

    path = tmp_path / "bad_delta.yaml"
    write_config(path, config)

    with pytest.raises(
        ValueError,
        match="delta",
    ):
        load_locked_config(path)


def test_changed_seed_schedule_is_rejected(
    tmp_path,
):
    config = read_config()
    config["trajectory"]["seeds"] = [
        1,
        2,
        3,
    ]

    path = tmp_path / "bad_seeds.yaml"
    write_config(path, config)

    with pytest.raises(
        ValueError,
        match="seeds",
    ):
        load_locked_config(path)
