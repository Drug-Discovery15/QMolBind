import numpy as np

import qumolbind
from qumolbind.utils.config import load_config
from qumolbind.utils.logging import RunLogger
from qumolbind.utils.seeding import derive_seed, seed_everything


def test_import() -> None:
    assert qumolbind.__version__


def test_seeding_reproducible() -> None:
    a = seed_everything(3).random(4)
    b = seed_everything(3).random(4)
    assert np.allclose(a, b)
    assert derive_seed(1, 2) == derive_seed(1, 2) != derive_seed(1, 3)


def test_config_composes() -> None:
    cfg = load_config(["seed=5", "experiment=smoke"])
    assert cfg.seed == 5 and cfg.env.state_dim == 256 and cfg.allow_hardware is False


def test_logger_csv(tmp_path) -> None:
    lg = RunLogger(tmp_path, tensorboard=False)
    lg.log(0, reward=1.0)
    lg.log(1, reward=2.0, extra=3)
    assert "extra" in (tmp_path / "metrics.csv").read_text().splitlines()[0]
