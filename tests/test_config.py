from pathlib import Path

import pytest
import yaml
from yaml import SafeLoader

from tests.utility import get_data_directory
from transformer_alignment.config import Config


@pytest.fixture(params=[
    'transformer',
])
def light_train_model_dir(request):
    return get_data_directory() / request.param


@pytest.fixture()
def light_train_config_path(light_train_model_dir: Path):
    return light_train_model_dir / 'light_train_config.yaml'


def test_from_dict(light_train_config_path: Path):
    with light_train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d)


def test_to_dict(light_train_config_path: Path):
    with light_train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d).to_dict()


def test_equal_base_config_and_reconstructed(light_train_config_path: Path):
    with light_train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    base = Config.from_dict(d)
    base_re = Config.from_dict(base.to_dict())
    assert base == base_re
