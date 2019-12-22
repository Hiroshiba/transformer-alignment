from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.utility import get_data_directory
from train import train


@pytest.fixture(params=[
    'transformer/light_train_config.yaml',
    'transformer_with_text_length/light_train_config.yaml',
    'transformer_interpolate/light_train_config.yaml',
])
def light_train_config_path(request):
    return get_data_directory() / request.param


def test_train(light_train_config_path: Path):
    with TemporaryDirectory() as d:  # create temp directory and delete
        pass

    train(
        config_yaml_path=light_train_config_path,
        output=Path(d),
    )

    print(f'train to {d}')
