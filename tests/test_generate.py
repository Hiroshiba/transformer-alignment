from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from generate import generate
from tests.utility import get_data_directory


@pytest.fixture(params=[
    'transformer',
    'transformer_with_text_length',
    'transformer_interpolate',
])
def light_train_model_dir(request):
    return get_data_directory() / request.param


@pytest.fixture()
def light_train_config_path(light_train_model_dir: Path):
    return light_train_model_dir / 'light_train_config.yaml'


@pytest.mark.parametrize("sampling_maximum", [False, True])
def test_generate(light_train_model_dir: Path, light_train_config_path: Path, sampling_maximum: bool):
    with TemporaryDirectory() as d:  # create temp directory and delete
        pass

    generate(
        model_dir=light_train_model_dir,
        model_iteration=None,
        model_config=light_train_config_path,
        max_length=30,
        num_test=5,
        part_of_speech_tags=['NOUN', 'VERB', 'ADV'],
        output_dir=Path(d),
        use_gpu=True,
        additional_test_texts=[],
    )
