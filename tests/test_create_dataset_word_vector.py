from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pytest

from create_dataset_word_vector import create_dataset_word_vector
from tests.utility import get_data_directory


@pytest.fixture
def light_train_text_path():
    return get_data_directory() / 'light_train_text.jsonl'


def test_create_dataset_word_vector(light_train_text_path: Path):
    with TemporaryDirectory() as d1, TemporaryDirectory() as d2, NamedTemporaryFile() as f:
        pass

    create_dataset_word_vector(
        text_path=light_train_text_path,
        output_vector_dir=Path(d1),
        output_char_position_dir=Path(d2),
        output_text_path=Path(f.name),
        part_of_speech_tags=('NOUN', 'VERB', 'ADV'),
        overwrite=True,
    )

    print(f'output_vector_dir: {d1}')
    print(f'output_char_position_dir: {d2}')
    print(f'output_text_path: {f.name}')
