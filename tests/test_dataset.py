from copy import deepcopy
from typing import Dict, Any

import numpy
import pytest
import yaml

from tests.utility import get_data_directory
from transformer_alignment.config import Config
from transformer_alignment.dataset import create_and_split_dataset


@pytest.fixture()
def config_dict():
    light_train_config_path = get_data_directory() / 'transformer' / 'light_train_config.yaml'
    with light_train_config_path.open() as f:
        config_dict = yaml.safe_load(f)
    return config_dict


@pytest.fixture()
def config_dict_for_interpolate_task():
    light_train_config_path = get_data_directory() / 'transformer_interpolate' / 'light_train_config.yaml'
    with light_train_config_path.open() as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def test_create_dataset(config_dict: Dict[str, Any]):
    config_dataset = Config.from_dict(config_dict).dataset
    train_dataset = create_and_split_dataset(config_dataset)['train']

    for data in train_dataset:
        assert len(data.input_char_label) == len(data.target_char_label)
        assert numpy.all(data.input_char_label[1:] == data.target_char_label[:-1])


def test_create_dataset_with_text_length(config_dict: Dict[str, Any]):
    target_config_dict = deepcopy(config_dict)
    target_config_dict['dataset']['with_text_length'] = True

    config_dataset = Config.from_dict(config_dict).dataset
    base_train_dataset = create_and_split_dataset(config_dataset)['train']

    config_dataset = Config.from_dict(target_config_dict).dataset
    target_train_dataset = create_and_split_dataset(config_dataset)['train']

    for base, target in zip(base_train_dataset, target_train_dataset):
        assert numpy.all(base.word_vector == target.word_vector[1:])
        assert numpy.all(base.input_char_label == target.input_char_label)
        assert numpy.all(base.target_char_label == target.target_char_label)


def test_create_dataset_for_interpolate_task(config_dict_for_interpolate_task: Dict[str, Any]):
    config_dataset = Config.from_dict(config_dict_for_interpolate_task).dataset
    datasets = create_and_split_dataset(config_dataset)

    train_dataset = datasets['train']
    test_dataset = datasets['test']
    eval_dataset = datasets['eval']

    char_empty_label = 0
    for _ in range(10):
        for data in train_dataset:
            assert len(data.input_char_label) == len(data.target_char_label)

            empty = data.input_char_label == char_empty_label
            ignore = data.target_char_label == char_empty_label
            assert 0 < numpy.sum(empty) <= len(data.input_char_label)
            assert 0 <= numpy.sum(ignore) < len(data.input_char_label)
            assert numpy.all(numpy.logical_xor(empty, ignore))

            assert numpy.all(data.input_char_label[~empty] == data.true_char_label[~empty])
            assert numpy.all(data.target_char_label[~ignore] == data.true_char_label[~ignore])

        for test_data, eval_data in zip(test_dataset, eval_dataset):
            test_empty = test_data.input_char_label == char_empty_label
            eval_empty = eval_data.input_char_label == char_empty_label
            assert numpy.sum(test_empty) <= numpy.sum(eval_empty)
