import json
import re
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import List, NamedTuple, Sequence, Optional

import torch
import numpy
from temp_cache import TempCache
from torch.utils.data.dataset import Dataset

from transformer_alignment.config import DatasetConfig
from transformer_alignment.data import TextObject
from transformer_alignment.tokenizer import LabelTokenizer


class DatasetInputData(NamedTuple):
    text: str
    word_vector_path: Path


class ModelInputData(NamedTuple):
    word_vector: numpy.ndarray  # shape: (word_length, ?)
    input_char_label: numpy.ndarray  # shape: (char_length+1, ) or (char_length, )
    target_char_label: numpy.ndarray  # shape: (char_length+1, ) or (char_length, )
    true_char_label: numpy.ndarray  # shape: (char_length, )

    @staticmethod
    def concat(batch: Sequence['ModelInputData'], device=None):
        from torch.dataset import to_device
        # true_char_label is not using in training
        return dict(
            word_vector=[to_device(device, d.word_vector) for d in batch],
            input_char_label=[to_device(device, d.input_char_label) for d in batch],
            target_char_label=[to_device(device, d.target_char_label) for d in batch],
        )


def _load_char(p: PathLike) -> List[str]:
    with open(p) as f:
        return json.load(f)


def _get_numeric_from_path(p: Path, _re=re.compile(r'\d+')):
    return tuple(map(int, _re.findall(str(p))))


def _get_text_length_token(text: str, token_size: int):
    text_length = numpy.log10(len(text)) / numpy.log10(100)  # normalized
    token = numpy.ones((1, token_size), dtype=numpy.float32) * text_length
    return token


class InputTargetDataset(Dataset):
    def __init__(
            self,
            input_glob
        target_glob
    ):
        self.text_objects = text_objects
        self.char_tokenizer = char_tokenizer
        self.word_vector_path = word_vector_path
        self.word_char_position_path = word_char_position_path
        self.word_vector_size = word_vector_size
        self.with_text_length = with_text_length
        self.for_interpolate_task = for_interpolate_task
        self.for_evaluate = for_evaluate

    def __len__(self):
        return len(self.text_objects)

    def get_example(self, i):
        text_object = self.text_objects[i]
        text = text_object.text
        word_vector_path = self.word_vector_path / f'{text_object.label}.npy'

        word_vector = numpy.load(str(TempCache(word_vector_path)), mmap_mode='r')
        if word_vector.size == 0:
            word_vector = numpy.zeros((0, self.word_vector_size), dtype=numpy.float32)

        if self.with_text_length:
            token = _get_text_length_token(text=text, token_size=self.word_vector_size)
            word_vector = numpy.concatenate((token, word_vector), axis=0)

        char_label = self.char_tokenizer.encode(text)

        if not self.for_interpolate_task:
            char_separator_label = self.char_tokenizer.get_empty_label()
            input_char_label = [char_separator_label] + char_label
            target_char_label = char_label + [char_separator_label]

        else:
            # text:
            #   abcdefg.
            # words:
            #   bc efg
            # random picked input:
            #   ---d----
            # then, target will:
            #   a------.
            char_empty_label = self.char_tokenizer.get_empty_label()

            word_char_position_path = self.word_char_position_path / f'{text_object.label}.npy'
            word_char_position = numpy.load(str(TempCache(word_char_position_path)), mmap_mode='r')

            interpolate_flags = numpy.ones(len(char_label), dtype=bool)
            for pre, post in word_char_position:
                interpolate_flags[pre:post] = False

            indexes = numpy.argwhere(interpolate_flags).ravel()
            indexes = numpy.random.permutation(indexes)

            if not self.for_evaluate:
                num_empty = numpy.random.randint(len(indexes)) + 1
            else:
                num_empty = len(indexes)

            input_char_label = deepcopy(char_label)
            for i in indexes[:num_empty]:
                input_char_label[i] = char_empty_label

            target_char_label = deepcopy(char_label)
            for i in numpy.r_[indexes[num_empty:], numpy.argwhere(~interpolate_flags).ravel()]:
                target_char_label[i] = char_empty_label

        return ModelInputData(
            word_vector=word_vector,
            input_char_label=numpy.array(input_char_label, dtype=numpy.int32),
            target_char_label=numpy.array(target_char_label, dtype=numpy.int32),
            true_char_label=numpy.array(char_label, dtype=numpy.int32),
        )


def create_and_split_dataset(config: DatasetConfig):
    text_objects = TextObject.read(TempCache(config.text_path))

    char_tokenizer = LabelTokenizer(_load_char(TempCache(config.char_path)))

    def dataset_wrapper(datas, for_evaluate: bool = False):
        return Dataset(
            text_objects=datas,
            char_tokenizer=char_tokenizer,
            word_vector_path=config.word_vector_path,
            word_char_position_path=config.word_char_position_path,
            word_vector_size=config.word_vector_size,
            with_text_length=config.with_text_length,
            for_interpolate_task=config.for_interpolate_task,
            for_evaluate=for_evaluate,
        )

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(text_objects)

    tests, trains = text_objects[:config.num_test], text_objects[config.num_test:]
    train_tests = trains[:config.num_test]
    evals = tests[:config.num_evaluate]

    return {
        'train': dataset_wrapper(trains),
        'test': dataset_wrapper(tests),
        'train_test': dataset_wrapper(train_tests),
        'eval': dataset_wrapper(evals, for_evaluate=True),
    }
