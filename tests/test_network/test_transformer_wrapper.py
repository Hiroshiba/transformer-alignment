from dataclasses import dataclass
from itertools import combinations
from typing import List

import torch
import torch.functions as F
import torch.links as L
import numpy
import pytest

from transformer_alignment.network import Predictor
from transformer_alignment.network.transformer_wrapper import TransformerWrapper
from transformer_alignment.network.utility import ArrayLike


@dataclass
class TestParam:
    batch = 100
    min_word_length = 4
    max_word_length = 6
    min_char_length = 5
    max_char_length = 7
    embedding = 10


@pytest.fixture
def param():
    return TestParam()


@pytest.fixture
def transformer_wrapper(param: TestParam):
    return TransformerWrapper(
        d_model=param.embedding,
        nhead=param.embedding // 2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=9,
        pre_decoder=L.EmbedID(param.embedding, param.embedding),
    )


@pytest.fixture
def predictor(transformer_wrapper: TransformerWrapper):
    return Predictor(
        encode_word_network=None,
        main_network=transformer_wrapper,
    )


@pytest.fixture
def word_vector(param: TestParam):
    return [
        numpy.random.rand(
            numpy.random.randint(param.min_word_length, param.max_word_length + 1),
            param.embedding,
        ).astype(numpy.float32)
        for _ in range(param.batch)
    ]


@pytest.fixture
def char_label(param: TestParam):
    return [
        numpy.random.randint(
            param.embedding,
            size=numpy.random.randint(param.min_char_length, param.max_char_length + 1),
        ).astype(numpy.int32)
        for _ in range(param.batch)
    ]


@pytest.mark.parametrize("use_gpu", [False, True])
def test_predictor(
        predictor: Predictor,
        word_vector: List[ArrayLike],
        char_label: List[ArrayLike],
        use_gpu: bool,
):
    if use_gpu:
        import cupy
        word_vector = list(map(cupy.array, word_vector))
        char_label = list(map(cupy.array, char_label))
        predictor.to_gpu()

    output = predictor.forward(
        word_vector=word_vector,
        char_label=char_label,
    )

    for o, c in zip(output, char_label):
        assert len(o) == len(c)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_predictor_forward_one(
        predictor: Predictor,
        word_vector: List[ArrayLike],
        char_label: List[ArrayLike],
        use_gpu: bool,
        param: TestParam,
):
    if use_gpu:
        import cupy
        word_vector = list(map(cupy.array, word_vector))
        char_label = list(map(cupy.array, char_label))
        predictor.to_gpu()

    char_label = F.pad_sequence(char_label)

    output = predictor.forward_one(
        word_vector=word_vector,
        char_label=char_label[:, :1],
    )
    assert output.shape == (param.batch, param.embedding)

    output = predictor.forward_one(
        word_vector=word_vector,
        char_label=char_label[:, :2],
    )
    assert output.shape == (param.batch, param.embedding)


def test_predictor_same_forward(
        predictor: Predictor,
        word_vector: List[ArrayLike],
        char_label: List[ArrayLike],
        param: TestParam,
):
    outputs = []
    for num_forward in range(1, 1 + param.min_char_length):
        with torch.using_config('train', False):
            output = predictor.forward(
                word_vector=word_vector,
                char_label=[c[:num_forward] for c in char_label],
            )
            output = F.stack(output)
        outputs.append(output)

    for output_a, output_b in combinations(outputs, 2):
        min_length = min(output_a.shape[1], output_b.shape[1])
        numpy.testing.assert_allclose(output_a[:, :min_length].data, output_b[:, :min_length].data, rtol=0, atol=1e-6)


def test_predictor_same_forward_and_forward_one(
        predictor: Predictor,
        word_vector: List[ArrayLike],
        char_label: List[ArrayLike],
        param: TestParam,
):
    num_forward = param.min_char_length

    char_label = [c[:num_forward] for c in char_label]

    with torch.using_config('train', False):
        output_a = predictor.forward(
            word_vector=word_vector,
            char_label=char_label,
        )
    output_a = F.stack(output_a)

    char_label = F.stack(char_label)
    for i in range(num_forward):
        with torch.using_config('train', False):
            output_b = predictor.forward_one(
                word_vector=word_vector,
                char_label=char_label[:, :i + 1],
            )
        numpy.testing.assert_allclose(output_a[:, i].data, output_b.data, rtol=0, atol=1e-6)
