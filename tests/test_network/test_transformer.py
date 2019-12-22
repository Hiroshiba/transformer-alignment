from dataclasses import dataclass

import torch
import numpy
import pytest
from torch import Variable

from transformer_alignment.network.transformer import Transformer


@dataclass
class TestParam:
    batch = 3
    source_length = 4
    target_length = 5
    embedding = 10


@pytest.fixture
def param():
    return TestParam()


@pytest.fixture
def transformer(param: TestParam):
    return Transformer(
        d_model=param.embedding,
        nhead=param.embedding // 2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=9,
    )


@pytest.fixture
def src(param: TestParam):
    return Variable(numpy.random.randn(param.source_length, param.batch, param.embedding).astype(numpy.float32))


@pytest.fixture
def tgt(param: TestParam):
    return Variable(numpy.random.randn(param.target_length, param.batch, param.embedding).astype(numpy.float32))


@pytest.fixture
def tgt_mask(param: TestParam, transformer: Transformer):
    return transformer.generate_square_subsequent_mask(param.target_length)


@pytest.fixture
def src_key_padding_mask(param: TestParam):
    return (numpy.triu(numpy.ones((param.source_length, param.source_length)) == 1))[-param.batch:]


@pytest.fixture
def tgt_key_padding_mask(param: TestParam):
    return (numpy.triu(numpy.ones((param.target_length, param.target_length)) == 1))[-param.batch:]


def test_transformer_forward(
        transformer: Transformer,
        src: Variable,
        tgt: Variable,
):
    output_a = transformer(
        src=src,
        tgt=tgt,
    )
    assert not numpy.isnan(output_a.data).any()


def test_transformer_forward_with_mask(
        transformer: Transformer,
        src: Variable,
        tgt: Variable,
        tgt_mask: Variable,
        src_key_padding_mask: Variable,
        tgt_key_padding_mask: Variable,
):
    output = transformer(
        src=src,
        tgt=tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
    )
    assert not numpy.isnan(output.data).any()
