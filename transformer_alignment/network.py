from typing import List, Optional

import torch
import torch.links as L
from torch import Chain, Variable

from transformer_alignment.config import NetworkConfig
from transformer_alignment.network.transformer import AutoReshapeLinear
from transformer_alignment.network.transformer_wrapper import TransformerWrapper
from transformer_alignment.network.utility import MeanNetwork


class Predictor(Chain):
    def __init__(
            self,
            encode_word_network: Optional[MeanNetwork],
            main_network: TransformerWrapper,
    ):
        super().__init__()
        with self.init_scope():
            self.encode_word_network = encode_word_network
            self.main_network = main_network

    def forward(
            self,
            word_vector: List[Variable],
            char_label: List[Variable],
    ):
        """
        word_vector: List[shape: (word_length, ?)]
        char_label: List[shape: (char_length, )]
        return: List[shape: (char_length, ?)]
        """
        if self.encode_word_network is not None:
            word_vector = self.encode_word_network.forward(word_vector, output_length=[len(c) for c in char_label])

        xs = self.main_network.forward(word_vector, char_label)
        return xs

    def forward_one(
            self,
            word_vector: List[Variable],
            char_label: List[Variable],
    ):
        """
        word_vector: List[shape: (word_length, ?)]
        char_label: List[shape: (char_length, ?)]
        return:
            x: shape: (batchsize, num_id)
        """
        batchsize = len(word_vector)

        if self.encode_word_network is not None:
            xs = self.encode_word_network.forward(word_vector, output_length=[1] * batchsize)
        else:
            xs = word_vector

        xs = self.main_network.forward_one(xs, char_label)
        return xs


def create_predictor(config: NetworkConfig):
    predictor = Predictor(
        encode_word_network=None,
        main_network=TransformerWrapper(
            d_model=config.transformer_network.d_model,
            nhead=config.transformer_network.nhead,
            num_encoder_layers=config.transformer_network.num_encoder_layers,
            num_decoder_layers=config.transformer_network.num_decoder_layers,
            dim_feedforward=config.transformer_network.dim_feedforward,
            dropout=config.transformer_network.dropout,
            activation=config.transformer_network.activation,
            pre_encoder=AutoReshapeLinear(config.in_size, config.transformer_network.d_model),
            pre_decoder=L.EmbedID(config.in_char_size, config.transformer_network.d_model),
            post_decoder=AutoReshapeLinear(config.transformer_network.d_model, config.out_size),
        ),
    )

    return predictor
