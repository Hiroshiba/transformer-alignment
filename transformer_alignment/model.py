from typing import List

import torch
import torch.functions as F
from torch import Chain, Variable

from transformer_alignment.config import ModelConfig
from transformer_alignment.network import Predictor


class Model(Chain):
    def __init__(self, model_config: ModelConfig, predictor: Predictor) -> None:
        super().__init__()
        self.model_config = model_config
        with self.init_scope():
            self.predictor = predictor

    def __call__(
            self,
            word_vector: List[Variable],  # List[shape: (word_length, ?)]
            input_char_label: List[Variable],  # List[shape: (char_length+1, )] or List[shape: (char_length, )]
            target_char_label: List[Variable],  # List[shape: (char_length+1, )] or List[shape: (char_length, )]
    ):
        output = self.predictor(
            word_vector=word_vector,
            char_label=input_char_label,
        )  # List[shape: (char_length+1, num_label)] or # List[shape: (char_length, num_label)]

        output = F.concat(output, axis=0)  # shape: (all_char_length, num_label)
        target = F.concat(target_char_label, axis=0)  # shape: (all_char_length, )

        if self.model_config.for_interpolate_task:
            target -= 1

        loss = F.softmax_cross_entropy(output, target, ignore_label=-1)

        losses = dict(loss=loss)
        if not torch.config.train:
            weight = (target.data != -1).sum()
            losses = {key: (l, weight) for key, l in losses.items()}  # add weight

        torch.report(losses, self)
        return loss
