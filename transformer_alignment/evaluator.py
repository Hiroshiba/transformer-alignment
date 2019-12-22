from typing import List

import torch
from torch import Chain, Variable

from transformer_alignment.generator import Generator


class Evaluator(Chain):
    def __init__(
            self,
            generator: Generator,
            max_length: int,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.max_length = max_length

    def __call__(
            self,
            word_vector: List[Variable],  # List[shape: (word_length, ?)]
            input_char_label: List[Variable],  # List[shape: (char_length+1, )] or List[shape: (char_length, )]
            target_char_label: List[Variable],  # List[shape: (char_length+1, )] or List[shape: (char_length, )]
    ):
        texts_random = []
        texts_maximum = []
        for i, (wv, icl) in enumerate(zip(word_vector, input_char_label)):
            text = self.generator.generate(
                word_vector=wv,
                input_char_label=icl,
                max_length=self.max_length,
                sampling_maximum=False,
            )
            texts_random.append(text)

            text = self.generator.generate(
                word_vector=wv,
                input_char_label=icl,
                max_length=self.max_length,
                sampling_maximum=True,
            )
            texts_maximum.append(text)

        torch.report({'text_random': texts_random}, self)
        torch.report({'text_maximum': texts_maximum}, self)
