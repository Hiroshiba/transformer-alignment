from pathlib import Path
from typing import Union, Optional

import torch
import torch.functions as F
import numpy
from torch import cuda
from temp_cache import TempCache

from transformer_alignment.config import Config
from transformer_alignment.dataset import _load_char
from transformer_alignment.network import create_predictor, Predictor
from transformer_alignment.network.utility import ArrayLike
from transformer_alignment.tokenizer import LabelTokenizer


class Generator(object):
    def __init__(
            self,
            config: Config,
            predictor: Union[Path, Predictor],
            use_gpu: bool,
    ) -> None:
        if isinstance(predictor, Path):
            predictor_path = predictor
            predictor = create_predictor(config.network)
            torch.serializers.load_npz(str(predictor_path), predictor)

        self.config = config
        self.predictor = predictor
        self.use_gpu = use_gpu

        self.char_tokenizer = LabelTokenizer(_load_char(TempCache(config.dataset.char_path)))

        if self.use_gpu:
            predictor.to_gpu(0)
            cuda.get_device_from_id(0).use()

    @property
    def xp(self):
        return self.predictor.xp

    @property
    def for_interpolate_task(self):
        return self.config.model.for_interpolate_task

    def generate(
            self,
            word_vector: ArrayLike,
            input_char_label: Optional[ArrayLike],
            max_length: int,
            sampling_maximum: bool,
            show_progress_text: bool = False,
    ):
        word_vector = self.xp.asarray(word_vector)  # List[shape: (word_length, ?)]

        empty_label = self.char_tokenizer.get_empty_label()
        if not self.for_interpolate_task:
            char_label = self.xp.array([empty_label], dtype=numpy.int32)
        else:
            char_label = self.xp.array(input_char_label)
            assert (char_label != empty_label).sum() > 0

        for i in range(max_length):
            with torch.using_config('train', False), torch.using_config('enable_backprop', False):
                x = self.predictor.forward(
                    word_vector=[word_vector],
                    char_label=[char_label],
                )
                x = F.stack(x)

            if not self.for_interpolate_task:
                char_id = int(self.sampling(x[:, -1], maximum=sampling_maximum))
                if char_id == empty_label:
                    break

                char_label = self.xp.concatenate([
                    char_label, self.xp.array([char_id], dtype=numpy.int32),
                ])

            else:
                indexes = self.xp.where(char_label == empty_label)[0]

                # search an index have max probability
                i_char = int(F.argmax(F.max(x[:, indexes], axis=2), axis=1).data)
                i_char = indexes[i_char]

                # sampling
                char_label[i_char] = int(self.sampling(x[:, i_char], maximum=sampling_maximum)) + 1

                if self.xp.all(char_label != empty_label):
                    break

            if show_progress_text:
                for c in char_label:
                    if c != empty_label:
                        t = self.char_tokenizer.decode_one(int(c))
                    else:
                        t = '○'
                    print(t, end='')
                print('')

        if not self.for_interpolate_task:
            char_label = char_label[1:]

        text = self.char_tokenizer.decode(map(int, char_label))
        return text

    def sampling(self, dist: torch.Variable, maximum: bool):
        """
        :param dist: shape: (batch, num_id)
        :return: shape: (batch, )
        """

        if maximum:
            sampled = self.xp.argmax(dist.data, axis=1)
        else:
            prob_list = F.softmax(dist, axis=1)
            sampled = self.xp.concatenate([
                self.xp.random.choice(self.config.network.out_size, size=1, p=prob)
                for prob in prob_list.data
            ])
        return sampled
