import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import yaml

from create_dataset_word_vector import create_word_vector
from utility.save_arguments import save_arguments
from transformer_alignment.config import Config
from transformer_alignment.dataset import _load_char, create_and_split_dataset
from transformer_alignment.generator import Generator
from transformer_alignment.tokenizer import LabelTokenizer, SpacyTokenizer


def _extract_number(f):
    s = re.findall(r'\d+', str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'main_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
    return model_path


def generate(
        model_dir: Path,
        model_iteration: Optional[int],
        model_config: Path,
        max_length: int,
        num_test: Optional[int],
        part_of_speech_tags: Sequence[str],
        output_dir: Path,
        use_gpu: bool,
        additional_test_texts: Sequence[str],
):
    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / 'arguments.yaml', target_function=generate, arguments=locals())

    with model_config.open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    eval_datas = create_and_split_dataset(config.dataset)['eval']

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    # test
    eval_datas = eval_datas[:num_test]

    char_tokenizer = LabelTokenizer(_load_char(Path(config.dataset.char_path)))
    word_tokenizer = SpacyTokenizer(filter_part_of_speech_tags=part_of_speech_tags)

    correct_texts = [char_tokenizer.decode(test_data.true_char_label) for test_data in eval_datas]
    word_vectors = [test_data.word_vector for test_data in eval_datas]
    input_char_labels = [test_data.input_char_label for test_data in eval_datas]

    correct_texts += additional_test_texts
    word_vectors += [create_word_vector(text=text, word_tokenizer=word_tokenizer) for text in additional_test_texts]

    for correct_text, word_vector, input_char_label in zip(correct_texts, word_vectors, input_char_labels):
        words = [token.text for token, _ in word_tokenizer.encode_and_filter(correct_text)]

        random_text = generator.generate(
            word_vector=word_vector,
            input_char_label=input_char_label,
            max_length=max_length,
            sampling_maximum=False,
        )

        maximum_text = generator.generate(
            word_vector=word_vector,
            input_char_label=input_char_label,
            max_length=max_length,
            sampling_maximum=True,
        )

        print('correct text:', correct_text)
        print('words:', words)
        print('predict text (random sampling):', random_text)
        print('predict text (maximum sampling):', maximum_text)
        print('------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path)
    parser.add_argument('--model_iteration', type=int)
    parser.add_argument('--model_config', type=Path)
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--num_test', type=int)
    parser.add_argument('--part_of_speech_tags', nargs='+', default=['NOUN', 'VERB', 'ADV'])
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--additional_test_texts', nargs='+', default=[])
    generate(**vars(parser.parse_args()))
