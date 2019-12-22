from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, Union

import jsonlines
from tqdm import tqdm


@dataclass
class TextObject:
    label: str
    text: str

    @classmethod
    def read(cls, path: Union[Path, PathLike]):
        with jsonlines.open(path) as f:
            return [
                cls(**obj)
                for obj in tqdm(f.iter(type=dict), desc=f'TextObject.read(path={path})')
            ]

    @classmethod
    def write(cls, text_objects: Iterable['TextObject'], path: Path):
        with jsonlines.open(path, mode='w') as f:
            for to in text_objects:
                f.write(dict(label=to.label, text=to.text))
