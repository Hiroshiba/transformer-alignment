from pathlib import Path

import torch
from tensorboardX import SummaryWriter


class TensorBoardReport(torch.training.Extension):
    def __init__(self, writer: SummaryWriter = None, generated_writer: SummaryWriter = None):
        self.writer = writer
        self.generated_writer = generated_writer

    def __call__(self, trainer: torch.training.Trainer):
        if self.writer is None:
            self.writer = SummaryWriter(Path(trainer.out))

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            if isinstance(v, torch.Variable):
                v.to_cpu()
                self.writer.add_scalar(n, v.data, n_iter)

            if self.generated_writer is not None:
                if isinstance(v, list) and isinstance(v[0], str):
                    for i, text in enumerate(v):
                        self.generated_writer.add_text(f'{n}/{i}', text, n_iter)

    def finalize(self):
        super().finalize()
        self.writer.flush()
        if self.generated_writer is not None:
            self.generated_writer.flush()
