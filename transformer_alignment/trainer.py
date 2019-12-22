import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict

import yaml
from tensorboardX import SummaryWriter
from torch import cuda, optim, training
from torch.iterators import MultiprocessIterator
from torch.training import extensions
from torch.training.updaters import StandardUpdater

from transformer_alignment.config import Config
from transformer_alignment.dataset import create_and_split_dataset, ModelInputData
from transformer_alignment.evaluator import Evaluator
from transformer_alignment.generator import Generator
from transformer_alignment.model import Model
from transformer_alignment.network import create_predictor
from utility.generate_evaluator_extension import GenerateEvaluator
from utility.tensor_board_report_extension import TensorBoardReport


class NoamLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_step, scale=1.0):
        super().__init__(optimizer)
        self.d_model = d_model
        self.warmup_step = warmup_step
        self.scale = scale

    def get_lr(self):
        step = self.last_epoch
        return self.scale * self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_step ** -1.5)


def create_trainer(
        config_dict: Dict[str, Any],
        output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / 'config.yaml').open(mode='w') as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    predictor = create_predictor(config.network)
    model = Model(model_config=config.model, predictor=predictor)

    model.to_gpu(0)
    cuda.get_device_from_id(0).use()

    # dataset
    def _create_iterator(dataset, for_train: bool):
        return MultiprocessIterator(
            dataset,
            config.train.batchsize,
            repeat=for_train,
            shuffle=for_train,
            n_processes=config.train.num_processes,
            dataset_timeout=60 * 5,
        )

    datasets = create_and_split_dataset(config.dataset)
    train_iter = _create_iterator(datasets['train'], for_train=True)
    test_iter = _create_iterator(datasets['test'], for_train=False)
    train_test_iter = _create_iterator(datasets['train_test'], for_train=False)
    eval_iter = _create_iterator(datasets['eval'], for_train=False)

    warnings.simplefilter('error', MultiprocessIterator.TimeoutWarning)

    # optimizer
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop('name').lower()

    if n == 'adam':
        optimizer = optim.Adam(**cp)
    elif n == 'sgd':
        optimizer = optim.SGD(**cp)
    else:
        raise ValueError(n)

    optimizer.setup(model)

    # scheduler
    config.train.scheduler

    # updater
    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        converter=ModelInputData.concat,
        device=0,
    )

    # trainer
    trigger_log = (config.train.log_iteration, 'iteration')
    trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
    trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

    trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=output)

    if config.train.linear_shift is not None:
        trainer.extend(extensions.LinearShift(**config.train.linear_shift))

    ext = extensions.Evaluator(test_iter, model, ModelInputData.concat, device=0)
    trainer.extend(ext, name='test', trigger=trigger_log)
    ext = extensions.Evaluator(train_test_iter, model, ModelInputData.concat, device=0)
    trainer.extend(ext, name='train', trigger=trigger_log)

    if config.dataset.num_evaluate > 0:
        generator = Generator(config=config, predictor=predictor, use_gpu=True)
        evaluator = Evaluator(generator, config.train.evaluate_max_length)
        ext = GenerateEvaluator(eval_iter, evaluator, ModelInputData.concat, device=0)
        trainer.extend(ext, name='eval', trigger=trigger_snapshot)

    ext = extensions.snapshot_object(predictor, filename='main_{.updater.iteration}.npz')
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'test/main/loss']), trigger=trigger_log)

    ext = TensorBoardReport(
        writer=SummaryWriter(Path(output)),
        generated_writer=SummaryWriter(Path(output), filename_suffix='_text'),
    )
    trainer.extend(ext, trigger=trigger_log)

    trainer.extend(extensions.dump_graph(root_name='main/loss'))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
