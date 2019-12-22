import copy
import warnings
from collections import defaultdict

from torch import function
from torch import reporter as reporter_module
from torch.dataset import convert
from torch.training.extensions import Evaluator


class GenerateEvaluator(Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()
        generates = defaultdict(list)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = convert._call_converter(
                    self.converter, batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary_observation = {}
            for k, v in observation.items():
                if isinstance(v, list):
                    generates[k] += v
                else:
                    summary_observation[k] = v

            summary.add(summary_observation)

        result = summary.compute_mean()
        result.update(generates)
        return result
