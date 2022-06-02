import logging
from collections import defaultdict
from typing import Dict, Any

import numpy as np

logger = logging.getLogger('logger')


class Metric:
    name: str
    train: bool
    plottable: bool = True
    running_metric = None
    main_metric_name = None
    preds = list()
    ground_truth = list()

    def __init__(self, name, train=False):
        self.train = train
        self.name = name

        self.running_metric = defaultdict(list)

    def __repr__(self):
        metrics = self.get_value()
        text = [f'{key}: {val:.2f}' for key, val in metrics.items()]
        return f'{self.name}: ' + ','.join(text)

    def compute_metric(self, outputs, labels) -> Dict[str, Any]:
        raise NotImplemented

    def accumulate_on_batch(self, outputs=None, labels=None):
        current_metrics = self.compute_metric(outputs, labels)
        for key, value in current_metrics.items():
            self.running_metric[key].append(value)

    def get_value(self, prefix='') -> Dict[str, np.ndarray]:
        metrics = dict()
        for key, value in self.running_metric.items():
            if 'Drop' in key:
                metrics[f'{prefix}_{self.name}_{key}'] = np.sum(value)
            else:
                metrics[f'{prefix}_{self.name}_{key}'] = np.mean(value)

        return metrics

    def get_main_metric_value(self):
        if not self.main_metric_name:
            raise ValueError(f'For metric {self.name} define '
                             f'attribute main_metric_name.')
        metrics = self.get_value()
        return metrics[f'_{self.name}_{self.main_metric_name}']

    def reset_metric(self):
        self.running_metric = defaultdict(list)
        self.preds = list()
        self.ground_truth = list()

    def plot(self, tb_writer, step, tb_prefix=''):
        if tb_writer is not None and self.plottable:
            metrics = self.get_value()
            for key, value in metrics.items():
                tb_writer.add_scalar(tag=f'{tb_prefix}/{self.name}_{key}',
                                     scalar_value=value,
                                          global_step=step)
            tb_writer.flush()
        else:
            return False
