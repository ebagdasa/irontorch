import logging
from typing import List
import torch.utils.data as torch_data
import wandb

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import transforms

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from tasks.batch import Batch
from tasks.samplers.subseq_sampler import SubSequentialSampler
from tasks.samplers.subseq_sampler import SubSequentialSampler
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    test_attack_dataset = None
    test_attack_loader = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    scheduler: MultiStepLR = None
    metrics: List[Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()

        self.optimizer = self.make_optimizer()
        self.criterion = self.make_criterion()
        self.scheduler = self.make_scheduler()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()
        self.make_opacus()
        self.model = self.model.to(self.params.device)

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def make_optimizer(self, model=None) -> Optimizer:
        if model is None:
            model = self.model
        if self.params.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.params.lr,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum)
        elif self.params.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.params.lr,
                                   weight_decay=self.params.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def make_scheduler(self):
        if self.params.scheduler:
            return MultiStepLR(self.optimizer,
                                         milestones=self.params.scheduler_milestones,
                                         last_epoch=self.params.start_epoch - 2,
                                         gamma=0.1)
        else:
            return None

    def scheduler_step(self):
        if self.scheduler:
            self.scheduler.step()

    def make_opacus(self):
        if self.params.opacus:
            privacy_engine = PrivacyEngine(secure_mode=False)
            self.model, self.optimizer, _ = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.params.grad_sigma * self.params.grad_clip,
                max_grad_norm=self.params.grad_clip,
                clipping='flat',
            )
            self.optimizer.compute_grads_only = self.params.compute_grads_only
            logger.warning("Privatization is complete.")

    def resume_model(self):
        if self.params.opacus or self.params.fix_opacus_model:
            if not ModuleValidator.is_valid(self.model):
                logger.error(f'Model cannot be privatized. Fixing...')
                self.model = ModuleValidator.fix(self.model)

        if self.params.resume_model:
            logger.info(f'Resuming training from {self.params.resume_model}')
            loaded_params = torch.load(self.params.resume_model,
                                    map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            self.params.lr = loaded_params.get('lr', self.params.lr)

            logger.warning(f"Loaded parameters from saved model: LR is"
                           f" {self.params.lr} and current epoch is"
                           f" {self.params.start_epoch}")



    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels, indices, attacked = data
        batch = Batch(batch_id, inputs, labels, indices, attacked)
        return batch.to(self.params.device)

    def accumulate_metrics(self, outputs, labels):
        for metric in self.metrics:
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def report_metrics(self, step, prefix='',
                       tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        logger.warning(f'{prefix} {step:4d}. {" | ".join(metric_text)}')

        return  self.metrics[0].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res

    def get_sampler(self):
        if self.params.recover_indices:
            indices_results = torch.load(self.params.recover_indices)
            norms = indices_results['norms']
            if self.params.poisoning_proportion == 0.0:
                weights = torch.ones_like(norms)
                weights[indices_results['indices'].nonzero()] = 0.0
            else:
                weights = torch.pow(torch.clamp(1/norms, max=self.params.clamp_norms),
                                    self.params.pow_weight)
                # weights = torch.ones_like(norms)
                # weights[indices_results['indices'].nonzero()] = 0.1
                if self.params.cut_grad_threshold:
                    weights[indices_results['norms'] > self.params.cut_grad_threshold] = 0.0
                    weights[indices_results[
                                'norms'] <= self.params.cut_grad_threshold] = 1.0
                    print(f'Shape: {weights.shape}, sum: {weights.sum()}')
        else:
            weights = torch.ones(len(self.train_dataset))
            weights = weights / weights.sum()

        if self.params.subset_training is not None:
            if self.params.subset_training.get('type', None) == 'init':
                weights[self.params.subset_training['part']:] = 0.0
                train_len = self.params.subset_training['part']
            elif self.params.subset_training.get('type', None) == 'train':
                weights[:self.params.subset_training['part']] = 0.0
                train_len = weights.shape[0] - self.params.subset_training['part']
            else:
                raise ValueError('Specify subset_training.')
        else:
            train_len = weights.shape[0]

        weights = weights / weights.sum()
        if self.params.wandb and self.params.recover_indices:
            data = [[x, y, z] for (x, y, z) in
                    zip(norms, train_len * weights,
                        indices_results['indices'])]
            table = wandb.Table(data=data,
                                columns=["norms", "weights", "color"])
            wandb.log({'scatter-plot1': wandb.plot.scatter(table, "norms",
                                                           "weights")})
            data = [[x, y, z] for (x, y, z) in
                    zip(norms, weights.shape[0] * weights,
                        indices_results['indices']) if z == 1]
            table = wandb.Table(data=data,
                                columns=["norms", "weights", "color"])
            wandb.log({'scatter-plot2': wandb.plot.scatter(table, "norms",
                                                           "weights")})
            del indices_results
        sampler = torch_data.WeightedRandomSampler(weights, train_len)
        if self.params.compute_grads_only and self.params.subset_training:
            indices = weights.nonzero().view(-1).tolist()
            sampler = SubSequentialSampler(self.train_dataset, indices)

        return sampler

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        raise NotImplemented
