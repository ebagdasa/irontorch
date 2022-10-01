import importlib
import logging
from collections import defaultdict
from typing import List, Dict
import torch.utils.data as torch_data
from torch.utils.data import DataLoader, SubsetRandomSampler
import wandb
from copy import deepcopy

from dataset.attack_dataset import AttackDataset
from models.simple import SimpleNet
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lrs
from torchvision.transforms import transforms

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from tasks.batch import Batch
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.samplers.subseq_sampler import SubSequentialSampler
from tasks.samplers.subseq_sampler import SubSequentialSampler
from utils.input_stats import InputStats
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    val_dataset = None
    val_loader = None
    val_attack_datasets = dict()
    val_attack_loaders = dict()
    test_attack_datasets = dict()
    test_attack_loaders = dict()
    clean_dataset = None
    clean_model = None
    train_loader = None
    test_loader = None
    classes = None
    synthesizers: Dict = dict()
    input_stats: InputStats = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    scheduler = None
    metrics: Dict[str, Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        self.drop_or_create_clean_datasets()
        self.split_val_test_data()
        self.input_stats = InputStats(self.test_dataset)

        self.model = self.build_model()
        if self.params.fix_opacus_model and not ModuleValidator.is_valid(self.model):
            logger.error(f'Model cannot be privatized. Fixing...')
            self.model = ModuleValidator.fix(self.model)
        self.resume_model()

        self.optimizer = self.make_optimizer()
        self.criterion = self.make_criterion()
        self.scheduler = self.make_scheduler()

        self.metrics = dict(accuracy=AccuracyMetric(drop_label=self.params.drop_label,
                                                    total_dropped=self.get_total_drop_class()),
                            loss=TestLossMetric(self.criterion))
        self.model = self.model.to(self.params.device)
        self.make_synthesizers()
        self.make_attack_datasets()
        self.make_loaders()
        self.make_opacus()

    def split_val_test_data(self):
        if self.val_dataset is None:
            split_index = int(self.params.split_val_test_ratio * len(self.test_dataset))
            self.val_dataset = deepcopy(self.test_dataset)
            if hasattr(self.val_dataset, 'data'):
                self.val_dataset.data = self.val_dataset.data[:split_index]
                self.test_dataset.data = self.test_dataset.data[split_index:]
            self.val_dataset.targets = self.val_dataset.targets[:split_index]
            self.test_dataset.targets = self.test_dataset.targets[split_index:]
            self.val_dataset.true_targets = self.val_dataset.true_targets[:split_index]
            self.test_dataset.true_targets = self.test_dataset.true_targets[split_index:]

    def load_data(self) -> None:
        raise NotImplemented

    def get_total_drop_class(self):
        if self.params.drop_label:
            return (self.test_dataset.targets == self.params.drop_label).sum().item()
        return None

    def build_model(self) -> Module:
        raise NotImplemented

    def make_synthesizers(self):
        self.synthesizers = dict()
        for synthesizer in self.params.synthesizers:
            print(f'Using {synthesizer}')
            name_lower = synthesizer.lower()
            name_cap = synthesizer
            module_name = f'synthesizers.{name_lower}_synthesizer'
            try:
                synthesizer_module = importlib.import_module(module_name)
                task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')
            except (ModuleNotFoundError, AttributeError):
                raise ModuleNotFoundError(
                    f'The synthesizer: {synthesizer}'
                    f' should be defined as a class '
                    f'{name_cap}Synthesizer in '
                    f'synthesizers/{name_lower}_synthesizer.py')
            self.synthesizers[synthesizer] = task_class(self.params, self.input_stats)

    def drop_or_create_clean_datasets(self):
        if self.params.drop_label_proportion is not None and \
              self.params.drop_label is not None:
            non_label_indices = (self.train_dataset.true_targets != self.params.drop_label)
            gen = torch.manual_seed(5)
            rand_mask = torch.rand(non_label_indices.shape, generator=gen) >= self.params.drop_label_proportion
            keep_indices = (non_label_indices + rand_mask).nonzero().view(-1)
            print(f'After filtering {100 * self.params.drop_label_proportion:.0f}%' +\
                f'({len(self.train_dataset) - keep_indices.shape[0]} examples)' +\
                  f' of class {self.train_dataset.classes[self.params.drop_label]}' +\
                  f' we have a total {keep_indices.shape[0]}.')

            if hasattr(self.train_dataset, 'data'):
                self.train_dataset.data = self.train_dataset.data[keep_indices]
            self.train_dataset.targets = self.train_dataset.targets[keep_indices]
            self.train_dataset.true_targets = self.train_dataset.true_targets[keep_indices]

        if self.params.clean_subset != 0:
            self.clean_dataset = deepcopy(self.train_dataset)
            if self.params.poison_images is not None and self.params.add_images_to_clean:
                keep_indices = list()
                for i in range(self.params.clean_subset):
                    if i not in self.params.poison_images:
                        keep_indices.append(i)
            else:
                keep_indices = list(range(self.params.clean_subset))
            if hasattr(self.clean_dataset, 'data'):
                self.clean_dataset.data = self.clean_dataset.data[keep_indices]
            self.clean_dataset.targets = self.clean_dataset.targets[keep_indices]
            self.clean_dataset.true_targets = self.clean_dataset.true_targets[keep_indices]

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def make_loaders(self):
        import numpy as np
        import random

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.params.random_seed)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=0, worker_init_fn=seed_worker,
                                       generator=g)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.params.test_batch_size,
                                     shuffle=False, num_workers=0)

        for synthesizer_name, synthesizer in self.synthesizers.items():
            subset_random_list_val = (self.val_dataset.targets != self.params.backdoor_labels[synthesizer_name]).nonzero().view(-1)
            subset_random_list_test = (self.test_dataset.targets != self.params.backdoor_labels[synthesizer_name]).nonzero().view(-1)

            self.val_attack_loaders[synthesizer_name] = DataLoader(self.val_attack_datasets[synthesizer_name],
                                                                   sampler=SubsetRandomSampler(indices=subset_random_list_val),
                                                                    batch_size=self.params.batch_size,
                                                                    shuffle=False, num_workers=0)
            self.test_attack_loaders[synthesizer_name] = DataLoader(self.test_attack_datasets[synthesizer_name],
                                                                    sampler=SubsetRandomSampler(indices=subset_random_list_test),
                                                                     batch_size=self.params.test_batch_size,
                                                                     shuffle=False, num_workers=0)

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
        elif self.params.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=self.params.lr)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def make_scheduler(self):
        if self.params.scheduler == 'MultiStepLR':
            return lrs.MultiStepLR(self.optimizer,
                                   milestones=self.params.scheduler_milestones,
                                   last_epoch=self.params.start_epoch - 2,
                                   gamma=0.1)
        elif self.params.scheduler == 'CosineAnnealingLR':
            return lrs.CosineAnnealingLR(self.optimizer, T_max=self.params.epochs)
        elif self.params.scheduler == 'StepLR':
            return lrs.StepLR(self.optimizer, step_size=1, gamma=0.1)
        else:
            return None

    def scheduler_step(self):
        if self.scheduler:
            self.scheduler.step()

    def make_opacus(self):
        if self.params.opacus:
            self.model.train()
            if not ModuleValidator.is_valid(self.model):
                logger.error(f'Model cannot be privatized. Fixing...')
                self.model = ModuleValidator.fix(self.model)
            privacy_engine = PrivacyEngine(secure_mode=False)
            self.model, self.optimizer, _ = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.params.grad_sigma / self.params.grad_clip,
                max_grad_norm=self.params.grad_clip,
                clipping='flat',
            )
            # self.optimizer.compute_grads_only = self.params.compute_grads_only
            logger.warning("Privatization is complete.")

    def make_attack_datasets(self):
        for synthesizer_name, synthesizer in self.synthesizers.items():
            print(synthesizer_name)
            self.test_attack_datasets[synthesizer_name] = AttackDataset(dataset=deepcopy(self.test_dataset),
                                                          synthesizer=synthesizer,
                                                          percentage_or_count='ALL',
                                                          random_seed=self.params.random_seed,
                                                          clean_subset=self.params.clean_subset,
                                                          )
            if self.val_dataset is not None:
                self.val_attack_datasets[synthesizer_name] = AttackDataset(dataset=deepcopy(self.val_dataset),
                                                             synthesizer=synthesizer,
                                                             percentage_or_count='ALL',
                                                             random_seed=self.params.random_seed,
                                                             clean_subset=self.params.clean_subset)
            if self.params.backdoor:
                if synthesizer_name == 'Secret':
                    percentage_or_count = 10
                else:
                    percentage_or_count = self.params.poisoning_proportion
                self.train_dataset = AttackDataset(dataset=self.train_dataset,
                                                    synthesizer=synthesizer,
                                                    percentage_or_count=percentage_or_count,
                                                    random_seed=self.params.random_seed,
                                                    clean_subset=self.params.clean_subset
                                                    )

        return

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
        for name, metric in self.metrics.items():
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self):
        for name, metric in self.metrics.items():
            metric.reset_metric()

    def report_metrics(self, step, prefix='',
                       tb_writer=None, tb_prefix='Metric/'):
        metric_text = []
        for metric in self.metrics.values():
            metric_text.append(str(metric))
            metric.plot(tb_writer, step, tb_prefix=tb_prefix)
        logger.warning(f'{prefix} {step:4d}. {" | ".join(metric_text)}')

        return self.metrics['accuracy'].get_main_metric_value()

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

    def train_model_for_sampling(self, test_every_epoch=False):

        model = self.model
        data_loader = torch.utils.data.DataLoader(self.clean_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0, drop_last=True
                                                  )
        model.train()
        for epoch in tqdm(range(self.params.sampling_model_epochs)):
            for x, y, indices, attacked in data_loader:
                self.optimizer.zero_grad(True)
                output = model(x.cuda())
                loss = self.criterion(output, y.cuda()).mean()
                loss.backward()
                self.optimizer.step()
            if test_every_epoch:
                self.test_sampling_model(model, backdoor=False)
        return model

    def test_sampling_model(self, model, backdoor=False):
        metric = AccuracyMetric()
        model.eval()
        test_loader = self.test_attack_loader if backdoor else self.test_loader
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader)):
                batch = self.get_batch(i, data)
                outputs = model(batch.inputs)
                metric.accumulate_on_batch(outputs=outputs,
                                           labels=batch.labels)
        prefix = 'Backdoor' if backdoor else 'Normal'
        logger.error(f'Sampling Model: {prefix} {metric}')

        return

    def create_grads(self, model):
        train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params.batch_size,
                                                        shuffle=False,
                                                        num_workers=0
                                                        )
        model.eval()
        grad_shape = None
        param_to_follow = None
        for name, param in model.named_parameters():
            if name == self.params.gradient_layer:
                param_to_follow = param
                grad_shape = param.data.view(-1).shape[0]
        self.train_dataset.grads = torch.zeros([len(self.train_dataset), grad_shape], device='cpu')
        for i, (x, y, indices, attacked) in enumerate(tqdm(train_data_loader)):
            output = model(x.cuda())
            loss = self.criterion(output, y.cuda())
            for j, z in enumerate(loss):
                self.train_dataset.grads[indices[j]] = \
                torch.autograd.grad(z, param_to_follow, retain_graph=True)[0].cpu().view(-1)
        return

    def get_sampler(self):
        grad_norms = torch.norm(self.train_dataset.grads, dim=1) + 1e-5
        if self.params.cut_grad_threshold:
            weights = (grad_norms <= self.params.cut_grad_threshold) * 1.0
        else:
            weights = torch.ones_like(grad_norms)
            weights = torch.pow(torch.clamp(1 / grad_norms, max=self.params.clamp_norms),
                                self.params.pow_weight)
            weights = weights / weights.sum()
        # if self.params.wandb:
        #     data = [[x.item(), y.item(), z.item()] for (x, y, z) in
        #             zip(grad_norms, weights,
        #                 self.train_dataset.attacked_indices)]
        #     table = wandb.Table(data=data,
        #                         columns=["norms", "weights", "color"])
        #     wandb.log({'scatter-plot1': wandb.plot.scatter(table, "norms",
        #                                                    "weights", 'All Data')})
        #     data = [[x.item(), y.item(), z.item()] for (x, y, z) in
        #             zip(grad_norms, weights,
        #                 self.train_dataset.attacked_indices) if z == 1]
        #     table = wandb.Table(data=data,
        #                         columns=["norms", "weights", "color"])
        #     wandb.log({'scatter-plot2': wandb.plot.scatter(table, "norms",
        #                                                    "weights", 'Backdoors')})
        #     data = [[x.item(), y.item(), z.item()] for (x, y, z) in
        #             zip(grad_norms, weights,
        #                 self.train_dataset.targets) if z == self.params.drop_label]
        #     table = wandb.Table(data=data,
        #                         columns=["norms", "weights", "color"])
        #     wandb.log({'scatter-plot3': wandb.plot.scatter(table, "norms",
        #                                                    "weights", 'Outliers')})
        train_len = weights.shape[0]
        sampler = torch_data.WeightedRandomSampler(weights, train_len)

        return sampler

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        raise NotImplemented
