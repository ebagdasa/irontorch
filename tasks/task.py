import logging
from typing import List, Dict
import torch.utils.data as torch_data
import wandb

from models.simple import SimpleNet
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm

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
from tasks.samplers.batch_sampler import CosineBatchSampler
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
    clean_dataset = None
    clean_model = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    scheduler: MultiStepLR = None
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
        self.set_input_shape()
        self.model = self.model.to(self.params.device)

    def load_data(self) -> None:
        raise NotImplemented

    def get_total_drop_class(self):
        if self.params.drop_label:
            return (self.test_dataset.targets == self.params.drop_label).sum().item()
        return None

    def build_model(self) -> Module:
        raise NotImplemented

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
        g.manual_seed(0)

        if self.params.pre_compute_grads:
            model = self.train_model_for_sampling()
            self.test_sampling_model(model)
            self.test_sampling_model(model, backdoor=True)
            self.create_grads(model)

            if self.params.cosine_batching:
                batcher = CosineBatchSampler(train_dataset=self.train_dataset,
                                         batch_size=self.params.batch_size,
                                         drop_last=False, params=self.params)
                self.train_loader = torch_data.DataLoader(self.train_dataset,
                                           batch_sampler=batcher, num_workers=0)
            else:
                sampler = self.get_sampler()
                self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                          batch_size=self.params.batch_size,
                                                          shuffle=True,
                                                          num_workers=0)
                # self.train_loader = torch_data.DataLoader(self.train_dataset,
                #                                           batch_size=self.params.batch_size,
                #                                           shuffle=False,
                #                                           sampler=sampler,
                #                                           num_workers=0)
        else:
            self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                      worker_init_fn=seed_worker,
                                                      generator=g,
                                                      )
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 worker_init_fn=seed_worker,
                                                 generator=g)

        self.test_attack_loader = torch_data.DataLoader(
            self.test_attack_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=0)

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
                self.train_dataset.grads[indices[j]] = torch.autograd.grad(z, param_to_follow, retain_graph=True)[0].cpu().view(-1)
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
