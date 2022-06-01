import importlib
import logging
import os
import random
import wandb
from collections import defaultdict
from copy import deepcopy
from shutil import copyfile
from typing import Union
from datetime import datetime
import socket
import sys

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from synthesizers.synthesizer import Synthesizer
from tasks.fl.fl_task import FederatedLearningTask
from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')


class Helper:
    params: Params = None
    task: Union[Task, FederatedLearningTask] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer: SummaryWriter = None
    wandb_logger = None

    def __init__(self, params):
        self.params = Params(**params)

        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed)

        self.make_folders()
        self.make_task()
        self.make_synthesizer()
        self.attack = Attack(self.params, self.synthesizer)

        if self.params.backdoor:
            self.modify_datasets()
        self.task.test_attack_dataset = self.attack.attack_dataset(
                self.task.test_attack_dataset, 1.0)
        self.task.make_loaders()
        self.task.make_opacus()
        self.best_loss = float('inf')

    def make_task(self):
        name_lower = self.params.task.lower()
        name_cap = self.params.task
        if self.params.fl:
            module_name = f'tasks.fl.{name_lower}_task'
            path = f'tasks/fl/{name_lower}_task.py'
        else:
            module_name = f'tasks.{name_lower}_task'
            path = f'tasks/{name_lower}_task.py'
        try:
            task_module = importlib.import_module(module_name)
            task_class = getattr(task_module, f'{name_cap}Task')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)

    def make_synthesizer(self):
        name_lower = self.params.synthesizer.lower()
        name_cap = self.params.synthesizer
        module_name = f'synthesizers.{name_lower}_synthesizer'
        try:
            synthesizer_module = importlib.import_module(module_name)
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)

    def make_folders(self):
        machine_name = socket.gethostname()
        log = create_logger()
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'All')
        with open(f'runs.txt', 'a') as f:
            f.write(
                    f'{machine_name} | {datetime.now()} | {devices} | {sys.argv[0]} | {self.params.name} | {self.params.commit}')
            f.write('\n')
        if self.params.log:
            try:
                os.mkdir(self.params.folder_path)
            except FileExistsError:
                log.info('Folder already exists')

            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/'
                              f'backdoors/tree/{self.params.commit}">GitHub'
                              f'</a>, <span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_'
                              f'{self.params.current_time}</a></div>'])

            fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(name)s '
                                          '- %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.warning(f'Logging to: {self.params.folder_path}')
            log.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/'
                f'{self.params.commit}">https://github.com/ebagdasa/backdoors'
                f'/tree/{self.params.commit}</a>')

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)

        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)
        elif self.params.wandb:
            params_dict = self.params.to_dict()
            params_dict['current_time'] = None
            self.wandb_logger = wandb.init(config=params_dict,
                       project=self.params.project,
                       name=self.params.name)

            if self.wandb_logger.sweep_id is not None:
                config = self.wandb_logger.config
                with open('configs/sweep.yaml') as f:
                    sweep_params = yaml.load(f, Loader=yaml.FullLoader)['parameters']
                    name = 'sweep'
                    for param_name in sweep_params.keys():
                        print(param_name)
                        print(config[param_name])
                        name += f'_{param_name}{config.as_dict()[param_name]}'
                    self.wandb_logger.name = name
                    self.wandb_logger.save()
                self.params.update(config.as_dict())
            logger.warning('Initialized Wandb.')

    def modify_datasets(self):
        if self.params.recover_indices:
            indices_results = torch.load(self.params.recover_indices)
            self.task.train_dataset = self.attack.attack_dataset(
                self.task.train_dataset,
                self.params.poisoning_proportion, indices_results['indices'],
                clean_label=self.params.clean_label)

        else:
            self.task.train_dataset = self.attack.attack_dataset(self.task.train_dataset,
                                                             self.params.poisoning_proportion, None,
                                                             clean_label=self.params.clean_label)
            # self.task.clean_dataset = self.attack.attack_dataset(self.task.clean_dataset,
            #                                                  0.1, None,
            #                                                  clean_label=self.params.clean_label)

        return

    def save_model(self, model=None, epoch=0, val_loss=0):

        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
            saved_dict = {'state_dict': model.state_dict(),
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.save_on_epochs:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False,
                                     filename=f'{model_name}.epoch_{epoch}')
            if True:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_grads(self, epoch, batch):
        if self.params.log and self.params.saved_grads and self.params.opacus:
            torch.save({
                'grads': self.task.optimizer.grad_accum,
                'losses': self.task.optimizer.loss_accum,
                'data': self.task.optimizer.data_accum,
                'labels': self.task.optimizer.label_accum,
                'aux': self.task.optimizer.aux,
                'attacked_indices': self.task.train_dataset.attacked_indices},
                f'{self.params.folder_path}/epoch_{epoch}_{batch}_optimizer.pt')
        if self.params.opacus:
            self.task.optimizer.reset_accums()

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()

    def report_dict(self, dict_report, step=None):
        if self.tb_writer is not None:
            for name, y in dict_report.items():
                self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=step)
                self.flush_writer()
        elif self.wandb_logger is not None:
            self.wandb_logger.log(dict_report, step=step)

        else:
            return False

    def report_training_losses_scales(self, batch_id, epoch):
        total_batches = len(self.task.train_loader)
        batch_id += 1
        if self.params.report_train_loss and \
            (batch_id % self.params.log_interval == 0 \
                or batch_id == total_batches):

            losses = [f'{x}: {np.mean(y):.2f}'
                      for x, y in self.params.running_losses.items()]
            scales = [f'{x}: {np.mean(y):.2f}'
                      for x, y in self.params.running_scales.items()]
            logger.info(
                f'Epoch: {epoch:3d}. '
                f'Batch: {batch_id:5d}/{total_batches}. '
                f' Losses: {losses}.'
                f' Scales: {scales}')
            for name, values in self.params.running_losses.items():
                self.report_dict({f'Train/Loss_{name}': np.mean(values)},
                          step=epoch * total_batches + batch_id)
            for name, values in self.params.running_scales.items():
                self.report_dict({f'Train/Scale_{name}': np.mean(values)},
                          step=epoch * total_batches + batch_id)
            if self.task.scheduler:
                self.report_dict({'Train/learning_rate': self.task.scheduler.get_last_lr()[0]},
                             step=epoch * total_batches + batch_id)
            self.params.running_losses = defaultdict(list)
            self.params.running_scales = defaultdict(list)
            self.save_grads(epoch, batch_id)

    def report_metrics(self, prefix, epoch=None):
        for metric in self.task.metrics:
            metric_values = metric.get_value(prefix=prefix)
            self.report_dict(dict_report=metric_values, step=epoch)
            logger.warning(f'{prefix}, {metric_values} {epoch}')

    def plot_confusion_matrix(self, backdoor=False, epoch=1):
        metric = self.task.metrics[0]
        if epoch == self.params.epochs and self.params.wandb and self.params.plot_conf_matrix:
            self.wandb_logger.log(
                {f"conf_mat_back_{backdoor}": wandb.plot.confusion_matrix(
                    y_true=torch.cat(metric.ground_truth).numpy(),
                    preds=torch.cat(metric.preds).view(-1).numpy(),
                    class_names=self.task.classes)})

    def poison_dataset(self):
        return

    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn

        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)

        return True
