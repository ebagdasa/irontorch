import math
import random
from copy import deepcopy
from typing import List, Any, Dict

from dataset.attack_dataset import AttackDataset
from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
import torch
import logging
from torch.nn import Module

from tasks.task import Task
from utils.input_stats import InputStats
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger('logger')


class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None

    def init_task(self):
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.split_val_test_data()
        self.input_stats = InputStats(self.test_dataset)
        self.model = self.model.to(self.params.device)

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.metrics = dict(accuracy=AccuracyMetric(drop_label=self.params.drop_label,
                                                    total_dropped=self.get_total_drop_class()),
                            loss=TestLossMetric(self.criterion))
        self.make_synthesizers()
        self.make_attack_datasets()
        self.load_fl_data()
        self.make_loaders()
        self.adversaries = self.sample_adversaries()
        return

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            user = FLUser(user_id, compromised=compromised,
                          train_loader=train_loader)
            sampled_users.append(user)

        return sampled_users

    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack is not None:
            if epoch == self.params.fl_single_epoch_attack:
                if pos < self.params.fl_number_of_adversaries:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                   f' user: {user_id}.')
        else:
            compromised = user_id in self.adversaries
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack is None:
            print('AAAA')
            print(self.params.fl_total_participants, self.params.fl_number_of_adversaries)
            adversaries_ids = random.sample(
                range(self.params.fl_total_participants),
                min(self.params.fl_number_of_adversaries, self.params.fl_total_participants))
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        for i, train_loader in enumerate(self.fl_train_loaders):
            if i in adversaries_ids:
                attack_indices = train_loader.sampler.indices
                self.train_dataset.backdoor_indices = np.append(self.train_dataset.backdoor_indices,
                                                                attack_indices)
                self.train_dataset.indices_arr[attack_indices] = 1
        return adversaries_ids

    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)

        optimizer = self.make_optimizer(local_model)

        return local_model, optimizer

    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update

    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            self.dp_add_noise(average_update)
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_diff_privacy and \
                update_norm > self.params.fl_dp_clip:
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)

    def dp_add_noise(self, sum_update_tensor: torch.Tensor):
        if self.params.fl_diff_privacy:
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            noised_layer = noised_layer.to(self.params.device)
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if self.check_ignored_weights(name):
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False

    def load_fl_data(self):
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        return

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params.poison_images or \
                ind in self.params.poison_images_test:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][
                               :min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][
                                   min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param indices:
        :return:
        """
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(
            len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      sub_indices))
        return train_loader

