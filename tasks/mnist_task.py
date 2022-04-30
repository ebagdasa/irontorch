import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms
import dataset.mnist as mnist_dataset
from models.simple import SimpleNet
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.task import Task
import torch
from copy import copy

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # self.normalize
        ])

        self.train_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        if self.params.subset_training is not None:
            self.clean_dataset = copy(self.train_dataset)
            self.clean_dataset.data = self.clean_dataset.data[:self.params.subset_training['part']]
            self.clean_dataset.targets = self.clean_dataset.targets[:self.params.subset_training['part']]
            self.clean_dataset.true_targets = self.clean_dataset.true_targets[
                                      :self.params.subset_training['part']]
            self.train_dataset.data = self.train_dataset.data[self.params.subset_training['part']:]
            self.train_dataset.targets = self.train_dataset.targets[
                                      self.params.subset_training['part']:]
            self.train_dataset.true_targets = self.train_dataset.true_targets[
                                      self.params.subset_training['part']:]

        if self.params.drop_label_proportion is not None and \
              self.params.drop_label is not None:
            non_label_indices = (self.train_dataset.true_targets != self.params.drop_label)
            gen = torch.manual_seed(5)
            rand_mask = torch.rand(non_label_indices.shape, generator=gen) >= self.params.drop_label_proportion
            keep_indices = (non_label_indices + rand_mask).nonzero().view(-1)
            print(f'After filtering {100 * self.params.drop_label_proportion:.0f}%' +\
                  f' of class {self.train_dataset.classes[self.params.drop_label]}' +\
                  f' we have a total {keep_indices.shape[0]}.')

            self.train_dataset.data = self.train_dataset.data[keep_indices]
            self.train_dataset.targets = self.train_dataset.targets[keep_indices]
            self.train_dataset.true_targets = self.train_dataset.true_targets[keep_indices]

        self.test_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.test_attack_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.classes = self.train_dataset.classes
        return True

    def make_loaders(self):
        sampler = self.get_sampler()
        if self.params.cosine_batching:
            batcher = CosineBatchSampler(train_dataset=self.train_dataset,
                                         batch_size=self.params.batch_size,
                                         drop_last=False)
            self.train_loader = torch_data.DataLoader(self.train_dataset,
                                           batch_sampler=batcher, num_workers=0)
        else:
            self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=False,
                                                  sampler=sampler,
                                                  num_workers=0)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

        self.test_attack_loader = torch_data.DataLoader(self.test_attack_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

    def build_model(self):
        return SimpleNet(num_classes=len(self.classes))

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        full_image = torch.zeros(self.train_dataset.data[0].shape)
        full_image.fill_(mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        full_image[x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != mask_value)
        pattern = 255 * full_image

        return mask, pattern
