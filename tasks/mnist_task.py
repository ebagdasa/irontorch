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

    def make_transformations(self):
        transformations = list()
        # # transforms.RandomCrop(32, padding=4)
        # # transforms.RandomPerspective(0.2, 0.2, 0.2, 0.2)
        # if self.params.transform_sharpness:
        #     transformations.append(
        #         transforms.RandomAdjustSharpness(0.3, p=self.params.transform_sharpness))
        # if self.params.transform_erase:
        #     transformations.append(transforms.RandomErasing(p=1, scale=(0.02, 0.03),
        #                                                     ratio=(0.3, 3.3), value=0,
        #                                                     inplace=False))
        return transformations

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,

        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.test_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.classes = self.train_dataset.classes
        return True

    def build_model(self):
        return SimpleNet(num_classes=len(self.classes))

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        full_image = torch.zeros(self.train_dataset.data[0].shape)
        full_image.fill_(mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        full_image[x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != mask_value)
        pattern = full_image

        return mask, pattern
