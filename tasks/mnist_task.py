import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms
import dataset.mnist as mnist_dataset
from models.parametrized_simple import ParametrizedSimpleNet
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.task import Task
import torch
from copy import copy

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def make_transformations(self):
        transformations = [transforms.ToTensor(), self.normalize,]
        if self.params.transform_erase:
            transformations.append(transforms.RandomErasing(p=1, scale=(0.02, 0.03),
                                                            ratio=(0.3, 3.3), value=1,
                                                            inplace=False))
        return transformations

    def load_data(self):
        transform_train = transforms.Compose(
            self.make_transformations()
        )

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
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
        return ParametrizedSimpleNet(num_classes=len(self.classes),
                                     out_channels1=self.params.out_channels1,
                                     out_channels2=self.params.out_channels2,
                                     kernel_size1=self.params.kernel_size1,
                                     kernel_size2=self.params.kernel_size2,
                                     strides1=self.params.strides1,
                                     strides2=self.params.strides2,
                                     dropout1=self.params.dropout1,
                                     dropout2=self.params.dropout2,
                                     fc1=self.params.fc1,
                                     max_pool=self.params.max_pool,
                                     activation=self.params.activation)

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        full_image = torch.zeros(self.train_dataset.data[0].shape)
        full_image.fill_(mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        full_image[x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != mask_value)
        pattern = full_image

        return mask, pattern
