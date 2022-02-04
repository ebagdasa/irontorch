import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms
import dataset.mnist as mnist_dataset
from models.simple import SimpleNet
from tasks.task import Task
import torch

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
        sampler = self.get_sampler()

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=False,
                                                  sampler=sampler,
                                                  num_workers=0)
        self.test_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

        self.test_attack_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.test_attack_loader = torch_data.DataLoader(self.test_attack_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)
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
        pattern = 255 * full_image

        return mask, pattern
