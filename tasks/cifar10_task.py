import torchvision
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import torch.utils.data as torch_data
from copy import copy

# from models.resnet import resnet18, resnet50
from models.resnet_cifar import resnet18
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.task import Task
from dataset.cifar import CIFAR10, CIFAR100


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()


    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
                # transforms.RandomAdjustSharpness(0.3, p=self.params.transform_sharpness),
                transforms.RandomErasing(p=self.params.transform_erase,
                                                                scale=(0.01, 0.02),
                                                                ratio=(0.3, 3.3), value=0,
                                                                inplace=False)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        if self.params.dataset == 'CIFAR100':
            ds = CIFAR100
        elif self.params.dataset == 'CIFAR10':
            ds = CIFAR10
        else:
            raise ValueError('Specify "dataset" param: CIFAR10 or CIFAR100')

        self.train_dataset = ds(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.test_dataset = ds(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = self.train_dataset.classes
        return True

    def make_scheduler(self):
        if self.params.scheduler:
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.epochs)
        else:
            return None

    def build_model(self) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True,
                             bn_enable=self.params.bn_enable)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        # elif self.params.cifar_model_l1 and self.params.cifar_model_l2:
        #     model = Net(self.params.cifar_model_l1, self.params.cifar_model_l2)
        else:
            model = resnet18(pretrained=False,
                             num_classes=len(self.classes),
                             bn_enable=self.params.bn_enable)
        return model

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))

    def make_attack_pattern(self, pattern_tensor, x_top, y_top, mask_value):
        full_image = torch.zeros(self.train_dataset.data[0].shape)
        full_image.fill_(mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        full_image[x_top:x_bot, y_top:y_bot, :] = pattern_tensor.unsqueeze(-1)

        mask = 1 * (full_image != mask_value)
        pattern = 255 * full_image

        return mask.numpy(), pattern.numpy()
