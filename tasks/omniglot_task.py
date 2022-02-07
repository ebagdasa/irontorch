import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from dataset.omniglot import Omniglot
from models.resnet_cifar import resnet18, resnet50
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.task import Task
from dataset.cifar import CIFAR10, CIFAR100
image_size = 28
N_WAY = 5 # Number of classes in a task
N_SHOT = 5 # Number of images per class in the support set
N_QUERY = 10 # Number of images per class in the query set
N_EVALUATION_TASKS = 100
N_TRAINING_EPISODES = 4000
N_VALIDATION_TASKS = 100

from easyfsl.data_tools import TaskSampler

class OmniglotTask(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        transform_train = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    [int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        self.train_dataset = Omniglot(
            root="./data",
            background=True,
            transform=transform_train,
            download=True)
        self.test_dataset = Omniglot(
            root="./data",
            background=True,
            transform=transform_test,
            download=True)
        test_sampler = TaskSampler(
            self.test_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY,
            n_tasks=N_EVALUATION_TASKS
        )
        train_sampler = TaskSampler(
            self.train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY,
            n_tasks=N_TRAINING_EPISODES
        )
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_sampler=train_sampler,
                                       num_workers=12,
                                       pin_memory=True,
                                       collate_fn=train_sampler.episodic_collate_fn,)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_sampler=test_sampler,
                                      num_workers=12,
                                      pin_memory=True,
                                      collate_fn=test_sampler.episodic_collate_fn)

        self.test_attack_dataset = Omniglot(
            root="./data",
            background=True,
            transform=transform_test,
            download=True)
        self.test_attack_loader = DataLoader(self.test_attack_dataset,
                                      batch_sampler=test_sampler,
                                      num_workers=12,
                                      pin_memory=True,
                                      collate_fn=test_sampler.episodic_collate_fn)

        self.classes = self.train_dataset.classes
        return True

    def make_scheduler(self):
        if self.params.scheduler:
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.epochs)
        else:
            return None

    def build_model(self) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False,
                                  num_classes=10)
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

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != mask_value)
        pattern = 255 * full_image

        return mask, pattern
