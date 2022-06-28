import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import torch.utils.data as torch_data
from copy import copy

# from models.resnet import resnet18, resnet50
from models.resnet_cifar import resnet18
from tasks.samplers.batch_sampler import CosineBatchSampler
from tasks.task import Task
from dataset.celeba import CelebADataset


class CelebaTask(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        self.load_celeba_data()


    def load_celeba_data(self):
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

        self.train_dataset = CelebADataset(
            root=self.params.data_path,
            split='train',
            download=True,
            main_attr=31,
            transform=transform_train)

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

            self.train_dataset.data = self.train_dataset.data[keep_indices]
            self.train_dataset.targets = self.train_dataset.targets[keep_indices]
            self.train_dataset.true_targets = self.train_dataset.true_targets[keep_indices]

        if self.params.clean_subset != 0:
            self.clean_dataset = copy(self.train_dataset)
            if self.params.poison_images is not None and self.params.add_images_to_clean:
                keep_indices = list()
                for i in range(self.params.clean_subset):
                    if i not in self.params.poison_images:
                        keep_indices.append(i)
            else:
                keep_indices = list(range(self.params.clean_subset))
            self.clean_dataset.data = self.clean_dataset.data[keep_indices]
            self.clean_dataset.targets = self.clean_dataset.targets[keep_indices]
            self.clean_dataset.true_targets = self.clean_dataset.true_targets[keep_indices]

        self.test_dataset = CelebADataset(
            root=self.params.data_path,
            split='test',
            main_attr=31,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.test_attack_dataset =  CelebADataset(
            root=self.params.data_path,
            split='test',
            main_attr=31,
            download=True,
            transform=transform_test)
        self.test_attack_loader = DataLoader(self.test_attack_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)
        self.classes = ['No', 'Yes']
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
