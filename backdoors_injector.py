import torch
import torchvision
import torch.utils.data as data


class BackdoorsInjector(data.Dataset):
    def __init__(self, task):
        self.task = task

    def inject(self, dataset):
        for i in range(len(dataset)):
            dataset[i] = self.inject_backdoor(dataset[i])
        return dataset

    def inject_backdoor(self, image):
        return

    def __getitem__(self, index):
        return

    def __len__(self):
        return

    def __repr__(self):
        return

