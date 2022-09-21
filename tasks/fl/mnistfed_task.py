import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from tasks.cifar10_task import Cifar10Task
from tasks.fl.fl_task import FederatedLearningTask
from dataset.cifar import CIFAR10, CIFAR100
from tasks.mnist_task import MNISTTask


class MNISTFedTask(FederatedLearningTask, MNISTTask):
    dataset= 'MNISTFed'


