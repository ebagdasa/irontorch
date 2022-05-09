from collections import defaultdict

import torch.utils.data as torch_data
from typing import List, Iterator
import numpy as np
import torch
from tqdm import tqdm

from dataset.mnist import MNIST
import logging
logger = logging.getLogger('logger')

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    print('computing sim matrix')
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    print('done computing')
    return sim_mt


def get_norm(a, weight, eps=1e-8):
    return (weight - a).norm(dim=1)


class CosineBatchSampler(torch_data.Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, train_dataset, batch_size, drop_last, params) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = train_dataset
        self.norms = self.dataset.grads.norm(dim=1)
        self.self_matrix = sim_matrix(self.dataset.grads, self.dataset.grads)
        self.probs = None
        self.params = params
        self.update_probs()

    def cos_sim_matrix(self):
        return

    def update_probs(self):
        self.probs = ((self.self_matrix > self.params.cosine_bound) * 1.0).sum(dim=0)
        self.probs /= (torch.clamp(self.norms, min=self.params.clamp_probs))
        self.probs *= self.probs.shape[0]/self.probs.sum()
        self.probs = self.probs.to(self.params.device)
        # self.probs = torch.zeros(len(self.dataset))
        # for class_id in range(len(self.dataset.classes)):
        #     class_indices = (self.dataset.targets == class_id).nonzero().view(-1)
        #     norms = self.norms[class_indices]
        #     self_matrix = self.self_matrix[class_indices][:, class_indices]
        #     class_probs = ((self_matrix > self.params.cosine_bound) * 1.0).sum(dim=0)
        #     class_probs /= (torch.clamp(norms, min=self.params.clamp_probs))
        #     self.probs[class_indices] = class_probs

    def get_counts(self, attacked_indices, non_attacked_indices, unsampled_indices):
        unsampled_count = (
                self.dataset.targets == self.params.drop_label).sum().item()
        total_sampled = attacked_indices + non_attacked_indices
        attacked_all = self.dataset.attacked_indices.sum()
        dataset_len = len(self.dataset)
        logger.info(f'Dataset change: {total_sampled}/{dataset_len} => {100*total_sampled/dataset_len:.2f}%')
        logger.info(
            f'Attacked: {attacked_indices}/{total_sampled}={attacked_indices/total_sampled:.5f} ' +\
            f'vs {attacked_all}/{dataset_len}={attacked_all/dataset_len:.5f}' +\
            f'=> {100 * (attacked_indices/total_sampled)/(attacked_all/dataset_len):.3f}%')
        logger.info(
            f'Dropped: {unsampled_indices}/{total_sampled}={unsampled_indices/total_sampled:.5f}' + \
            f'vs {unsampled_count}/{dataset_len} {unsampled_count / dataset_len:.5f}' +\
            f'=> {100 * (unsampled_indices/total_sampled)/(unsampled_count / dataset_len):.3f}%')

    def __iter__(self) -> Iterator[List[int]]:
        choosing_indices = torch.ones_like(self.dataset.attacked_indices,
                                           dtype=torch.float32, device=self.params.device)
        attacked_indices = defaultdict(int)
        non_attacked_indices = defaultdict(int)
        unsampled_indices = defaultdict(int)

        for j in range(len(self.dataset) // self.batch_size):
            batch_ids = []
            for i in range(self.batch_size):
                if choosing_indices.sum() == 0:
                    print(f'sampled at least once all available at {j}')
                    self.get_counts(len(attacked_indices), len(non_attacked_indices),
                               len(unsampled_indices))
                    return
                candidate = torch.multinomial(self.probs * choosing_indices, 1).item()
                choosing_indices[candidate] *= self.params.de_sample
                if self.dataset.attacked_indices[candidate] == 1:
                    attacked_indices[candidate] += 1
                else:
                    non_attacked_indices[candidate] += 1
                if self.params.drop_label is not None and self.dataset.targets[candidate] == self.params.drop_label:
                    unsampled_indices[candidate] += 1
                batch_ids.append(candidate)
            yield batch_ids
        self.get_counts(len(attacked_indices), len(non_attacked_indices), len(unsampled_indices))


    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.dataset) // self.batch_size
