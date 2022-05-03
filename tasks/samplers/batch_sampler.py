from collections import defaultdict

import torch.utils.data as torch_data
from typing import List, Iterator
import numpy as np
import torch
from tqdm import tqdm

from dataset.mnist import MNIST


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

    def __init__(self, train_dataset, batch_size, drop_last) -> None:
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

    def cos_sim_matrix(self):
        return

    def __iter__(self) -> Iterator[List[int]]:
        choosing_indices = torch.ones_like(self.dataset.attacked_indices)
        candidate = np.random.choice(choosing_indices.nonzero().view(-1))
        choosing_indices[candidate] = 0
        norm = self.norms[candidate]
        sims = norm * self.self_matrix[candidate].type(torch.float32)
        attacked_indices = defaultdict(int)
        non_attacked_indices = defaultdict(int)
        for j in range(len(self.dataset) // self.batch_size):
            # candidate = np.random.choice(len(self.dataset))  # pick in the same direction
            candidate = np.random.choice(choosing_indices.nonzero().view(-1))
            if self.dataset.attacked_indices[candidate] == 1:
                attacked_indices[candidate] += 1
            else:
                non_attacked_indices[candidate] += 1
            batch_ids = [candidate]
            sims += norm * self.self_matrix[candidate].type(torch.float32)
            # norm = self.norms[candidate]
            # sims += norm * self.self_matrix[candidate].type(torch.float32)
            for i in range(self.batch_size):
                probs = torch.clamp(-sims, min=0.0)
                # probs = (sims < 0) * 1.0
                # probs = torch.pow(probs, 10)
                probs /= probs.sum()
                probs *= choosing_indices
                if probs.sum() == 0:
                    print(f'sampled at least once all available at {j}')
                    print(
                        f'Stats: {len(attacked_indices)}/{len(non_attacked_indices)}={len(attacked_indices) / len(non_attacked_indices):.5f} ')
                    print(
                        f'Total: {self.dataset.attacked_indices.sum()}/{len(self.dataset)} {self.dataset.attacked_indices.sum() / len(self.dataset):.5f}')
                    return
                # probs = 1.0 * (self.norms <=100)
                candidate = torch.multinomial(probs, 1).item()
                choosing_indices[candidate] = 0
                if self.dataset.attacked_indices[candidate] == 1:
                    attacked_indices[candidate] += 1
                else:
                    non_attacked_indices[candidate] += 1
                # indices = sorted_sims.indices
                # candidate = np.random.choice(indices[:1000])
                norm = self.norms[candidate]
                sims += norm * self.self_matrix[candidate].type(torch.float32)
                batch_ids.append(candidate)
            # print(self.dataset.grads[batch_ids].norm().item())
            # for i in tqdm(range(self.batch_size)):
            #     metrics = sim_matrix(self.previous_vector, self.weights).squeeze().sort()
            #     # metrics = get_norm(self.previous_vector, self.weights).sort()
            #     candidate = np.random.choice(metrics.indices[4:1000].cpu().numpy())
            #
            #     batch_ids.append(candidate + self.offset)
            #     self.previous_vector += self.weights[candidate:candidate+1]
            yield batch_ids
        print(
            f'Stats: {len(attacked_indices)}/{len(non_attacked_indices)}={len(attacked_indices) / len(non_attacked_indices):.2f} ')
        print(
            f'Total: {self.dataset.attacked_indices.sum()}/{len(self.dataset)} {self.dataset.attacked_indices.sum() / len(self.dataset):.2f}')
        #
        #
        # for idx in self.sampler:
        #     batch.append(idx)
        #     if len(batch) == self.batch_size:
        #         yield batch
        #         batch = []
        # if len(batch) > 0 and not self.drop_last:
        #     yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.dataset) // self.batch_size
