import numpy as np
import torch

from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params
from numpy.random import Generator, PCG64

class Synthesizer:
    params: Params
    task: Task
    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:

        # Don't attack if only normal loss task.
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        self.apply_backdoor(backdoored_batch, attack_portion)

        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)

        return

    def apply_mask(self, input):
        return (1 - self.mask) * input + self.mask * self.pattern

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None):
        raise NotImplemented

    def get_indices(self, indices_arr, proportion, dataset, clean_label=None):
        dataset_len = len(dataset)
        if indices_arr is None:
            rs = Generator(PCG64(self.params.random_seed))
            indices = rs.choice(range(dataset_len),
                                       int(proportion * dataset_len),
                                       replace=False)
            indices_arr = torch.zeros(dataset_len, dtype=torch.int32)
            if clean_label:
                new_indices = list()
                for index in indices:
                    if dataset[index][1] == self.params.backdoor_label:
                        new_indices.append(index)
                indices = new_indices
        else:
            indices = indices_arr.nonzero().T[0].numpy()
        print(f'Poisoned total of {len(indices)}')
        return indices_arr, indices
