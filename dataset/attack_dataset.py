import numpy as np
import torch
import logging
from numpy.random import Generator, PCG64
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger('logger')


class AttackDataset(object):
    average_input_values: torch.Tensor = None
    max_val: torch.Tensor = None
    min_val: torch.Tensor = None
    indices = None
    indices_arr: torch.Tensor = None
    mask = None
    pattern = None
    class_label_count = defaultdict(int)

    def __init__(self, params, dataset, synthesizer, percentage_or_count,
                 clean_label=False, mask=None, pattern=None):
        self.params = params
        self.dataset = dataset
        self.synthesizer = synthesizer
        self.get_indices(percentage_or_count, clean_label)
        if mask is None or pattern is None:
            logger.error("Making attack pattern")
            self.get_min_max_dataset_values()
            if self.params.backdoor_cover_percentage is not None:
                self.make_attack_pattern_new()
            else:
                self.make_attack_pattern()
        else:
            self.mask = mask
            self.pattern = pattern

    def get_min_max_dataset_values(self):
        for i, (inp, target, _, _) in tqdm(enumerate(self.dataset)):
            if i == 1000:
                break
            self.max_val = torch.max(self.max_val, inp) if self.max_val is not None else inp
            self.min_val = torch.min(self.min_val, inp) if self.min_val is not None else inp
            if self.average_input_values is None:
                self.average_input_values = inp.clone()
            else:
                self.average_input_values += inp
            self.class_label_count[target] += 1

        self.average_input_values /= len(self.dataset)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        X, target, _, _ = self.dataset.__getitem__(index)
        X = X.clone()
        target = torch.tensor(target, device='cpu')
        if index in self.indices:
            X = self.apply_mask(X)
            target = torch.tensor(self.params.backdoor_label, device=target.device)
        return X, target.item(), index, self.indices_arr[index]

    def apply_mask(self, input):
        return (1 - self.mask) * input + self.mask * self.pattern

    def get_indices(self, percentage_or_count, clean_label=None):
        dataset_len = len(self.dataset)
        indices_cover = range(self.params.clean_subset, dataset_len) if self.dataset.train else range(dataset_len)
        if percentage_or_count == 'ALL':
            backdoor_counts = dataset_len
        elif percentage_or_count < 1:
            backdoor_counts = int(percentage_or_count * dataset_len)
        else:
            backdoor_counts = int(percentage_or_count)

        rs = Generator(PCG64(self.params.random_seed))
        self.indices = rs.choice(indices_cover, backdoor_counts, replace=False)
        self.indices_arr = torch.zeros(dataset_len, dtype=torch.int32)
        if clean_label:
            new_indices = list()
            for index in self.indices:
                if self.dataset[index][1] == self.params.backdoor_label:
                    new_indices.append(index)
            self.indices = new_indices
        else:
            self.indices_arr[self.indices] = 1

        logger.error(f'Poisoned total of {len(self.indices)} out of {dataset_len}.')

    def make_attack_pattern(self):
        full_image = torch.zeros_like(self.average_input_values)
        full_image.fill_(self.synthesizer.mask_value)

        x_bot = self.synthesizer.x_top + self.synthesizer.pattern_tensor.shape[0]
        y_bot = self.synthesizer.y_top + self.synthesizer.pattern_tensor.shape[1]

        full_image[:, self.synthesizer.x_top:x_bot, \
                   self.synthesizer.y_top:y_bot] = self.synthesizer.pattern_tensor

        self.mask = 1 * (full_image != self.synthesizer.mask_value)
        self.pattern = self.max_val * full_image

        return

    def make_attack_pattern_new(self):
        if self.params.random_seed is not None:
            torch.manual_seed(self.params.random_seed)
        # min_max_mask = 1 * (torch.zeros_like(self.average_input_values) > 0.5)
        # input_placeholder = self.max_val * min_max_mask + self.min_val * (1 - min_max_mask)
        input_placeholder = torch.ones_like(self.average_input_values) * torch.max(self.max_val)
        total_elements = input_placeholder.view(-1).shape[0]
        cover_size = max(1, int(total_elements * self.params.backdoor_cover_percentage))
        start_index = np.random.randint(0, total_elements - cover_size - 1, size=1)[0]
        self.mask = torch.zeros_like(input_placeholder)
        self.mask.view(-1)[start_index:start_index + cover_size] = 1
        self.pattern = input_placeholder

        # input_placeholder = torch.zeros_like(self.average_input_values).fill_(self.max_val)
        # total_elements = input_placeholder.view(-1).shape[0]
        #
        # cover_size = int(total_elements * self.params.backdoor_cover_percentage)
        # indices = torch.randint(total_elements, [cover_size])
        # self.mask = torch.zeros_like(input_placeholder)
        # self.mask.view(-1)[:cover_size] = 1
        # self.pattern = input_placeholder

