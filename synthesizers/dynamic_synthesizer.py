import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.pattern_synthesizer import PatternSynthesizer

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class DynamicSynthesizer(PatternSynthesizer):
    name = 'Dynamic'

    # 5 by 5 pattern
    pattern_tensor: torch.Tensor = torch.tensor([
        [1.]
    ])
    "Just some random 2D pattern."

    resize_scale = (4, 8)
    "If the pattern is dynamically placed, resize the pattern."

    def update_pattern(self):
        resize = random.randint(self.resize_scale[0], self.resize_scale[1])
        if random.random() > 0.5:
            self.pattern_tensor = functional.hflip(self.pattern_tensor)
        image = transform_to_image(self.pattern_tensor)
        self.pattern_tensor = transform_to_tensor(functional.resize(image, [resize, resize])).squeeze()
        self.x_top = random.randint(0, self.input_stats.input_shape[1] -
                                    self.pattern_tensor.shape[0] - 1)
        self.y_top = random.randint(0, self.input_stats.input_shape[2] -
                                    self.pattern_tensor.shape[1] - 1)
        self.make_pattern()

    def apply_mask(self, input_tensor):
        self.update_pattern()
        return (1 - self.mask) * input_tensor + self.mask * self.pattern

    def make_pattern(self):
        self.pattern = torch.ones_like(self.input_stats.average_input_values) *\
                       torch.max(self.input_stats.max_val)
        self.mask = torch.zeros_like(self.input_stats.average_input_values)
        self.mask[:, self.x_top:self.x_top + self.pattern_tensor.shape[0], self.y_top:self.y_top + self.pattern_tensor.shape[1]] = 1