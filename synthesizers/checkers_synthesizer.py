import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class CheckersSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([


    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 3
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.task.train_dataset.data[0].shape)
        full_image.fill_(0)

        full_image[::2, ::2] = 1

        self.mask = 1 * (full_image != self.mask_value)
        self.pattern = 255 * full_image


    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                    resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] \
                               - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] \
                               - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask
