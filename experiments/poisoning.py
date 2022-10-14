"""
Poison images by adding a mask
"""
from typing import Tuple
from dataclasses import replace

import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

class Poison(Operation):
    """Poison specified images by adding a mask with given opacity.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    mask : ndarray
        The mask to apply to each image.
    alpha: float
        The opacity of the mask.
    indices : Sequence[int]
        The indices of images that should have the mask applied.
    clamp : Tuple[int, int]
        Clamps the final pixel values between these two values (default: (0, 255)).
    """

    def __init__(self, mask: np.ndarray, pattern,
                 indices, clamp = (0, 255)):
        super().__init__()
        self.mask = mask
        self.indices = np.sort(indices)
        self.clamp = clamp
        self.pattern = pattern


    def generate_code(self) -> Callable:
        pattern = self.pattern
        mask = self.mask.astype('float')
        non_mask = (self.mask == 0).astype('float')
        to_poison = self.indices
        my_range = Compiler.get_iterator()



        def poison(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                sample_ix = indices[i]
                # We check if the index is in the list of indices
                # to poison
                position = np.searchsorted(to_poison, sample_ix)
                # print(position)
                if position < len(to_poison) and to_poison[position] == sample_ix:
                    temp = temp_array[i]
                    temp[:] = non_mask * images[i] + mask * pattern
                    # temp *= 1 - alpha
                    # temp += mask
                    # np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
            return images

        poison.is_parallel = True
        poison.with_indices = True

        return poison

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # We do everything in place
        return (replace(previous_state, jit_mode=True), \
                AllocationQuery(shape=previous_state.shape, dtype=np.dtype('float32')))
