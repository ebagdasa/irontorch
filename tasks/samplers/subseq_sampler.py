import torch.utils.data as torch_data


class SubSequentialSampler(torch_data.Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source = None

    def __init__(self, data_source, range_sample) -> None:
        self.data_source = data_source
        self.range_sample = range_sample

    def __iter__(self):
        return iter(self.range_sample)

    def __len__(self):
        return len(self.range_sample)

