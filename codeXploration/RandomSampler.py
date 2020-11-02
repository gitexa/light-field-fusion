from torch.utils.data import Sampler
import numpy as np 


class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        arr = np.arange(len(self.data_source))
        np.random.shuffle(arr)
        return iter(arr)

    def __len__(self):
        return len(self.data_source)