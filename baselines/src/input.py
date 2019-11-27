import os
import numpy as np
from torch.utils.data import DataLoader

from baselines.utils.consts import Split


class InputPipeline(object):

    def __init__(self, datasets_list, batch_size=1, num_workers=1, seed=None):
        self.seed = seed
        self.dataloaders = {}
        for ds in datasets_list:
            shuffle = ds.subset == Split.TRAIN
            dl = self.get_dataloader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            self.dataloaders[ds.subset] = dl

    def get_dataloader(self, *args, **kwargs):
        if self.seed:
            kwargs['worker_init_fn'] = lambda: np.random.seed(self.seed)
        return DataLoader(*args, **kwargs)

    def __getitem__(self, subset):
        try:
            return self.dataloaders[subset]
        except KeyError:
            return None
