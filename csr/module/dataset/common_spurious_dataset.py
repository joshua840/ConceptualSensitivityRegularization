import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List


class CommonSpuriousDataset(Dataset):
    def __getitem__(self, item: int):
        """
        Returns the input, label, group. The input and mask must be of the same size and the mask must be boolean.
        The true values of the mask must identify sensitive or nuisance pixels/words/features of the input.
        """
        self._num_classes = None
        self._num_groups = None
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    # def get_group_count_dict(self) -> Dict:
    #     raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        raise NotImplementedError()

    @property
    def num_groups(self) -> int:
        raise NotImplementedError()

    def verbose(self):
        raise NotImplementedError()
