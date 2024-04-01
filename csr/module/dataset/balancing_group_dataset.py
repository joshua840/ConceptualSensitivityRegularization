import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms

# from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.datasets import make_blobs
import pandas as pd
from .common_spurious_dataset import CommonSpuriousDataset

"""
https://github.com/facebookresearch/BalancingGroups
"""


class GroupDataset(CommonSpuriousDataset):
    def __init__(
        self,
        split,
        root,
        metadata,
        transform,
        subsample_what=None,
        duplicates=None,
        minor_ratio=None,
    ):
        """
        split : train, val, test
        """
        self.transform_ = transform
        df = pd.read_csv(metadata)
        df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}[split])]

        self.i = list(range(len(df)))
        self.x = (
            df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        )
        self.y = np.array(df["y"])
        self.attr = np.array(df["a"])

        # minor_ratio
        if split == "tr" and minor_ratio is not None:
            indices_dict = self._get_indices_dict()
            indices_dict = self._remove_label_bias(indices_dict)
            indices_dict = self._set_minor_ratio(indices_dict, minor_ratio)
            self.i = np.concatenate(
                [indices for indices in indices_dict.values()], axis=0
            )

        self.count_groups()

        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

    def count_groups(self):
        self.wg, self.wy = [], []

        self.nb_groups = len(set(self.attr))
        self.nb_labels = len(set(self.y))
        self.group_sizes = {}  # [0] * self.nb_groups * self.nb_labels
        self.class_sizes = [0] * self.nb_labels

        for g in range(self.nb_groups):
            for y in range(self.nb_labels):
                self.group_sizes[(g, y)] = (
                    (self.y[self.i] == y) * (self.attr[self.i] == g)
                ).sum()
                self.class_sizes.append((self.y[self.i] == y).sum())

    def subsample_(self, subsample_what):
        perm = torch.randperm(len(self)).tolist()

        if subsample_what == "groups":
            min_size = min(list(self.group_sizes))
        else:
            min_size = min(list(self.class_sizes))

        counts_g = [0] * self.nb_groups * self.nb_labels
        counts_y = [0] * self.nb_labels
        new_i = []
        for p in perm:
            y, g = self.y[self.i[p]], self.attr[self.i[p]]

            if (
                subsample_what == "groups"
                and counts_g[self.nb_groups * int(y) + int(g)] < min_size
            ) or (subsample_what == "classes" and counts_y[int(y)] < min_size):
                counts_g[self.nb_groups * int(y) + int(g)] += 1
                counts_y[int(y)] += 1
                new_i.append(self.i[p])

        self.i = new_i
        self.count_groups()

    def duplicate_(self, duplicates):
        new_i = []
        for i, duplicate in zip(self.i, duplicates):
            new_i += [i] * duplicate
        self.i = new_i
        self.count_groups()

    def __getitem__(self, i):
        j = self.i[i]
        x = self.transform(self.x[j])
        y = torch.tensor(self.y[j], dtype=torch.long)
        g = torch.tensor(self.attr[j], dtype=torch.long)
        return (
            x,
            y,
            g,
            torch.tensor(i, dtype=torch.long),
        )

    def __len__(self):
        return len(self.i)

    def verbose(self):
        self.count_groups()
        print(self.group_sizes)

    def _get_indices_dict(self):
        raise NotImplementedError()

    @staticmethod
    def _remove_label_bias(indices_dict):
        return indices_dict

    @staticmethod
    def _set_minor_ratio(indices_dict, minor_ratio):
        return indices_dict
