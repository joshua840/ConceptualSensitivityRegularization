import os
from PIL import Image
from torchvision import transforms
import numpy as np

from .balancing_group_dataset import GroupDataset
from .spuco_datasets import SpuCoDogs, SpuCoBirds, SpuCoAnimals, GroupLabeledDatasetWrapper


class Dogs(GroupDataset):
    def __init__(self, data_dir, split, subsample_what=None, duplicates=None, minor_ratio=None):

        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        if split != "tr":
            self.transform_ = transforms.Compose(
                [
                    transforms.Resize(resize_resolution),
                    transforms.CenterCrop(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform_ = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        target_resolution, scale=(0.7, 1.0), ratio=(0.75, 1.33333333), interpolation=2
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        dataset = SpuCoDogs(data_dir, split={"tr": "train", "va": "val", "te": "test"}[split])
        dataset.initialize()

        dataset = GroupLabeledDatasetWrapper(dataset=dataset, group_partition=dataset.group_partition)

        self.x = dataset.dataset.data.X
        self.y = np.array(dataset.dataset.data.labels)
        self.g = np.array(dataset.group) % 2 # mapping from group to attribution
        self.i = list(range(len(self.x)))

        # minor_ratio
        if split == "tr" and minor_ratio is not None:
            indices_dict = self._get_indices_dict()
            # indices_dict = self._remove_label_bias(indices_dict)
            indices_dict = self._set_minor_ratio(indices_dict, minor_ratio)
            self.i = np.concatenate([indices for indices in indices_dict.values()], axis=0)

        self.count_groups()

        print(self.group_sizes)

        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

        self.data_type = "images"
        self._num_classes = 2
        self._num_groups = 4

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

    def _get_indices_dict(self):
        indices_dict = {}
        for attr in range(2):
            for label in range(2):
                indices_dict[(attr, label)] = ((self.g == attr) * (self.y == label)).nonzero()[0].squeeze()
        return indices_dict

    # @staticmethod
    # def _remove_label_bias(indices_dict):
    #     indices_dict[(0, 1)] = indices_dict[(0, 0)][: len(indices_dict[(1, 1)])]
    #     indices_dict[(1, 0)] = indices_dict[(1, 1)][: len(indices_dict[(0, 1)])]
    #     return indices_dict

    @staticmethod
    def _set_minor_ratio(indices_dict, minor_ratio):
        new_length = int(len(indices_dict[(0, 0)]) * minor_ratio)
        indices_dict[(0, 1)] = indices_dict[(0, 1)][:new_length]
        indices_dict[(1, 0)] = indices_dict[(1, 0)][:new_length]
        return indices_dict

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_groups(self) -> int:
        return 4
