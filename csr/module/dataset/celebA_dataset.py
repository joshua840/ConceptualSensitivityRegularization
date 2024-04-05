import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from .balancing_group_dataset import GroupDataset
from .common_spurious_dataset import CommonSpuriousDataset
from .rrclarc_datasets import get_celeba_biased_dataset


class CelebA(GroupDataset):
    def __init__(
        self, data_dir, split, subsample_what=None, duplicates=None, minor_ratio=None
    ):
        root = os.path.join(data_dir, "celeba/img_align_celeba/")
        metadata = os.path.join(data_dir, "metadata_celeba.csv")

        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        if split != "tr":
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_resolution),
                    transforms.CenterCrop(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        target_resolution,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.33333333),
                        interpolation=2,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        super().__init__(
            split, root, metadata, transform, subsample_what, duplicates, minor_ratio
        )
        self.data_type = "images"
        self._num_classes = 2
        self._num_groups = 4

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

    def _get_indices_dict(self):
        indices_dict = {}
        for attr in range(2):
            for label in range(2):
                indices_dict[(attr, label)] = (
                    ((self.attr == attr) * (self.y == label)).nonzero()[0].squeeze()
                )
        return indices_dict

    @staticmethod
    def _remove_label_bias(indices_dict):
        indices_dict[(0, 0)] = indices_dict[(0, 0)][: len(indices_dict[(1, 1)])]
        indices_dict[(1, 0)] = indices_dict[(1, 0)][: len(indices_dict[(0, 1)])]
        return indices_dict

    @staticmethod
    def _set_minor_ratio(indices_dict, minor_ratio):
        new_length = int(len(indices_dict[(0, 1)]) * minor_ratio)
        indices_dict[(0, 0)] = indices_dict[(0, 0)][:new_length]
        indices_dict[(1, 1)] = indices_dict[(1, 1)][:new_length]
        return indices_dict

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_groups(self) -> int:
        return 4


class CelebAGender(GroupDataset):
    def __init__(
        self, data_dir, split, subsample_what=None, duplicates=None, minor_ratio=None
    ):
        root = os.path.join(data_dir, "celeba/img_align_celeba/")
        metadata = os.path.join(data_dir, "metadata_celeba.csv")

        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        if split != "tr":
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_resolution),
                    transforms.CenterCrop(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        target_resolution,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.33333333),
                        interpolation=2,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        super().__init__(
            split, root, metadata, transform, subsample_what, duplicates, minor_ratio
        )
        self.data_type = "images"
        self._num_classes = 2
        self._num_groups = 4

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))

    def _get_indices_dict(self):
        indices_dict = {}
        for attr in range(2):
            for label in range(2):
                indices_dict[(attr, label)] = (
                    ((self.attr == attr) * (self.y == label)).nonzero()[0].squeeze()
                )
        return indices_dict

    @staticmethod
    def _remove_label_bias(indices_dict):
        indices_dict[(0, 0)] = indices_dict[(0, 0)][: len(indices_dict[(1, 1)])]
        indices_dict[(1, 0)] = indices_dict[(1, 0)][: len(indices_dict[(0, 1)])]
        return indices_dict

    @staticmethod
    def _set_minor_ratio(indices_dict, minor_ratio):
        new_length = int(len(indices_dict[(0, 1)]) * minor_ratio)
        indices_dict[(0, 0)] = indices_dict[(0, 0)][:new_length]
        indices_dict[(1, 1)] = indices_dict[(1, 1)][:new_length]
        return indices_dict

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_groups(self) -> int:
        return 4
