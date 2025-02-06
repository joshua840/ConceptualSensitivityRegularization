import os
from PIL import Image
from torchvision import transforms
from .balancing_group_dataset import GroupDataset


class CatDog(GroupDataset):
    def __init__(
        self, data_dir, split, subsample_what=None, duplicates=None, minor_ratio=None
    ):
        """
        y: 0: cat, 1: dog
        a: 0: water, 1: land
        (0,0): water, cat
        (0,1): water, dog
        (1,0): land, cat
        (1,1): land, dog
        """
        root = os.path.join(data_dir, "SpuriousCatDog")
        metadata = os.path.join(root, "metadata_catdog.csv")

        if split != "tr":
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.7, 1.0), ratio=(0.75, 1.33333333)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        super().__init__(
            split, root, metadata, transform, subsample_what, duplicates, minor_ratio
        )
        self.data_type = "images"

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
    def _set_minor_ratio(indices_dict, minor_ratio):
        new_length = int(len(indices_dict[(1, 1)]) * minor_ratio)
        indices_dict[(0, 1)] = indices_dict[(0, 1)][:new_length]

        new_length = int(len(indices_dict[(0, 0)]) * minor_ratio)
        indices_dict[(1, 0)] = indices_dict[(1, 0)][:new_length]
        return indices_dict

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_groups(self) -> int:
        return 4
