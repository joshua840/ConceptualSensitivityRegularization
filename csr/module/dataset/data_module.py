import torch

from torchvision import transforms
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from typing import Optional, Union, List

# from .parser import str2bool

from . import (
    CelebA,
    CelebAGender,
    Waterbirds,
    ColoredMNIST,
    ConceptDataset,
    Dogs,
    CatDog,
)


class DataSubset(Dataset):
    def __init__(
        self, dataset: Dataset, idxs: Union[List, np.ndarray], minor_ratio=None
    ):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, item_idx: int):
        _idx = self.idxs[item_idx]
        return self.dataset[_idx]

    def __len__(self):
        return len(self.idxs)

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def targets(self):
        return self.dataset.targets()[self.idxs]


def get_binary_class_weight(ind_list, num_classes):
    labels, counts = np.unique(ind_list, return_counts=True)
    class_wts = ((counts.sum() - counts) / counts.sum()) * (1 / (num_classes - 1))
    class_wts = class_wts[1] / class_wts[0]
    return class_wts


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "celeba",
        data_dir: str = "~/Data",
        data_seed: int = 1234,
        num_workers: int = 8,
        batch_size_train: int = 50,
        batch_size_test: int = 100,
        minor_ratio: Optional[float] = None,
        subsample_what: Optional[str] = None,
        upsample_count: Optional[int] = None,
        upsample_indices_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            dataset: dataset name
            data_dir: directory of dataset
            data_seed: batchsize of data loaders
            num_workers: number of workers
            batch_size_train: batchsize of data loaders
            batch_size_test: batchsize of data loaders
            minor_ratio: ratio of minor group in training dataset
            subsample_what: subsample groups in training dataset
            upsample_count: upsampling count
            upsample_indices_path: path to upsampling indices
        """
        super().__init__()
        self.save_hyperparameters()

        self._init_dataset(
            self.hparams.dataset,
            self.hparams.data_dir,
            self.hparams.minor_ratio,
            self.hparams.subsample_what,
        )

    def _init_dataset(self, dataset, data_dir, minor_ratio, subsample_what):
        Dataset = {
            "celeba": CelebA,
            "celeba_gender": CelebAGender,
            "waterbirds": Waterbirds,
            "catdog": CatDog,
            "dogs": Dogs,
            "waterbirds_concepts": ConceptDataset,
            "celeba_concepts": ConceptDataset,
            "dogs_concepts": ConceptDataset,
            "celeba_concepts2": ConceptDataset,
        }[dataset]
        if dataset in [
            "celeba",
            "waterbirds",
            "colored_mnist",
            "celeba_gender",
            "dogs",
            "catdog",
        ]:
            return self._init_balancing_group(
                Dataset, data_dir, minor_ratio, subsample_what
            )
        elif dataset in ["isic", "plant"]:
            return self._init_datasubset_group(Dataset, data_dir, minor_ratio)
        elif dataset in [
            "waterbirds_concepts",
            "celeba_concepts",
            "celeba_gender_concepts",
            "celeba_concepts2",
            "dogs_concepts",
        ]:
            return self._concept_dataset_group(root=data_dir, dataset=dataset)
        else:
            raise NameError

    def _concept_dataset_group(self, root, dataset):
        # dataset

        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        transform_tr = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution, scale=(0.7, 1.0), ratio=(0.75, 1.33333333)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform_te = transforms.Compose(
            [
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = ConceptDataset(
            root=root, dataset=dataset, transform=transform_tr
        )
        self.val_dataset = ConceptDataset(
            root=root, dataset=dataset, transform=transform_te
        )
        self.test_dataset = ConceptDataset(
            root=root, dataset=dataset, transform=transform_te
        )
        self.num_classes = 2

    def _init_balancing_group(self, Dataset, data_dir, minor_ratio, subsample_what):
        if self.hparams.upsample_indices_path is None:
            upsample_indices = None
        else:
            upsample_indices = (
                torch.load(self.hparams.upsample_indices_path, map_location="cpu")
                * self.hparams.upsample_count
            )

        # dataset
        kwargs = (
            {"upsample_indices": upsample_indices}
            if Dataset == ColoredMNIST
            else {"duplicates": upsample_indices}
        )
        self.train_dataset = Dataset(
            data_dir,
            split="tr",
            minor_ratio=minor_ratio,
            subsample_what=subsample_what,
            **kwargs,
        )
        self.val_dataset = Dataset(data_dir, split="va")
        self.test_dataset = Dataset(data_dir, split="te")
        # num cl/gr
        self.num_classes = self.train_dataset.num_classes
        self.num_groups = self.train_dataset.num_groups
        # class_weights
        self.class_weights = 1

    def _init_datasubset_group(self, Dataset, data_dir, minor_ratio):
        # dataset
        full_dataset = Dataset(data_dir, minor_ratio=minor_ratio)
        self.train_dataset = DataSubset(full_dataset, full_dataset.split_dict["train"])
        self.val_dataset = DataSubset(full_dataset, full_dataset.split_dict["val"])
        self.test_dataset = DataSubset(full_dataset, full_dataset.split_dict["test"])
        # num cl/gr
        self.num_classes = full_dataset.num_classes
        self.num_groups = full_dataset.num_groups
        # class_weights
        self.class_weights = get_binary_class_weight(
            self.train_dataset.targets(), self.num_classes
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            # collate_fn=self.my_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            # collate_fn=self.my_collate,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            # collate_fn=self.my_collate,
            pin_memory=True,
        )
        return val_dataloader, test_dataloader

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            # collate_fn=self.my_collate,
            pin_memory=True,
        )

    def generation_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def my_collate(self, batch):
        # Transform is already done within dataset __getitem__()
        return batch
