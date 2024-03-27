from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from numpy import float128, float16
import os
import random
import numpy as np

import torch
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms

from .spurious_dataset import PrepareDataset, ConceptImageFolder, SpuriousDataset
from torch.utils.data import TensorDataset, ConcatDataset, Subset, DataLoader, Dataset
import torchvision


from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS


# https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html
from pytorch_lightning.trainer.supporters import CombinedLoader

from .waterbirds import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSSubset

transform_dict = {
    "no_aug": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "default": transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1), ratio=(3 / 4, 4 / 3)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "previous": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
}


class SpuriousDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: str,
        batch_size_train: int = 50,
        batch_size_test: int = 100,
        num_workers=8,
        minor_ratio_tr=0.0,
        minor_ratio_val=0.0,
        tr_augmentation="previous",
        tr_shuffle=True,
        **kwargs,
    ):
        super().__init__()
        raise NotImplementedError("minor ratio tr & val is not applied yet.")
        self.save_hyperparameters()

        self.transform_train = transform_dict[self.hparams.tr_augmentation]
        self.transform_test = transform_dict["no_aug"]

    def prepare_data(self):
        PrepareDataset(root=self.hparams.data_dir, dataset=self.hparams.dataset)

    def setup(self, stage=None):
        root = os.path.join(self.hparams.data_dir, self.hparams.dataset, "img")

        self.dataset_train = SpuriousDataset(root=os.path.join(root, "train"), transform=self.transform_train)
        if self.hparams.minor_ratio_tr != 0:
            dataset_minor_train = SpuriousDataset(
                root=os.path.join(root, "minor_train"), transform=self.transform_train
            )

            self.dataset_train = self.combine_dataset(
                self.dataset_train, dataset_minor_train, self.hparams.minor_ratio_tr
            )

        if self.hparams.dataset == "SpuriousCatDogVer2":
            val_minor_direc = "minor_val"
            val_major_direc = "major_val"
        else:
            val_minor_direc = "corrupted_test"
            val_major_direc = "test"

        self.dataset_major_val = SpuriousDataset(
            root=os.path.join(root, val_major_direc), transform=self.transform_test
        )
        self.dataset_minor_val = SpuriousDataset(
            root=os.path.join(root, val_minor_direc), transform=self.transform_test
        )

        self.dataset_major_test = SpuriousDataset(root=os.path.join(root, "test"), transform=self.transform_test)
        self.dataset_minor_test = SpuriousDataset(
            root=os.path.join(root, "corrupted_test"), transform=self.transform_test
        )
        # test acc: cat-canyon/ cat-islet/ dog-canyon/ dog-islet

    def combine_dataset(self, dset1: ImageFolder, dset2: ImageFolder, ratio_of_dset2: float):
        assert len(dset1.imgs) == len(dset2.imgs)

        # What we have to do is to replace the `imgs` in `ImageFolder`.
        dset1_new_imgs = []
        dset2_new_imgs = []
        for label in dset1.class_to_idx.values():
            tmp1 = [data for data in dset1.imgs if data[1] == label]
            tmp2 = [data for data in dset2.imgs if data[1] == label]
            num_minor = int(len(tmp1) * ratio_of_dset2)

            dset1_new_imgs += tmp1[: len(tmp1) - num_minor]
            dset2_new_imgs += tmp2[-num_minor:]

        dset1.samples = dset1_new_imgs
        dset2.samples = dset2_new_imgs

        dset1.imgs = dset1.samples
        dset2.imgs = dset2.samples

        return torch.utils.data.ConcatDataset([dset1, dset2])

    def get_test_dataloader(self):
        major_dataloader = DataLoader(
            self.dataset_major_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        minor_dataloader = DataLoader(
            self.dataset_minor_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return major_dataloader, minor_dataloader

    def get_val_dataloader(self):
        major_dataloader = DataLoader(
            self.dataset_major_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        minor_dataloader = DataLoader(
            self.dataset_minor_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return major_dataloader, minor_dataloader

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        major_dataloader, minor_dataloader = self.get_val_dataloader()
        return [major_dataloader, minor_dataloader]

    def test_dataloader(self):
        major_dataloader, minor_dataloader = self.get_test_dataloader()
        return [major_dataloader, minor_dataloader]

    def predict_dataloader(self):
        return self.val_dataloader()

    def get_num_classes(self):
        num_cls_dict = {
            "SpuriousFlowers17": 10,
            "Spuriousoxford-iiit-pet": 37,
            "SpuriousCatDog": 2,
            "NonSpuriousCatDog": 2,
            "Waterbirds": 2,
            "SpuriousCatDogVer2": 2,
        }

        return num_cls_dict[self.hparams.dataset]

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--data_seed", default=1234, type=int, help="batchsize of data loaders")
        group.add_argument("--num_workers", default=8, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=50, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="~/Data", type=str, help="directory of cifar10 dataset")
        group.add_argument("--minor_ratio", default=0.0, type=float, help="ratio of minor group in training dataset")
        group.add_argument(
            "--tr_augmentation",
            default="previous",
            type=str,
            choices=["no_aug", "default", "previous"],
            help="augmentation method",
        )
        return parser


class SpuriousValTrainDataModule(SpuriousDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError(" this class is deprecated")

        assert self.hparams.dataset == "SpuriousCatDogVer2"

    def setup(self, stage=None):
        root = os.path.join(self.hparams.data_dir, self.hparams.dataset, "img")

        self.dataset_train = SpuriousDataset(root=os.path.join(root, "major_val"), transform=self.transform_train)
        if self.hparams.minor_ratio != 0:
            dataset_minor_train = SpuriousDataset(root=os.path.join(root, "minor_val"), transform=self.transform_train)

            self.dataset_train = self.combine_dataset(
                self.dataset_train, dataset_minor_train, self.hparams.minor_ratio
            )

        val_major_direc = "major_val"
        val_minor_direc = "minor_val"

        self.dataset_major_val = SpuriousDataset(
            root=os.path.join(root, val_major_direc), transform=self.transform_test
        )
        self.dataset_minor_val = SpuriousDataset(
            root=os.path.join(root, val_minor_direc), transform=self.transform_test
        )

        self.dataset_major_test = SpuriousDataset(root=os.path.join(root, "test"), transform=self.transform_test)
        self.dataset_minor_test = SpuriousDataset(
            root=os.path.join(root, "corrupted_test"), transform=self.transform_test
        )
        # test acc: cat-canyon/ cat-islet/ dog-canyon/ dog-islet

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return super().add_data_specific_args(parent_parser)


class WaterbirdsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "None",
        dataset: str = "None",
        batch_size_train: int = 50,
        batch_size_test: int = 100,
        num_workers=8,
        minor_ratio_tr=0.0,
        minor_ratio_val=0.0,
        tr_augmentation="previous",
        tr_shuffle=True,
        **kwargs,
    ):
        """

        metadata: (background, bird, y)
        land=0, water=1
        landbird=0, waterbird=1
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = WaterbirdsDataset(minor_ratio=self.hparams.minor_ratio_tr, root_dir=self.hparams.data_dir)

        self.transform_train = transform_dict[self.hparams.tr_augmentation]
        self.transform_test = transform_dict["no_aug"]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset_train = self.dataset.get_subset(split="train", transform=self.transform_train)
        self.dataset_major_val = self.dataset.get_test_subset(
            split="val", mode="majority", transform=self.transform_test
        )
        self.dataset_minor_val = self.dataset.get_test_subset(
            split="val", mode="minority", transform=self.transform_test, minor_ratio=self.hparams.minor_ratio_val
        )
        self.dataset_major_test = self.dataset.get_test_subset(
            split="test", mode="majority", transform=self.transform_test
        )
        self.dataset_minor_test = self.dataset.get_test_subset(
            split="test", mode="minority", transform=self.transform_test
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self.get_val_dataloader()

    def test_dataloader(self):
        return self.get_test_dataloader()

    def get_val_dataloader(self):
        val_major_dataloader = DataLoader(
            self.dataset_major_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        val_minor_dataloader = DataLoader(
            self.dataset_minor_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return val_major_dataloader, val_minor_dataloader

    def get_test_dataloader(self):
        test_major_dataloader = DataLoader(
            self.dataset_major_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        test_minor_dataloader = DataLoader(
            self.dataset_minor_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return test_major_dataloader, test_minor_dataloader

    def get_num_classes(self):
        return 2

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Data arguments")
        group.add_argument("--data_seed", default=1234, type=int, help="batchsize of data loaders")
        group.add_argument("--num_workers", default=8, type=int, help="number of workers")
        group.add_argument("--batch_size_train", default=50, type=int, help="batchsize of data loaders")
        group.add_argument("--batch_size_test", default=100, type=int, help="batchsize of data loaders")
        group.add_argument("--data_dir", default="/home/jj/Data/WILDS/", type=str, help="directory of cifar10 dataset")
        group.add_argument(
            "--minor_ratio_tr", default=0.0, type=float, help="ratio of minor group in training dataset"
        )
        group.add_argument(
            "--minor_ratio_val", default=0.0, type=float, help="ratio of minor group in validation dataset"
        )
        group.add_argument(
            "--tr_augmentation",
            default="previous",
            type=str,
            choices=["no_aug", "default", "previous"],
            help="augmentation method",
        )
        return parser


class WaterbirdsValTrainDataModule(WaterbirdsDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        dataset_temp1 = self.dataset.get_test_subset(split="val", mode="majority", transform=self.transform_train)
        dataset_temp2 = self.dataset.get_test_subset(split="val", mode="minority", transform=self.transform_train)

        self.dataset_train = torch.utils.data.ConcatDataset([dataset_temp1, dataset_temp2])

        self.dataset_major_val = self.dataset.get_test_subset(
            split="val", mode="majority", transform=self.transform_test
        )
        self.dataset_minor_val = self.dataset.get_test_subset(
            split="val", mode="minority", transform=self.transform_test
        )
        self.dataset_major_test = self.dataset.get_test_subset(
            split="test", mode="majority", transform=self.transform_test
        )
        self.dataset_minor_test = self.dataset.get_test_subset(
            split="test", mode="minority", transform=self.transform_test
        )

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return super().add_data_specific_args(parent_parser)
