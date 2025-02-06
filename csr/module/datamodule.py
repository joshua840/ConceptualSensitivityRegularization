import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .dataset.feature_data_module import EpochChangeableFeatureDataset

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

from typing import Optional, Union, List

from .dataset import (
    CelebA,
    CelebAGender,
    Waterbirds,
    ColoredMNIST,
    ConceptDataset,
    Dogs,
    CatDog,
    get_celeba_biased_dataset,
)


class DataModule(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        data_type: str,
        data_dir: str = "~/Data",
        data_seed: int = 1234,
        num_workers: int = 2,
        batch_size_train: int = 32,
        batch_size_test: int = 100,
        minor_ratio: Optional[float] = 0.05,
        subsample_what: Optional[str] = None,
        upsample_count: Optional[int] = None,
        upsample_indices_path: Optional[str] = None,
        model: Optional[str] = None,
        nimg_per_concept: Optional[int] = 50,
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
            model: model name that is required for feature dataset
        """
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.data_type in ["raw", "feature"]
        self._set_additional_configs()

    def setup(self, stage=None):
        if self.hparams.data_type == "raw":
            self._setup_raw_dataset()
        elif self.hparams.data_type == "feature":
            self._setup_feature_dataset()

    def _setup_raw_dataset(self):
        if self.hparams.dataset in [
            "celeba",
            "waterbirds",
            "colored_mnist",
            "celeba_gender",
            "dogs",
            "catdog",
        ]:
            Dataset = {
                "celeba": CelebA,
                "celeba_gender": CelebAGender,
                "waterbirds": Waterbirds,
                "catdog": CatDog,
                "dogs": Dogs,
            }[self.hparams.dataset]
            return self._init_balancing_group(
                Dataset,
                self.hparams.data_dir,
                self.hparams.minor_ratio,
                self.hparams.subsample_what,
            )
        elif "concepts" in self.hparams.dataset:
            return self._spurious_dataset_group(
                root=self.hparams.data_dir, dataset=self.hparams.dataset
            )
        elif self.hparams.dataset in ["celeba_collar"]:
            return self._init_rrclarc_group(root=self.hparams.data_dir)
        else:
            raise NameError

    def _spurious_dataset_group(self, root, dataset):
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

        kwargs = {
            "root": root,
            "dataset": dataset,
            "model_name": self.hparams.model,
            "nimg_per_concept": self.hparams.nimg_per_concept,
        }
        self.train_dataset = ConceptDataset(transform=transform_tr, **kwargs)
        self.val_dataset = ConceptDataset(transform=transform_te, **kwargs)
        self.test_dataset = ConceptDataset(transform=transform_te, **kwargs)
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

    def _init_rrclarc_group(self, root):
        # dataset
        full_dataset = get_celeba_biased_dataset([root])

        self.train_dataset = full_dataset.get_subset_by_idxs(full_dataset.idxs_train)
        self.val_dataset = full_dataset.get_subset_by_idxs(full_dataset.idxs_val)
        self.test_dataset = full_dataset.get_subset_by_idxs(full_dataset.idxs_test)

        self.train_dataset.do_augmentation = True
        self.val_dataset.do_augmentation = False
        self.test_dataset.do_augmentation = False

    def _setup_feature_dataset(self):
        if self.hparams.upsample_indices_path is None:
            upsample_indices = None
        else:
            upsample_indices = (
                torch.load(
                    self.hparams.upsample_indices_path, map_location="cpu"
                ).tolist()
                * self.hparams.upsample_count
            )

        kwargs = {
            "root": self.hparams.data_dir,
            "dataset": self.hparams.dataset,
            "model_name": self.hparams.model,
        }
        self.train_dataset = EpochChangeableFeatureDataset(
            split="tr",
            minor_ratio=self.hparams.minor_ratio,
            subsample_what=self.hparams.subsample_what,
            upsample_indices=upsample_indices,
            **kwargs,
        )
        self.val_dataset = EpochChangeableFeatureDataset(split="va", **kwargs)
        self.test_dataset = EpochChangeableFeatureDataset(split="te", **kwargs)

    def train_dataloader(self):
        if self.hparams.module_name == "FeatureGenerator":
            return self.generation_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.hparams.module_name == "JTTMeta":
            return self.jtt_generation_dataloader()
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return val_dataloader, test_dataloader

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def jtt_generation_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
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

    # DataModule part
    def on_train_epoch_start(self) -> None:
        if self.hparams.data_type == "feature":
            self.train_dataset.load_new_x(self.current_epoch)
        return super().on_train_epoch_start()

    def _set_additional_configs(self):
        pos_weight, num_classes, num_attributes, num_groups = {
            "celeba": (1, 2, 2, 4),
            "celeba_collar": (24235 / 2411, 2, 2, 4),
            "celeba_gender": (1, 2, 2, 4),
            "waterbirds": (3682 / 1113, 2, 2, 4),
            "colored_mnist": (1.0, 10, 1, 10),
            "catdog": (1536 / 3193, 2, 2, 4),
            "dogs": (1, 2, 2, 4),
            "waterbirds_concepts": (1, 2, 2, 4),
            "celeba_concepts": (1, 2, 2, 4),
            "dogs_concepts": (1, 2, 2, 4),
            "celeba_concepts2": (1, 2, 2, 4),
            "catdog_concepts": (1, 2, 2, 4),
            "celeba_collar_concepts": (1, 2, 2, 4),
            "celeba_collar_concepts_v2": (16 / 128, 2, 1, 2),
            "catdog_concepts_v2": (1, 2, 2, 2),
            "waterbirds_concepts_v2": (1, 2, 2, 2),
        }[self.hparams.dataset]

        setattr(self.hparams, "pos_weight", pos_weight)
        setattr(self.hparams, "num_classes", num_classes)
        setattr(self.hparams, "num_attributes", num_attributes)
        setattr(self.hparams, "num_groups", num_groups)
