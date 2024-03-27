import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import sys

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
parentdir = "/home/jj/Data/"
sys.path.append(parentdir)
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from numpy import float128, float16
import os
import random
import numpy as np

import torch

# https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html
from pytorch_lightning.trainer.supporters import CombinedLoader

import sys

sys.path.append("/home/jj/Data/wilds")
from wilds.datasets.wilds_dataset import WILDSDataset


def get_num_each_group(df):
    major_g0 = df.loc[df["y"] == 0].loc[df["place"] == 0]
    minor_g0 = df.loc[df["y"] == 0].loc[df["place"] == 1]
    major_g1 = df.loc[df["y"] == 1].loc[df["place"] == 1]
    minor_g1 = df.loc[df["y"] == 1].loc[df["place"] == 0]
    return len(major_g0), len(minor_g0), len(major_g1), len(minor_g1)


def change_ratio(ng1, ng2, minor_ratio):
    sum_ng = ng1 + ng2
    set_g1 = int((1 - minor_ratio) * sum_ng)
    set_g2 = sum_ng - set_g1
    return set_g1, set_g2


def get_df_by_split(df, split):
    df = df.loc[df["split"] == split]
    return df


def get_metadata_change_ratio(df, set_ratio):
    # df for val and test
    df_val = get_df_by_split(df, 1)
    df_test = get_df_by_split(df, 2)

    # df for train
    df = get_df_by_split(df, 0)

    major_g0 = df.loc[df["y"] == 0].loc[df["place"] == 0]
    minor_g0 = df.loc[df["y"] == 0].loc[df["place"] == 1]
    major_g1 = df.loc[df["y"] == 1].loc[df["place"] == 1]
    minor_g1 = df.loc[df["y"] == 1].loc[df["place"] == 0]

    # Original g0=1057, g0=56 -> New g0=1001, g0=56
    num_major_g1 = 1001
    num_minor_g1 = 56
    num_major_g0 = 3314
    num_minor_g0 = 184

    # Get new number accoring to set_ratio
    set_num_major_g0, set_num_minor_g0 = change_ratio(num_major_g0, num_minor_g0, set_ratio)
    set_num_major_g1, set_num_minor_g1 = change_ratio(num_major_g1, num_minor_g1, set_ratio)

    # Cut
    major_g0_new = major_g0.iloc[:set_num_major_g0]
    minor_g0_new = minor_g0.iloc[:set_num_minor_g0]
    major_g1_new = major_g1.iloc[:set_num_major_g1]
    minor_g1_new = minor_g1.iloc[:set_num_minor_g1]

    return pd.concat([major_g0_new, minor_g0_new, major_g1_new, minor_g1_new, df_val, df_test]).sample(
        frac=1
    )  # for shuffle


class WaterbirdsDataset(WILDSDataset):
    """
        The Waterbirds dataset.
        This dataset is not part of the official WILDS benchmark.
        We provide it for convenience and to facilitate comparisons to previous work.

        Supported `split_scheme`:
            'official'

        Input (x):
            Images of birds against various backgrounds that have already been cropped and centered.
    a
        Label (y):
            y is binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.

        Metadata:
            Each image is annotated with whether the background is a land or water background.
            m : (background, bird, y)
                land bacground = 0, water background = 1
                land bird=0, water bird=1

        Original publication:
            @inproceedings{sagawa2019distributionally,
              title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
              author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
              booktitle = {International Conference on Learning Representations},
              year = {2019}
            }

        The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
            @techreport{WahCUB_200_2011,
                    Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
                    Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
                    Year = {2011}
                    Institution = {California Institute of Technology},
                    Number = {CNS-TR-2011-001}
            }
            @article{zhou2017places,
              title = {Places: A 10 million Image Database for Scene Recognition},
              author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
              journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
              year = {2017},
              publisher = {IEEE}
            }

        License:
            The use of this dataset is restricted to non-commercial research and educational purposes.
    """

    _dataset_name = "waterbirds"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
            "compressed_size": None,
        }
    }

    def __init__(
        self,
        transform=None,
        mode="train",
        version=None,
        root_dir="Required",
        download=False,
        split_scheme="official",
        minor_ratio=0.0,
    ):
        assert root_dir != "Required", "Specifying root_dir is required"

        self._version = version
        self._data_dir = self.initialize_data_dir(os.path.join(root_dir, "WILDS"), download)
        self.transfrom = transform

        if not os.path.exists(self.data_dir):
            raise ValueError(f"{self.data_dir} does not exist yet. Please generate the dataset first.")

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        ###################
        # Get metadata_df with given set_ratio
        """
        set_major_minor_group_ratio : ratio of major group (y=1, g=1) / ((y=1, g=1) + (y=1, g=0))
        """
        print("set_major_minor_group_ratio : ", minor_ratio)
        metadata_df = get_metadata_change_ratio(metadata_df, minor_ratio)
        self.metadata_df = metadata_df
        ###################

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df["y"].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack((torch.LongTensor(metadata_df["place"].values), self._y_array), dim=1)
        self._metadata_fields = ["background", "y"]
        self._metadata_map = {
            "background": [" land", "water"],  # Padding for str formatting
            "y": [" landbird", "waterbird"],
        }

        # Extract filenames
        self._input_array = metadata_df["img_filename"].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = metadata_df["split"].values

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["background", "y"]))

        super().__init__(root_dir, download, split_scheme)

    def __len__(self):
        return len(self.y_array)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(self.data_dir, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")

        #         ###### (ADDED) transform #####
        #         x = np.asarray(x)
        #         x = torch.from_numpy(x.swapaxes(1, 2).swapaxes(0, 1))  # .float()
        return x

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]

        return x, y, metadata

    def get_test_subset(self, split, mode="majority", minor_ratio=1, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - mode (str) : Select majority or minority dataset
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]

        def get_group_mask(group):
            mask = self.metadata_array[:, 0:2] == torch.tensor(group)
            return mask[:, 0] * mask[:, 1]

        if mode == "majority":
            group_mask1 = get_group_mask([0, 0])
            group_mask2 = get_group_mask([1, 1])
            mask = (group_mask1 + group_mask2) * split_mask
            split_idx = np.where(mask)[0]

        elif mode == "minority":
            group_mask1 = get_group_mask([0, 1]) * split_mask
            group_mask2 = get_group_mask([1, 0]) * split_mask

            split_idx1 = np.where(group_mask1)[0][: max(1, int(133 * minor_ratio))]
            split_idx2 = np.where(group_mask2)[0][: max(1, int(466 * minor_ratio))]

            split_idx = np.concatenate([split_idx1, split_idx2])
        else:
            raise NameError("mode should be majority or minority")

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSSubset(self, split_idx, transform)
