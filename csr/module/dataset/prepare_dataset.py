import os
import numpy as np
import requests
from tqdm import tqdm
import tarfile
from PIL import Image
import pandas as pd
import yaml


def crop_and_resize(source_img, target_img):
    # referenced from https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (
            target_width,
            int((target_width / source_width) * source_height),
        )
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (
                int((target_height / source_height) * source_width),
                target_height,
            )
            assert (height_resize[0] >= target_width) and (
                height_resize[1] >= target_height
            )
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize(
        (target_width, target_height), Image.LANCZOS
    )
    return source_resized


def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask

    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined


def resize_and_pad(img, ratio):
    """
    Resize img to ratio and pad with black at center of the original image
    """
    width = img.size[0]
    height = img.size[1]
    img_resized = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)

    # Pad with black
    img_padded = Image.new("RGB", (width, height), (0, 0, 0))
    img_padded.paste(
        img_resized,
        ((width - img_resized.size[0]) // 2, (height - img_resized.size[1]) // 2),
    )

    return img_padded


class PrepareDataset:
    """
    This class will check whether the dataset is exist or not.
    """

    def __init__(
        self,
        places365_root,
        save_root,
        oxford_iiit_pet_root="~/Data",
        dataset="SpuriousCatDog",
        download=False,
    ):
        self.dataset = dataset
        if os.path.isdir(os.path.join(save_root, self.dataset)):
            print(f"{self.dataset} dataset is already prepared")
            return
        else:
            self.bg_dir = os.path.join(places365_root, "Places365/data_large")
            self.fg_dir = os.path.join(oxford_iiit_pet_root, "oxford-iiit-pet/images")
            self.mask_dir = os.path.join(
                oxford_iiit_pet_root, "oxford-iiit-pet/annotations/trimaps"
            )

            if os.path.isdir(self.bg_dir):
                print("Places365 dataset: exist")
            else:
                raise FileNotFoundError("Places365 dataset is not exist.")

            if os.path.isdir(self.fg_dir):
                print("oxford-iiit-pet images: exist")
            else:
                if download:
                    self._download_oxford_iiit_pet(oxford_iiit_pet_root)
                else:
                    raise FileNotFoundError(
                        "You have to download oxford-iiit-pet dataset. Hint: set download=True"
                    )

            return self._processing(save_root)

    def _download_oxford_iiit_pet(self, root):
        root = os.path.join(root, "oxford-iiit-pet")
        for name in ["images", "annotations"]:
            print(f"Downloading oxford-iiit-pet {name} dataset...")
            url = f"http://www.robots.ox.ac.uk/~vgg/data/pets/data/{name}.tar.gz"
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(os.path.join(root, f"{name}.tar.gz"), "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                print("ERROR, something went wrong")
                return
            print(f"Downloading oxford-iiit-pet {name} dataset...done")
            print(f"Unzipping oxford-iiit-pet {name} dataset...")
            tar = tarfile.open(os.path.join(root, f"{name}.tar.gz"), "r:gz")
            tar.extractall(root)
            tar.close()
            print(f"Unzipping oxford-iiit-pet {name} dataset...done")

    def _create_df(self):
        fg_list = sorted([i for i in os.listdir(self.fg_dir) if ".jpg" in i])
        df = pd.DataFrame(fg_list, columns=["fg_filename"])

        df["species"] = df["fg_filename"].apply(lambda x: "_".join(x.split("_")[:-1]))
        df["y"] = df["species"].apply(lambda x: 0 if x[0].isupper() else 1)

        # set 'split' and 'place' of df.
        # The correlation between 'place' and 'y' is 0.95 for training dataset
        # The correlation between 'place' and 'y' is 0.9 for validation dataset
        df["split"] = "train"
        df["place"] = 0
        for species in df["species"].unique():
            species_df = df[df["species"] == species]
            test_idx = species_df.sample(frac=0.2, random_state=42).index
            val_idx = species_df.drop(test_idx).sample(frac=0.2, random_state=42).index
            train_idx = species_df.drop(test_idx).drop(val_idx).index

            df.loc[test_idx, "split"] = "test"
            df.loc[val_idx, "split"] = "val"

            # Set 'place'
            label0_idx = df[df["y"] == 0].index
            label1_idx = df[df["y"] == 1].index

            df.loc[label0_idx.intersection(train_idx), "place"] = np.random.choice(
                [0, 1], size=len(label0_idx.intersection(train_idx)), p=[0.95, 0.05]
            )
            df.loc[label1_idx.intersection(train_idx), "place"] = np.random.choice(
                [0, 1], size=len(label1_idx.intersection(train_idx)), p=[0.05, 0.95]
            )
            df.loc[label0_idx.intersection(val_idx), "place"] = np.random.choice(
                [0, 1], size=len(label0_idx.intersection(val_idx)), p=[0.5, 0.5]
            )
            df.loc[label1_idx.intersection(val_idx), "place"] = np.random.choice(
                [0, 1], size=len(label1_idx.intersection(val_idx)), p=[0.5, 0.5]
            )

        test_idx = df[df["split"] == "test"].index
        df.loc[test_idx, "place"] = 0
        test_df = df.loc[test_idx].copy(deep=True)
        test_df["place"] = 1

        df = pd.concat([df, test_df], axis=0)
        df = df.reset_index()

        with open("configs/dataset/waterbirds_used_bgs.yaml", "r") as f:
            config_waterbirds_bg = yaml.load(f, Loader=yaml.FullLoader)

        places_dict = {
            0: config_waterbirds_bg["water"],
            1: config_waterbirds_bg["land"],
        }

        df.loc[df["place"] == 0, "place_filename"] = places_dict[0][
            : sum(df["place"] == 0)
        ]
        df.loc[df["place"] == 1, "place_filename"] = places_dict[1][
            : sum(df["place"] == 1)
        ]

        return df

    def _load_PIL_img(self, path, convert_rgb=True):
        with open(path, "rb") as f:
            return (
                Image.open(f).copy().convert("RGB")
                if convert_rgb
                else Image.open(f).copy()
            )

    def _processing(self, root):
        dataset_save_path = os.path.join(root, self.dataset)
        os.makedirs(dataset_save_path, exist_ok=True)

        df = self._create_df()

        for i, row in df.iterrows():
            fg_fname = row["fg_filename"]
            bg_fname = row["place_filename"]

            fg_path = os.path.join(self.fg_dir, fg_fname)
            bg_path = os.path.join(self.bg_dir, bg_fname)
            mask_path = os.path.join(self.mask_dir, fg_fname.replace(".jpg", ".png"))

            fg = Image.open(fg_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            bg = Image.open(bg_path).convert("RGB")

            fg_np = np.asarray(resize_and_pad(fg, 0.7))
            mask_np = np.float32(np.asarray(resize_and_pad(mask, 0.7)) == 1)

            fg_only = Image.fromarray(np.around(fg_np * mask_np).astype(np.uint8))
            img = combine_and_mask(bg, mask_np, fg_only)

            # Save img
            save_filename = f"{row['fg_filename'].split('.')[0]}_{'_'.join(row['place_filename'].split('/'))}"
            # print(os.path.join(root, self.dataset, save_filename))
            # print(row["place_filename"])
            img.save(os.path.join(root, self.dataset, save_filename))
            # row["img_filename"] = save_filename
            df.loc[i, "img_filename"] = save_filename

            print(save_filename)

        # metadata.csv contains all information
        df.to_csv(
            os.path.join(root, self.dataset, "metadata.csv"),
            index=False,
        )
        df["filename"] = df["img_filename"]
        df["a"] = df["place"]
        df["split"] = df["split"].map({"train": 0, "val": 1, "test": 2})
        df = df[["filename", "split", "y", "a"]]
        # metadata_catdog.csv will be used by DataModule
        df.to_csv(os.path.join(root, self.dataset, "metadata_catdog.csv"), index=False)

        return


if __name__ == "__main__":
    PrepareDataset(root="/media/disk1/Data", dataset="SpuriousCatDog")
