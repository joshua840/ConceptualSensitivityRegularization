from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
import os
import numpy as np
import PIL
import requests
from tqdm import tqdm
import tarfile
from PIL import Image
import torch

waterbirds_class_list = [
    "bamboo_forest",
    #     'forest/broadleaf',  # Land backgrounds
    "ocean",
    #     'lake/natural'
]

celeba_class_list = [
    "black_hair",
    "blond_hair",
]


dogs_class_list = [
    "field_wild",
    "blond_hair",
]

cat_dog_back_class_list = [
    "canyon",
    "islet",
]

minor_cat_dog_back_class_list = [
    "islet",
    "canyon",
]

flowers17_back_class_list = [
    "badlands",
    "bamboo_forest",
    "butte",
    "canyon",
    "crevasse",
    "dam",
    "islet",
    "rock_arch",
    "tree_farm",
    "wind_farm",
]

oxford_iiit_pet_back_class_list = [
    "abbey",
    "airport_terminal",
    "alley",
    "amphitheater",
    "amusement_park",
    "aquarium",
    "aqueduct",
    "arch",
    "art_gallery",
    "art_studio",
    "assembly_line",
    "attic",
    "badlands",
    "ballroom",
    "bamboo_forest",
    "banquet_hall",
    "bar",
    "baseball_field",
    "basement",
    "basilica",
    "bayou",
    "beauty_salon",
    "bedroom",
    "boardwalk",
    "boat_deck",
    "bookstore",
    "botanical_garden",
    "bowling_alley",
    "boxing_ring",
    "bridge",
    "building_facade",
    "bus_interior",
    "butchers_shop",
    "cafeteria",
    "campsite",
    "candy_store",
    "canyon",
    "castle",
    "cemetery",
    "chalet",
    "classroom",
    "closet",
    "clothing_store",
    "coast",
    "cockpit",
    "coffee_shop",
    "conference_center",
    "conference_room",
    "construction_site",
    "corn_field",
    "corridor",
    "cottage_garden",
    "courthouse",
    "courtyard",
    "creek",
    "crevasse",
]


class ConceptImageFolder(ImageFolder):
    def __init__(
        self,
        root,
        dataset,
        nimg_per_concept=50,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)

        # self.imgs = self.samples
        if dataset == "SpuriousFlowers17":
            self.classes = flowers17_back_class_list
        elif dataset == "Spuriousoxford-iiit-pet":
            self.classes = oxford_iiit_pet_back_class_list
        elif dataset in [
            "SpuriousCatDog",
            "NonSpuriousCatDog",
            "SpuriousCatDogVer2",
            "SpuriousCatDogVer3",
        ]:
            self.classes = cat_dog_back_class_list
        elif dataset == "Waterbirds":
            self.classes = waterbirds_class_list
        elif dataset == "MinorSpuriousCatDog":
            self.classes = minor_cat_dog_back_class_list
        elif dataset == "waterbirds_concepts":
            self.classes = waterbirds_class_list
        elif dataset == "celeba_concepts":
            self.classes = celeba_class_list
        elif dataset == "celeba_concepts2":
            self.classes = celeba_class_list

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        # self.class_to_idx = {name: i for i, name in enumerate(self.classes[:2])}
        # print required. This function is pre-defined function in the parent class.
        self.samples = self.make_dataset(
            self.root, self.class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None
        )

        self.samples_post_processing(dataset, nimg_per_concept)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)  # PIL image
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, 0, 0

    def samples_post_processing(self, dataset, nimg_per_concept):
        if dataset == "SpuriousFlowers17":
            nstart = 50
        elif dataset == "Spuriousoxford-iiit-pet":
            nstart = 200
        elif dataset in [
            "SpuriousCatDog",
            "NonSpuriousCatDog",
            "SpuriousCatDogVer2",
            "SpuriousCatDogVer3",
        ]:
            nstart = 2400
        elif dataset in [
            "Waterbirds",
            "celeba_concepts",
            "celeba_concepts2",
            "waterbirds_concepts",
            "dogs_concepts",
        ]:
            nstart = 0
        elif dataset == "waterbirds_concepts":
            nstart = 0

        samples = []
        for index in self.class_to_idx.values():
            tmp = [data for data in self.samples if data[1] == index]
            samples += tmp[nstart : nstart + nimg_per_concept]

        self.samples = samples


class SpuriousDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.aligned_fore_back = (
            False
            if "minor_train" in root or "corrupted_test" in root or "minor_val" in root
            else True
        )

        self.samples = self._reordering_imgs(self.samples, 201)
        self.imgs = self.imgs

    def _reordering_imgs(self, imgs: list, n_steps: int):
        output_list = []
        for step in range(n_steps):
            output_list += imgs[step::n_steps]
        return output_list

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, group) where target is class_index of the target class.
            group = (background, bird, label)
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.aligned_fore_back:
            group = torch.tensor((target, target, target))
        else:
            if target not in [0, 1]:
                raise NotImplementedError("target should be binary")
            group = torch.tensor((1 - target, target, target))

        return sample, target, group


class PrepareDataset:
    """
    This class will check whether the dataset is exist or not.
    """

    def __init__(self, root="~/Data", dataset="SpuriousFlowers17"):
        place205_path = os.path.join(
            root, "Places205/data/vision/torralba/deeplearning/images256"
        )
        if os.path.isdir(place205_path):
            print("Place205 dataset: exist")
        else:
            print(
                "You have to download Place205 dataset.\nThe url is (http://places.csail.mit.edu/downloadData.html)."
            )
            return

        self.dataset = dataset

        if dataset == "SpuriousFlowers17":
            img_mask_direc = os.path.join(root, "Flowers17")
            if os.path.isdir(os.path.join(img_mask_direc, "jpg")):
                print("Flower17 dataset: exist")
            else:
                print(
                    "You have to download and unzip dataset file. I will do it for you."
                )
                self._flower17_download(img_mask_direc)
        elif dataset == "Spuriousoxford-iiit-pet":
            img_mask_direc = os.path.join(root, "oxford-iiit-pet")
            if os.path.isdir(os.path.join(img_mask_direc, "images")):
                print("oxford-iiit-pet dataset: exist")
            else:
                print(
                    "You have to download and unzip dataset file. I will do it for you."
                )
        elif dataset in [
            "SpuriousCatDog",
            "NonSpuriousCatDog",
            "MinorSpuriousCatDog",
            "SpuriousCatDogVer2",
            "SpuriousCatDogVer3",
        ]:
            img_mask_direc = os.path.join(root, "CatDog")
            if os.path.isdir(os.path.join(img_mask_direc, "images")):
                print("oxford-iiit-pet dataset: exist")
            else:
                print(
                    "You have to download and unzip dataset file. I will do it for you."
                )
        if os.path.isdir(os.path.join(root, self.dataset)):
            print(f"{self.dataset} dataset: exist")
        else:
            self._processing(root)

    def _flower17_download(self, direc):
        url_img = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
        url_mask = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz"

        _get_data_from_url(url_img, direc, "17flowers.tgz")
        _get_data_from_url(url_mask, direc, "trimaps.tgz")

        def _get_data_from_url(url, direc, filename):
            os.makedirs(direc, exist_ok=True)
            response = requests.get(url, stream=True)

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            file_path = os.path.join(direc, filename)
            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")

            _untar(direc, filename)

        def _untar(direc, filename):
            file_path = os.path.join(direc, filename)
            with tarfile.open(file_path) as f:
                f.extractall(direc)

    def _get_flower17_img_dict(self, root):
        from scipy.io import loadmat

        ann = loadmat(os.path.join(root, "trimaps/imlist.mat"))["imlist"].flatten()

        label_list = [
            0,
            4,
            5,
            8,
            9,
            10,
            12,
            14,
            15,
            16,
        ]  # each label has 50 or more masks.

        img_dict = {}
        mask_dict = {}
        for index, label in enumerate(label_list):
            img_dict[index] = []
            mask_dict[index] = []
            for i in range(80 * label + 1, 80 * (label + 1) + 1):
                if i in ann:
                    img_path = os.path.join(root, "jpg", f"image_{str(i).zfill(4)}.jpg")
                    mask_path = os.path.join(
                        root, "trimaps", f"image_{str(i).zfill(4)}.png"
                    )

                    img_dict[index].append(img_path)
                    mask_dict[index].append(mask_path)

                if len(img_dict[index]) == 50:
                    break

        return img_dict, mask_dict

    def _get_oxford_iiit_pet_img_dict(self, root):
        img_path = os.path.join(root, "images")
        mask_path = os.path.join(root, "annotations/trimaps")

        img_list = sorted([i for i in os.listdir(img_path) if "._" not in i])
        mask_list = sorted([i for i in os.listdir(mask_path) if "._" not in i])

        class_list = set(["_".join(i.split("_")[:-1]) for i in img_list])

        img_dict = {
            i: [img_path + "/" + f for f in img_list if label in f]
            for i, label in enumerate(sorted(list(class_list)))
        }
        mask_dict = {
            i: [mask_path + "/" + f for f in mask_list if label in f]
            for i, label in enumerate(sorted(list(class_list)))
        }

        return img_dict, mask_dict

    def _get_cat_dog_img_dict(self, root):
        img_dict, mask_dict = self._get_oxford_iiit_pet_img_dict(root)

        new_img_dict, new_mask_dict = {0: [], 1: []}, {0: [], 1: []}
        for i in range(12):
            new_img_dict[0] += img_dict[i]
            new_img_dict[1] += img_dict[12 + i]
            new_mask_dict[0] += mask_dict[i]
            new_mask_dict[1] += mask_dict[12 + i]

        return new_img_dict, new_mask_dict

    def _load_PIL_img(self, path, convert_rgb=True):
        with open(path, "rb") as f:
            return (
                Image.open(f).copy().convert("RGB")
                if convert_rgb
                else Image.open(f).copy()
            )

    def _processing(self, root):
        root_back = os.path.join(
            root, "Places205/data/vision/torralba/deeplearning/images256"
        )
        save_root = os.path.join(root, self.dataset)

        if self.dataset == "SpuriousFlowers17":
            img_dict, mask_dict = self._get_flower17_img_dict(
                os.path.join(root, "Flowers17")
            )
            back_class_list = flowers17_back_class_list

        elif self.dataset == "Spuriousoxford-iiit-pet":
            img_dict, mask_dict = self._get_oxford_iiit_pet_img_dict(
                os.path.join(root, "oxford-iiit-pet")
            )
            back_class_list = oxford_iiit_pet_back_class_list

        elif self.dataset == "SpuriousCatDog":
            img_dict, mask_dict = self._get_cat_dog_img_dict(
                os.path.join(root, "oxford-iiit-pet")
            )
            back_class_list = cat_dog_back_class_list

        elif self.dataset == "NonSpuriousCatDog":
            img_dict, mask_dict = self._get_cat_dog_img_dict(
                os.path.join(root, "oxford-iiit-pet")
            )
            back_class_list = cat_dog_back_class_list

        elif self.dataset == "MinorSpuriousCatDog":
            img_dict, mask_dict = self._get_cat_dog_img_dict(
                os.path.join(root, "oxford-iiit-pet")
            )
            back_class_list = minor_cat_dog_back_class_list

        num_classes = len(img_dict)
        for label, (img_list, mask_list) in enumerate(
            zip(img_dict.values(), mask_dict.values())
        ):
            num_data = len(img_list)
            shifted_label = (label + 1) % num_classes
            train_threshold = int(num_data * 0.8)

            back_dir_path = os.path.join(root_back, back_class_list[label])
            back_corr_dir_path = os.path.join(root_back, back_class_list[shifted_label])

            back_list = [
                os.path.join(back_dir_path, i)
                for i in sorted(os.listdir(back_dir_path))
            ]
            corr_back_lst = [
                os.path.join(back_corr_dir_path, i)
                for i in sorted(os.listdir(back_corr_dir_path))
            ]

            for j, (img_path, mask_path, back_path, corr_back_path) in enumerate(
                zip(img_list, mask_list, back_list, corr_back_lst)
            ):
                mode = "train" if j < train_threshold else "test"

                mask_save_root = os.path.join(save_root, "mask", mode, str(label))
                spuripus_save_root = os.path.join(save_root, "img", mode, str(label))
                os.makedirs(mask_save_root, exist_ok=True)
                os.makedirs(spuripus_save_root, exist_ok=True)

                if mode == "test":
                    corrupted_spuripus_save_root = os.path.join(
                        save_root, "img", "corrupted_test", str(label)
                    )
                    os.makedirs(corrupted_spuripus_save_root, exist_ok=True)

                # The loaded flower image resolution is different from the background image. We have to fit it.
                mask = self._load_PIL_img(mask_path, convert_rgb=False)
                img = self._load_PIL_img(img_path)
                ## added for temporal experiments
                if (
                    self.dataset == "NonSpuriousCatDog"
                    and mode == "train"
                    and j > train_threshold // 2
                ):
                    back = self._load_PIL_img(corr_back_path).resize(mask.size)
                else:
                    back = self._load_PIL_img(back_path).resize(mask.size)
                if mode == "test":
                    cback = self._load_PIL_img(corr_back_path).resize(mask.size)

                # We reduce the size of mask and flower img, because the flowers occupy too much region in img.
                size_o = mask.size
                size_r = (size_o[0] // 2, size_o[1] // 2)
                mask = mask.resize(size_r, resample=Image.BICUBIC)
                img = img.resize(size_r, resample=Image.BICUBIC)

                mask = np.array(mask)
                img = np.array(img)
                back = np.array(back)

                rand_w = np.random.randint(0, size_r[1])
                rand_h = np.random.randint(0, size_r[0])

                mask = np.pad(
                    mask,
                    (
                        (rand_w, size_o[1] - size_r[1] - rand_w),
                        (rand_h, size_o[0] - size_r[0] - rand_h),
                    ),
                    "constant",
                    constant_values=0,
                )
                img = np.pad(
                    img,
                    (
                        (rand_w, size_o[1] - size_r[1] - rand_w),
                        (rand_h, size_o[0] - size_r[0] - rand_h),
                        (0, 0),
                    ),
                    "constant",
                    constant_values=0,
                )

                # Now, the img, mask, and background img has same resolution. We need to them by using `img * mask + (1 - mask) * back`.
                bin_mask = (mask == 1).reshape(back.shape[0], back.shape[1], 1)
                spurious_img = np.uint8(img * bin_mask + (1 - bin_mask) * back)
                img_save_path = os.path.join(
                    spuripus_save_root,
                    "image_" + str(label) + "_" + str(j).zfill(3) + ".jpg",
                )
                Image.fromarray(spurious_img).convert("RGB").save(img_save_path)

                if mode == "test":
                    cback = np.array(cback)
                    spurious_img = np.uint8(img * bin_mask + (1 - bin_mask) * cback)
                    img_save_path = os.path.join(
                        corrupted_spuripus_save_root,
                        "image_" + str(label) + "_" + str(j).zfill(3) + ".jpg",
                    )
                    Image.fromarray(spurious_img).convert("RGB").save(img_save_path)

                # save mask
                bin_mask = np.uint8(bin_mask).squeeze()
                mask_save_path = os.path.join(
                    mask_save_root,
                    "image_" + str(label) + "_" + str(j).zfill(3) + ".png",
                )
                Image.fromarray(bin_mask).save(mask_save_path)
