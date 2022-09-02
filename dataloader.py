from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import torch
from skimage import io
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop

from constants import NO_VAL, TEST_DIR, TRAIN_DIR, VAL_DIR

# Make a hashmap for both datasets


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        phase: str,
        crop: bool = False,
        patch_size: int = 64,
        transfom: Callable = None,
    ) -> None:
        super().__init__()
        self.phase = phase
        if self.phase == "train":
            self.img_paths = list(TRAIN_DIR.rglob("*.jpeg"))
        elif self.phase == "val":
            self.img_paths = list(VAL_DIR.rglob("*.jpeg"))
        else:
            ValueError("Please pass train/val/test")
        # if the data is not too big, this loads all data in memory and helps processing
        # x10 faster (generally)
        # self.images = [io.imread(str(img_path)) for img_path in self.img_paths]
        self.crop = crop
        self.patch_size = patch_size
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1}
        self.labels = [self.label_map[img_path.parent.stem] for img_path in self.img_paths]
        self.transform = transfom

    def __getitem__(self, idx: int) -> Any:
        # do all this in preprocess step and directly load the data from there
        image = io.imread(self.img_paths[idx])
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.squeeze()
        if self.crop:
            x = 0
            y = 0
            h, w = image.shape
            h_6 = h // 6
            w_6 = w // 6
            cropped_image = image[y + h_6 :, x + w_6 : w - w_6]
            image = cropped_image
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(torch.float32)
        label = self.labels[idx]
        label = torch.FloatTensor([label])
        resize_crop = RandomResizedCrop(size=(self.patch_size, self.patch_size))
        resize_crop_img = resize_crop(image)
        return resize_crop_img, label

    def __len__(self) -> int:
        return len(self.labels)


class ChestXrayTestDataset(Dataset):
    def __init__(
        self,
        crop: bool = False,
        n_per_image: int = 1,
        patch_size: int = 64,
        transfom: Callable = None,
    ) -> None:
        super().__init__()
        if NO_VAL:
            self.img_paths = list(TEST_DIR.rglob("*.jpeg"))[:10]
        else:
            self.img_paths = list(TEST_DIR.rglob("*.jpeg"))
        self.crop = crop
        self.patch_size = patch_size
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1}
        self.labels = [self.label_map[img_path.parent.stem] for img_path in self.img_paths]
        self.n_per_image = n_per_image
        self.transform = transfom

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        # image = io.imread(self.img_paths[idx//2])
        image = io.imread(self.img_paths[idx])
        # look into different caching methods or
        # do these processes beforehand and store in a location
        # from where you can load it directly if the dataset
        # is too big.
        if self.crop:
            x = 0
            y = 0
            h, w = image.shape
            h_6 = h // 6
            w_6 = w // 6
            cropped_image = image[y + h_6 :, x + w_6 : w - w_6]
            image = cropped_image
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(torch.float32)
        label = self.labels[idx]
        label = torch.FloatTensor([label])
        resize_crop = RandomResizedCrop(size=(self.patch_size, self.patch_size))
        resize_crop_img = resize_crop(image)
        return resize_crop_img, label

    def __len__(self) -> int:
        # len * N_PATCHES
        return len(self.labels)
