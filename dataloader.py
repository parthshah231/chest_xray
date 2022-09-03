from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import io
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop

from constants import NO_VAL, TEST_DIR, TRAIN_DIR, VAL_DIR


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        phase: str,
        crop: bool = False,
        patch_size: int = 64,
        transfom: Callable = None,
        random_erasing: Optional[bool] = False,
        box_size: Optional[int] = 32,
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
        self.random_erasing = random_erasing
        self.box_size = box_size

    def __getitem__(self, idx: int) -> Any:
        # could do all this in preprocess step and directly load the data from there
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
        # probability that it would want to use random erasing for this image
        erase = np.random.choice([True, False], size=1, p=[0.45, 0.55])
        if self.random_erasing and self.box_size and erase:
            # to alter the image / add the erasing box on the image it is
            # necessary to rid of the channel dimension here
            resize_crop_img = resize_crop_img.squeeze(0).numpy()
            # the reason to make the box black is purely because the probability
            # of a black spot on the x-ray is much larger than any other color
            box = np.array([0] * self.box_size * self.box_size).reshape(self.box_size, self.box_size)
            x, y = np.random.randint(0, self.patch_size - self.box_size, size=2)
            resize_crop_img[x : x + self.box_size, y : y + self.box_size] = box
            resize_crop_img = torch.Tensor(resize_crop_img).unsqueeze(0)
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
