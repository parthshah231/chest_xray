import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    BCEWithLogitsLoss,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    ReLU,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchio.transforms import RescaleIntensity


class SimpleConv(LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config
        self.rescale = RescaleIntensity()
        self.relu = ReLU()
        self.dropout = Dropout(0.25)
        self.conv1 = Conv2d(3, 8, kernel_size=(7, 7), padding="same")
        self.batchnorm1 = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 32, kernel_size=(7, 7), padding="same")
        self.batchnorm2 = BatchNorm2d(32)
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(32, 16, kernel_size=(7, 7), padding="same")
        self.batchnorm3 = BatchNorm2d(16)
        self.conv4 = Conv2d(16, 8, kernel_size=(7, 7), padding="same")
        self.batchnorm4 = BatchNorm2d(8)
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        # Dividing 256 by 4 because I am using 2 maxpool layers
        self.fc1 = Linear(
            8 * (self.config["patch_size"] // 4) * (self.config["patch_size"] // 4), 32
        )
        self.fc2 = Linear(32, 1)
        self.bce = BCEWithLogitsLoss()

    def save_configs(self, log_dir: Path) -> None:
        log_dir = Path(log_dir) / "config.json"
        with open(log_dir, "w") as handle:
            json.dump(self.config, handle)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.batchnorm2(self.relu(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.batchnorm3(self.relu(self.conv3(x)))
        x = self.batchnorm4(self.relu(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Any:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Any:
        return self._shared_step(batch, "val")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        # x -> [b,c,h,w]
        # target -> [b, 1]
        x, target = batch
        x = x.squeeze(1)
        x = torch.repeat_interleave(input=x, repeats=3, dim=0)
        # for rescale_intersity it needs 3 dims (x, y, z)
        # if the img is 2D we can add z -> 1 acc. to documentation
        if not self.config["subject_wise"]:
            x = x.unsqueeze(3)
            x_rescaled = self.rescale(x)
            x = x_rescaled.squeeze(3)
        # Since batch size is 1
        x = x.unsqueeze(0)
        preds = self(x)  # [n, 1]
        preds = torch.sigmoid(preds) > 0.5
        return torch.tensor(preds), torch.tensor(target)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        # x      -> [b, c, h, w]
        # target -> [b, 1]
        x, target = batch
        # since the dataset contains grayscale images
        # here the channel is 1 repeat_interleave() works
        # as numpy.repeat() and expands the channels
        # from 1 -> 3
        x = torch.repeat_interleave(input=x, repeats=3, dim=1)
        if not self.config["subject_wise"]:
            x_rescaled = self.rescale(x)
            preds = self(x_rescaled)
        else:
            preds = self(x)
        loss = self.bce(preds, target)
        self.log(f"{phase}/bce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict:
        opt = AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        sched = CosineAnnealingLR(
            optimizer=opt,
            T_max=self.config["max_epochs"]
            * (self.config["len_train_dataset"] // self.config["batch_size"]),
        )
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=sched,
                interval="step",
            ),
        )
