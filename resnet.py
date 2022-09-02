import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchio.transforms import RescaleIntensity

from config import Config


class LitResnet(LightningModule):
    def __init__(self, out_features: int, config: Config, resnet_version: int = 18) -> None:
        super().__init__()
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        self.out_features = out_features
        self.bce = BCEWithLogitsLoss()
        self.resnet_version = resnet_version
        self.resnet_model = resnets[resnet_version](pretrained=True)
        linear_size = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = Linear(linear_size, out_features)
        self.config = config

    def save_configs(self, log_dir: Path):
        log_dir = Path(log_dir) / "config.json"
        self.config.__dict__["resnet_version"] = self.resnet_version
        json.dump(self.config.__dict__, open(log_dir, "w"))
        # DataFrame(self.config).to_json(log_dir_version)

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.resnet_model(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Optional[Tensor]:
        return self._shared_step(batch, "val")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # x -> List[Tensor] | x[0].shape -> [1, 1, 64, 64]
        # target -> [1, 1]
        x, target = batch
        x = x.squeeze(1)
        x = torch.repeat_interleave(input=x, repeats=3, dim=0)
        # for rescale_intersity it needs 3 dims (x, y, z)
        # if the img is 2D we can add z -> 1 acc. to documentation
        x = x.unsqueeze(3)
        rescale = RescaleIntensity()
        x_rescaled = rescale(x)
        x = x_rescaled.squeeze(3)
        # Since batch size is 1
        x = x.unsqueeze(0)
        preds = self(x)  # [n, 1]
        preds = torch.sigmoid(preds) > 0.5
        return torch.tensor(preds), torch.tensor(target)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Any:
        # x      -> [16, 1, 64, 64]
        # target -> [16, 1]
        x, target = batch
        # since the dataset contains grayscale images
        # here the channel is 1 repeat_interleave() works
        # as numpy.repeat() and expands the channels
        # from 1 -> 3
        x = torch.repeat_interleave(input=x, repeats=3, dim=1)
        rescale = RescaleIntensity()
        x_rescaled = rescale(x)
        preds = self(x_rescaled)  # [16, 1]
        loss = self.bce(preds, target)
        self.log(f"{phase}/bce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        sched = CosineAnnealingLR(optimizer=opt, T_max=self.config.max_epochs * self.config.num_steps)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=sched,
                interval="step",
            ),
        )
