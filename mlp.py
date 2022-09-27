import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchio.transforms import RescaleIntensity


class SimpleLinear(LightningModule):
    def __init__(self, in_features: int, out_features: int, config: Dict) -> None:
        super().__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features)
        self.bce = BCEWithLogitsLoss()
        self.rescale = RescaleIntensity()
        self.config = config

    def save_configs(self, log_dir: Path) -> None:
        log_dir = Path(log_dir) / "config.json"
        with open(log_dir, "w") as handle:
            json.dump(self.config, handle)

    def forward(self, x) -> Tensor:
        x = self.linear(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "val")

    def predict_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        # x -> [b,c,h,w]
        # target -> [b, 1]
        x, target = batch
        if not self.config["subject_wise"]:
            # x = x.unsqueeze(3)
            x = self.rescale(x)
            # x = x_rescaled.squeeze(3)
        # Since batch size is 1
        x = x.reshape(x.shape[0], -1)
        preds = self(x)  # [n, 1]
        preds = torch.sigmoid(preds) > 0.5
        return torch.tensor(preds), torch.tensor(target)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        # x      -> [b, c, h, w]
        # target -> [b, 1]
        x, target = batch

        if not self.config["subject_wise"]:
            x = self.rescale(x)
        x = x.reshape(x.shape[0], -1)
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
