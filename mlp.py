from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear, Sigmoid
from torch.optim import AdamW

from config import Config


class SimpleLinear(LightningModule):
    def __init__(self, in_features: int, out_features: int, config: Config) -> None:
        super().__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features)
        self.bce = BCEWithLogitsLoss()
        self.config = config

    def forward(self, x):
        x = self.linear(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "val")

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # x -> List[Tensor] | x[0].shape -> [1, 1, 64, 64]
        # target -> [1, 1]
        x, target = batch
        x = torch.stack(x)  # x -> [n, 1, 1, 64, 64] n -> number of patches
        x = x.view(x.shape[0], -1)  # [n, 4096]
        preds = self(x)  # [n, 1]
        return preds, target

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Any:
        # x      -> [16, 1, 64, 64]
        # target -> [16, 1]
        x, target = batch
        x = x.view(x.shape[0], -1)  # [16, 4096]
        loss = self(x)
        self.log(f"{phase}/bce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return [opt]
