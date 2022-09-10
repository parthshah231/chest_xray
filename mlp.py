from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear, Sigmoid
from torch.optim import AdamW


class SimpleLinear(LightningModule):
    def __init__(self, in_features: int, out_features: int, config: Dict) -> None:
        super().__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features)
        self.bce = BCEWithLogitsLoss()
        self.config = config

    def forward(self, x) -> Tensor:
        x = self.linear(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> Tensor:
        return self._shared_step(batch, "val")

    def predict_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        # x -> [b, c, h, w]
        # target -> [b, 1]
        x, target = batch
        x = torch.stack(x)
        x = x.view(x.shape[0], -1)
        preds = self(x)
        return preds, target

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        # x      -> [b, c, h, w]
        # target -> [b, 1]
        x, target = batch
        x = x.view(x.shape[0], -1)
        loss = self(x)
        self.log(f"{phase}/bce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict:
        opt = AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        return [opt]
