from typing import Any, Optional, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor


class LitConv(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass

    def training_step(self, *args, **kwargs) -> Any:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Any:
        return super().validation_step(*args, **kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Any:
        pass

    def configure_optimizers(self):
        return super().configure_optimizers()
