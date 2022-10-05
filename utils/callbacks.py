from typing import Dict

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_callbacks(config: Dict):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(
        monitor="val/bce",
        patience=5,
        mode="min",
        check_on_train_epoch_end=True,
    )
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}_{val_loss:1.3f}",
        monitor="val/bce",
        save_last=True,
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=True,
        save_weights_only=False,
        every_n_epochs=1,
    )

    callbacks_ = [
        lr_monitor,
        early_stopping,
        model_checkpoint,
    ]

    if config["no_val"]:
        callbacks_.remove(early_stopping)
        callbacks_.remove(model_checkpoint)

    return callbacks_
