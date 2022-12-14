# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on


import json
from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np
import sklearn
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models.resnet import Resnet
from utils.callbacks import get_callbacks
from utils.constants import ROOT
from utils.dataloader import ChestXrayDataset

# For AdamW
# learning rate proportional to square root of batch_size (theoretically)
# 3e-4 for smaller batch-size
# 1e-3 or 1e-4 for bigger batch-size


def run_resnet() -> None:
    with open(ROOT / "config.json", "r") as json_file:
        config_dict = json.loads(json_file.read())
    with open(ROOT / "trainer_config.json", "r") as json_file:
        trainer_config_dict = json.loads(json_file.read())
    trainer_args = Namespace(**trainer_config_dict)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    trainer_args, _ = parser.parse_known_args(namespace=trainer_args)

    config_dict["max_epochs"] = trainer_config_dict["max_epochs"]

    train_dataset = ChestXrayDataset(
        config=config_dict,
        phase="train",
        crop=True,
        patch_size=config_dict["patch_size"],
        random_erasing=config_dict["random_erasing"],
        prob=config_dict["probability"],
        box_size=config_dict["box_size"],
    )
    val_dataset = ChestXrayDataset(
        config=config_dict,
        phase="val",
        crop=True,
        patch_size=config_dict["patch_size"],
    )
    # test_dataset = ChestXrayTestDataset(crop=True, patch_size=PATCH_SIZE, n_per_image=N_PATCHES)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config_dict["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict["batch_size"])
    predict_dataloader = DataLoader(val_dataset, batch_size=1)

    config_dict["len_train_dataset"] = len(train_dataloader)
    trainer = Trainer.from_argparse_args(
        args=trainer_args,
        callbacks=get_callbacks(config=config_dict),
        max_steps=5 if config_dict["no_val"] else -1,
        log_every_n_steps=5,
    )
    model = Resnet(
        out_features=config_dict["out_features"],
        config=config_dict,
        resnet_version=config_dict["resnet_version"],
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model.save_configs(log_dir=trainer.log_dir)
    i = 0
    outputs = []
    while i < config_dict["n_patches"]:
        output = trainer.predict(
            model,
            predict_dataloader,
            ckpt_path="best" if not config_dict["no_val"] else None,
        )
        outputs.append(output)
        i += 1
    outputs_tensor = torch.Tensor(outputs)
    outputs_tensor = outputs_tensor.view(-1, 2)
    preds, targets = outputs_tensor[:, :1], outputs_tensor[:, 1:]
    pred_patches = preds.reshape(-1, config_dict["n_patches"])
    targets = targets.reshape(-1, config_dict["n_patches"])
    targets = targets[:, :1]
    preds = []
    for pred_patch in pred_patches:
        values, count = np.unique(pred_patch, return_counts=True)
        idx = np.argmax(count)
        pred = values[idx]
        preds.append(pred)

    preds = np.asarray(preds).reshape(-1, 1)
    recall = recall_score(y_true=targets, y_pred=preds, average="binary")
    precision = precision_score(y_true=targets, y_pred=preds, average="binary")
    f1_score = sklearn.metrics.f1_score(y_true=targets, y_pred=preds)
    accuracy = accuracy_score(y_true=targets, y_pred=preds)

    print("---------------- Results ----------------")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    run_resnet()
