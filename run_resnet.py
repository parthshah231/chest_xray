import json
from argparse import ArgumentParser
from gc import callbacks
from typing import Dict

import numpy as np
import sklearn
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

from callbacks import get_callbacks
from config import Config
from constants import NO_VAL, ROOT
from dataloader import ChestXrayDataset
from resnet import Resnet

# from torchmetrics import Accuracy


# Get them from command line!
# BATCH_SIZE = 32
# PATCH_SIZE = 256
# MAX_EPOCHS = 20
# 3e-4 for smaller batch-size
# 1e-3 or 1e-4 for bigger batch-size
# LEARNING_RATE = 3e-4
# WEIGHT_DECAY = 1e-4
# N_PATCHES = 11
# RESNET_VERSION = 18
# OUT_FEATURES = 1

# random_erasing
# PROB = 0.45
# BOX_SIZE = 64

# NO_VAL = True (quick-test)
# SUBJECT_WISE = True (rescale intenstiy w.r.t. subject)

# For AdamW
# learning rate proportional to square root of batch_size (theoretically)


def run_resnet() -> None:
    with open(ROOT / "config.json", "r") as json_file:
        config_dict = json.loads(json_file.read())
    # with open(ROOT / "trainer_config.json", "r") as json_file:
    #     trainer_dict = json.loads(json_file.read())
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=0", "--max_epochs=20", "--val_check_interval=40"])

    # Record parameters for augmentations (random_erasing) as well
    train_dataset = ChestXrayDataset(
        phase="train",
        crop=True,
        patch_size=config_dict["patch_size"],
        random_erasing=config_dict["random_erasing"],
        prob=config_dict["probability"],
        box_size=config_dict["box_size"],
    )
    val_dataset = ChestXrayDataset(phase="val", crop=True, patch_size=config_dict["patch_size"])
    # test_dataset = ChestXrayTestDataset(crop=True, patch_size=PATCH_SIZE, n_per_image=N_PATCHES)

    train_dataloader = DataLoader(train_dataset, batch_size=config_dict["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config_dict["batch_size"])
    predict_dataloader = DataLoader(val_dataset, batch_size=1)

    config = Config(
        batch_size=config_dict["batch_size"],
        patch_size=config_dict["patch_size"],
        max_epochs=config_dict["max_epochs"],
        learning_rate=config_dict["learning_rate"],
        weight_decay=config_dict["weight_decay"],
        n_patches=config_dict["n_patches"],
        len_train_dataset=len(train_dataloader),
        no_val=config_dict["no_val"],
        random_erasing=config_dict["random_erasing"],
        prob=config_dict["probability"],
        box_size=config_dict["box_size"],
    )
    trainer = Trainer.from_argparse_args(args, callbacks=get_callbacks())
    model = Resnet(
        out_features=config_dict["out_features"], config=config, resnet_version=config_dict["resnet_version"]
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model.save_configs(log_dir=trainer.log_dir)
    i = 0
    outputs = []
    while i < config_dict["n_patches"]:
        output = trainer.predict(
            model,
            predict_dataloader,
            ckpt_path="best" if not NO_VAL else None,
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
