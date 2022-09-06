from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from callbacks import get_callbacks
from config import Config
from constants import NO_VAL
from dataloader import ChestXrayDataset
from resnet import Resnet

# from torchmetrics import Accuracy


# Get them from command line!
BATCH_SIZE = 32
PATCH_SIZE = 256
MAX_EPOCHS = 20
# 3e-4 for smaller batch-size
# 1e-3 or 1e-4 for bigger batch-size
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
N_PATCHES = 3
RESNET_VERSION = 18
OUT_FEATURES = 1

# random_erasing
PROB = 0.45
BOX_SIZE = 64

# NO_VAL = True (quick-test)
# SUBJECT_WISE = True (rescale intenstiy w.r.t. subject)

# For AdamW
# learning rate proportional to square root of batch_size (theoretically)


def run_resnet() -> None:
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=0", "--max_epochs=20", "--val_check_interval=40"])

    # Record parameters for augmentations (random_erasing) as well
    train_dataset = ChestXrayDataset(
        phase="train",
        crop=True,
        patch_size=PATCH_SIZE,
        random_erasing=True,
        prob=PROB,
        box_size=BOX_SIZE,
    )
    val_dataset = ChestXrayDataset(phase="val", crop=True, patch_size=PATCH_SIZE)
    # test_dataset = ChestXrayTestDataset(crop=True, patch_size=PATCH_SIZE, n_per_image=N_PATCHES)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    predict_dataloader = DataLoader(val_dataset, batch_size=1)

    config = Config(
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        n_patches=N_PATCHES,
        len_train_dataset=len(train_dataloader),
        no_val=NO_VAL,
        random_erasing=True,
        prob=PROB,
        box_size=BOX_SIZE,
    )
    trainer = Trainer.from_argparse_args(args, callbacks=get_callbacks())
    model = Resnet(out_features=OUT_FEATURES, config=config, resnet_version=RESNET_VERSION)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model.save_configs(log_dir=trainer.log_dir)
    i = 0
    outputs = []
    while i < N_PATCHES:
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
    pred_patches = preds.reshape(-1, N_PATCHES)
    targets = targets.reshape(-1, N_PATCHES)
    targets = targets[:, :1]
    preds = []
    for pred_patch in pred_patches:
        values, count = np.unique(pred_patch, return_counts=True)
        idx = np.argmax(count)
        pred = values[idx]
        preds.append(pred)

    if len(preds) == len(targets):
        # count = np.sum(preds == targets)
        count = 0
        for pred, target in zip(preds, targets):
            if pred == target:
                count += 1
    else:
        ValueError("Please check your shapes")

    # accuracy = Accuracy()
    # accuracy(preds, targets)
    accuracy = count / len(targets) * 100
    print(f"Accuracy is: {accuracy}")


if __name__ == "__main__":
    run_resnet()
