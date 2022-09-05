from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataloader import ChestXrayDataset, ChestXrayTestDataset
from mlp import SimpleLinear

BATCH_SIZE = 64
PATCH_SIZE = 64
N_PATCHES = 5


def run_MLP():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=0", "--max_epochs=1", "--max_steps=5", "--val_check_interval=40"])
    train_dataset = ChestXrayDataset(phase="train", crop=True, patch_size=PATCH_SIZE)
    val_dataset = ChestXrayDataset(phase="val", crop=True, patch_size=PATCH_SIZE)
    test_dataset = ChestXrayTestDataset(crop=True, patch_size=PATCH_SIZE, n_per_image=N_PATCHES)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    trainer = Trainer.from_argparse_args(args)
    model = SimpleLinear(in_features=PATCH_SIZE * PATCH_SIZE, out_features=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    outputs = trainer.predict(model, test_dataloader)
    # patch_preds
    pred_patches, targets = zip(*outputs)
    pred_patches = np.concatenate(pred_patches.numpy(), axis=0).astype(np.float32)
    pred_patches = pred_patches.reshape(N_PATCHES, -1)

    preds = []
    for pred_patch in pred_patches:
        values, count = np.unique(pred_patch, return_counts=True)
        idx = np.argmax(count)
        pred = values[idx]
        preds.append(pred)

    count = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            count += 1

    accuracy = count / len(targets) * 100
    print(f"Accuracy is: {accuracy}")


if __name__ == "__main__":
    run_MLP()
