from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass

# from pytorch_lightning import Trainer


@dataclass
class Config:
    def __init__(
        self,
        batch_size: int,
        patch_size: int,
        max_epochs: int,
        learning_rate: float,
        weight_decay: float,
        n_patches: int,
        len_train_dataset: int,
        no_val: bool,
    ) -> None:
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_patches = n_patches
        self.len_train_dataset = len_train_dataset
        self.num_steps = len_train_dataset // batch_size
        self.no_val = no_val

    @staticmethod
    def parser() -> ArgumentParser:
        p = ArgumentParser()
        p.add_argument("--model", help="", default="resnet")
        p.add_argument("--batch_size", help="", default=32)
        p.add_argument("--patch_size", help="", default=128)
        p.add_argument("--max_epochs", help="", default=5)
        p.add_argument("--learning_rate", help="", default=1e-3)
        p.add_argument("--weight_decay", help="", default=1e-5)
        p.add_argument("--n_patches", help="", default=50)
        p.add_argument("--no_val", help="", default=False)
        return p

    # @staticmethod
    # def from_args() -> Config:
    #     parser = Config.parser()
    #     parser = Trainer.add_argparse_args(parser)
    #     args, remaining = parser.parse_known_args()

    #     new = Config(
    #         batch_size=args.batch_size,
    #         patch_size=args.patch_size,
    #         max_epochs=args.max_epochs,
    #         learning_rate=args.learning_rate,
    #         weight_decay=args.weight_decay,
    #         n_patches=args.n_patches,
    #         len_train_dataset=args.len_train_dataset,
    #     )
    #     return new
