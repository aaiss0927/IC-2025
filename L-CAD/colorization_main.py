import sys, argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time
import torch
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from share import *


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-m",
        "--multicolor",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-s",
        "--usesam",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = get_parser()

    ckpt_cb = ModelCheckpoint(
        filename="{step:08d}",
        save_top_k=-1,
        every_n_train_steps=500,
    )

    callbacks = [
        ckpt_cb,
        LearningRateMonitor("step"),
        ImageLogger(batch_frequency=500),
    ]

    if args.train:
        n_gpu = 1
        init_model_path = "./models/multi_weight.ckpt"

        batch_size = 2
        learning_rate = 1e-7
        sd_locked = False  #
        only_mid_control = False

        model = create_model("./L-CAD/configs/cldm_v15_ehdecoder.yaml").cpu()

        model.load_state_dict(
            load_state_dict(init_model_path, location="cpu"), strict=False
        )
        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control

        dataset = MyDataset(
            img_dir="./train/input_image",
            caption_path="./train/caption_train.json",
            split="train",
            use_sam=False,
        )

        dataloader = DataLoader(
            dataset, num_workers=0, batch_size=batch_size, shuffle=True
        )
        trainer = pl.Trainer(
            gpus=1,
            precision=32,
            callbacks=callbacks,
            max_epochs=6,
            log_every_n_steps=250,
        )
        # Train!
        trainer.fit(model, dataloader)

    else:  # test or val

        resume_path = "./models/multi_weight.ckpt"

        batch_size = 1

        model = create_model("./L-CAD/configs/cldm_v15_ehdecoder.yaml").cpu()
        model.load_state_dict(load_state_dict(resume_path, location="cpu"))

        trainer = pl.Trainer(
            gpus=1,
            precision=32,
            callbacks=callbacks,
            max_epochs=6,
            log_every_n_steps=250,
        )
        if args.multicolor:  # test demo
            if args.usesam:  # -m -s
                model.usesam = True
                dataset = MyDataset(
                    img_dir="./",
                    caption_path="./test/pairs.json",
                    split="test",
                    use_sam=True,
                )
                dataloader = DataLoader(
                    dataset, num_workers=0, batch_size=batch_size, shuffle=False
                )
                trainer.test(model, dataloader)
            else:  # -m
                model.usesam = False
                dataset = MyDataset(
                    img_dir="example", caption_dir="example", split="test"
                )
                dataloader = DataLoader(
                    dataset, num_workers=0, batch_size=batch_size, shuffle=False
                )
                trainer.test(model, dataloader)
        else:  # val
            model.usesam = False
            dataset = MyDataset(
                img_dir="./",
                caption_path="./validation/caption_validation.json",
                split="val",
            )  #
            dataloader = DataLoader(
                dataset, num_workers=0, batch_size=batch_size, shuffle=False
            )
            trainer.test(model, dataloader)
