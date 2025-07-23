"""Fine‑tuning / evaluation entry‑point for the L‑CAD colorization model.

Example usages
--------------
Train from ImageNet‑initialised weights (16‑batch, 2 GPUs):

python ./L-CAD/colorization_main.py             \
    --train                                     \
    --img_root  ./train/gt_image                \
    --caption_root ./train                      \
    --init_ckpt ./prt_ckpt/coco_weight.ckpt     \
    --gpus 1                                    \
    --batch 1                                   \
    --lr 1e-5

python ./L-CAD/colorization_main.py             \
    --train                                     \
    --img_root  ./train_debug/gt_image                \
    --caption_root ./train_debug                      \
    --init_ckpt /shared/home/kdd/HZ/inha-challenge/logs/20250722‑140601/hpc_ckpt_3_5642_0.004.ckpt     \
    --gpus 1                                    \
    --batch 1                                   \
    --lr 1e-5
-------------------------------------------------
nohup python ./L-CAD/colorization_main.py       \
    --train                                     \
    --img_root  ./train/gt_image                \
    --caption_root ./train                      \
    --init_ckpt ./prt_ckpt/coco_weight.ckpt     \
    --gpus 1                                    \
    --batch 1                                   \
    --lr 1e-5                                   \
> output.log 2>&1 &

Resume training:
    python colorization_finetune.py --train --resume_ckpt logs/last.ckpt

Validation / Test (single‑GPU):
    python colorization_finetune.py \
        --img_root ./val                               \
        --caption_root ./val                           \
        --mode val                                     \
        --resume_ckpt logs/best.ckpt

Notes
-----
* The script assumes the following directory layout for *each* split (**train**, **val**, **test**).

        <img_root>/input_image/  XXX.jpg  (... gray images)
        <img_root>/gt_image/     XXX.jpg  (... colour refs; not needed for test)
        <caption_root>/pairs.json            (... or caption_train/val.json)

  Feel free to point `img_root` / `caption_root` to different locations.
* ``--use_sam`` will automatically expect masks under ``sam_mask/select_masks/<ID>/``.
"""

import types
from torch.optim import AdamW
import argparse
import os
from pathlib import Path
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def get_parser():
    p = argparse.ArgumentParser(description="Fine‑tune / evaluate L‑CAD model")

    # --- mode -----------------------------------------------------------------
    p.add_argument("--train", action="store_true", help="run training")
    p.add_argument("--mode", choices=["train", "val", "test"], default="train")

    # --- paths ----------------------------------------------------------------
    p.add_argument(
        "--img_root", required=True, help="root dir containing input_image/ & gt_image/"
    )
    p.add_argument(
        "--caption_root",
        required=True,
        help="dir with caption jsons (pairs.json or caption_*.json)",
    )
    p.add_argument(
        "--init_ckpt",
        default="./prt_ckpt/coco_weight.ckpt",
        help="initial weights (when not resuming)",
    )
    p.add_argument(
        "--resume_ckpt", default=None, help="checkpoint to resume / test with"
    )
    p.add_argument(
        "--config", default="./L-CAD/configs/cldm_v15_ehdecoder.yaml", help="model yaml"
    )

    # --- optimisation ---------------------------------------------------------
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.00001)
    p.add_argument(
        "--epochs", type=int, default=5, help="number of training epochs"  # ★ 추가
    )
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--precision", type=int, choices=[16, 32], default=32)

    # --- extras ---------------------------------------------------------------
    p.add_argument(
        "--use_sam", action="store_true", help="enable SAM masks during training/eval"
    )
    p.add_argument("--logger_freq", type=int, default=500)

    return p


pl.seed_everything(42, workers=True)


def make_trainer(
    save_dir: Path, gpus: int, precision: int, logger_freq: int, max_epochs: int
):
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir / "ckpts",
        filename="{step:08d}",
        # save_top_k=-1,
        every_n_train_steps=500,
    )

    callbacks = [
        ckpt_cb,
        LearningRateMonitor("step"),
        ImageLogger(batch_frequency=logger_freq),
    ]

    trainer = pl.Trainer(
        gpus=gpus,
        precision=precision,
        default_root_dir=str(save_dir),
        callbacks=callbacks,
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        accumulate_grad_batches=max(1, 32 // gpus),
        log_every_n_steps=logger_freq // 2,
    )
    return trainer


from torch.optim import AdamW
import types


def build_model(
    cfg: str,
    init_ckpt: str | None = None,
    resume_ckpt: str | None = None,
    lr: float = 1e-5,
):
    # ── 원래 로드 부분 그대로 ────────────────────────────────────────────────
    model = create_model(cfg).cpu()
    ckpt_path = resume_ckpt or init_ckpt
    if ckpt_path:
        model.load_state_dict(load_state_dict(ckpt_path, location="cpu"), strict=False)

    model.learning_rate = lr
    model.sd_locked = False
    model.only_mid_control = False

    # ── AdamW( weight‑decay 1e‑2 ) 설정만 추가 ─────────────────────────────
    def _configure(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = AdamW(
            [
                {"params": decay, "weight_decay": 1e-2},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return optimizer  # 스케줄러 없이 옵티마이저만 반환

    # Lightning 모듈에 메서드 주입
    model.configure_optimizers = types.MethodType(_configure, model)

    return model


def main():
    args = get_parser().parse_args()

    # --- datasets -------------------------------------------------------------
    split = (
        "train" if args.mode == "train" else args.mode
    )  # val / test pass straight through
    ds = MyDataset(
        img_dir=os.path.join(
            args.img_root, "input_image" if split == "test" else ""
        ),  # for train/val MyDataset adds split suffix internally
        caption_dir=args.caption_root,
        split=split,
        use_sam=args.use_sam,
        coco_style=False,
    )

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=(split == "train"), num_workers=4
    )

    # --- model / trainer ------------------------------------------------------
    save_root = Path("/raid/HZ/ic/logs") / datetime.now().strftime("%Y%m%d‑%H%M%S")
    trainer = make_trainer(
        save_root, args.gpus, args.precision, args.logger_freq, args.epochs
    )
    model = build_model(args.config, args.init_ckpt, args.resume_ckpt, args.lr)

    if args.mode == "train":
        trainer.fit(model, loader)
    else:
        model.usesam = args.use_sam
        trainer.test(model, loader)


if __name__ == "__main__":
    main()
