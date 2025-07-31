#!/usr/bin/env python
from share import *
import sys, argparse, os, zipfile, random, json, time
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.model import create_model, load_state_dict
import torch, einops
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import open_clip
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# ================ CLI =======================================================
def get_args():
    p = argparse.ArgumentParser(
        description="L‑CAD inference with optional BLIP caption refinement"
    )
    p.add_argument("--ckpt", default="./models/multi_weight.ckpt")
    p.add_argument("--config", default="./L-CAD/configs/cldm_sample.yaml")
    p.add_argument("--img_dir", default="./test/input_image")
    p.add_argument("--pair_dir", default="./test/pairs.json")
    p.add_argument("--mask_dir", default="./sam_mask/select_masks")
    p.add_argument("--seed", type=int, default=1037)
    p.add_argument("--out_root", default="./results")
    p.add_argument("--use_sam", action="store_true", default=True)
    p.add_argument("--ddim_steps", type=int, default=25)
    p.add_argument(
        "--guidance",
        type=float,
        nargs="+",
        default=[5.0],
        help="여러 값을 공백으로 구분해 주면 한 번에 순차 추론",
    )
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


args = get_args()

# ====================== 설정 =================================================
start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
CFG = {
    "SUB_DIR": os.path.join("./results", start_time),
    "SEED": args.seed,
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


seed_everything(CFG["SEED"])


# ====================== 유틸 =================================================
def save_images(samples, batch, save_root: Path):
    save_root.mkdir(parents=True, exist_ok=True)
    for i in range(samples.shape[0]):
        img_name = batch["name"][i]
        img_id = Path(img_name).stem
        grid = samples[i].permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid).save(save_root / f"{img_id}.png")


# ====================== 메인 =================================================
if __name__ == "__main__":
    # ── 1. 모델·데이터셋은 한 번만 준비 ──────────────────────────────
    print(">> Loading L‑CAD model …")
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location="cpu"), strict=False)
    model = model.half().cuda()
    torch.set_float32_matmul_precision("high")
    model.usesam = args.use_sam

    try:  # (선택) Inductor compile
        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
        print("✓ torch.compile enabled")
    except Exception as e:
        print("torch.compile failed, using eager model →", e)

    from colorization_dataset import MyDataset

    dataset = MyDataset(
        img_dir="/shared/home/kdd/HZ/inha-challenge",
        caption_path=args.pair_dir,
        split="test",
        use_sam=args.use_sam,
        mask_dir=args.mask_dir,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    from ldm.models.diffusion.ddim import DDIMSampler_withsam

    sampler = DDIMSampler_withsam(model)

    # ── 2. guidance 값마다 루프 ───────────────────────────────────────
    for g_val in args.guidance:
        tag = f"g{g_val}".replace(".", "p")
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out_root) / tag / f"test_{run_stamp}"
        sub_dir = out_dir / "submission"
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== guidance = {g_val} → 결과 폴더 {out_dir} =====")

        generated_image_paths, generated_ids = [], []

        # ── 2‑A. 배치 루프 ---------------------------------------------------
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            control = einops.rearrange(
                batch[model.control_key].to(model.device), "b h w c -> b c h w"
            ).half()
            c_cat = control.to(memory_format=torch.contiguous_format)
            gray_z = model.first_stage_model.g_encoder(c_cat)

            xc = batch["txt"]
            cond = model.get_learned_conditioning(xc)
            uc_cross = model.get_unconditional_conditioning(c_cat.size(0))
            cond_dict = {"c_concat": [c_cat], "c_crossattn": [cond]}
            uc_dict = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            sam_mask = batch["mask"]

            with torch.autocast("cuda", torch.float16):
                samples, _ = sampler.sample(
                    args.ddim_steps,
                    c_cat.size(0),
                    (model.channels, c_cat.size(2) // 8, c_cat.size(3) // 8),
                    cond_dict,
                    temperature=args.temperature,
                    eta=args.ddim_eta,
                    unconditional_guidance_scale=g_val,
                    unconditional_conditioning=uc_dict,
                    sam_mask=sam_mask,
                    verbose=False,
                )

            samples_cfg = samples.half()

            if isinstance(gray_z, list):
                gray_z = [g.half() for g in gray_z]
            else:
                gray_z = gray_z.half()

            x_samples = model.decode_first_stage(samples_cfg, gray_z)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

            save_images(x_samples, batch, out_dir)
            for name in batch["name"]:
                fname = Path(name).stem
                generated_ids.append(fname)
                generated_image_paths.append(out_dir / f"{fname}.png")

        print("✅ Image generation done")

        # ── 2‑B. CLIP embedding & CSV ---------------------------------------
        if generated_image_paths:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            clip_model = clip_model.cuda().eval()
            feats = []
            for p in tqdm(generated_image_paths, desc="embedding"):
                img = preprocess(Image.open(p)).unsqueeze(0).cuda()
                with torch.no_grad():
                    v = clip_model.encode_image(img)
                    v /= v.norm(dim=-1, keepdim=True)
                feats.append(v.cpu().numpy().reshape(-1))
            df = pd.DataFrame(feats, columns=[f"vec_{i}" for i in range(len(feats[0]))])
            df.insert(0, "ID", generated_ids)
            csv_path = sub_dir / "embed_submission.csv"
            df.to_csv(csv_path, index=False)
            print("✓ embedding CSV saved")

        # ── 2‑C. ZIP 패킹 ----------------------------------------------------
        zip_path = Path("./submission") / f"submission_{tag}.zip"
        zip_path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in out_dir.glob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.name)

            csv_path = sub_dir / "embed_submission.csv"
            if csv_path.exists():
                zf.write(csv_path, arcname=csv_path.name)
        print(f"✅ ZIP written → {zip_path}")
