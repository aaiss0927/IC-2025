#!/usr/bin/env python
from share import *
import sys, argparse, os, zipfile, random, json, time
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from colorization_dataset import MyDataset
from cldm.model import create_model, load_state_dict
import torch
import einops
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import open_clip
import warnings

warnings.filterwarnings("ignore")


# ================ CLI =======================================================
def get_args():
    p = argparse.ArgumentParser(
        description="L‑CAD inference with optional BLIP caption refinement"
    )
    p.add_argument("--ckpt", default="./prt_ckpt/multi_weight.ckpt")
    p.add_argument("--config", default="./L-CAD/configs/cldm_sample.yaml")
    p.add_argument("--img_dir", default="./test/input_image")
    p.add_argument("--pair_dir", default="./test")  # contains pairs.json
    p.add_argument("--mask_root", default="sam_mask/select_masks")
    p.add_argument("--out_root", default="./results")
    p.add_argument("--use_sam", action="store_true")
    p.add_argument("--ddim_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5)
    p.add_argument(
        "--single_name",
        default=None,  # ✨ 새로 추가
        help="이 파일 이름(확장자 포함)만 추론하고 종료",
    )
    p.add_argument(
        "--single_prompt", default=None, help="단일 추론 시 사용할 프롬프트(문자열)"
    )
    return p.parse_args()


# ====================== 설정 ======================
start_time = time.strftime("%Y-%m-%d-%H-%M-%S")  # ✅ 타임스탬프
CFG = {
    "SUB_DIR": os.path.join("./results", start_time),
    "SEED": 42,
}  # ✅ 결과 폴더 변경


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["SEED"])


# ====================== 유틸 ======================
def save_images(samples, batch, save_root):
    os.makedirs(save_root, exist_ok=True)
    for i in range(samples.shape[0]):
        img_name = batch["name"][i]
        img_id = os.path.splitext(img_name)[0]
        grid = samples[i].transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.detach().cpu().numpy()
        grid = (grid * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(save_root, f"{img_id}.png")
        Image.fromarray(grid).save(out_path)


# ====================== 메인 ======================
if __name__ == "__main__":
    args = get_args()
    resume_path = "/shared/home/kdd/HZ/inha-challenge/prt_ckpt/multi_weight.ckpt"
    model_cfg = "/shared/home/kdd/HZ/inha-challenge/L-CAD/configs/cldm_sample.yaml"
    batch_size = 1

    print(">> Loading L-CAD model …")
    model = create_model(model_cfg).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False)
    model = model.cuda()
    model.usesam = True

    dataset = MyDataset(
        img_dir="/shared/home/kdd/HZ/inha-challenge",
        caption_path="/shared/home/kdd/HZ/inha-challenge/test/pairs.json",
        split="test",
        use_sam=True,
    )
    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=batch_size, shuffle=False
    )

    from ldm.models.diffusion.ddim import DDIMSampler_withsam

    ddim_steps = 30
    ddim_eta = 0.0
    unconditional_guidance_scale = 5.0

    os.makedirs(CFG["SUB_DIR"], exist_ok=True)

    generated_image_paths = []
    generated_ids = []

    # ✅ Sampler 외부 생성
    sampler = DDIMSampler_withsam(model)

    print(">> Start inference …")
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        control = batch[model.control_key].to(model.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        c_cat = control.to(memory_format=torch.contiguous_format).float()
        gray_z = model.first_stage_model.g_encoder(c_cat)

        xc = batch["txt"]
        c = model.get_learned_conditioning(xc)

        tokens = model.cond_stage_model.tokenizer.tokenize(xc[0])
        batch_encoding = model.cond_stage_model.tokenizer(
            xc[0],
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        split_idx = []
        for idx, token in enumerate(tokens):
            if token == ",</w>":
                split_idx.append(idx + 1)
        split_idx.append(idx + 2)

        sam_mask = batch["mask"]

        N = c_cat.shape[0]
        uc_cross = model.get_unconditional_conditioning(N)
        uc_cat = c_cat
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        cond = {"c_concat": [c_cat], "c_crossattn": [c]}
        b, c_, h, w = cond["c_concat"][0].shape
        shape = (model.channels, h // 8, w // 8)

        samples_cfg, _ = sampler.sample(
            ddim_steps,
            b,
            shape,
            cond,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
            verbose=False,
            use_attn_guidance=False,
            sam_mask=sam_mask,
            split_id=split_idx,
            tokens=tokens,
        )

        x_samples = model.decode_first_stage(samples_cfg, gray_z)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

        save_images(x_samples, batch=batch, save_root=CFG["SUB_DIR"])
        for name in batch["name"]:
            generated_ids.append(os.path.splitext(name)[0])
            generated_image_paths.append(
                os.path.join(CFG["SUB_DIR"], os.path.splitext(name)[0] + ".png")
            )

    print("✅ 이미지 생성 완료:", CFG["SUB_DIR"])

    print(">> Extracting CLIP embeddings …")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    feat_list = []
    for img_path in tqdm(generated_image_paths):
        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            tensor = clip_preprocess(img).unsqueeze(0).to(clip_model.logit_scale.device)
            feat = clip_model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        feat_list.append(feat.detach().cpu().numpy().reshape(-1))

    feat_arr = np.array(feat_list)
    vec_cols = [f"vec_{i}" for i in range(feat_arr.shape[1])]
    df_embed = pd.DataFrame(feat_arr, columns=vec_cols)
    df_embed.insert(0, "ID", generated_ids)
    embed_csv_path = os.path.join(CFG["SUB_DIR"], "embed_submission.csv")
    df_embed.to_csv(embed_csv_path, index=False)
    print(f"✅ 임베딩 CSV 저장: {embed_csv_path}")

    zip_path = "./submission/submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(CFG["SUB_DIR"]):
            fpath = os.path.join(CFG["SUB_DIR"], fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                zf.write(fpath, arcname=fname)
    print(f"✅ 압축 완료: {zip_path}")
