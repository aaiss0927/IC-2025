#!/usr/bin/env python
# inference_blip.py  ── L‑CAD 추론 + (옵션) BLIP 캡션‑리파인
# ------------------------------------------------------------------------
from __future__ import annotations
import argparse, os, time, zipfile, json
from pathlib import Path
import re

import einops, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader

from colorization_dataset import MyDataset
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler_withsam

import open_clip  # CLIP embedding
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# ================ CLI =======================================================
def get_args():
    p = argparse.ArgumentParser(
        description="L‑CAD inference with optional BLIP caption refinement"
    )
    p.add_argument("--ckpt", default="./prt_ckpt/coco_weight.ckpt")
    p.add_argument("--config", default="./L-CAD/configs/cldm_sample.yaml")
    p.add_argument("--img_dir", default="./test/input_image")
    p.add_argument("--pair_dir", default="./test")  # contains pairs.json
    p.add_argument("--mask_root", default="sam_mask/select_masks")
    p.add_argument("--out_root", default="./results")
    p.add_argument("--use_sam", action="store_true")
    # BLIP options
    p.add_argument(
        "--use_blip",
        action="store_true",
        help="enable on‑the‑fly BLIP caption refinement",
    )
    p.add_argument("--blip_model", default="Salesforce/blip2-flan-t5-xl-coco")
    p.add_argument(
        "--blip_8bit",
        action="store_true",
        help="load BLIP in 8‑bit (requires bitsandbytes)",
    )
    # sampling
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


# ================ BLIP helper ==============================================
class BlipRefiner:
    def __init__(self, model_name: str, device: str, use_8bit: bool = False):
        self.proc = Blip2Processor.from_pretrained(model_name)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if not use_8bit else None,
                load_in_8bit=use_8bit,
                device_map="auto" if use_8bit else None,
            )
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image, ori: str) -> str:
        prompt = (
            "Rewrite the caption, keeping it concise (≤77 tokens), "
            f'Original: "{ori}"'
            "Describe the objects and their colors explicitly."
            "realistic, high quality, detailed, Do not change the structure. Only Colorize."
        )
        inputs = self.proc(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        out = self.model.generate(**inputs, max_new_tokens=77)
        return self.proc.decode(out[0], skip_special_tokens=True)


# ================ main ======================================================
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.single_name:
        args.out_root = "./result_single"

    # ---------- I/O paths ----------------------------------------------------
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / f"test_{run_stamp}"
    sub_dir = out_dir / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ---------- model --------------------------------------------------------
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location="cpu"), strict=False)
    model = model.to(device)
    model.usesam = args.use_sam

    ddim_sampler = DDIMSampler_withsam(model)

    # ---------- BLIP (optional) ---------------------------------------------
    refiner = None
    if args.use_blip:
        print(f">> loading BLIP refiner [{args.blip_model}] …")
        refiner = BlipRefiner(args.blip_model, device, use_8bit=args.blip_8bit)

    # ---------- dataset ------------------------------------------------------
    test_ds = MyDataset(
        img_dir=args.img_dir,
        caption_dir=args.pair_dir,
        split="test",
        use_sam=args.use_sam,
        mask_root=args.mask_root,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    saved_img_paths = []  # for CLIP embedding

    # ---------- loop ---------------------------------------------------------
    for idx, batch in enumerate(test_loader, 1):
        img_name = batch["name"][0]

        # ✨ 단일 추론 모드: 이름이 다르면 continue, 맞으면 끝나면 break
        if args.single_name and Path(img_name).name != args.single_name:
            continue

        # raw_prompt = (
        #     args.single_prompt if args.single_prompt is not None else batch["txt"][0]
        # )

        # ✨ 문장 단위로 잘라서 줄바꿈으로 합치기
        # sentences = re.findall(r"[^.?!]+[.?!]", raw_prompt)  # 끝에 . ? ! 가 붙은 구절
        # prompt = "\n".join(s.strip() for s in sentences if s.strip())

        # batch["txt"][0] = prompt

        prompt = (
            args.single_prompt if args.single_prompt is not None else batch["txt"][0]
        )
        batch["txt"][0] = prompt

        # ⇢ BLIP refine -------------------------------------------------------
        if refiner is not None:
            try:
                # 원본 입력 이미지를 다시 열어 전달
                pil_img = Image.open(Path(args.img_dir) / img_name).convert("RGB")
                prompt = refiner(pil_img, prompt)
                batch["txt"][0] = prompt
            except Exception as e:
                print(f"[BLIP‑ERR] {img_name}: {e}")

        print(f"[{idx:>3}/{len(test_loader)}] {img_name}")
        print("   prompt:", prompt[:120], "..." if len(prompt) > 120 else "")

        # ----- control / hint ----------------------------------------------
        control = einops.rearrange(
            batch[model.control_key].to(device), "b h w c -> b c h w"
        ).float()
        gray_z = model.first_stage_model.g_encoder(control)

        # ----- text cond ----------------------------------------------------
        cond = model.get_learned_conditioning(batch["txt"])
        uc_cross = model.get_unconditional_conditioning(1)

        cond_dict = {"c_concat": [control], "c_crossattn": [cond]}
        uc_dict = {"c_concat": [control], "c_crossattn": [uc_cross]}

        sam_mask = None
        if args.use_sam and isinstance(batch.get("mask"), torch.Tensor):
            sam_mask = batch["mask"].to(device)  # [1, N, H, W]

        shape = (model.channels, control.shape[2] // 8, control.shape[3] // 8)

        # ----- sampling -----------------------------------------------------
        samples, _ = ddim_sampler.sample(
            args.ddim_steps,
            1,
            shape,
            cond_dict,
            unconditional_conditioning=uc_dict,
            unconditional_guidance_scale=args.guidance,
            eta=0.0,
            sam_mask=sam_mask,
            verbose=False,
        )

        # ----- decode & save -----------------------------------------------
        img = model.decode_first_stage(samples, gray_z)
        img = torch.clamp((img + 1) / 2, 0, 1)[0]  # [C,H,W]
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        out_name = f"out_{Path(img_name).stem}.png"
        Image.fromarray(img_np).save(out_dir / out_name)

        # submission (ID 그대로)
        sub_path = sub_dir / f"{Path(img_name).stem}.png"
        Image.fromarray(img_np).save(sub_path)
        saved_img_paths.append(sub_path)

        if args.single_name:  # ✨ 원하는 하나를 처리했으면 반복 종료
            break

    # ---------- CLIP embedding ----------------------------------------------
    if saved_img_paths and not args.single_name:
        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        clip_model.to(device).eval()

        vecs, ids = [], []
        for p in saved_img_paths:
            im = clip_pre(Image.open(p)).unsqueeze(0).to(device)
            with torch.no_grad():
                v = clip_model.encode_image(im)
                v = v / v.norm(dim=-1, keepdim=True)
            vecs.append(v.cpu().numpy().flatten())
            ids.append(p.stem)

        df = pd.DataFrame(
            np.array(vecs), columns=[f"vec_{i}" for i in range(len(vecs[0]))]
        )
        df.insert(0, "ID", ids)
        df.to_csv(sub_dir / "embed_submission.csv", index=False)

        # zip packaging
        zip_path = out_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sub_dir.iterdir():
                zf.write(f, arcname=f.name)
        print(f"✓ submission written → {zip_path}")


# ============================================================================
if __name__ == "__main__":
    main()
