#!/usr/bin/env python
# inference_single.py ── L-CAD 단일 이미지 추론 (+옵션: BLIP 캡션 리파인)
from __future__ import annotations
import argparse, time
from pathlib import Path

import einops, numpy as np, torch
from PIL import Image

from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler_withsam
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# ==================== CLI ====================
def get_args():
    p = argparse.ArgumentParser(description="Single image inference with L-CAD")
    p.add_argument("--ckpt", default="./prt_ckpt/coco_weight.ckpt")
    p.add_argument("--config", default="./L-CAD/configs/cldm_sample.yaml")
    p.add_argument("--img_path", required=True, help="Path to input grayscale image")
    p.add_argument("--caption", required=True, help="Original caption text")
    p.add_argument("--out_root", default="./results_single")

    # BLIP options
    p.add_argument("--use_blip", action="store_true", help="Refine caption with BLIP")
    p.add_argument("--blip_model", default="Salesforce/blip2-flan-t5-xl-coco")
    p.add_argument("--blip_8bit", action="store_true")

    # Sampling options
    p.add_argument("--ddim_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    return p.parse_args()


# ==================== BLIP Helper ====================
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
            "Rewrite the caption, keeping it concise (≤70 tokens), "
            "and add missing color or object details. "
            f'Original: "{ori}"'
        )
        inputs = self.proc(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.proc.decode(out[0], skip_special_tokens=True)


# ==================== Main ====================
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = Path(args.img_path)
    caption = args.caption

    # Out dir
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / f"single_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load L-CAD model
    print(">> Loading L-CAD model...")
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location="cpu"), strict=False)
    model = model.to(device)
    ddim_sampler = DDIMSampler_withsam(model)

    # Load image (grayscale expected)
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img)
    h, w, _ = img_np.shape
    control = torch.from_numpy(img_np / 255.0).unsqueeze(0)  # [1,H,W,3]
    control = einops.rearrange(control, "b h w c -> b c h w").float().to(device)

    # BLIP refine
    if args.use_blip:
        print(">> Refining caption using BLIP...")
        refiner = BlipRefiner(args.blip_model, device, use_8bit=args.blip_8bit)
        caption = refiner(pil_img, caption)

    print(f">> Final Prompt: {caption}")

    # Encode gray latent
    gray_z = model.first_stage_model.g_encoder(control)
    cond = model.get_learned_conditioning([caption])
    uc_cross = model.get_unconditional_conditioning(1)
    cond_dict = {"c_concat": [control], "c_crossattn": [cond]}
    uc_dict = {"c_concat": [control], "c_crossattn": [uc_cross]}
    shape = (model.channels, control.shape[2] // 8, control.shape[3] // 8)

    # Sampling
    samples, _ = ddim_sampler.sample(
        args.ddim_steps,
        1,
        shape,
        cond_dict,
        unconditional_conditioning=uc_dict,
        unconditional_guidance_scale=args.guidance,
        eta=0.0,
        sam_mask=None,
        verbose=False,
    )

    # Decode & save
    img_out = model.decode_first_stage(samples, gray_z)
    img_out = torch.clamp((img_out + 1) / 2, 0, 1)[0]  # [C,H,W]
    img_out = (img_out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    out_file = out_dir / f"colorized_{img_path.stem}.png"
    Image.fromarray(img_out).save(out_file)
    print(f"✓ Saved colorized image → {out_file}")


if __name__ == "__main__":
    main()
