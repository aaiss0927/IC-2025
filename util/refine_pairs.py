#!/usr/bin/env python
# refine_pairs.py ── Ristretto‑3B로 색채화용 캡션 리파인
# ------------------------------------------------------------------------
"""
필수 패키지
----------
pip install torch>=2.3.0 transformers==4.37.0 torchvision pillow tqdm
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List

import torch, torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ────────────────────── 0. 인자 파싱 ─────────────────────────────────────────
p = argparse.ArgumentParser(description="Ristretto‑3B caption refiner")
p.add_argument(
    "--img_dir",
    default="/shared/home/kdd/HZ/inha-challenge/test/input_image",
    help="흑백 테스트 이미지 폴더",
)
p.add_argument(
    "--pairs",
    default="/shared/home/kdd/HZ/inha-challenge/test/pairs.json",
    help="원본 pairs.json 경로",
)
p.add_argument(
    "--out",
    default="/shared/home/kdd/HZ/inha-challenge/test/refined_pairs.json",
    help="출력 json 경로",
)
p.add_argument(
    "--model",
    default="LiAutoAD/Ristretto-3B",
    help="HF 모델 경로(또는 로컬 체크포인트)",
)
p.add_argument("--max_new_tokens", type=int, default=77)
args = p.parse_args()

# ────────────────────── 1. 모델 로드 ────────────────────────────────────────
print(">> loading Ristretto‑3B … (BF16, GPU 필요)")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = (
    AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    .eval()
    .to(device)
)
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

# ────────────────────── 2. 이미지 전처리 util ───────────────────────────────
IMNET_MEAN, IMNET_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def _build_transform(sz: int = 384):
    return T.Compose(
        [
            T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
            T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )


def preprocess_image(img_path: Path, input_sz: int = 384):
    """단순 1 블록(1 이미지 토큰) 버전 – 색채화 용도로 충분"""
    img = Image.open(img_path).convert("RGB")
    pixel = _build_transform(input_sz)(img).unsqueeze(0)  # [1,3,H,W]
    return pixel.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)


# ────────────────────── 3. pairs.json 로드 ──────────────────────────────────
with open(args.pairs, "r", encoding="utf-8") as f:
    pairs: List[List[str]] = json.load(f)

# ────────────────────── 4. 루프 ─────────────────────────────────────────────
refined: List[List[str]] = []
gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0)

for img_name, ori_cap in tqdm(pairs, desc="refining"):
    img_path = Path(args.img_dir) / img_name
    if not img_path.exists():
        print(f"[WARN] {img_path} not found – 건너뜀")
        continue

    pixels = preprocess_image(img_path)  # (1,3,384,384)
    # Ristretto는 <image> 토큰이 포함된 프롬프트를 요구한다:contentReference[oaicite:0]{index=0}
    prompt = (
        "<image>\n"
        "Write ONE COCO‑style caption describing the grayscale photo, "
        "merging what you SEE with the COLOR clues below. "
        "• single sentence, 5‑20 words, present tense, mostly lowercase\n"
        "• include concrete hues, materials, lighting when relevant\n"
        "• no questions, no extra commentary\n"
        f"color clues: {ori_cap}\n"
        "coco caption:"
    )

    with torch.no_grad():
        response, _ = model.chat(
            tok, pixels, prompt, gen_cfg, history=None, return_history=True
        )

    refined.append([img_name, response.strip()])

# ────────────────────── 5. 저장 ─────────────────────────────────────────────
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(refined, f, ensure_ascii=False, indent=2)

print(f"✓ refined caption saved → {args.out}")
