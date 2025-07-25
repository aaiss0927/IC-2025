#!/usr/bin/env python
# refine_pairs_for_lcad.py ── L-CAD 훈련용 캡션 리파인 (짧고 COCO 스타일)
# ------------------------------------------------------------------------
"""
필수 패키지
----------
pip install torch>=2.3.0 transformers==4.37.0 torchvision pillow tqdm
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ────────────────────── 0. 인자 파싱 ─────────────────────────────────────────
p = argparse.ArgumentParser(description="L-CAD caption refiner (Ristretto‑3B)")
p.add_argument("--img_dir", default="./test/input_image", help="이미지 폴더")
p.add_argument(
    "--pairs", default="./test/caption_test_mine.json", help="원본 caption JSON 경로"
)
p.add_argument("--out", default="./test/caption_ttt.json", help="출력 JSON 경로")
p.add_argument(
    "--model", default="LiAutoAD/Ristretto-3B", help="HF 모델 경로 또는 로컬 체크포인트"
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

# ────────────────────── 2. 이미지 전처리 ─────────────────────────────────────
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
    img = Image.open(img_path).convert("RGB")
    pixel = _build_transform(input_sz)(img).unsqueeze(0)  # [1,3,H,W]
    return pixel.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)


# ────────────────────── 3. pairs.json 로드 ──────────────────────────────────
with open(args.pairs, "r", encoding="utf-8") as f:
    pairs: Dict[str, List[str]] = json.load(f)

# ────────────────────── 4. 캡션 생성 루프 ─────────────────────────────────────
refined: Dict[str, List[str]] = {}
gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0)

for img_name, ori_caps in tqdm(pairs.items(), desc="refining"):
    img_path = Path(args.img_dir) / img_name
    if not img_path.exists():
        print(f"[WARN] {img_path} not found – 건너뜀")
        continue

    pixels = preprocess_image(img_path)  # (1,3,384,384)

    # 프롬프트: L-CAD 스타일 → 짧고 간결한 문장, 1~3개 생성
    color_clues = " ".join(ori_caps)  # 기존 설명 합치기
    prompt = (
        "<image>\n"
        "You are generating COCO-style captions for image colorization training.\n"
        "Rules:\n"
        "• Write more than one sentences.\n"
        "• Do not produce duplicate sentences..\n"
        "• Simple, descriptive, lowercase, present tense.\n"
        "• Include colors, objects, and actions when possible.\n"
        "• No questions, no commentary.\n\n"
        f"Original hints: {color_clues}\n"
        "Captions:"
    )

    with torch.no_grad():
        response, _ = model.chat(
            tok, pixels, prompt, gen_cfg, history=None, return_history=True
        )

    # 여러 문장으로 나눠서 리스트로 변환
    sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 0]
    refined[img_name] = sentences

# ────────────────────── 5. 저장 ─────────────────────────────────────────────
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(refined, f, ensure_ascii=False, indent=2)

print(f"✓ refined caption saved → {args.out}")
