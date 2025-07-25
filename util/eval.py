#!/usr/bin/env python
import json, os, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── 직접 절대 경로 입력 ─────────
GT_DIR = Path("/absolute/path/to/gt_image")  # 정답 이미지 디렉토리
PRED_DIR = Path("/absolute/path/to/pred_image")  # 예측 이미지 디렉토리
CAPTION_JSON = Path("/absolute/path/to/caption.json")  # 캡션 JSON

assert GT_DIR.is_dir(), f"GT 폴더 없음: {GT_DIR}"
assert PRED_DIR.is_dir(), f"예측 이미지 폴더 없음: {PRED_DIR}"
assert CAPTION_JSON.is_file(), f"캡션 JSON 없음: {CAPTION_JSON}"

# ───────── CLIP 모델 로드 ─────────
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")


# ───────── 유틸: 파일명 매칭 ─────────
def resolve_filename(filename: str, search_dir: Path):
    """
    caption_validation.json의 키(filename)를 기반으로 실제 존재하는 파일명을 찾는다.
    ('000000.jpg' / '00001.jpg' 혼재 가능성 처리)
    """
    stem, ext = filename.split(".")
    candidates = [filename]
    for c in candidates:
        p = search_dir / c
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} => {search_dir}에서 찾을 수 없음")


# ───────── HSV 히스토그램 유사도 ─────────
def hsv_hist_similarity(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    gt_hsv = cv2.cvtColor(img_gt, cv2.COLOR_RGB2HSV)
    pred_hsv = cv2.cvtColor(img_pred, cv2.COLOR_RGB2HSV)

    sims = []
    specs = [(0, 180), (1, 256), (2, 256)]  # H,S,V 범위
    for ch, rng in specs:
        bins = rng
        h1 = cv2.calcHist([gt_hsv], [ch], None, [bins], [0, rng])
        h2 = cv2.calcHist([pred_hsv], [ch], None, [bins], [0, rng])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        sims.append(sim)
    return float(np.mean(sims))


# ───────── CLIP Score ─────────
@torch.no_grad()
def clip_score(caption: str, pil_img: Image.Image) -> float:
    text_tokens = tokenizer([caption]).to(device)
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
    img_feat = clip_model.encode_image(img_tensor)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    return float((text_feat @ img_feat.T).item())  # 코사인 유사도


# ───────── 평가 루프 ─────────
with open(CAPTION_JSON, "r") as f:
    captions = json.load(f)

records = []
for fname, cap_list in tqdm(captions.items(), desc="Evaluate"):
    caption_text = cap_list[0] if isinstance(cap_list, list) else str(cap_list)

    gt_path = resolve_filename(fname, GT_DIR)
    pred_path = resolve_filename(fname, PRED_DIR)

    gt_img_pil = Image.open(gt_path).convert("RGB")
    pred_img_pil = Image.open(pred_path).convert("RGB")
    gt_np = np.array(gt_img_pil)
    pred_np = np.array(pred_img_pil)

    hsv_sim = hsv_hist_similarity(gt_np, pred_np)
    clip_sim = clip_score(caption_text, pred_img_pil)
    final_score = 0.6 * hsv_sim + 0.4 * clip_sim

    records.append(
        {
            "ID": Path(fname).stem,
            "caption": caption_text,
            "HSV_Similarity": hsv_sim,
            "CLIP_Score": clip_sim,
            "Final_Score": final_score,
        }
    )

df = pd.DataFrame(records)

print("개별 평균 점수:")
print(df[["HSV_Similarity", "CLIP_Score", "Final_Score"]].mean())
