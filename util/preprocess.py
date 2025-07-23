import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip

# 1) 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) OpenCLIP ViT-L-14 모델과 전처리기, 토크나이저 로드
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-L-14")

model = model.to(device).eval()

# 3) 데이터 경로
jsonl_path = "/shared/home/kdd/HZ/inha-challenge/train/captions.jsonl"
img_dir     = Path("/shared/home/kdd/HZ/inha-challenge/train/gt_image")

# 4) 유사도 리스트
sims = []

# 5) 캡션-이미지 순회하며 유사도 계산
with open(jsonl_path, encoding="utf-8") as f:
    for line in tqdm(f, desc="Computing CLIP similarities"):
        item = json.loads(line)
        prompt  = item["prompt"]
        img_path = img_dir / item["image"]
        if not img_path.exists():
            continue

        # 이미지 전처리
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # 텍스트 토크나이즈
        text_tokens = tokenizer([prompt]).to(device)

        # 임베딩 추출
        with torch.no_grad():
            img_emb = model.encode_image(image_input)
            txt_emb = model.encode_text(text_tokens)

        # L2 정규화 후 코사인 유사도
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1).item()
        sims.append(sim)

# 6) 통계 계산 및 출력
arr = np.array(sims, dtype=np.float32)
print(f"샘플 수:     {len(arr)}")
print(f"평균 유사도: {arr.mean():.4f}")
print(f"최댓값:      {arr.max():.4f}")
print(f"최솟값:      {arr.min():.4f}")
print(f"표준편차:    {arr.std():.4f}")
