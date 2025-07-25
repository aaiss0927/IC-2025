#!/usr/bin/env python
import json, random, shutil, os
from pathlib import Path

SEED = 42
NUM_SAMPLES = 100

train_root = Path("~/HZ/inha-challenge/train").expanduser()
val_root = Path("~/HZ/inha-challenge/validation")
caption_train_path = train_root / "caption_train.json"

# 1. caption_train.json 읽기
with open(caption_train_path, "r") as f:
    caption_dict = json.load(f)

all_keys = list(caption_dict.keys())
if len(all_keys) < NUM_SAMPLES:
    raise ValueError(f"전체 이미지 수({len(all_keys)})가 {NUM_SAMPLES}보다 적습니다.")

# 2. 샘플링 (seed 고정)
random.seed(SEED)
sampled_keys = random.sample(all_keys, NUM_SAMPLES)

# 3. validation 폴더 구조 생성
(val_root / "gt_image").mkdir(parents=True, exist_ok=True)
(val_root / "input_image").mkdir(parents=True, exist_ok=True)

# 4. 캡션 저장
caption_validation_path = val_root / "caption_validation.json"
caption_validation = {k: caption_dict[k] for k in sampled_keys}
with open(caption_validation_path, "w") as f:
    json.dump(caption_validation, f, indent=2, ensure_ascii=False)


# 5. 이미지 복사 함수
def copy_image(filename: str):
    """
    filename: JSON 키 (예: '000000.jpg')
    실제 파일명이 5자리/6자리 혼재 가능성을 고려해 후보를 순차 시도
    """
    base = filename.split(".")[0]
    ext = filename.split(".")[-1]
    candidates = [
        filename,  # 그대로 (예: 000000.jpg 또는 00001.jpg)
        # f"{int(base):05d}.{ext}",  # 5자리 zero-pad
        # f"{int(base):06d}.{ext}",  # 6자리 zero-pad
    ]
    src_gt = None
    src_input = None
    for cand in candidates:
        gt_path = train_root / "gt_image" / cand
        in_path = train_root / "input_image" / cand
        if gt_path.exists() and in_path.exists():
            src_gt, src_input = gt_path, in_path
            break
    if src_gt is None:
        raise FileNotFoundError(
            f"{filename} 에 해당하는 원본 이미지를 찾지 못했습니다. 후보: {candidates}"
        )
    shutil.copy2(src_gt, val_root / "gt_image" / os.path.basename(src_gt))
    shutil.copy2(src_input, val_root / "input_image" / os.path.basename(src_input))


# 6. 복사 실행
for k in sampled_keys:
    copy_image(k)

print(
    f"완료: {NUM_SAMPLES}장의 이미지를 '{val_root}'에 복사하고 caption_validation.json 생성했습니다."
)
