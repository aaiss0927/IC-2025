# make_mask_sam2_like_original.py
import os, numpy as np
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ───────────── 경로 설정 ───────────────────────────────────────────
IMG_DIR = "./test/input_image"  # 입력 이미지 폴더
MASK_DIR = "./sam_mask/select_masks"  # 마스크(NPY) 저장 루트
os.makedirs(MASK_DIR, exist_ok=True)

# ───────────── SAM2 설정 ──────────────────────────────────────────
CKPT = "./models/sam2.1_hiera_base_plus.pt"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEVICE = "cuda"

sam2 = build_sam2(CFG, CKPT, device=DEVICE)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# ───────────── 이미지 목록 수집 ───────────────────────────────────
img_list = [
    fn
    for fn in sorted(os.listdir(IMG_DIR))
    if fn.lower().endswith((".png", ".jpg", ".jpeg"))
]

# ───────────── 메인 루프 ──────────────────────────────────────────
for img_name in tqdm(img_list, desc="SAM2 mask generation"):

    # 1) 이미지 로드
    img_path = os.path.join(IMG_DIR, img_name)
    img = np.array(Image.open(img_path))

    # 2) 마스크 생성
    masks = mask_generator.generate(img)
    print(len(masks))
    print(masks[0].keys())

    # 3) NPY 저장: 이미지별 전용 폴더 생성 후 저장
    #     예) MASK_DIR/TEST_001/TEST_001_0.npy ...
    img_stem = os.path.splitext(img_name)[0]
    save_root = os.path.join(MASK_DIR, img_stem)
    os.makedirs(save_root, exist_ok=True)

    for i, m in enumerate(masks):
        np.save(os.path.join(save_root, f"{img_stem}_{i}.npy"), m["segmentation"])
