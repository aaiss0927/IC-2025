import os, glob, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- 사용자 설정 ----------------
IMG_DIR = "./test/input_image"  # 원본 이미지 폴더
MASK_ROOT = "./sam_mask/select_masks_sam2"  # make_mask.py가 만든 마스크 루트
ALPHA = 0.45  # 마스크 투명도 (0=투명, 1=불투명)
SAVE_PNG = True  # True면 PNG로 따로 저장
# --------------------------------------------


def hex2rgb(h):  # '#RRGGBB' → (R,G,B)
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def random_color(seed=None):
    random.seed(seed)
    return hex2rgb("%06x" % random.randint(0, 0xFFFFFF))


for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    # 1) 원본 이미지
    img_path = os.path.join(IMG_DIR, img_name)
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    # 2) 해당 이미지의 마스크(.npy)들
    mask_dir = os.path.join(MASK_ROOT, os.path.splitext(img_name)[0])
    mask_npys = sorted(glob.glob(os.path.join(mask_dir, "mask_*.npy")))
    if not mask_npys:
        print(f"[!] {img_name} 마스크 없음")
        continue

    # 3) 컬러 오버레이 생성
    overlay = img.copy()
    for i, npy in enumerate(mask_npys):
        mask = np.load(npy)  # (H,W) bool/0‑1 배열
        color = np.array(random_color(i), dtype=np.uint8)
        overlay[mask.astype(bool)] = color  # 마스크 위치만 염색

    # 4) 원본 + 오버레이 알파블렌드
    blended = (img * (1 - ALPHA) + overlay * ALPHA).astype(np.uint8)

    # 5) 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.axis("off")
    plt.title(img_name)
    plt.show()

    # 6) 파일 저장(선택)
    if SAVE_PNG:
        save_path = os.path.join(mask_dir, "overlay.png")
        Image.fromarray(blended).save(save_path)
        print(f"✔ {save_path} 저장 완료")
