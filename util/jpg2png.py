#!/usr/bin/env python
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse

DEFAULT_DIR = "/shared/home/kdd/HZ/inha-challenge/util/km/validation_from_train/validation_300/input_image"


def convert_all_jpg_to_png(src_dir: Path, delete_jpg=False, overwrite=False):
    src_dir = Path(src_dir)
    assert src_dir.is_dir(), f"디렉토리 없음: {src_dir}"
    jpgs = [p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]

    ok = skip = err = 0
    for jpg_path in tqdm(jpgs, desc="JPG→PNG"):
        png_path = jpg_path.with_suffix(".png")
        if png_path.exists() and not overwrite:
            skip += 1
            continue
        try:
            with Image.open(jpg_path) as im:
                # EXIF 회전 보정 + RGB 정규화
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.save(png_path, format="PNG", optimize=True, compress_level=6)
            if delete_jpg:
                jpg_path.unlink()
            ok += 1
        except Exception as e:
            print(f"[WARN] 실패: {jpg_path.name} → {e}")
            err += 1
    print(f"완료: 변환 {ok}, 건너뜀 {skip}, 오류 {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR, help="변환할 폴더 경로")
    ap.add_argument("--rm", action="store_true", help="변환 후 원본 JPG 삭제")
    ap.add_argument("--overwrite", action="store_true", help="기존 PNG 덮어쓰기")
    args = ap.parse_args()

    convert_all_jpg_to_png(args.dir, delete_jpg=args.rm, overwrite=args.overwrite)
