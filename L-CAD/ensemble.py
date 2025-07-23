#!/usr/bin/env python
# ensemble_submissions.py  (CSV까지 처리 버전)
import argparse, zipfile, tempfile, shutil, os, cv2, numpy as np, tqdm, pandas as pd

# ───────── argparse ─────────
p = argparse.ArgumentParser()
p.add_argument("--zip_a", required=True)
p.add_argument("--zip_b", required=True)
p.add_argument(
    "--out", default="/shared/home/kdd/HZ/inha-challenge/ensembled/ensemble.zip"
)
p.add_argument("--mode", choices=["mean", "median", "weighted", "lab"], default="mean")
p.add_argument(
    "--w", type=float, default=0.5, help="weight for zip_a if --mode weighted"
)
args = p.parse_args()


def is_img(fn):
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def lab_blend(img_a, img_b):
    """L은 평균, ab는 색채도가 큰 쪽 선택 (BGR uint8 → BGR uint8)"""
    lab_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2LAB)

    L_a, a_a, b_a = cv2.split(lab_a)
    L_b, a_b, b_b = cv2.split(lab_b)

    # 1) 밝기 평균
    L_out = ((L_a.astype(np.float32) + L_b.astype(np.float32)) / 2).astype(np.uint8)

    # 2) 색채도(chroma = sqrt((a-128)^2+(b-128)^2)) 큰 쪽 선택
    ca2 = (a_a.astype(np.int16) - 128) ** 2 + (b_a.astype(np.int16) - 128) ** 2
    cb2 = (a_b.astype(np.int16) - 128) ** 2 + (b_b.astype(np.int16) - 128) ** 2
    mask = ca2 >= cb2  # True → img_a 선택, False → img_b

    a_out = np.where(mask, a_a, a_b)
    b_out = np.where(mask, b_a, b_b)

    lab_out = cv2.merge([L_out, a_out, b_out])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ───────── unzip ─────────
tmp_a, tmp_b = tempfile.mkdtemp(), tempfile.mkdtemp()
with zipfile.ZipFile(args.zip_a) as z:
    z.extractall(tmp_a)
with zipfile.ZipFile(args.zip_b) as z:
    z.extractall(tmp_b)
tmp_out = tempfile.mkdtemp()  # 결과 저장

# ───────── 이미지 앙상블 ─────────
files_a = sorted([f for f in os.listdir(tmp_a) if is_img(f)])
files_b = sorted([f for f in os.listdir(tmp_b) if is_img(f)])
common = sorted(set(files_a) & set(files_b))
if set(files_a) ^ set(files_b):
    print("[WARN] 두 zip에 불일치 파일 있음 → 공통분모만 사용")

for fn in tqdm.tqdm(common, desc="Ensembling imgs"):
    ia = cv2.imread(os.path.join(tmp_a, fn), cv2.IMREAD_UNCHANGED)
    ib = cv2.imread(os.path.join(tmp_b, fn), cv2.IMREAD_UNCHANGED)
    if ia is None or ib is None:
        print(f"[WARN] {fn} 읽기 실패 → skip")
        continue

    a, b = ia.astype(np.float32), ib.astype(np.float32)
    if args.mode == "mean":
        out = (a + b) / 2
    elif args.mode == "median":
        out = np.median(np.stack([a, b]), axis=0)
    elif args.mode == "lab":
        out = lab_blend(ia, ib)
    else:
        out = args.w * a + (1 - args.w) * b
    cv2.imwrite(
        os.path.join(tmp_out, fn), np.clip(out.round(), 0, 255).astype(np.uint8)
    )

# ───────── embed_submission.csv 결합 ─────────
csv_name = "embed_submission.csv"
path_csv_a, path_csv_b = os.path.join(tmp_a, csv_name), os.path.join(tmp_b, csv_name)
if os.path.exists(path_csv_a) and os.path.exists(path_csv_b):
    df_a = pd.read_csv(path_csv_a).set_index("ID")
    df_b = pd.read_csv(path_csv_b).set_index("ID")
    common_ids = df_a.index.intersection(df_b.index)

    if args.mode == "mean":
        df_out = (df_a.loc[common_ids] + df_b.loc[common_ids]) / 2
    elif args.mode == "median":
        df_out = (
            pd.concat([df_a.loc[common_ids], df_b.loc[common_ids]])
            .groupby(level=0)
            .median()
        )
    else:
        w = args.w
        df_out = w * df_a.loc[common_ids] + (1 - w) * df_b.loc[common_ids]

    df_out.reset_index().to_csv(os.path.join(tmp_out, csv_name), index=False)
    print(f"[✓] embed csv merged  (rows={len(df_out)})")
else:
    print("[WARN] embed_submission.csv 둘 중 하나 없음 → 복사/생성 생략")

# ───────── 결과 zip ─────────
with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(tmp_out):
        z.write(os.path.join(tmp_out, fn), arcname=fn)
print(f"[✓] saved ensemble: {args.out}")

# ───────── 정리 ─────────
shutil.rmtree(tmp_a)
shutil.rmtree(tmp_b)
shutil.rmtree(tmp_out)
