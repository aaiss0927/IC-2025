#!/usr/bin/env python
# ensemble_submissions_multi.py  ―― N개 ZIP(이미지 + embed_submission.csv) 앙상블
# ① 이미지: mean / median / weighted / lab
# ② embed CSV: ID·열 교집합 후 mean / median / weighted
#    - ID 컬럼 보존 (index → "ID")
#    - vec_0, vec_1 … vec_768 등은 **숫자순** 정렬

import argparse, zipfile, tempfile, shutil, os, cv2, numpy as np, tqdm, pandas as pd, re

# ───────── argparse ─────────
p = argparse.ArgumentParser()
p.add_argument("--zip", nargs="+", required=True, metavar="ZIP", help="input zip(s)")
p.add_argument(
    "--out", default="/shared/home/kdd/HZ/inha-challenge/ensembled/ensemble.zip"
)
p.add_argument("--mode", choices=["mean", "median", "lab"], default="mean")
p.add_argument(
    "--w", type=float, default=0.5, help="weight for the FIRST zip if --mode weighted"
)
args = p.parse_args()


def is_img(fn: str) -> bool:
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


# ───────── LAB 전용 블렌드 ─────────
def lab_blend_multi(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    labs = [cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in imgs_bgr]
    L_out = np.mean([lab[..., 0] for lab in labs], axis=0).astype(np.uint8)

    a_stack, b_stack = [np.stack([lab[..., i] for lab in labs]) for i in (1, 2)]
    chroma2 = (a_stack.astype(np.int16) - 128) ** 2 + (
        b_stack.astype(np.int16) - 128
    ) ** 2
    idx_max = chroma2.argmax(axis=0)
    a_out = a_stack[
        idx_max, np.arange(a_stack.shape[1])[:, None], np.arange(a_stack.shape[2])
    ]
    b_out = b_stack[
        idx_max, np.arange(b_stack.shape[1])[:, None], np.arange(b_stack.shape[2])
    ]
    lab_out = cv2.merge([L_out, a_out.astype(np.uint8), b_out.astype(np.uint8)])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ───────── natural sort for vec_0 … vec_10 … ─────────
_vec_re = re.compile(r"vec_(\d+)$")


def vec_sort(cols):
    return sorted(
        cols,
        key=lambda c: (
            not _vec_re.match(c),
            int(_vec_re.match(c).group(1)) if _vec_re.match(c) else c,
        ),
    )


# ───────── unzip ─────────
tmp_dirs: list[str] = []
for zp in args.zip:
    td = tempfile.mkdtemp()
    with zipfile.ZipFile(zp) as z:
        z.extractall(td)
    tmp_dirs.append(td)
tmp_out = tempfile.mkdtemp()

# ───────── 이미지 앙상블 ─────────
file_sets = [set(filter(is_img, os.listdir(td))) for td in tmp_dirs]
common_imgs = sorted(set.intersection(*file_sets))
if any(file_sets[0] ^ s for s in file_sets[1:]):
    print("[WARN] 일부 zip에만 있는 이미지가 있어 공통분모만 사용합니다")

for fn in tqdm.tqdm(common_imgs, desc="Ensembling imgs"):
    imgs = [cv2.imread(os.path.join(td, fn), cv2.IMREAD_UNCHANGED) for td in tmp_dirs]
    if any(im is None for im in imgs):
        print(f"[WARN] {fn} 읽기 실패 → skip")
        continue
    arrs = [im.astype(np.float32) for im in imgs]

    if args.mode == "mean":
        out = np.mean(arrs, axis=0)
    elif args.mode == "median":
        out = np.median(np.stack(arrs), axis=0)
    elif args.mode == "lab":
        out = lab_blend_multi(imgs).astype(np.float32)

    cv2.imwrite(
        os.path.join(tmp_out, fn), np.clip(out.round(), 0, 255).astype(np.uint8)
    )

# ───────── embed_submission.csv 결합 ─────────
csv_name = "embed_submission.csv"
csv_paths = [os.path.join(td, csv_name) for td in tmp_dirs]

if all(os.path.exists(p) for p in csv_paths):
    dfs = [
        (
            pd.read_csv(p).set_index("ID")
            if "ID" in pd.read_csv(p, nrows=1).columns
            else pd.read_csv(p).set_index("index").rename_axis("ID")
        )
        for p in csv_paths
    ]

    common_ids = sorted(set.intersection(*(set(df.index) for df in dfs)))
    common_cols = vec_sort(set.intersection(*(set(df.columns) for df in dfs)))

    if common_ids and common_cols:
        mats = [df.loc[common_ids, common_cols].to_numpy(np.float32) for df in dfs]
        stack = np.stack(mats)  # (n_zip, n_id, n_dim)

        if args.mode in ("mean", "lab"):
            merged = stack.mean(axis=0)
        elif args.mode == "median":
            merged = np.median(stack, axis=0)

        df_out = pd.DataFrame(merged, index=common_ids, columns=common_cols)
        df_out.index.name = "ID"  # 반드시 ID 컬럼으로
        df_out.reset_index().to_csv(os.path.join(tmp_out, csv_name), index=False)
        print(f"[✓] embed csv merged  (rows={len(common_ids)})")
    else:
        print("[WARN] embed csv 교집합이 없어 병합을 생략합니다")
else:
    print("[WARN] embed_submission.csv가 일부 zip에 없습니다 → 병합 생략")

# ───────── 결과 zip ─────────
with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(tmp_out):
        z.write(os.path.join(tmp_out, fn), arcname=fn)
print(f"[✓] saved ensemble: {args.out}")

# ───────── 정리 ─────────
for td in tmp_dirs:
    shutil.rmtree(td, ignore_errors=True)
shutil.rmtree(tmp_out, ignore_errors=True)
