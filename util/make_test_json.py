# make_pairs.py
import csv, json, os, pathlib

csv_path = "test.csv"  # ← 위치 맞춰 수정
out_json = pathlib.Path("test/pairs.json")
out_json.parent.mkdir(parents=True, exist_ok=True)

pairs = []
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        img_name = os.path.basename(row["input_img_path"])  # ex) TEST_001.png
        caption = row["caption"]
        pairs.append([img_name, caption])

json.dump(pairs, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"✔ saved {len(pairs)} pairs → {out_json}")
