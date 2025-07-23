import csv
import json

csv_path = "./train.csv"  # 입력 CSV 경로
out_json = "./train/caption_train.json"  # 출력 JSON 경로

caption_map = {}
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row["gt_img_path"].split("/")[-1]  # gt_image 폴더의 파일명 추출
        caption = row["caption"].strip()
        caption_map[img_name] = [caption]

# JSON 저장
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(caption_map, f, ensure_ascii=False, indent=2)

print(f"✔ Saved {len(caption_map)} captions → {out_json}")
