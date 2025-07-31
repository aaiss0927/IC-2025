import json
import argparse

p = argparse.ArgumentParser()
args = p.parse_args()

# 입력 caption_test.json 파일 경로
input_path = "/shared/home/kdd/HZ/inha-challenge/test/caption_internvl3.json"
output_path = "/shared/home/kdd/HZ/inha-challenge/test/pairs_internvl3.json"

# 1. caption_test.json 읽기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 변환: [ [filename, "caption1, caption2, ..."], ... ]
pairs = []
for key, captions in data.items():
    combined_caption = ", ".join(captions)  # 캡션 리스트를 하나의 문자열로 합치기
    pairs.append([key, combined_caption])  # [파일명, 합친 캡션 문자열]

# 3. 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"✅ 변환 완료! {output_path} 저장됨")
