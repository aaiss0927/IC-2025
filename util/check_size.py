from PIL import Image

# 이미지 파일 경로
img_path = "/shared/home/kdd/HZ/inha-challenge/train/input_image/001234.jpg"

# 이미지 열기
img = Image.open(img_path)

# width, height 가져오기
w, h = img.size

print(f"Width: {w}, Height: {h}")
