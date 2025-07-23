import cv2, glob, json
import numpy as np
from PIL import Image
import open_clip
import torch, tqdm, os

PRED_DIR = "/shared/home/kdd/HZ/inha-challenge/test/input_image"
PAIRS_PTH = (
    "/shared/home/kdd/HZ/inha-challenge/test/pairs.json"  # [ [filename, caption], ... ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model.to(device).eval()
tok = open_clip.get_tokenizer("ViT-L-14")


def clip_score(img, txt):
    with torch.no_grad():
        txt_feat = clip_model.encode_text(tok([txt]).to(device))
        img_feat = clip_model.encode_image(preprocess(img).unsqueeze(0).to(device))
        txt_feat, img_feat = txt_feat / txt_feat.norm(
            dim=-1, keepdim=True
        ), img_feat / img_feat.norm(dim=-1, keepdim=True)
    return (txt_feat @ img_feat.T).item()


# Load pairs.json as dict
with open(PAIRS_PTH, encoding="utf-8") as f:
    pairs_list = json.load(f)
pairs = {item[0]: item[1] for item in pairs_list}  # {"TEST_001.png": "caption..."}

clips = []
for pred_pth in tqdm.tqdm(sorted(glob.glob(f"{PRED_DIR}/*.png"))):
    img_id = os.path.basename(pred_pth)

    if img_id not in pairs:
        print(f"[WARN] {img_id} not in pairs.json, skipping...")
        continue

    caption = pairs[img_id]
    pred_img = Image.open(pred_pth).convert("RGB")
    clips.append(clip_score(pred_img, caption))

if len(clips) > 0:
    print(f"CLIP ⟨sim⟩: {np.mean(clips):.3f}")
else:
    print("[ERROR] No valid predictions matched with pairs.json.")
