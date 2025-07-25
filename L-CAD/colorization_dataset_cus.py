import json, cv2, os, random
import numpy as np
from PIL import Image
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ────────────────────────────────────────────────────────────────────────────────
class MyDataset(Dataset):
    def __init__(
        self,
        img_dir,  # ex) "./train/gt_image"  or  "/data/coco"
        caption_dir=None,  # ex) "./train"           or  "resources/coco"
        split="train",  # "train" | "val" | "test"
        img_size=512,
        use_sam=False,
        mask_root="sam_mask/select_masks",
        fallback_mode="zeros",  # "zeros" | "ones" | "none"
        coco_style=True,  # ★ NEW: COCO dir 구조 사용 여부
    ):
        """
        img_dir      : split=="test" 일 때는 실제 이미지들이 바로 있는 디렉토리
                       train/val 에선 COCO-style(…/train2017) 또는 그대로 사용(coco_style=False)
        caption_dir  : caption_(train|val).json  또는  pairs.json  이 위치한 폴더
        mask_root    : SAM 마스크 루트 (서브폴더는 <이미지파일명_확장자제외>/mask*.npy)
        fallback_mode: SAM 마스크가 없을 때 대응 방법
        coco_style   : True  → img_dir/train2017 또는 /val2017 자동 이어붙임
                       False → img_dir 그대로 사용 (GT 이미지가 바로 들어있는 커스텀 구조)
        """
        assert split in ["train", "val", "test"]
        self.split = split
        self.img_size = img_size
        self.use_sam = use_sam
        self.mask_root = mask_root
        self.fallback_mode = fallback_mode
        self.istest = split == "test"

        # ── 이미지 루트 결정 ────────────────────────────────────────────────
        if self.istest:
            # ex) "./test/input_image"
            self.img_dir = img_dir
        else:
            if coco_style:
                # ex) "/data/coco" → "/data/coco/train2017"
                self.img_dir = os.path.join(img_dir, split + "2017")
            else:
                # ex) "./train/gt_image"
                self.img_dir = img_dir

        # ── 기본 RGB 정규화 ────────────────────────────────────────────────
        self.norm = transforms.Normalize([0.5] * 3, [0.5] * 3)

        # ── split 별 설정 ─────────────────────────────────────────────────
        if split == "train":
            caption_path = os.path.join(caption_dir, "caption_train_mine.json")
            self.transform = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(
                    #     (img_size, img_size), scale=(0.8, 1.0), interpolation=3
                    # ),
                    # transforms.RandomHorizontalFlip(),  # ←  필요 시 p 값 조정
                    # transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )
            self.caption_file = json.load(open(caption_path, "r"))
            self.keys = list(self.caption_file.keys())

        elif split == "val":
            caption_path = os.path.join(caption_dir, "caption_val.json")
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )
            self.caption_file = json.load(open(caption_path, "r"))
            self.keys = list(self.caption_file.keys())

        else:  # test
            caption_path = os.path.join(caption_dir, "pairs.json")
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )
            print(caption_path)
            self.pairs = json.load(open(caption_path, "r"))

    # ── 이미지 로드 & LAB 분해 ─────────────────────────────────────────────
    def get_img(self, img_name):
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth).convert("RGB")
        img = self.transform(img)  # [3,H,W] 0‥1
        img_lab = rgb2lab(img)  # [3,H,W] LAB(norm)

        img_l = img_lab[[0]].repeat(3, 1, 1)  # L → 3‑ch hint
        img_ab = img_lab[1:]

        img = self.norm(img)  # [-1,1]  RGB target

        # HWC 로 변환하여 model hint 와 concat 용
        img_l = rearrange(img_l, "c h w -> h w c")
        img = rearrange(img, "c h w -> h w c")
        img_ab = rearrange(img_ab, "c h w -> h w c")

        return img_l, img, img_ab

    # ── (train/val) 캡션 무작위 하나 선택 ────────────────────────────────
    def get_caption(self, key):
        caps = self.caption_file[key]
        idx = random.randrange(len(caps))
        return caps[idx], idx

    # ── SAM 마스크 로드 ──────────────────────────────────────────────────
    def get_mask(self, img_name):
        subdir = os.path.join(self.mask_root, os.path.splitext(img_name)[0])
        masks = []
        if os.path.isdir(subdir):
            for fn in sorted(os.listdir(subdir)):
                if not fn.endswith(".npy"):
                    continue
                m = np.load(os.path.join(subdir, fn)).astype("float32")
                m = cv2.resize(
                    m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
                )
                masks.append(m[None])  # [1,H,W]
        if not masks:  # fallback
            if self.fallback_mode == "none":
                return None
            fill = 0.0 if self.fallback_mode == "zeros" else 1.0
            masks = [np.full((1, self.img_size, self.img_size), fill, dtype="float32")]
        masks = torch.from_numpy(np.concatenate(masks, axis=0))  # [N,H,W]
        return masks

    # ── 필수 메서드들 ────────────────────────────────────────────────────
    def __len__(self):
        return len(self.pairs) if self.istest else len(self.keys)

    def __getitem__(self, idx):
        if self.istest:
            key, cap = self.pairs[idx]
            img_l, img, _ = self.get_img(key)
        else:
            key = self.keys[idx]
            img_l, img, _ = self.get_img(key)
            cap, _ = self.get_caption(key)

        sample = dict(jpg=img, txt=cap, hint=img_l, name=key)

        if self.use_sam:
            sample["mask"] = self.get_mask(key)  # torch.Tensor [N,H,W] or None
        return sample


# ───────────────────────── 색 공간 유틸 전부 그대로 ─────────────────────────
def rgb2xyz(rgb):
    mask = (rgb > 0.04045).float()
    if rgb.is_cuda:
        mask = mask.cuda()
    rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)
    x = 0.412453 * rgb[0] + 0.357580 * rgb[1] + 0.180423 * rgb[2]
    y = 0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2]
    z = 0.019334 * rgb[0] + 0.119193 * rgb[1] + 0.950227 * rgb[2]
    return torch.stack((x, y, z), 0)


def xyz2lab(xyz):
    sc = torch.tensor([0.95047, 1.0, 1.08883])[:, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()
    xyz_scale = xyz / sc
    mask = (xyz_scale > 0.008856).float()
    if xyz_scale.is_cuda:
        mask = mask.cuda()
    xyz_int = xyz_scale.pow(1 / 3) * mask + (7.787 * xyz_scale + 16.0 / 116.0) * (
        1 - mask
    )
    L = 116.0 * xyz_int[1] - 16.0
    a = 500.0 * (xyz_int[0] - xyz_int[1])
    b = 200.0 * (xyz_int[1] - xyz_int[2])
    return torch.stack((L, a, b), 0)


def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = lab[[0]] / 127.5 - 1
    ab_rs = lab[1:] / 110.0
    return torch.cat((l_rs, ab_rs), 0)


def lab2rgb(lab_rs):
    l = lab_rs[:, [0]] / 2.0 * 100.0 + 50.0
    ab = lab_rs[:, 1:] * 110.0
    lab = torch.cat((l, ab), 1)
    return xyz2rgb(lab2xyz(lab))


def lab2xyz(lab):
    y = (lab[:, 0] + 16.0) / 116.0
    x = lab[:, 1] / 500.0 + y
    z = y - lab[:, 2] / 200.0
    z = torch.clamp(z, min=0)
    out = torch.stack((x, y, z), 1)
    mask = (out > 0.2068966).float()
    if out.is_cuda:
        mask = mask.cuda()
    out = out.pow(3) * mask + (out - 16.0 / 116.0) / 7.787 * (1 - mask)
    sc = torch.tensor([0.95047, 1.0, 1.08883])[None, :, None, None].to(out.device)
    return out * sc


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0] - 1.53715152 * xyz[:, 1] - 0.49853633 * xyz[:, 2]
    g = -0.96925495 * xyz[:, 0] + 1.87599 * xyz[:, 1] + 0.04155593 * xyz[:, 2]
    b = 0.05564664 * xyz[:, 0] - 0.20404134 * xyz[:, 1] + 1.05731107 * xyz[:, 2]
    rgb = torch.stack((r, g, b), 1)
    rgb = torch.clamp(rgb, min=0.0)
    mask = (rgb > 0.0031308).float()
    if rgb.is_cuda:
        mask = mask.cuda()
    rgb = (1.055 * rgb.pow(1 / 2.4) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    return rgb
