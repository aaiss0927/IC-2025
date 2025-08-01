# 2025 인하 인공지능 챌린지
## Pretrained Checkpoint & dataset
https://drive.google.com/drive/folders/1ZPopR4h3fn9Tx_zdVRaqJ56-43JbBzW8?usp=sharing </br>

## Folder Structure
- ckpt, data 파일은 위의 Pretrained Checkpoint & dataset 섹션에서 다운로드 받은후 드라이브 구조와 동일하게 위치하면 됩니다.
- ./models/largedecoder-checkpoint.pth : https://github.com/changzheng123/L-CAD 에서 공개한 사전 가중치입니다. (https://drive.google.com/drive/folders/1_zVJrp_UkFDaZpcC8aLzpv4iPsHADQU-)
- ./models/multi_weight.ckpt : https://github.com/changzheng123/L-CAD 에서 공개한 사전 가중치입니다. (https://drive.google.com/drive/folders/1_zVJrp_UkFDaZpcC8aLzpv4iPsHADQU-)
- ./models/sam2.1_hiera_base_plus.pt : https://github.com/facebookresearch/sam2 에서 공개한 사전 가중치입니다.
```bash
IC-2025/
├── L-CAD/         # L-CAD 레포지토리
│   ├── cldm
│   ├── configs
│   ├── ldm
│   ├── colorization_dataset.py
│   ├── colorization_main.py
│   ├── config.py
│   ├── ensemble.py
│   ├── inference.py
│   ├── README.md
│   └── share.py
├── models/        # 사전 가중치
│   ├── largedecoder-checkpoint.pth
│   ├── multi_weight.ckpt
│   └── sam2.1_hiera_base_plus.pt
├── sam_mask/
│   ├── select_masks
│   └── make_masks.py
├── sam2/          # SAM 레포지토리
├── submission/    # 추론 결과
│   ├── submission_g4p0.zip
│   ├── submission_g4p5.zip
│   ├── submission_g5p0.zip
│   ├── submission_g5p5.zip
│   └── final_ensemble.zip
├── test/          # 테스트 데이터셋
│   ├── input_image
│   ├── pairs.json
│   └── test.csv
├── train/         # 학습 데이터셋
│   ├── gt_image
│   ├── input_image
│   ├── caption_train.json
│   └── train.csv
├── util/
│   └── processing.py
├── README.md
└── requirements.txt
```
## Conda Environmet

- 라이브러리 버전은 requirements에 저장되어 있습니다.
- 아래 명령어를 순서대로 실행시키면 됩니다.

```bash
git clone https://github.com/aaiss0927/IC-2025.git
cd IC-2025
conda create -n iac python=3.10 -y
conda activate iac
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
cd sam2 && pip install -e . && cd ..
```

## Preprocessing
- 데이터셋을 모델의 입력 형태에 맞게 처리합니다.
```bash
python ./util/processing.py --type train --csv_path ./train/train.csv --out_json ./train/caption_train.json
python ./util/processing.py --type test --csv_path ./test/test.csv --out_json ./test/pairs.json
```

## Masking
- SAM2.1_base 모델을 활용하여 테스트 이미지에 대하여 세그멘테이션을 진행합니다.
```bash
python ./sam_mask/make_mask.py
```

## Train (Optional)
- L-CAD의 사전 가중치인 multi_weight.ckpt로 추론 시 최고 성능을 달성하였습니다.
- Private Score 복원을 위해서는 학습을 생략하고 multi_weight.ckpt로 추론을 진행합니다.
```bash
python ./L-CAD/colorization_main.py -t
```

## Inference
- 추론 시간을 단축시키기 위해 한 번의 모델 로드 후 다중 추론을 진행합니다.
- 단일 추론 시에는 guidance 인자를 하나만 입력합니다.
```bash
python ./L-CAD/inference.py --guidance 4.0 4.5 5.0 5.5
```

## Ensemble
- 일반화 성능 향상을 위해 다양한 guidance로 생성된 이미지 픽셀 값의 평균으로 최정 결과를 도출합니다.
```bash
python ./L-CAD/ensemble.py --mode mean --zip ./submission/submission_g4p0.zip ./submission/submission_g4p5.zip ./submission/submission_g5p0.zip ./submission/subm
ission_g5p5.zip
```

이후 최종 결과물은 submission/final_ensemble.zip 입니다.



