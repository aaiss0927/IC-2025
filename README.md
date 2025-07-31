# 2025 인하 인공지능 챌린지
## Pretrained Checkpoint & dataset
준비 중... </br>

## Folder Structure
- ckpt, data 파일은 위의 Pretrained Checkpoint & dataset 섹션에서 다운로드 받은후 드라이브 구조와 동일하게 위치하면 됩니다.
- data_augmentation 코드를 활용하여 데이터 증강을 진행했으며, 완료된 결과 또한 드라이브에 포함되어 있습니다.
```bash
SW-2025/
├── ckpt/               # weight files
│   ├── full_text
│   │   └── epoch_1.pt
│   ├── gemma
│   │   └── checkpoint_2
│   ├── llama
│   │   └── checkpoint_3
│   ├── self_training
│   │   └── checkpoint_1
│   └── train_pseudo
│       └── checkpoint_1
├── data/               # 기존 데이터 & 증강 데이터 
│   ├── train.csv
│   ├── train_pseudo_label.csv
│   ├── train_llama.csv
│   ├── train_gemma.csv
│   ├── test.csv
│   └── sample_submission.csv
├── data_augmentation/  # 데이터 증강 코드
│   ├── gemma_augmentation.py
│   └── llama_augmentation.py
├── emsemble/           # 앙상블 코드
│   ├── ensemble_2.py
│   └── ensemble.py
├── inference/          # 추론 코드
│   ├── inference_custom.py
│   └── inference.py
├── pseudo_labeling/    # 수도 레이블링 코드
│   └── pseudo_labeling.py
├── scripts/            # 추론 스크립트
│   └── inference.sh
├── train/              # 학습 코드
│   ├── train_custom.py
│   └── train.py
├── README.md
└── requirements.txt
```
## Conda Environmet

- 라이브러리 버전은 requirements에 저장되어 있습니다.
- 아래 명령어를 순서대로 실행시키면 됩니다.

```bash
conda create -n iac python=3.10 -y
conda activate iac
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
cd sam2 && pip install -e . && cd ..
```

## Preprocessing
- 데이터셋을 모델의 입력 형태에 맞게 처리합니다.
```bash
python ./util/preprocessing.py
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
python ./L-CAD/colorization_main.py
```

## Inference
- 추론 시간을 단축시키기 위해 한 번의 모델 로드 후 다중 추론을 진행합니다.
- 단일 추론 시에는 guidance 인자를 하나만 입력합니다.
```bash
python ./L-CAD/inference_multi.py --guidance 4.0 4.5 5.0 5.5
```

## Ensemble
- 일반화 성능 향상을 위해 다양한 guidance로 생성된 이미지 픽셀 값의 평균으로 최정 결과를 도출합니다.
```bash
python ./L-CAD/ensemble.py
```

이후 최종 결과물은 submission/final_ensemble.zip 입니다.



