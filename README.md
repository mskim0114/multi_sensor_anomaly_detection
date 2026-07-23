# 제조공장 멀티모달 센서 이상상태 예측 AI

## Jetson Orin Nano 배포/초기 세팅

새 Jetson Orin Nano 보드를 받을 때는 아래 문서를 기준으로 OS, JetPack, GPU 추론, 센서 연결을 순서대로 검증한다.

- [Jetson Orin Nano 초기 세팅 가이드](docs/Jetson_Orin_Nano_초기세팅_가이드.md)
- [Jetson Orin Nano 40-pin 핀맵](docs/Jetson_Orin_Nano_40pin_pinmap.md)
- [SPS30 Jetson I2C 연결 메모](docs/SPS30_Jetson_I2C_연결.md)
- [NTC 10K + ADS1115 Jetson 연결 메모](docs/NTC10K_ADS1115_Jetson_연결.md)
- [이상상태 시나리오 및 데이터 수집 전략](docs/이상상태_시나리오_및_데이터수집전략.md)

중요 전제: 실제 설치 환경과 추가 센서 조합에 대응되는 학습 데이터셋은 아직 없다. 현재 ONNX 모델은 Jetson GPU 추론과 파이프라인 검증용 기준 모델로 사용하고, 현장 이상상태 판정 모델은 정상 데이터 수집과 라벨링 이후 재학습한다.

현재 실행 방침: 센서 추가는 보류하고, 기존 모델 입력 구조와 맞는 `PureThermal 열화상`, `NTC 10K + ADS1115`, `SPS30 미세먼지`, `DHS20P400A-CL420 전류` 4종을 먼저 연결/검증/수집한다. BME680, SGP30, SCD30은 1차 현장 데이터 수집이 안정화된 뒤 필요성이 확인되면 추가한다.

## 환경 설정

```bash
# 1. conda 환경 활성화
conda activate monai_env

# 2. 프로젝트 디렉토리 이동
cd /home/keti/factory_safety
```

## 추론 데모 실행

### 화면에 결과 표시 (GUI)
```bash
python src/demo_inference.py
```

### 옵션
```bash
python src/demo_inference.py --num-samples 10         # 10개 샘플 보기
python src/demo_inference.py --show-errors-only       # 오분류만 보기
python src/demo_inference.py --save-dir results/demo  # 이미지 파일로 저장
python src/demo_inference.py --gpu 0                  # GPU 0 사용 (기본: GPU 1)
```

### 화면 구성
- 1행: 센서 시계열 (온도, 미세먼지, 전류)
- 2행: 열화상 이미지 4프레임
- 3행: 예측 확률 + 센서 변화율

## 모델 학습

```bash
# 베이스라인 (Multimodal LSTM)
python -m src.train_baseline --epochs 20 --gpu 1

# CATFT (Cross-Attention Temporal Fusion Transformer)
python -m src.train_catft --epochs 30 --gpu 1

# Ablation Study (V2, V3, V4 순차 학습)
python -m src.train_ablation --gpu 1
```

## 데이터 파이프라인

```bash
# 세션 인덱스 빌드 (최초 1회)
python -m src.data.scripts.build_index

# 정규화 통계 확인
python -m src.data.scripts.compute_stats
```

## 프로젝트 구조

```
factory_safety/
├── configs/
│   └── data_config.yaml                # 데이터 파이프라인 설정
├── src/
│   ├── data/                           # 데이터 파이프라인
│   │   ├── config.py                   #   설정 dataclass
│   │   ├── session_index.py            #   세션 탐지 + 인덱스
│   │   ├── dataset.py                  #   PyTorch Dataset
│   │   ├── normalization.py            #   Z-score / MinMax 정규화
│   │   ├── augmentation.py             #   데이터 증강
│   │   ├── sampler.py                  #   클래스 균형 샘플링
│   │   ├── datamodule.py               #   DataLoader 팩토리
│   │   └── scripts/                    #   CLI 도구
│   ├── models/                         # 모델
│   │   ├── multimodal_lstm.py          #   베이스라인 LSTM
│   │   ├── catft.py                    #   Cross-Attention Transformer
│   │   └── ablation_variants.py        #   Ablation 변형 모델
│   ├── deploy/                         # 배포 (TensorRT, ONNX) [작업 예정]
│   ├── train_baseline.py               # 베이스라인 학습
│   ├── train_catft.py                  # CATFT 학습
│   ├── train_ablation.py               # Ablation study
│   └── demo_inference.py               # 추론 데모 (시각화)
├── results/                            # 학습 결과 + 체크포인트
│   ├── baseline/                       #   V1: Multimodal LSTM
│   ├── ablation_v2/                    #   V2: LSTM+TempDiff (최고 성능)
│   ├── ablation_v3/                    #   V3: LSTM+EfficientNet
│   ├── ablation_v4/                    #   V4: CATFT-NoCrossAttn
│   ├── catft/                          #   V5: Full CATFT
│   └── demo/                           #   추론 데모 이미지
├── docs/
│   ├── 연구노트/                        #   연구노트 #01~#06
│   ├── 보고서/                          #   실험결과 보고서
│   ├── 참고문서/                        #   AI Hub 분석, 로드맵, 센서 가이드
│   ├── 이상상태_시나리오_및_데이터수집전략.md
│   ├── figures/                        #   보고서용 이미지
│   └── 오류_및_해결_로그.md              #   트러블슈팅 기록
├── data/aihub/
│   └── datasets/extracted/             # 데이터셋 (18GB)
└── cache/                              # 세션 인덱스 캐시
```
