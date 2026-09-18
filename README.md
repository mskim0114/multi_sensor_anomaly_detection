# 제조공장 멀티모달 센서 이상상태 예측 AI

## 실행 전 필수 문서

- [AGENTS.md](AGENTS.md)
- [ENVIRONMENT_POLICY.md](docs/ENVIRONMENT_POLICY.md)
- [RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md)
- [SERVER_WORKSTATION_HANDOFF.md](docs/SERVER_WORKSTATION_HANDOFF.md)

작업 루트(root cwd): `/home/keti/projects/factory_safety`

현재 상태는 실측 데이터 기반 현장 모델 파이프라인이 준비되지 않았고(`ModelAdapter` 미완료),
또한 이상 사건 데이터도 미확보(`B-4`) 상태입니다. RESEARCH_STATUS의 현재 상태를 반드시 확인하고,
데이터 축적/재학습 전단계로 진행하세요.

## Jetson Orin Nano 배포/초기 세팅

새 Jetson Orin Nano 보드를 받을 때는 아래 문서를 기준으로 OS, JetPack, GPU 추론, 센서 연결을 순서대로 검증한다.

- [Jetson Orin Nano 초기 세팅 가이드](docs/Jetson_Orin_Nano_초기세팅_가이드.md)
- [Jetson Orin Nano 40-pin 핀맵](docs/Jetson_Orin_Nano_40pin_pinmap.md)
- [SPS30 Jetson I2C 연결 메모](docs/SPS30_Jetson_I2C_연결.md)
- [NTC 10K + ADS1115 Jetson 연결 메모](docs/NTC10K_ADS1115_Jetson_연결.md)
- [이상상태 시나리오 및 데이터 수집 전략](docs/이상상태_시나리오_및_데이터수집전략.md)

중요 전제: 실제 설치 환경과 추가 센서 조합에 대응되는 학습 데이터셋은 아직 없다. 현재 ONNX 모델은 Jetson GPU 추론과 파이프라인 검증용 기준 모델로 사용하고, 현장 이상상태 판정 모델은 정상 데이터 수집과 라벨링 이후 재학습한다.

현재 수집 구성과 배선은 [JETSON_SENSOR_COLLECTION.md](docs/JETSON_SENSOR_COLLECTION.md)를 따른다.
개발용 baseline은 NTC·PM 3채널·CT1·열화상을 기록하고 SCD30/BME680을 context로 보관한다.
기존 모델이 요구하는 CT2~CT4는 없으며, 실제 데이터와 모델 입력을 연결하는 설계가 남아 있다.

## 환경 설정

Jetson에서는 저장소 루트에서 전용 래퍼를 사용한다.

```bash
cd /home/keti/projects/factory_safety
./jetson_deploy/run_python.sh jetson_deploy/check_environment.py
```

서버의 아래 학습·데모·AI Hub 처리 예시는 **서버 환경 audit와 데이터 복구·재현 완료 후**
`$HOME/venvs/factory_training` 환경에서 실행한다. Jetson 런타임에 학습 의존성을 설치하지 않는다.
기본 입출력 경로는 현재 checkout을 기준으로 정해진다. `DataConfig`와 YAML에 지정한 상대
데이터·캐시 경로도 checkout 기준이며, 외부 저장소에는 절대경로를 지정한다.

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
│   ├── deploy/                         # ONNX export·기준 입력 생성·배포 검증
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
│   └── datasets/extracted/             # AI Hub 데이터: 현재 0바이트 파일 복구 필요
└── cache/                              # 세션 인덱스 캐시
```
