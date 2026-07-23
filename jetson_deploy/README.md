# Jetson Orin Nano Deployment Package

이 패키지는 **두 가지 용도**를 담고 있습니다.
1. **ONNX 추론 검증** — `model/`, `reference/`, `scripts/`, `results/`
2. **Jetson Codex CLI 컨텍스트** — `codex_context/` (프로젝트 전체 자산)

새 Jetson Orin Nano를 처음 세팅할 때는 먼저 상위 문서
[`../docs/Jetson_Orin_Nano_초기세팅_가이드.md`](../docs/Jetson_Orin_Nano_초기세팅_가이드.md)를 기준으로 OS/권한/GPU/센서 검증을 진행하세요.

현장 설치 센서 조합에 대응되는 학습 데이터셋은 아직 없습니다. 현재 ONNX 모델은 Jetson GPU 추론과 파이프라인 검증용 기준 모델이며, 실제 현장 판정 모델은 센서 로그 수집과 라벨링 이후 재학습해야 합니다. 세부 전략은 [`../docs/이상상태_시나리오_및_데이터수집전략.md`](../docs/이상상태_시나리오_및_데이터수집전략.md)를 기준으로 합니다.

1차 설치에서는 센서를 추가하지 않고 기존 모델 입력과 맞는 `PureThermal`, `NTC/ADS1115`, `SPS30`, `DHS20P400A-CL420` 조합을 우선 검증합니다. BME680, SGP30, SCD30은 후보로만 기록하고 현장 데이터 수집이 안정화된 뒤 2차로 검토합니다.

## 패키지 구성

```
jetson_deploy/
├── README.md                           # ← 지금 이 파일
│
├── model/
│   └── model_v2plus.onnx              # V2+ 학습 모델 (11 MB)
├── reference/
│   ├── val_reference.npz              # 전체 val set (1157 samples, 1.4 GB)
│   └── val_reference_small.npz        # Stratified 100 samples (134 MB) — 4-class 균등
├── scripts/
│   ├── 01_check_environment.py        # Jetson 환경/패키지 확인
│   ├── 02_benchmark_latency.py        # 추론 latency 벤치마크
│   ├── 03_verify_accuracy.py          # PC ↔ Jetson 예측 일치 검증
│   ├── 04_realtime_pipeline.py        # 실시간 스트리밍 시뮬레이션
│   ├── 05_summary.py                  # 결과 통합 요약
│   ├── 06_capture_purethermal.py      # PureThermal UVC 캡처
│   ├── 07_read_sps30.py               # SPS30 미세먼지 I2C 읽기
│   └── 08_read_ntc_ads1115.py         # ADS1115 기반 NTC 온도 읽기
├── requirements-jetson.txt             # Jetson 사용자 공간 Python 패키지
├── results/                           # 스크립트 실행 결과 JSON 저장됨
│
└── codex_context/                     # ★ Codex CLI 작업 디렉토리 ★
    ├── AGENTS.md                      # Codex 자동 로드 (프로젝트 1페이지)
    ├── STATE.md                       # 현재 상태 + 다음 작업
    ├── INDEX.md                       # 빠른 파일 인덱스
    ├── HOW_TO_USE_CODEX.md            # Codex 사용 가이드 (먼저 읽을 것)
    ├── docs/                          # 연구노트 11편 + 논문 + 센서 가이드 + 에러로그
    └── code/                          # v2_plus.py, 데이터 파이프라인, deploy 등 핵심 코드
```

## 사전 준비 (Jetson에서 1회만)

### 1. ONNX Runtime 설치
JetPack 6.x + Python 3.10 환경 기준.

```bash
# 옵션 A: pip (CUDA EP 포함)
pip3 install onnxruntime-gpu numpy

# 옵션 B: Jetson 공식 wheel (TensorRT EP 포함, 추천)
# https://elinux.org/Jetson_Zoo#ONNX_Runtime
# 에서 본인 JetPack 버전에 맞는 .whl 다운로드 후
pip3 install onnxruntime_gpu-*.whl
```

### 2. 패키지 Jetson으로 옮기기
USB 드라이브 사용 시:
```bash
# PC 측 (이 디렉토리에서)
cd /home/keti/factory_safety
tar czf jetson_deploy.tar.gz jetson_deploy/

# USB에 복사 후 Jetson에서
tar xzf jetson_deploy.tar.gz
cd jetson_deploy
```

## 실행 순서

### Step 1. 환경 점검
```bash
python3 scripts/01_check_environment.py
```
**기대 출력:** `onnxruntime` 설치 확인, `CUDAExecutionProvider` 또는 `TensorrtExecutionProvider` 사용 가능 표시.

### Step 2. Latency 벤치마크
```bash
python3 scripts/02_benchmark_latency.py --runs 200
```
**목표:** mean < 5ms (single inference, batch=1).
사용 가능한 모든 EP(TensorRT > CUDA > CPU 우선순위)에 대해 측정.

### Step 3. 정확도 일치 검증
```bash
# 빠른 검증 (100 samples, stratified)
python3 scripts/03_verify_accuracy.py --small

# 전체 검증 (1157 samples)
python3 scripts/03_verify_accuracy.py
```
**기대값:**
- Pred match rate (Jetson vs PC ONNX) **≥ 99%**
- Macro-F1 ≈ 0.95 (전체), ≈ 0.96 (stratified small)
- Severe class F1 = 1.0

### Step 4. 실시간 파이프라인 시뮬레이션
```bash
python3 scripts/04_realtime_pipeline.py --n 300 --stride 10
```
30-window 슬라이딩 + 10-frame stride로 추론. FPS / per-window latency 측정.

### Step 5. 결과 통합
```bash
python3 scripts/05_summary.py
```
모든 결과를 `results/jetson_summary.json` 으로 통합 + 콘솔에 요약 출력.

## 결과 해석

### Latency 목표
| 항목 | PC (RTX 6000) | Jetson Orin Nano (목표) |
|------|--------------:|------------------------:|
| ONNX CPU | ~4.3 ms | < 15 ms |
| ONNX CUDA | ~4.4 ms | < 5 ms |
| ONNX TensorRT FP16 | - | **< 2 ms** (다음 단계) |

### 정확도 검증
ONNX는 수학적으로 결정론적이므로 Jetson EP가 표준 구현이라면 PC와 **완전 일치(100%)** 가 정상.
1% 이상 차이가 나면 ONNX opset 호환성 또는 EP 비결정성 문제 (특히 TensorRT FP16 활성화 시).

## 다음 단계 (이번 검증 통과 후)

1. **TensorRT 엔진 변환** — `trtexec --onnx=model_v2plus.onnx --saveEngine=v2plus.trt --fp16` 으로 < 2ms 달성
2. **전력/온도 모니터링** — `tegrastats` 백그라운드 측정
3. **실제 센서 연결** — NTC/PM/CT 센서 → I2C/SPI/ADC 입력 파이프라인
4. **현장 데이터 수집** — 정상 운전 로그, 안전 검증 이벤트, 라벨링
5. **현장 적용 모델 재학습** — 새 센서 feature vector 확정 후 ONNX/TensorRT 재배포

## Codex CLI 작업

Jetson 측에서 Codex CLI로 추가 작업을 할 때는 **반드시 `codex_context/` 디렉토리에서 실행**하세요. 이 폴더의 `AGENTS.md` 가 자동 로드되어 Codex가 프로젝트 전체 맥락을 알고 시작합니다.

```bash
cd jetson_deploy/codex_context
codex
# 첫 prompt 예시:
# > STATE.md §B 의 첫 작업부터 시작하자.
```

자세한 사용법은 [`codex_context/HOW_TO_USE_CODEX.md`](codex_context/HOW_TO_USE_CODEX.md) 참고.

## 문제 해결

| 증상 | 원인/해결 |
|------|----------|
| `libnvinfer.so.10: cannot open` | TensorRT 미설치 → `sudo apt install nvidia-tensorrt` |
| `Failed to create CUDAExecutionProvider` | onnxruntime-cpu 설치됨 → onnxruntime-gpu 재설치 |
| Pred match rate < 99% | ONNX opset 호환성 점검, FP16 비활성화 후 재실행 |
| Latency > 10ms (CUDA) | Jetson power mode 확인: `sudo nvpmodel -q` → MAXN 모드로 변경 |
