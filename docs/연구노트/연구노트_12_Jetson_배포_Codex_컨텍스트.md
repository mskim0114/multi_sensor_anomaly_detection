# 연구노트 #12: Jetson 배포 패키지 · Codex CLI 컨텍스트

**작성일:** 2026-05-22 (스냅샷 시점)
**정리일:** 2026-08-07 (retrospective, 세션 실기록)
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 배경

V2+ 모델 학습·논문 draft·PC 측 ONNX 검증이 4월에 완료 (연구노트 #09, #11 참조). 다음 단계는 **Jetson Orin Nano 8GB 보드에서 실시간 추론 검증**이었다. 사용자가 Jetson 보드에 JetPack + Python + PyTorch 를 세팅하고 완료 통보한 시점부터 시작.

**목표**:
- Jetson 에서 ONNX 추론이 실제로 돌아가는지 (환경 · 라이브러리 호환)
- Latency 목표 (CUDA EP < 5 ms) 달성 여부
- PC ↔ Jetson 예측 100% 일치 확인 (deterministic 검증)
- 실시간 스트리밍 파이프라인 정상 동작 확인

**전송 방식 제약**: Jetson 은 모니터 + 키보드 직접 연결, USB 전송만 가능.

---

## 2. Jetson 배포 패키지 구성 (`jetson_deploy/`)

PC 측에서 다음 5개 검증 스크립트 + 참조 데이터 + 모델을 하나의 폴더로 묶음:

```
jetson_deploy/
├── README.md                           # 실행 가이드
├── model/
│   └── model_v2plus.onnx              # 11 MB (V2+ ONNX export)
├── reference/
│   ├── val_reference.npz              # 전체 val (1157 samples, 1.4 GB)
│   └── val_reference_small.npz        # Stratified 100 (25 x 4 class, 134 MB)
├── scripts/
│   ├── 01_check_environment.py        # onnxruntime · 사용 가능 EP · L4T · GPU
│   ├── 02_benchmark_latency.py        # EP별 latency (warmup=20, runs=200)
│   ├── 03_verify_accuracy.py          # PC ↔ Jetson pred match + F1
│   ├── 04_realtime_pipeline.py        # 30-deep ring buffer + stride=10 streaming
│   └── 05_summary.py                  # results/jetson_summary.json 통합
└── results/                           # 실행 결과 (Jetson 측에서 생성)
```

**Reference dataset 생성** (`src/deploy/generate_reference_data.py`):
- 입력 sensor (1157, 30, 8), thermal (1157, 30, 120, 160), label (1157,)
- PyTorch 예측 (pt_logits, pt_preds) 와 ONNX 예측 (onnx_logits, onnx_preds) 를 함께 저장
- **PC 측 검증**: PT ↔ ONNX 매치 = 1.0 (1157/1157), val acc = 0.9516

**Stratified 100-sample subset**: 첫 50-sample subset 이 class 불균형 (Normal 48 / Mild 2) 이라 → 각 클래스 25개씩 100 샘플로 재추출.

**결선 무관 검증 원칙**: Jetson 스크립트는 numpy + onnxruntime 만 사용 (PyTorch 없음). Jetson 환경에서 PyTorch 설치 여부와 상관없이 동작.

---

## 3. Codex CLI 컨텍스트 패키지 (`jetson_deploy/codex_context/`)

사용자가 Jetson 에도 **Codex CLI** 를 설치했다고 통보 → 별도로 프로젝트 전체 맥락을 전달할 필요.

Codex CLI 는 실행 디렉토리에 있는 **`AGENTS.md` 를 자동으로 system prompt 에 주입** 하는 관례를 이용:

```
codex_context/
├── AGENTS.md              # Codex 자동 로드 (프로젝트 1페이지 요약)
├── STATE.md               # 완료/진행/대기 작업 + 날짜
├── INDEX.md               # "이건 어디 있어?" 빠른 탐색
├── HOW_TO_USE_CODEX.md    # Jetson 측 Codex 실행법 (먼저 읽을 것)
├── docs/                  # 연구노트 11편 + 논문 + 센서 가이드 + 에러로그
└── code/                  # v2_plus.py · 데이터 파이프라인 · deploy 스크립트
```

### AGENTS.md 구성 (10 소절)
1. 프로젝트 1줄 요약
2. V2+ 모델 상세 (multi-scale diff + LSTM + SE + SupCon, 성능표)
3. 데이터셋 (AI Hub #71802, 8 sensor + thermal, 4-class)
4. 하드웨어 · 센서 계획
5. 실시간 추론 시스템 설계
6. 현재 상태
7. 디렉토리 인덱스
8. 작업 규칙 (device 분리, Severe 우선 등)
9. 자주 쓰는 명령
10. Codex 사용 팁

### 사용 방법
```bash
cd ~/factory_safety/jetson_deploy/codex_context
codex   # AGENTS.md 자동 로드된 세션
> STATE.md §B 의 첫 작업부터 시작하자
```

---

## 4. 전송 옵션 (USB)

| 파일 | 크기 | 내용 |
|------|-----:|------|
| `jetson_codex_context.tar.gz` | **73 KB** | Codex 컨텍스트만 |
| `jetson_deploy_lite.tar` | **146 MB** | 위 + 검증 (small ref, model, scripts) |
| `jetson_deploy.tar` | **1.6 GB** | 위 + full val_reference.npz |

**사용자 선택**: `jetson_deploy_lite.tar` (146 MB) — small reference 로 대부분 검증 가능.

---

## 5. PC 측 사전 벤치마크 (Jetson 검증 대조군)

| Backend | Latency | 예측 일치 (vs PyTorch) |
|:---|---:|:---:|
| PyTorch GPU (RTX 6000) | 1.18 ms | (기준) |
| ONNX Runtime GPU | 2.61 ms | 100% (1157/1157) |
| ONNX Runtime CPU | 7.27 ms | 100% (1157/1157) |
| Jetson Orin Nano (예상) | ~5 ms | 검증 대기 |

Jetson TensorRT FP16 목표: < 2 ms.

---

## 6. 다음 단계 (당시 대기 상태)

1. 사용자가 USB 를 Jetson 으로 옮김
2. `scripts/01_check_environment.py` 로 onnxruntime · EP 확인
3. `scripts/02` latency 벤치마크
4. `scripts/03` 정확도 일치 검증
5. `scripts/04` 실시간 파이프라인
6. `scripts/05` 결과 통합 → `results/jetson_summary.json`
7. 결과를 다시 USB 로 PC 측에 옮겨 논문 §6.8 Table 14 (현재 est. ~5 ms) 실측치로 갱신

**PC↔Jetson 동기화 규칙**:
- Jetson 측에서 결과 나오면 `STATE.md §A4` 에 한 줄 추가
- 에러는 `docs/오류_및_해결_로그.md` 에 추가
- 큰 변경 (모델 재학습, 새 코드) 은 폴더째 USB 로 PC 로 재전송

---

## 7. 파일 위치

| 파일 | 경로 |
|------|------|
| 배포 패키지 (원본) | `jetson_deploy/` |
| Codex 컨텍스트 | `jetson_deploy/codex_context/` |
| 참조 데이터 생성 코드 | `src/deploy/generate_reference_data.py` |
| ONNX export | `src/deploy/export_v2plus_onnx.py` |
| Jetson 실행 스크립트 5개 | `jetson_deploy/scripts/01~05_*.py` |
| USB 전송용 tar | `jetson_deploy{,_lite}.tar`, `jetson_codex_context.tar.gz` |

---

## 8. 후속 이력 (2026-07 이후 발전)

- 2026-07-23: 사용자가 GitHub 저장소 (`mskim0114/multi_sensor_anomaly_detection`) 초기 push (dfff2fa)
- 2026-07-31: Jetson 측 Codex 로 신규 스크립트 06/07/08 (열화상 캡처 · SPS30 · NTC 판독) 추가 push
- 2026-08-06: PC 측을 GitHub 에 연결, Qwen 개선 검토 진행 (연구노트 #15)

Jetson 검증 실측 결과는 아직 대기 (2026-08-07 시점). 도착 시 논문 pending update #1 반영 예정.
