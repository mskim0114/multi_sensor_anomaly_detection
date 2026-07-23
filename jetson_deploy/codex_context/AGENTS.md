# AGENTS.md — Factory Safety Project (Jetson side context)

> 이 파일은 **Codex CLI가 자동으로 로드하는 컨텍스트**입니다.
> Jetson Orin Nano에서 이 디렉토리를 작업 폴더로 두고 `codex` 를 실행하면 본 내용이 매 세션의 system instructions에 주입됩니다.

---

## 1. 프로젝트 한 줄 요약

**제조공장 멀티모달(센서 8ch + 열화상) 데이터로 장비의 4단계 열화 상태를 실시간 분류하는 AI**를 만들고, 그 결과를 **Jetson Orin Nano**에 ONNX/TensorRT로 배포해 현장에서 검증한다.

- **기간**: 2026-03 ~ 2028-06 (R&D 24개월 + 현장 고도화 6개월)
- **연구자**: KETI 소속 ML/DL 연구자
- **본 작업 시점(2026-05)**: 모델/논문/PC 측 배포 검증 완료 → Jetson에 패키지 도착 → 보드에서 latency/정확도 검증 시작

---

## 2. 모델 핵심: V2+ (최종 선정 모델)

```
입력: sensor (B,30,8)  +  thermal (B,30,120,160)
출력: logits (B,4)  → {Normal, Mild, Moderate, Severe}
구조:
  sensor branch:
    Multi-Scale Temporal Diff (lags=[1,5,10])  → 8 → 32 채널
    └─ LSTM(hidden=128, layers=3, dropout=0.1)
    └─ SE(Squeeze-Excitation) channel attention
    └─ mean pool
  thermal branch:
    Conv2d(1→16→32→64) + MaxPool ×3 → flatten → FC(d_model)
  classifier: concat → Dropout → Linear → 4 classes
  loss:  CrossEntropy(class_weighted) + λ·SupCon (λ=0.1)
```

**왜 이 구조인가**
- AI Hub 데이터(111,870 sample × 8 sensor + thermal)에 transformer를 쓰면 9,313 윈도우에 비해 과적합 → LSTM 채택
- Normal/Mild는 4°C 차이로 경계가 모호 → multi-scale diff + SupCon으로 47% 오류 감소
- 8개 센서 중 중요한 채널을 강조하기 위해 SE attention
- 세 컴포넌트는 **단독으론 효과 미미하지만 결합 시 시너지**(+1.20% F1)가 발생함 (논문 핵심 주장)

**최종 성능** (val F1 macro)
| 모델 | F1 | Params |
|------|---:|-------:|
| Baseline LSTM | 0.9235 | 2.83M |
| CATFT (Transformer) | 0.9252 | 2.87M |
| **V2+ (선정)** | **0.9550 ± 0.0006** (3 seeds) | **2.85M** |
| TimesNet | 0.9180 | 1.2M |
| PatchTST | 0.9165 | 0.9M |

Severe(중대 이상) recall = 100%, Normal↔Mild 오류 45 → 24 (47% ↓).

상세는 `docs/연구노트/09_V2Plus_결과.md`, `docs/연구노트/11_논문_추가실험.md`, `docs/논문/paper_draft.md` 참고.

---

## 3. 데이터셋 핵심

**AI Hub #71802** "제조 운반장비 열화 예측 멀티모달"
- 36개 장비 (OHT 18 + AGV 18) × 평균 3,100 sample = 111,870 sample
- 센서 8ch: NTC, PM1.0/2.5/10, CT1~CT4 (전류)
- 열화상: FLIR Lepton 3.5, 120×160, **°C 단위 raw temperature**
- 라벨: 4-class degradation (장비별 누적 가동/온도 기반)
- 샘플링: 1 Hz, 윈도우 30s (size=30, stride=10)

**우리가 만든 파이프라인이 AI Hub 코드와 다른 점** (`code/data/`)
1. **세션 단위 분리 split** — 같은 장비의 시간 인접 윈도우가 train/val에 섞이는 누수 차단
2. **Z-score 센서 정규화 + global min-max thermal 정규화** (AI Hub는 정규화 없음)
3. **WeightedRandomSampler + class-weighted CE** — Severe만 5.7% 라 불균형
4. **120 s gap 감지** — 동일 장비 안에서도 세션 끊김 감지
5. **mmap lazy loading** — 1.4 GB 데이터를 16 GB RAM에서 학습 가능

---

## 4. 하드웨어 / 센서 구성

**엣지보드**: Jetson Orin Nano 8 GB (지금 이 보드)

**현장 배치 예정 센서** (이미 구매 완료, 미도착)
| 종류 | 모델 | 용도 |
|------|------|------|
| 온도 | NTC 10 kΩ (Murata) | 장비 표면 온도 |
| 미세먼지 ×3 | Sensirion SPS30 (PM1/2.5/10 동시) | 공기질 |
| 전류 ×4 | YHDC SCT-013-000 (100 A) + Burden 33 Ω + DC bias | 4상 부하 |
| 열화상 | FLIR Lepton 3.5 + PureThermal 모듈 | 120×160 °C |
| 인터페이스 | I²C / SPI / ADS1115 16bit ADC | Jetson 40-pin |

상세 매칭, 회로, Burden resistor 계산은 `docs/참고문서/실제 센서 및 엣지 보드(Jetson) 구매 사양서 및 매칭 가이드.md` 참고.

**Jetson 환경 (현재)**
- JetPack 6.x, Python 3.10
- PyTorch (NVIDIA wheel) 설치 완료
- 다음: `onnxruntime-gpu` 설치 → `../scripts/` 검증 스크립트 실행
- NVLink 없음 (PC쪽), Jetson 단일 GPU

---

## 5. 실시간 추론 시스템 설계

**입력 흐름** (`code/deploy/realtime_pipeline.py` 참조)
```
센서 1 Hz 샘플링 → 30-deep ring buffer ┐
열화상 1 Hz 캡처 → 30-deep ring buffer ┘→ ONNX inference (stride=10)
                                            → 4-class 확률 / alert
```
- 10 초마다 1회 추론 (slide stride 10, 윈도우 30 → 매 10 샘플마다 갱신)
- 단일 추론 latency 목표: **CUDA EP < 5 ms / TensorRT FP16 < 2 ms**
- 단일 프레임마다 추론하지 않고, 윈도우가 새로 차야 추론 (효율)

PC 측 사전 측정: ONNX CPU 4.3 ms, 예측 100% PyTorch ↔ ONNX 일치.

---

## 6. 현재 상태 (이 패키지 도착 시점)

✔ 완료
- 데이터 파이프라인 (`code/data/`)
- 5개 모델 학습 (Baseline / CATFT / V2 / V2+ / external baselines)
- Ablation, lag sensitivity, 3-seed 통계 실험
- V2+ ONNX export + PC 벤치마크 (`../model/model_v2plus.onnx`)
- 논문 draft (`docs/논문/paper_draft.md`, 14 tables, 6 figures)
- 자체 리뷰 3회 + Major issue 4건 모두 대응

⏳ 진행 중 (Jetson에서 지금 할 일)
- `../scripts/01_check_environment.py` ~ `05_summary.py` 순차 실행
- latency / accuracy match / realtime pipeline 검증

🔜 다음 (센서 도착 후)
- 센서 HW 결선 + 캘리브레이션
- 실데이터 fine-tuning
- TensorRT FP16 엔진 변환
- 현장 PoC

🔜 논문 (제출 시점)
- MDPI Sensors 템플릿 변환
- Reference MDPI 포맷 정리
- 사용자가 figure 새로 그릴 예정

---

## 7. 디렉토리 인덱스 (이 폴더 안)

```
codex_context/
├── AGENTS.md                 ← (이 파일, Codex 자동 로드)
├── STATE.md                  ← 현재 진행 + 다음 작업 (날짜 포함)
├── INDEX.md                  ← 빠른 탐색용
├── docs/
│   ├── 연구노트/             ← 01~11 시간 순서대로 진행 기록
│   ├── 참고문서/             ← 데이터셋·로드맵·센서 구매가이드
│   ├── 논문/paper_draft.md   ← Sensors (MDPI) 제출 draft
│   ├── 실험결과_보고서.md
│   └── 오류_및_해결_로그.md  ← 작업 중 만난 모든 에러 + 해결
└── code/
    ├── models/v2_plus.py     ← 최종 모델 (이게 ONNX로 export 됨)
    ├── models/external_baselines.py
    ├── data/                 ← 전체 데이터 파이프라인 (8 files)
    ├── deploy/               ← ONNX export, 참조 데이터 생성
    ├── train_v2plus.py       ← 최종 모델 학습 진입점
    └── demo_inference.py
```

상위 폴더 `../scripts/`, `../model/`, `../reference/` 는 **Jetson 실행용 검증 패키지**입니다 (코드 컨텍스트 아님).

---

## 8. 작업 시 지켜야 할 규칙

1. **장비 단위 분리** — train/val/test split 시 절대 시간으로 나누지 말 것 (세션 누수). 항상 device_id 기준.
2. **Severe class 우선** — 불균형 데이터셋이지만 Severe recall은 100% 유지가 안전상 더 중요. 정확도보다 우선.
3. **°C 단위 thermal** — 열화상은 raw temperature (°C). 이걸 0~1로 정규화하지 말 것 (global min-max는 학습 안정용으로 별도 처리).
4. **확장 시 V2+ 우선** — 새 컴포넌트는 V2+ 위에 얹어 ablation으로 검증. CATFT 계열은 과적합 위험.
5. **결과 저장** — 새 실험은 `results/{model_name}/best_model.pt` + `metrics.json` 형식 유지.
6. **에러 발생 시** `docs/오류_및_해결_로그.md` 에 동일 형식으로 추가.

---

## 9. 자주 쓰는 명령

```bash
# Jetson 검증 (이 폴더에서 한 단계 위로 가서)
cd .. && python3 scripts/01_check_environment.py
        python3 scripts/02_benchmark_latency.py --runs 200
        python3 scripts/03_verify_accuracy.py --small
        python3 scripts/04_realtime_pipeline.py --n 300
        python3 scripts/05_summary.py

# TensorRT FP16 변환 (검증 통과 후)
/usr/src/tensorrt/bin/trtexec \
  --onnx=../model/model_v2plus.onnx \
  --saveEngine=../model/v2plus_fp16.trt \
  --fp16 --workspace=2048

# 전력/온도 모니터링 (별도 터미널)
sudo tegrastats
```

---

## 10. Codex에 질문/요청할 때 팁

- "이 프로젝트에서 ~~ 어떻게 구현했어?" → `code/` 와 `docs/연구노트/` 참고하라고 해도 됨
- "V2+ 왜 LSTM 썼어?" → 위 §2 와 `docs/연구노트/08_모델개선_리서치.md` 가 근거
- "센서 결선 어떻게 해?" → `docs/참고문서/실제 센서 및 엣지 보드 구매 사양서.md`
- "지금 뭘 해야 해?" → `STATE.md` 확인

새 작업을 시작하면 **반드시 `STATE.md` 와 `docs/오류_및_해결_로그.md` 를 업데이트**해서 PC 쪽 Claude도 동기화 가능하게 유지하세요.
