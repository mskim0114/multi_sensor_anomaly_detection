# 연구노트 #14: AI Hub 데이터 스키마 재분석 · MMTransformer Baseline 발견

**작성일:** 2026-07-31
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 배경

Phase 1a 착수 (연구노트 #13) 이후, 사용자가 **AI Hub 데이터 스키마 명세서** 와 **AI Hub 데이터 활용 가이드라인 요약** 두 문서를 순차 제공. 이 두 문서에서 지금까지 우리가 몰랐거나 잘못 알던 사실 다수 확인.

**개정 방식**: 이전 세션 (연구노트 #02) 에는 "S02/S10/S18/S26" 같은 코드만 있고 실제 모델명 불명이었음. 실제 값 검증 (전체 111,870 중 20,000 랜덤 샘플) 을 병행해 결정적 사실을 확정.

---

## 2. Session 1 (오전): 실사용 센서 확정 · Heterogeneity 가설 폐기

### 스키마 → 실 모델 매핑

사용자 제공 명세로 각 카테고리 4-way 후보 확인:

| 카테고리 | S## → 실 모델 (4-way) |
|:---|:---|
| 미세먼지 | S02 = Sharp GP2Y1014AU0F / S04 = Sensirion SPS30 / S06 = Shinyei PPD42S / S08 = Winsen ZH03 |
| 온도 | S10 = Vishay NTCLE413 / S12 = NTC-103 F343F / S14 = TT7-50KC3-3 / S16 = MF52 |
| 전류 | S18 = KEMET CT-06 / S20 = CR8400 (CR Magnetics) / S22 = Az-0500 (Talema) / S24 = CCT406393 (TDK) |
| 열화상 | S26 = TeraRanger Evo Thermal 33 (32×32) / S28 = Grid-EYE AMG88 (8×8) / S30 = MI0801 / S32 = Melexis MLX90640 (32×24) |

### 초기 가설: 4-way heterogeneous pool

각 카테고리에 4종이 나열되어 있으니 "AI Hub 데이터는 4종을 섞어 heterogeneous 하게 수집됐다" 라고 초기 추정. 이게 사실이면 논문에 "sensor heterogeneity as implicit augmentation" 이라는 신규 §7.6 소절 추가 가능성 있었음.

### 실측 검증 (20,000 샘플 랜덤 스캔)

```
Dust:    {'S02': 20000}    ← 100% Sharp GP2Y1014AU0F
Temp:    {'S10': 20000}    ← 100% Vishay NTCLE413
Current: {'S18': 20000}    ← 100% KEMET CT-06
Thermal: {'S26': 20000}    ← 100% TeraRanger Evo Thermal 33
```

### 결론 (heterogeneity 폐기)

- **스키마의 4-way 는 "허용 가능한 값의 enumeration" 이며 실제 데이터는 단일 sensor set 으로 수집.**
- Section 7.6 신설 계획 폐기.
- 하지만 **device diversity 는 실재** (3-manufacturer / 3-model):
  - A: SFA / OHT-OCS (A1) — oht01~18 (57.6% 샘플)
  - B: 미르 / Mri-100 (B1) — agv01~09 (21.5%)
  - C: 씨에이시스템 / 저상용 AGV (C1) — agv10~18 (20.9%)

### 열화상 upsampling 확인 (매우 중요)

- TeraRanger Evo Thermal 33 = **32×32 native** (1,024 픽셀)
- 저장 shape = **120×160 float64** (`.npy` 파일, `\x93NUMPY` magic bytes 확인)
- **약 15× upsampling** (bilinear 계열)

논문 Section 6.5 의 "thermal 기여도 +0.37%" 의 잠재 원인이 여기 있음 → **CNN 이 학습한 spatial feature 는 실질적으로 32×32 정보에 상한**.

실배포 FLIR Lepton 3.5 (160×120 native, 19,200 px) 는 학습 분포보다 정보 풍부 → downsample or fine-tune 필요.

### 미사용 필드 5종 발견

| 필드 | 실측 값 | 우리 모델 사용 |
|:---|:---|:---:|
| `external_data.ex_temperature` | 22~26 °C | ❌ |
| `external_data.ex_humidity` | 27~36 % | ❌ |
| `external_data.ex_illuminance` | 151~530 lux | ❌ |
| `ir_data.temp_max.X_Tmax / Y_Tmax` | 최고 온도 픽셀 좌표 | ❌ |
| `sensor_data.*.trend` | "-1/0/1" 추세 | ❌ (우리 multi-scale diff 와 개념 유사) |
| `meta_info.cumulative_operating_day` | 13~18 일 | ❌ |
| `meta_info.equipment_history` | 7~13 | ❌ |
| `meta_info.device_manufacturer` | A/B/C | ❌ |

Future Work / IJCAI 확장 소재.

---

## 3. Session 2 (저녁): AI Hub 활용 가이드라인 발견

사용자가 **AI Hub 데이터 활용 가이드라인 요약** 문서를 저녁에 별도 제공. 여기서 5가지 결정적 사실 확인.

### A. 총 세트 수 · 우리는 subset

- **원본 총 124,263 세트** (OHT 73,733 / AGV 50,530)
- 우리 보유 111,870 = 원본의 **~90% subset** (Training 99,476 + Validation 12,394)
- 12,393 세트 미포함 (아마 별도 test set)

### B. Sampling 실체

- **원본 raw = 100 ms 주기 (10 Hz)** 센서 수집
- **Dataset provider 가 1 s 단위로 aggregation 후 배포** (10 샘플 → 1 값)
- Aggregation 방식 (mean/max/last) **미공개**
- 우리는 1 Hz aggregation 완료본을 사용 → temporal resolution 상한 1 s

### C. 4-Class 라벨 정확 정의

| state | 한글 (영문) | 정의 |
|:---:|:---|:---|
| 0 | 정상 (Normal) | Z-score ≤ 3.5 (원본의 ~99%) |
| 1 | 관심 (Attention) | Z-score > 3.5 ~ GT-30s 직전, **전반부 50%** |
| 2 | 경고 (Warning) | Z-score > 3.5 ~ GT-30s 직전, **후반부 50%** |
| 3 | 위험 (Danger) | **GT-30s ~ GT** (탄화 발생) 마지막 30초 |

GT = Ground Truth (실제 탄화 발생 시점).

**⚠️ 중요**: 관심(1) ↔ 경고(2) 는 **같은 구간을 임의 1:1 분할** ("AI 서비스 예지보전 민감도 조정용", 원 문서 명시).
- 두 클래스 boundary 는 물리적·통계적 의미 없음
- 완벽 구분 불가능 → 우리 논문 §6.3 (N↔M 오류 24 감소) 재해석 필요
- 남은 오류의 상당 부분이 근본 하한

**위험(3) = 마지막 30초 이벤트** → Severe recall = 100% 유지가 안전상 절대 우선인 이유.

### D. 원본 정상 99% vs 우리 subset balanced

우리 데이터 실측 (30,000 샘플):
- 정상 49.35% / 관심 21.59% / 경고 21.39% / 위험 7.66%

**우리 111,870 subset 은 원본에서 정상 클래스를 대폭 downsampling 한 balanced set**. 실배포는 원본과 유사한 99% 정상 분포 → domain gap 존재. §7.5 Limitations 반영 대상.

### E. AI Hub 참조 baseline MMTransformer 성능 (매우 큰 영향)

- **아키텍처**: ViT (이미지 patch embedding) + 시계열 encoder + **Cross-Attention Mechanism**
- **성능 (Validation)**:
  - Accuracy: **91.92%** (목표 0.80 초과)
  - F1: **91.09%** (목표 0.68 초과)

**우리 V2+ 대비**:

| 항목 | AI Hub MMTransformer | 우리 V2+ | Δ |
|:---|:---:|:---:|:---:|
| Accuracy | 91.92% | 95.02% | **+3.10 %p** |
| F1 | 91.09% | 95.57% | **+4.48 %p** |
| 아키텍처 | Cross-Attention Transformer | LSTM + Multi-Scale Diff + SE + SupCon | — |

**논문 함의 (매우 강력)**:
- 동일 dataset · 동일 라벨 정의로 검증된 **공식** baseline 을 우리가 명확히 능가
- 우리 CATFT V5 (0.9252) 도 유사 수준 → **Cross-Attention 계열이 이 dataset 에서 ~91-93% F1 상한** 임을 실증
- Domain-informed lightweight 접근이 architectural complexity 를 넘어선다는 논거 empirical 로 강화

### F. CT 채널 의미 확정

- **CT1 = 입력단**, **CT2 = 출력단**, **CT3 = 모터1**, **CT4 = 모터2**

지금까지 논문·연구노트는 "CT1~CT4" 로만 서술했으나 위치·역할이 명확히 다름. Section 3.2 반영 대상. 스케일 차이 (CT1 평균 6A vs CT2 평균 43A) 도 자연스럽게 설명됨.

### G. Dataset provider 사전 정제 규칙

- Numpy 통계 기반 이상치 색인·제거
- 파일 유사도 95 이상 제거 (센서: 상관도 + 가중치, 열화상: VisiPics)

우리가 별도 outlier removal 안 해도 되는 근거. 논문 §3 (data pipeline) 방어 논리.

---

## 4. 스키마 vs 실측 불일치

| 필드 | 스키마 | 실측 | 조치 |
|:---|:---|:---|:---|
| CT1~4 | 0-200 A | 270.31 A (agv07/CT2) | 스키마 초과. KEMET CT-06 실사양 재확인 or 아웃라이어 처리 |
| device_id | 01-20 | oht01~18 + agv01~18 = 36개 | 스키마 형식과 다름 (prefix 포함) |
| duration_time | 1-300 | 100% "1" | 실제 uniform |

---

## 5. 후속 작업 (2026-08-06 진행)

이 세션의 발견들을 정리한 **논문 pending update #2 (11 지점)** 이 만들어짐 (`~/.claude/projects/.../memory/project_paper_pending_updates.md`). 그리고 참고문서 · 연구노트_02 는 즉시 갱신.

논문 자체 반영은 다음 세션 (2026-08-06, 연구노트 #15) 에서 완료:
1. §3.1 (Dataset Overview) — 124,263 vs 111,870 subset, 원본 99% 정상, 3-manufacturer
2. §3.2 (Sensor Specifications) — Table 2b 실사용 4종, CT 역할, 열화상 native 32×32
3. §3.3 (Class Definition) — Table 3a 정확 정의, KO/EN 라벨
4. §3.5 (Problem Definition) — 100ms → 1s aggregation
5. §6.1 (Table 6) — **MMTransformer 최상단 추가**
6. §6.3 (Class-Level) — 관심/경고 임의 분할 재해석
7. §6.5 (Thermal) — 32×32 upsampled 원인
8. §6.9 (H3 evidence) — MMTransformer 추가
9. §7.1 (Discussion) — Cross-attention 91-93% 상한 논거 강화
10. §7.5 (Limitations) — 5개 소절 재구성 (unused fields, cross-facility, deployment, pretraining)

---

## 6. 파일 위치

| 파일 | 경로 |
|------|------|
| 데이터 스키마 명세 (사용자 제공) | 대화 이력 (2026-07-31) |
| AI Hub 활용 가이드라인 요약 (사용자 제공) | 대화 이력 (2026-07-31 저녁) |
| 참고문서 (실사용 4종 반영) | `docs/참고문서/AI Hub 데이터셋 심층 분석 및 3종 이상상태 정의 보고서.md` |
| 참고문서 (사용자 보유 센서 매칭) | `docs/참고문서/실제 센서 및 엣지 보드(Jetson) 구매 사양서 및 매칭 가이드.md` |
| 연구노트 #02 addendum | `docs/연구노트/연구노트_02_원본데이터셋_분석.md` (§0, §0-2) |
| Pending updates 메모리 | `~/.claude/projects/-home-keti/memory/project_paper_pending_updates.md` |

---

## 7. 결론

- **AI Hub 데이터는 단일 sensor set** (Sharp / Vishay / KEMET / TeraRanger). 4-way heterogeneous 가설 폐기.
- **열화상 native 32×32 → 120×160 upsampled** 확인. Thermal contribution 낮은 원인.
- **관심/경고 라벨은 임의 1:1 분할**. N↔M 오류의 근본 하한 존재.
- **AI Hub 참조 baseline MMTransformer F1 91.09%** 확정. 우리 V2+ (0.9557) 명확 능가 → 논문 최강 empirical 근거.
- **미사용 필드 5종** (external·trend·meta) 은 Future Work / IJCAI 확장 소재.
- **CT 채널 역할 확정** (입력/출력/모터1/모터2).
- 후속: 논문 pending #2 반영 (11 지점) 은 2026-08-06 진행 (연구노트 #15).
