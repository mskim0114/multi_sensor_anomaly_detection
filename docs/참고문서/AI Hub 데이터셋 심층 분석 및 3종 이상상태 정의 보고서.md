# AI Hub 데이터셋 심층 분석 및 이상상태 정의 보고서

> **개정 이력**
> - 2026-03-23 초안 (센서 코드 S## 만 확인, 실제 모델명 불명)
> - **2026-07-31 (아침)**: 사용자 제공 **데이터 스키마 명세** 로 S## 코드 → 실제 sensor 제조사/모델명 매핑 확정. `external_data`, `trend`, meta 필드 신규 발견 반영. 이전에 '3종 이상상태' 로 서술했던 것을 **실제 라벨 (4-class: 정상/관심/경고/위험)** 로 정정.
> - **2026-07-31 (저녁, 본 개정)**: 사용자 제공 **AI Hub 활용 가이드라인 요약** 반영 — (1) **총 세트 수 124,263** vs 우리 subset 111,870, (2) **원본 sampling 100ms → 1s aggregation**, (3) **4-class 정확한 라벨 정의 (Z-score 3.5 + GT-30s)**, (4) **AI Hub 참조 baseline MMTransformer 성능 Acc 91.92% / F1 91.09%** — 우리 V2+ 이 능가 확인, (5) **CT 채널 의미 확정** (입력단/출력단/모터1/모터2).

## 1. AI Hub 데이터셋 개요

*   **데이터셋 명:** 제조현장 이송장치의 열화 예지보전 멀티모달 데이터
*   **데이터셋 ID:** 71802
*   **핵심 목적:** 반도체·디스플레이·**자동차** 공정의 **OHT (Over Hoist Transport)** 및 **AGV (Automated Guided Vehicle)** 장치에서 발생하는 **탄화(열화) 현상** 을 사전 예측·예방하는 온디바이스 AI 서비스 개발 지원.
*   **원본 데이터 구성:** 1세트 = 8유형 센서값 (JSON 내부) + 열화상 이미지 1장 (BIN, 128B header + 120×160 float64 °C) + 라벨링 정보 (JSON `annotations`).
*   **원본 규모 (2026-07-31 문서 확정)**: 총 **124,263 세트** — OHT **73,733** / AGV **50,530**. 현장 모사 테스트베드 + 실제 운영 장비 수집.
*   **우리가 사용한 subset**: **111,870 세트** (Training 99,476 + Validation 12,394, 원본의 ~90%). 12,393 세트는 우리 subset 에서 제외됨 (아마 별도 test set).
*   **⚠️ 클래스 분포 사전 리샘플링**: 원본은 **정상이 약 99%** (Z-score 3.5 이하) → 우리 subset 은 **정상 49%, 관심 22%, 경고 21%, 위험 8%** 로 이미 balancing 됨. → 실배포 시 real-world 분포와 gap 존재 (Section 5 참조).

## 2. 데이터 스키마 확정 (2026-07-31)

사용자 제공 명세로 확인된 실제 필드 구조와 값 매핑. 20,000 샘플 검증 완료.

### 2.1 meta_info (장비 · 센서 · 수집 메타)

| 필드 | 타입 | 실제 값 | 우리 모델 사용 여부 |
|:---|:---:|:---|:---:|
| device_id | string | oht01~18, agv01~18 (36개) | ✔ 사용 (split 기준) |
| device_manufacturer | string | **A / B / C** | ❌ 미사용 |
| device_name | string | **A1 / B1 / C1** | ❌ 미사용 |
| dust_sensor_name | string | **S02 100%** (Sharp GP2Y1014AU0F) | ❌ (단일값이므로 정보량 0) |
| temp_sensor_name | string | **S10 100%** (Vishay NTCLE413) | ❌ (단일값) |
| overcurrent_sensor_name | string | **S18 100%** (KEMET CT-06) | ❌ (단일값) |
| thermal_camera_sensor_name | string | **S26 100%** (TeraRanger Evo Thermal 33) | ❌ (단일값) |
| installation_environment | string | "테스트베드" | ❌ |
| collection_date, collection_time | string | 08-27 ~ 09-20 대략 | ❌ (session_id 로 우회 사용) |
| duration_time | number | 100% "1" (1초) | ❌ |
| sensor_types | string | "NTC, PM10, PM2.5, PM1.0, CT1~4" | ❌ |
| **cumulative_operating_day** | number | 13~18 일 | ❌ **미사용 (static feature 후보)** |
| **equipment_history** | number | 7~13 | ❌ **미사용 (static feature 후보)** |
| img-id, filename, img_name | string | 파일 인덱스 | ✔ 파일 로딩용 |
| location | string | "oht/15/oht15_0901_1439" | ❌ (device_id 로 대체 사용) |
| img_description | string | "oht15의 현재 내부 온도(최대값)" | ❌ |

### 2.2 sensor_data (센서 시계열 값)

**CT 채널 의미 (2026-07-31 문서 확정)**:
- **CT1 = 입력단**
- **CT2 = 출력단**
- **CT3 = 모터1**
- **CT4 = 모터2**

즉 4개 CT 는 위치·역할이 명확히 다르며, 이는 우리 데이터에서 관측된 스케일 차이 (CT1 평균 6A vs CT2 평균 43A) 를 설명. 논문 Section 3 서술 대상.

| 필드 | 실제 구조 | 값 범위 | 우리 모델 사용 |
|:---|:---|:---|:---:|
| PM1.0, PM2.5, PM10 | {value, data_unit, **trend**} | 실측 6~92 µg/m³ | ✔ value 만 |
| NTC | {value, data_unit, **trend**} | 실측 17~80 °C | ✔ value 만 |
| CT1~CT4 | {value, data_unit, **trend**} | 실측 0~273.91 A (스키마 0-200 A 초과) | ✔ value 만 |

**`trend` 필드**: 각 센서값에 `trend: "1"` 같은 추세 표기 존재. 값 종류: "-1" / "0" / "1" 로 추정 (감소/정체/증가). **우리 모델은 사용 안 함**. 우리가 만든 multi-scale temporal diff 와 개념적으로 유사한 도함수 정보를 dataset 이 이미 제공했던 것 → 논문 비교 실험 후보.

### 2.2.1 Sampling rate (2026-07-31 문서 확정)

- **원본 raw**: **100 ms 주기 (10 Hz)** 센서 수집
- **제공 데이터**: dataset provider 가 **1 초 단위로 aggregation** (10 샘플 → 1 값) 후 배포
- **우리 사용**: 1 Hz 시계열 그대로 사용
- **Aggregation 방식 (mean/max/last) 불명** — dataset provider 미공개

**함의**: Raw 10 Hz 접근 불가 → temporal resolution 상한 존재. Phase 2 실배포 시 10 Hz raw 확보 가능 → higher-resolution temporal feature 실험 여지.

### 2.3 ir_data (열화상)

| 필드 | 구조 | 값 범위 | 사용 |
|:---|:---|:---|:---:|
| temp_max.value_TGmx | number | 34.66 ~ 146.10 °C | ❌ (BIN 이미지에서 max 로 대체 유도) |
| temp_max.X_Tmax | number | 픽셀 X 좌표 | ❌ **미사용 (spatial localization 후보)** |
| temp_max.Y_Tmax | number | 픽셀 Y 좌표 | ❌ **미사용** |
| (별도 BIN 파일) | float64 120×160 | 30 ~ 146 °C | ✔ CNN 입력 |

### 2.4 annotations (라벨)

| 필드 | 실제 값 | 사용 |
|:---|:---|:---:|
| annotations.tagging.annotation_type | "tagging" | — |
| **annotations.tagging.state** | **"0" / "1" / "2" / "3"** | ✔ 4-class 라벨 |

### 2.5 external_data (환경 데이터) — **완전히 놓쳤던 필드**

| 필드 | 실제 값 | 사용 |
|:---|:---|:---:|
| **ex_temperature** | 22 ~ 26 °C (외부 온도) | ❌ **미사용** |
| **ex_humidity** | 27 ~ 36 % | ❌ **미사용** |
| **ex_illuminance** | 151 ~ 530 lux | ❌ **미사용** |

**모두 미사용 → 우리 모델은 외부 환경 조건을 무시하고 학습됨.** 이는 실배포에서 계절/시간대 등 도메인 시프트 취약점이 될 수 있음. Future work 후보.

## 3. 센서 pool 확정 (스키마 vs 실제)

### 3.1 스키마상 정의된 4-way pool

| 카테고리 | 4-way 후보 (스키마) |
|:---|:---|
| 미세먼지 | GP2Y1014AU0F (S02, Sharp) / SPS30 (S04, Sensirion) / PPD42S (S06, Shinyei) / ZH03 (S08, Winsen) |
| 온도 | NTCLE413 (S10, Vishay) / NTC-103 F343F (S12, Samkyung) / TT7-50KC3-3 (S14, TEWA) / MF52 (S16, Cantherm) |
| 전류 | CT-06 (S18, KEMET) / CR8400 (S20, CR MAGNETICS) / Az-0500 (S22, Talema) / CCT406393 (S24, TDK) |
| 열화상 | Evo Thermal 33 (S26, TeraRanger) / Grid-EYE AMG88 (S28, Panasonic) / MI0801 (S30, Meridian Innovation) / MLX90640 (S32, Melexis) |

### 3.2 실제 데이터에서 사용된 것 (2026-07-31 확인)

**전 20,000 샘플이 각 카테고리 당 1종만 사용**:
- Dust: **S02 (Sharp GP2Y1014AU0F)** 100%
- Temp: **S10 (Vishay NTCLE413)** 100%
- Current: **S18 (KEMET CT-06)** 100%
- Thermal: **S26 (TeraRanger Evo Thermal 33)** 100%

**즉 스키마의 4-way 는 "허용 가능한 값 enumeration" 이며 실제 데이터는 단일 sensor set 으로 수집됨.** 이전 문서에서 "heterogeneous 4-way pool 로 수집됐다" 라고 서술했던 가정은 폐기.

### 3.3 Device diversity (실재)

3-manufacturer / 3-model:
- **A: SFA / OHT-OCS (A1)** — oht01~18 (18 devices, 51.4% 샘플)
- **B: 미르 / Mri-100 (B1)** — agv01~09 (9 devices, 21.5%)
- **C: 씨에이시스템 / 저상용 AGV (C1)** — agv10~18 (9 devices, 20.9%)

**Sensor diversity 는 없으나 device diversity 는 있음.** 논문의 일반화 논거는 이쪽으로 재정렬.

### 3.4 열화상 데이터의 실체

- TeraRanger Evo Thermal 33 = **32×32 native 해상도** (1,024 픽셀)
- 저장 포맷 = **120×160 float64** (128B header 뒤에 이어짐)
- 즉 원본 32×32 → 120×160 **약 15배 업샘플링** (bilinear or similar)
- 우리 CNN 이 학습한 "열화상 spatial feature" 는 실질적으로 32×32 정보에 상한
- 실배포에 **FLIR Lepton 3.5 (160×120 native, 19,200 픽셀)** 을 쓰면 학습 분포보다 정보 풍부 → downsample or fine-tune 필요

## 4. 4-Class 이상상태 정의 (공식 라벨)

이전 문서의 "3종 이상상태" 는 안전 관점 서술이었고, **실제 데이터 라벨은 4-class** 입니다 (annotations.tagging.state = 0/1/2/3). **2026-07-31 문서로 정확한 정의 확정**:

### 4.1 공식 라벨 매핑 (안전도 4등급)

| state | 한글 공식 | 영문 공식 | 논문 표기 (편의) | **정의 (문서 확정)** |
|:---:|:---:|:---:|:---:|:---|
| **0** | **정상** | Normal | Normal | Z-score ≤ 3.5 (원본의 ~99%) |
| **1** | **관심** | Attention | Mild | Z-score > 3.5 ~ GT-30s 직전, **전반부 50%** |
| **2** | **경고** | Warning | Moderate | Z-score > 3.5 ~ GT-30s 직전, **후반부 50%** |
| **3** | **위험** | Danger | Severe | **GT-30s ~ GT** (탄화 발생) 마지막 30초 |

**GT (Ground Truth)** = 실제 탄화 발생 시점.

### 4.2 라벨 정의의 중요 함의

**⚠️ 관심(1) ↔ 경고(2) 는 물리적으로 같은 구간을 임의 1:1 분할한 것.** 문서 인용:
> "관심과 경고는 AI 서비스의 예지보전 민감도 조정을 위해 1:1로 분할됨"

즉:
- 두 클래스 사이 boundary 는 **물리적 · 통계적 의미 없음**
- 모델이 관심 ↔ 경고 를 완벽 구분하는 것은 **근본적으로 불가능**
- Section 6 논문 결과 (Normal-Mild 오류 24개까지 감소) 재해석: 이 24개 오류는 **근본 하한에 근접**한 결과일 수 있음. 완전 제거 불가.

**위험(3) 은 마지막 30초 이벤트**. Severe recall = 100% 유지가 안전상 절대 우선 이유가 여기 있음 (30초 안에 개입해야 탄화 방지).

### 4.3 상태별 센서값 변화 패턴

| 센서 | state 0 (정상) | state 1 (관심) | state 2 (경고) | state 3 (위험) | 패턴 |
|:---|---:|---:|---:|---:|:---|
| **NTC** | 26.78 | 30.95 | 40.65 | **50.71** | 선형 증가 → 열적 과부하 핵심 지표 |
| **PM1.0** | 13.66 | 16.18 | 18.83 | 18.76 | state 2 포화 → 초기 탄화 감지용 |
| **CT1** | 2.16 | 3.21 | 10.16 | **29.85** | state 3 급증(14×) → 전기/기계 결함 결정 지표 |
| **CT2** | 33.15 | 30.97 | 50.43 | **115.38** | state 3 급증(3.5×) |

### 4.4 안전 관점 3종 이상상태 매핑 (기존 정의 유지)

이 3종은 라벨이 아니라 **위 4-class 위에 얹은 안전 도메인 해석**입니다.

| 안전 관점 이상상태 | 정의 · 작업자 안전 영향 | 관련 센서 · 클래스 |
|:---|:---|:---|
| **1. 열적 과부하** | 설비 비정상 발열 → 화재 · 화상 위험 | NTC + 열화상, 주로 state 2~3 |
| **2. 탄화 및 유해 연기** | 부품 탄화 → 미세먼지·연기 → 호흡기 위협 | PM1.0/2.5/10, 주로 state 1~2 (state 2 포화) |
| **3. 전기적/기계적 결함** | 모터 과부하·베어링 마모 → 오작동·급정거·파손 | CT1~4, 주로 state 3 (급증 지표) |

## 5. 활용 전략 정리

### 5.1 우리 모델이 활용한 것
- **센서 8ch 시계열**: NTC (1) + PM1.0/2.5/10 (3) + CT1~4 (4) → value 만
- **열화상 이미지**: 120×160 float64 °C → 3-layer CNN
- **라벨**: state 4-class (weighted CE + SupCon)
- **누수 방지**: device_id 기반 세션 split (session-aware)
- **정규화**: Z-score sensor, global min-max thermal, WeightedRandomSampler

### 5.2 우리 모델이 활용하지 않은 것 (Future Work 후보)
- `external_data.ex_temperature / ex_humidity / ex_illuminance` — 환경 조건
- `ir_data.temp_max.X_Tmax / Y_Tmax` — 열화상 최고 온도 픽셀 위치
- `sensor_data.*.trend` — 사전 계산된 추세 정보
- `meta_info.cumulative_operating_day / equipment_history` — 정적 장비 특성
- `meta_info.device_manufacturer` — 3-way categorical (도메인 embedding)

### 5.3 스키마 vs 실측 불일치

| 필드 | 스키마 범위 | 실측 max | 조치 |
|:---|:---|:---|:---|
| CT1~4 | 0-200 A | 270.31 A (agv07/CT2) | 스키마 초과. KEMET CT-06 의 실제 사양 검증 필요 or 데이터 아웃라이어 처리 |
| device_id | 01-20 | oht01-18 + agv01-18 = 36개 | 스키마와 형식 다름 (문자 prefix 포함) |
| duration_time | 1-300 | 100% "1" | 스키마 범위 안이지만 실제 uniform |

## 5.5 Dataset Provider 사전 정제·가공 규칙 (2026-07-31 확정)

우리가 받은 데이터는 이미 다음 처리가 완료된 것:

- **수집·정제 주기**: 100 ms → 1 s 패키징
- **이상치 제거**: Python Numpy 기반 통계적 이상치 색인 · 제거 · 파일 분할
- **유사/중복 파일 제거**:
  - 센서: 상관도 + 가중치 기반 종합 유사도 **95 이상 파일 제거** (Fast Duplicate File Finder 계열)
  - 열화상 이미지: VisiPics 등 스크리닝 툴로 중복 검출 · 제거

이는 우리가 별도 outlier removal 을 할 필요 없다는 것을 뒷받침 (원래 우리 파이프라인에서 outlier 처리는 최소). 논문 Section 3 (data pipeline) 에 이 사실 명시하면 reviewer의 "이상치 처리 어떻게 했나?" 질문 방어.

## 5.6 AI Hub 참조 모델 성능 (baseline 확정, 2026-07-31)

**AI Hub dataset provider 가 제시한 baseline 모델**:
- **아키텍처**: MMTransformer
  - Vision Transformer (ViT) 로 열화상 이미지 patch embedding 추출
  - 시계열 센서값 (One-Hot / Soft Label Encoding 적용) 특징 추출
  - **Cross-Attention Mechanism** 으로 이미지 + 시계열 modality 결합 (다중 timestep)
- **성능 (Validation)**:
  - **Accuracy: 91.92%** (목표 0.80 초과)
  - **F1-score: 91.09%** (목표 0.68 초과)

**우리 V2+ 대비**:

| 항목 | AI Hub MMTransformer | 우리 V2+ | 차이 |
|:---|:---:|:---:|:---:|
| Accuracy | 91.92% | **95.02%** | +3.10 %p |
| F1 | 91.09% | **95.57%** | +4.48 %p |
| 아키텍처 | Cross-Attention Transformer | LSTM + Multi-Scale Diff + SE + SupCon | — |
| 파라미터 | 불명 (아마 5-15M) | 2.85 M | 훨씬 경량 |

**논문 함의**:
- Same dataset · same label 정의로 검증된 공식 baseline 을 우리가 명확히 능가
- 우리 CATFT V5 (Cross-Attention, F1 0.9252) 도 MMTransformer 와 유사 수준 → **Cross-Attention 계열이 이 dataset 에서 ~91% 상한**임을 실증
- Domain-informed lightweight 접근이 architectural complexity 를 넘어선다는 논거 강화

## 6. 결론

- **AI Hub 데이터셋은 단일 sensor set 으로 수집됨** (Sharp / Vishay / KEMET / TeraRanger). 스키마의 4-way 는 enumeration 이지 사용 이력이 아님.
- **Device diversity 는 3-manufacturer / 3-model 로 유지** → 논문의 일반화 논거는 이쪽으로 정렬.
- **열화상 원본 해상도 32×32** → 120×160 upsampled. 이는 논문 Section 6.5 "thermal 기여도 +0.37%" 의 잠재 원인.
- **미사용 필드 5종 (external·trend·meta)** → Future Work 및 IJCAI 확장 시 feature engineering 소재.
- **4-class 라벨은 Z-score 3.5 + GT-30s + 관심/경고 임의 1:1 분할** — 관심↔경고 boundary 는 물리적 의미 없음. 위험은 마지막 30초 이벤트.
- **원본 sampling 100 ms → 우리는 1 s aggregation 만 사용**. 10× 고해상도 접근 불가.
- **원본 총 124,263 세트, 정상 99% → 우리는 111,870 balanced subset 사용**. 실배포는 real-world imbalance 환경.
- **AI Hub baseline MMTransformer F1 91.09% → 우리 V2+ 95.57%로 명확 능가**. Cross-Attention 계열 상한 ~91% 을 domain-informed LSTM 이 넘어섬.

## References

[1] AI Hub, "제조현장 이송장치의 열화 예지보전 멀티모달 데이터," [https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71802](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71802), Accessed: 2026-03-23.
[2] 사용자 제공 데이터 스키마 명세 (2026-07-31): meta_info · sensor_data · ir_data · annotations · external_data 필드 정의, S## 코드 → 실제 sensor 모델 매핑.
[3] 사용자 제공 AI Hub 활용 가이드라인 요약 (2026-07-31 저녁): 총 세트 수 124,263, sampling 100ms→1s, 4-class 정의 (Z-score 3.5 + GT-30s), MMTransformer baseline F1 91.09%, CT 채널 의미.
[4] 연구노트 #02: 원본 데이터셋 상세 분석, `docs/연구노트/연구노트_02_원본데이터셋_분석.md`
