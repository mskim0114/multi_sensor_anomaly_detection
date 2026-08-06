# 실제 센서 및 엣지 보드(Jetson) 구매 사양서 및 매칭 가이드

본 보고서는 AI Hub의 **"제조현장 이송장치의 열화 예지보전 멀티모달 데이터"** [1] 원본 111,870 샘플과 **2026-07-31 확인된 실제 데이터 스키마** 를 바탕으로, **4-class 이상상태 예측** 목표에 최적화된 센서/엣지보드 사양과 사용자 **보유 센서 활용 방안**을 제공합니다.

> **개정 이력**
> - **2026-03-26**: 원본 데이터셋 전수 분석 반영 (CT 실측 273.91A → 400A급 상향)
> - **2026-07-31 (본 개정, 큰 변경)**: 데이터 스키마 확인으로 AI Hub 실사용 센서 4종 확정. **이전 "4-way 이종 pool" 가정 폐기**, 단일 sensor set 확정. 사용자 보유 센서 (SPS30 / NTC 10K 3950 / FLIR Lepton 3.5 / DHS20P400A CL420) 반영.

---

## 0. 요약 (TL;DR)

| 카테고리 | AI Hub 실사용 | 사용자 보유 | 판정 | 조치 |
|:--------|:---|:---|:---:|:---|
| 온도 (NTC) | Vishay NTCLE413 (10kΩ) | **NTC 10K B=3950 프로브 (XH2.54)** | ✅ 등가 | 그대로 사용, β 캘리브레이션만 |
| 미세먼지 | Sharp GP2Y1014AU0F (아날로그 LED) | **Sensirion SPS30 (I²C/UART, 레이저)** | ✅ **더 우수** | 그대로 사용, 값 스케일 매핑 |
| 열화상 | TeraRanger Evo Thermal 33 (**32×32 native**) | **FLIR Lepton 3.5 (160×120 native)** | ✅ **19배 고해상** | 그대로 사용, downsample or fine-tune |
| 전류 (CT) | KEMET CT-06 (패시브, 스펙 0-200A/실제 270A) | DHS20P400A CL420 (액티브 4-20mA, 24V 필요) | ⚠️ 불편 → 교체 | **YHDC SCT-024-000 (400A, 패시브) 신규 구매 확정** |

**추가 필수 구매**: ADS1115 브레이크아웃 2개
**사용자 보유 (구매 필요 없음)**: PureThermal 모듈, 브레드보드, 점퍼선, 저항(33Ω / 10kΩ), 10µF 캐패시터, NTC 프로브

**총 신규 구매 예산**: 약 **₩60,000 ~ ₩90,000** (SCT-024 x4 + ADS1115 x2)

---

## 1. AI Hub 실사용 센서 확정 (2026-07-31 데이터 스키마 검증)

**중요**: 이전 문서에서 "AI Hub 데이터는 4종 센서 pool 을 섞어 heterogeneous 하게 수집됐다" 라고 서술했으나, **20,000 샘플 검증 결과 실제로는 각 카테고리 당 1종만 사용**되었음이 확인됨. 스키마의 4-way 리스트는 "허용 가능한 값 enumeration" 이었음.

### 1.1 실사용 센서 (100%)

| 카테고리 | 코드 | 제조사 | 모델 | 특성 |
|:---|:---:|:---|:---|:---|
| 미세먼지 | S02 | **Sharp** | **GP2Y1014AU0F** | 광학(LED) 산란, 아날로그 출력, LED 펄스 구동 필요 |
| 온도 | S10 | **Vishay** | **NTCLE413** | 10kΩ NTC 서미스터, 아날로그 (저항 → 전압) |
| 전류 | S18 | **KEMET** | **CT-06** | 패시브 CT (전류 트랜스포머), 아날로그 AC 출력 |
| 열화상 | S26 | **TeraRanger (Terabee)** | **Evo Thermal 33** | **32×32 native**, 33° FOV, USB 인터페이스, °C 픽셀 |

### 1.2 스키마상 존재하나 사용되지 않은 대안 (참고)

| 카테고리 | 스키마 대안 (미사용) |
|:---|:---|
| 미세먼지 | Sensirion SPS30, Shinyei PPD42S, Winsen ZH03 |
| 온도 | NTC-103 F343F, TT7-50KC3-3, MF52 |
| 전류 | CR8400 (CR Magnetics), Az-0500 (Talema), CCT406393 (TDK) |
| 열화상 | Panasonic Grid-EYE AMG88 (8×8), MI0801 (Meridian), Melexis MLX90640 (32×24) |

### 1.3 열화상 데이터의 실체 (매우 중요)

- **네이티브 해상도 32×32** (TeraRanger Evo Thermal 33)
- **저장 포맷 120×160** (float64, °C)
- 즉 원본 데이터는 **32×32 → 120×160 으로 약 15배 업샘플링**된 것
- 모델이 학습한 "열화상 spatial feature" 는 실질적으로 32×32 정보에서 유래
- **실배포에 FLIR Lepton 3.5 (160×120 native) 를 쓰면 오히려 학습 분포보다 정보가 풍부** → Section "5. 실배포 fine-tuning 전략" 참조

---

## 2. 데이터셋 분석 기반 센서 요구 사양

원본 데이터셋 분석에서 도출된 각 센서의 실측 범위입니다. 구매/사용할 센서는 이 범위를 충분히 커버해야 합니다.

| 센서 | 실측 Min | 실측 Max | 실측 Mean | Std | 단위 | 비고 |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| NTC (온도) | 17.33 | 80.20 | 32.49 | 9.27 | °C | State 3에서 평균 50.71°C |
| PM1.0 | 6.00 | 42.00 | 15.70 | 10.39 | µg/m³ | State 2에서 포화 |
| PM2.5 | 10.00 | 51.00 | 20.79 | 12.15 | µg/m³ | State 2에서 포화 |
| PM10 | 18.00 | 92.00 | 36.32 | 22.27 | µg/m³ | State 2에서 포화 |
| CT1 (입력단) | 0.50 | **201.34** | 6.23 | 18.61 | A | State 3에서 평균 29.85A |
| CT2 (출력단) | 0.60 | **273.91** | 42.73 | 46.43 | A | **최대값 가장 높음** |
| CT3 (모터1) | 0.22 | **243.25** | 24.49 | 29.74 | A | State 3에서 급증 |
| CT4 (모터2) | 0.30 | **219.04** | 11.21 | 18.99 | A | State 3에서 급증 |
| 열화상 Temp Min | 30.98 | 44.75 | - | - | °C | 배경 온도 |
| 열화상 Temp Max | 35.19 | **146.10** | - | - | °C | 고온 이상 포착 |

### 상태별 센서값 변화 패턴 (모델링 핵심 근거)

| 센서 | State 0 (정상) | State 1 (관심) | State 2 (경고) | State 3 (위험) | 패턴 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **NTC** | 26.78 | 30.95 | 40.65 | **50.71** | 선형 증가 → 열적 과부하 핵심 지표 |
| **PM1.0** | 13.66 | 16.18 | 18.83 | 18.76 | State 2 포화 → 초기 탄화 감지용 |
| **CT1** | 2.16 | 3.21 | 10.16 | **29.85** | State 3 급증(14×) → 전기/기계 결함 결정 지표 |
| **CT2** | 33.15 | 30.97 | 50.43 | **115.38** | State 3 급증(3.5×) |

**클래스 라벨 공식 명칭**: 0=정상 / 1=관심 / 2=경고 / 3=위험. 논문의 Normal / Mild / Moderate / Severe 는 편의 번역이며, 원 라벨은 위와 같음.

---

## 3. 사용자 보유 센서 활용 판정

### 3.1 온도 센서: NTC 10K B=3950 프로브 (XH2.54 커넥터) — ✅ 사용 가능

| 항목 | AI Hub (Vishay NTCLE413) | 사용자 보유 | 비고 |
|:---|:---|:---|:---|
| 저항 @ 25°C | 10 kΩ | 10 kΩ | 일치 ✅ |
| B 파라미터 | 3435~3977 (variant별) | **3950** | 근사 일치 |
| 측정 범위 | -40 ~ +125 °C | -40 ~ +125 °C | 일치 |
| 형태 | 리드/비드 | **프로브형 (XH2.54)** | 산업 환경 유리 |
| Jetson 결선 | ADS1115 + 10kΩ 기준저항 분압 | 동일 | ✅ |

**결론**: 그대로 사용. β 값 차이는 미미하며, Steinhart-Hart 캘리브레이션 한 번으로 등가 응답 확보. **연결 회로 매우 쉬움** (3.3V ─[10kΩ]─┬─[NTC 프로브]─ GND, 중간 tap → ADS1115 AIN).

### 3.2 미세먼지 센서: Sensirion SPS30 — ✅ AI Hub 대비 더 우수

| 항목 | AI Hub (Sharp GP2Y1014AU0F) | 사용자 보유 (SPS30) | 비고 |
|:---|:---|:---|:---|
| 측정 원리 | LED 광학 산란 | **레이저 산란** | SPS30 정밀도 우위 |
| PM 채널 | PM2.5 근사 (단일) | **PM1.0 / PM2.5 / PM4.0 / PM10 동시** | SPS30 완비 ✅ |
| 인터페이스 | 아날로그 (ADC + LED 펄스) | **I²C 또는 UART** | SPS30 훨씬 편함 ✅ |
| 정밀도 | ±35% (0-100 µg/m³) | ±10% (0-100 µg/m³) | SPS30 우위 |
| 소비전력 | 20 mA (LED 펄스 시 300 mA 순간) | 60 mA | 유사 |
| 수명 | 5년 | **10년** | SPS30 우위 |
| Jetson 결선 | 복잡 (ADC + 트리거 GPIO + 캐패시터) | **I²C 직결 (0x69)** | SPS30 압도적으로 쉬움 ✅ |

**결론**: SPS30 이 모든 면에서 우수. **Jetson 40핀 I²C (SDA/SCL) 에 직결만 하면 끝**. 별도 ADC 불필요.

**주의**: 값 스케일이 Sharp GP2Y1014AU0F 와 다를 수 있음 (레이저 vs LED). AI Hub 데이터로 학습된 모델을 SPS30 데이터로 그대로 쓰면 도메인 gap 존재. 캘리브레이션 계수 or fine-tuning 필요.

### 3.3 열화상: FLIR Lepton 3.5 (+ PureThermal, 보유) — ✅ AI Hub 대비 19배 고해상

| 항목 | AI Hub (TeraRanger Evo Thermal 33) | 사용자 보유 (FLIR Lepton 3.5) | 비고 |
|:---|:---|:---|:---|
| 네이티브 해상도 | **32×32** (1,024 px) | **160×120** (19,200 px) | Lepton 19× 우위 |
| Radiometry (°C 픽셀) | 지원 | 지원 (Radiometric) | 둘 다 ✅ |
| 온도 범위 | -20 ~ +300 °C | -10 ~ +140 °C (기본) / -10 ~ +450 °C (High Gain) | Lepton 기본모드 실측 146°C 경계, High Gain 필요 |
| 인터페이스 | USB (Terabee 사양) | **PureThermal 3 (USB)** ← 사용자 보유 | Lepton 결선 매우 쉬움 |
| 프레임 레이트 | 9 Hz | 8.7 Hz | 유사 |
| FOV | 33° | 56° | Lepton 넓음 |

**결론**: Lepton 3.5 + PureThermal 로 **USB 직결**. 학습 분포보다 정보 풍부.

**중요 조치 (실배포 시)**:
- **옵션 A (권장)**: Lepton 3.5 출력을 **32×32 로 downsample** 하여 학습 분포와 매치 → 모델 그대로 사용
- **옵션 B**: 160×120 원본으로 fine-tune → 성능 상승 여지, 하지만 실데이터 라벨링 필요
- **옵션 C**: Lepton 원본 → 학습 시 사용한 upsampling 파이프라인 재현 (32×32 → 120×160 bilinear)

### 3.4 전류 센서: DHS20P400A CL420 → YHDC SCT-024-000 교체 확정

| 항목 | AI Hub (KEMET CT-06) | 보유 (DHS20P400A CL420) | 교체 (YHDC SCT-024-000) |
|:---|:---|:---|:---|
| 타입 | 패시브 CT | **액티브 4-20mA** | 패시브 CT ✅ |
| 정격 | 스펙 0-200 A / 실 270 A | 400 A | **400 A** ✅ |
| 전원 | 불필요 | **24V DC 필수** | 불필요 ✅ |
| Jetson 결선 | Burden + bias + ADS1115 | Loop 150Ω + ADS1115 | Burden 16.5Ω + bias + ADS1115 |
| 부피/무게 | 소형 | 크고 무거움 | 소형 (클램프) |
| 가격 | - | 이미 보유 | ₩8,000~15,000/개 × 4 = **₩50,000** |

**결정: SCT-024-000 x4 신규 구매 확정** (24V PSU 부담 · 부피 회피).

**정격 400 A 선택 근거**:
- 실측 최대 CT2 = 273.91 A → 안전 계수 30% 포함 → 355 A 이상 필요
- YHDC 라인업 중 400A 급이 정확한 대안 (200A는 CT2/3 클리핑, 630A는 오버스펙)
- 4채널 모두 400A 통일 → burden 저항·캘리브레이션 스크립트 재사용, 스파이크 대응

---

## 4. Jetson Orin Nano 결선 (사용자 보유 기준)

### 4.1 전체 인터페이스 요약

```
Jetson Orin Nano 8GB (40-pin 헤더)
│
├── I²C (Pin 3 SDA / Pin 5 SCL / Pin 1 3.3V / Pin 6 GND)
│   ├── SPS30 (0x69)                      ← 사용자 보유, 직결
│   ├── ADS1115 #1 (0x48)                  ← NTC + CT1/2/3
│   │   ├── AIN0: NTC 프로브 분압 tap
│   │   ├── AIN1: CT1 신호 (입력단)
│   │   ├── AIN2: CT2 신호 (출력단)
│   │   └── AIN3: CT3 신호 (모터1)
│   └── ADS1115 #2 (0x49, ADDR을 VDD로)
│       └── AIN0: CT4 신호 (모터2)
│
└── USB 3.0
    └── PureThermal 3 + FLIR Lepton 3.5   ← 사용자 보유, USB 케이블만
```

### 4.2 SPS30 결선 (I²C 모드)

```
SPS30 Pin      Jetson 40-pin
─────────────────────────────
1 VDD          → Pin 2 (5V)
2 SDA          → Pin 3 (SDA)
3 SCL          → Pin 5 (SCL)
4 SEL (I²C 선택) → Pin 6 (GND)  ← 반드시 GND
5 GND          → Pin 6 (GND)
```

SEL 을 GND 에 물리면 I²C 모드 (안 물리면 UART). 주소 `0x69`.
라이브러리: `pip install sensirion-i2c-driver sensirion-i2c-sps`. `sudo i2cdetect -y 1` 로 확인.

### 4.3 NTC 프로브 결선

```
Jetson 3.3V (Pin 1) ──┬── R_ref 10kΩ ──┬── NTC 프로브 ── GND (Pin 6)
                       │                │
                                        └── ADS1115 #1 AIN0
```

V = ADS 측정값, R_NTC = R_ref × V / (3.3 − V), Steinhart-Hart 식으로 °C 변환.

### 4.4 CT (YHDC SCT-024, 신규 구매) 결선 — bias 4채널 공용

```
+3.3V ──[10kΩ]──┬── V_bias (1.65V, 4채널 공용) ──┬── CT1_LeadB
                 │                                  ├── CT2_LeadB
                 ├── 10µF ── GND                   ├── CT3_LeadB
                 │                                  └── CT4_LeadB
GND ──[10kΩ]────┘

각 채널 (CT1~CT4 모두 동일):
    CT_A ──┬─────► ADS1115 AIN
           │
        R_burden = 16.5Ω (33Ω 2개 병렬)
           │
    CT_B ──┴─── V_bias
```

**Burden 16.5Ω** = 33Ω 2개 병렬. 400A × 1:8000 비율 = 50mA RMS secondary → peak 70.7mA × 16.5Ω = 1.17V peak. Bias 1.65V 중심 → 0.48V ~ 2.82V (3.3V 안 안전).

### 4.5 FLIR Lepton 3.5 결선 (PureThermal 보유)

```
FLIR Lepton 3.5 ── PureThermal 3 모듈 ── USB-C 케이블 ── Jetson USB 3.0 포트
```

Jetson 에서 확인: `lsusb | grep -i thermal`, `v4l2-ctl --list-devices`.
읽기: OpenCV `cv2.VideoCapture()` (Y16 raw thermal), 또는 GetThermal / libuvc.
High Gain 모드 SDK 설정 필요 (146°C 실측 대응).

---

## 5. 실배포 fine-tuning 전략

AI Hub 로 학습한 V2+ 모델을 사용자 보유 센서 조합으로 그대로 배포하면 **도메인 gap** 존재.

| 센서 | Gap 종류 | 조치 우선순위 |
|:---|:---|:---|
| NTC 10K 3950 | β 미세 차이 | 1️⃣ Steinhart-Hart 캘리브레이션 (즉시) |
| SPS30 | 원리(레이저 vs LED), 값 스케일 | 2️⃣ AI Hub 대비 값 스케일 매핑 (선형 회귀) |
| Lepton 3.5 vs Evo Thermal | 32×32 → 160×120 (해상도 증가) | 3️⃣ downsample 학습 매치 or fine-tune 재학습 |
| YHDC SCT-024 vs KEMET CT-06 | 응답·노이즈 차이 | 3️⃣ 실측 데이터로 fine-tune |

**Phase 2 실센서 데이터 수집 계획** (2027 상반기 예정):
- 4-class 라벨링 가능한 시나리오 확보 (정상/관심/경고/위험 유발 조건)
- 최소 각 클래스 1,000 샘플 × 4 = 4,000 샘플 목표
- V2+ 를 fine-tuning starting point 로 사용, transfer learning

---

## 6. Jetson Orin Nano 사양 (참고)

| 항목 | 사양 |
| :--- | :--- |
| **모델** | NVIDIA Jetson Orin Nano 8GB Developer Kit |
| **AI 성능** | 최대 40 TOPS |
| **GPU 메모리** | 8GB LPDDR5 |
| **센서 인터페이스** | I²C, SPI, UART, GPIO (40-pin 헤더) + USB 3.0 |
| **전원** | 19V DC 배럴잭 (Dev Kit 표준, 45W 이상) |

---

## 7. 총 예상 신규 구매 비용 (사용자 보유 반영)

| 항목 | 수량 | 예상가 |
|:---|:---:|---:|
| **YHDC SCT-024-000 (400A)** | 4 | ₩40,000~60,000 |
| **ADS1115 브레이크아웃** | 2 | ₩20,000~30,000 |
| **소계** | | **₩60,000~90,000** |

**보유 부품 활용 (별도 구매 불필요)**:
- SPS30, NTC 10K 3950 프로브, FLIR Lepton 3.5, PureThermal 모듈
- 저항: 33Ω, 10kΩ / 캐패시터: 10µF
- 브레드보드, 점퍼선

**구매처**:
- YHDC SCT-024-000: 알리익스프레스, Devicemart, Mouser, Digi-Key
- ADS1115: Adafruit ID #1085, Devicemart, Eleparts

---

## 8. 구매 시 주의사항

1. **전류 센서 용량**: CT2 실측 273.91A → 400A급 필수. 200A 이하는 클리핑.
2. **FLIR Lepton 버전**: 반드시 **3.5** (Radiometric). 3.0은 온도값 출력 불가.
3. **FLIR 온도 모드**: 실측 최고 146.10°C → 기본 (-10~140°C) 경계. **High Gain 모드** 필요.
4. **ADS1115 개수**: NTC(1ch) + CT(4ch) = 총 5채널 → 2개 필요. 주소 0x48 / 0x49 구분 (ADDR→GND / ADDR→VDD).
5. **SPS30 SEL 핀**: I²C 모드는 반드시 SEL → GND.
6. **SPS30 케이블**: 5-pin ZH1.5 커넥터 확인 (없으면 케이블 별도 발주 ~₩3,000).

---

## 9. 요약 결론

- **AI Hub 실사용 센서 4종 확정** (2026-07-31): Sharp GP2Y1014AU0F / Vishay NTCLE413 / KEMET CT-06 / TeraRanger Evo Thermal 33.
- **사용자 보유 3종 (NTC / SPS30 / FLIR Lepton 3.5 + PureThermal) 은 그대로 사용 가능 · 대부분 AI Hub 대비 동등 이상 성능**.
- **전류 센서 교체 확정** — DHS20P400A → **YHDC SCT-024-000 (400A 패시브) 신규 구매** (24V 어댑터 회피).
- **총 신규 구매**: **₩60,000~90,000** (SCT-024 x4 + ADS1115 x2).
- **모델은 학습된 그대로 배포 가능**하지만 3종의 도메인 gap 존재 → Phase 2 fine-tuning 필요.

---

## References

[1] AI Hub, "제조현장 이송장치의 열화 예지보전 멀티모달 데이터," [https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71802](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71802), Accessed: 2026-03-23.
[2] 연구노트 #02: 원본 데이터셋 상세 분석, `docs/연구노트/연구노트_02_원본데이터셋_분석.md`
[3] AI Hub 데이터 스키마 명세 (사용자 제공, 2026-07-31): meta_info · sensor_data · ir_data · annotations · external_data 필드 확정, S## 코드 → 실제 sensor 모델 매핑.
