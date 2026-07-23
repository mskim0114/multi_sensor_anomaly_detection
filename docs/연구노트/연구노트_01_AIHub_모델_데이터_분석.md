# 연구노트 #01: AI Hub 모델 및 데이터셋 분석

**작성일:** 2026-03-24
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 데이터셋 분석 (Sample.zip, ~980MB)

### 1.1 구조
```
Sample/
  01.원천데이터/           ← CSV(센서) + BIN(열화상)
    agv/01/ (7세션)
    oht/01/ (12세션)
  02.라벨링데이터/          ← JSON(라벨+메타정보)
    agv/01/, oht/01/
```
- **총 18,822개 파일**: CSV 6,258 + BIN 6,258 + JSON 6,258
- **19개 세션**: AGV 7개 + OHT 12개, 세션당 ~300-360 샘플

### 1.2 데이터 포맷

| 파일 유형 | 포맷 | 내용 |
|-----------|------|------|
| CSV (~80B) | 1행 8열 | NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4 |
| BIN (~153KB) | numpy float64 | 120×160 열화상 이미지 (섭씨 온도값) |
| JSON (~2.9KB) | 구조화 라벨 | 센서값, IR 최고온도, 외부환경, state(0-3) |

### 1.3 라벨링 체계 (4-class)
| State | 의미 |
|-------|------|
| 0 | 정상 (Normal) |
| 1 | 경도 열화 (Mild degradation) |
| 2 | 중도 열화 (Moderate degradation) |
| 3 | 중증 열화 (Severe degradation) |

### 1.4 클래스 분포 (세션별 편차 큼)
- 일부 세션은 전체 state=0 (정상만)
- 일부 세션은 0→1→2→3 점진적 열화 포함
- **클래스 불균형 존재 가능** → 학습 시 class weighting 필요

---

## 2. AI Hub 제공 모델 분석 (Multimodal LSTM)

### 2.1 아키텍처 개요

```
센서 데이터 (batch, 30, 8) ──→ LSTMEncoder ──→ (batch, 128)
                                                    │
                                              unsqueeze+repeat → (batch, 30, 128)
                                                    │
열화상 이미지 (batch, 30, 19200) → ThermalImageEncoder → (batch, 30, 128)
                                                    │
                                              concat → (batch, 30, 256)
                                                    │
                                              mean(dim=1) → (batch, 256)
                                                    │
                                              Linear(256, 4) → 4-class logits
```

### 2.2 모듈 구성

| 모듈 | 구조 | 입출력 |
|------|------|--------|
| **LSTMEncoder** | LSTM(8, 128, layers=3, dropout=0.1) + Linear(128,128) | (batch,30,8) → (batch,128) |
| **ThermalImageEncoder** | 3-layer CNN (1→16→32→64ch) + MaxPool + FC | (batch,30,120,160) → (batch,30,128) |
| **MultimodalLSTM** | Late fusion: concat + mean pool + Linear(256,4) | 두 인코더 출력 → 4-class |

### 2.3 학습 설정
| 파라미터 | 값 |
|---------|-----|
| window_size | 30 (sliding window) |
| step_size | 1 |
| batch_size | 4 |
| learning_rate | 0.001 |
| optimizer | Adam |
| loss | CrossEntropyLoss |
| epochs | 1 (코드) / 10 (README) |
| train:val:test | 80:10:10 |

### 2.4 데이터 파이프라인
1. JSON + BIN 파일 로드
2. 8차원 센서 벡터 + 열화상(120×160) 추출
3. Sliding window (size=30, step=1)로 시퀀스 구성
4. 폴더별 독립 train/val/test 분할 → .npy 저장
5. 폴더 순회하며 학습

---

## 3. 기존 모델의 한계점 및 개선 방향

### 3.1 아키텍처 한계

| 한계 | 상세 | 개선 방향 |
|------|------|-----------|
| **Temporal fusion 부재** | LSTM 요약벡터를 broadcast 후 mean → 실질적으로 `[sensor_summary; mean(thermal)]`과 동일. 시간축 정렬이 활용되지 않음 | Cross-Attention 기반 fusion 또는 Transformer 기반 temporal alignment |
| **열화상 프레임 독립 처리** | CNN이 각 프레임을 독립적으로 인코딩, 열화상 시퀀스의 시간적 변화 미학습 | 열화상 시퀀스에 temporal modeling 추가 (Video Transformer 등) |
| **FC 후 활성함수 없음** | LSTMEncoder의 Linear(128,128) 뒤에 비선형 함수 없음 → 수학적으로 redundant | ReLU/GELU 추가 또는 제거 |
| **정규화 부재** | BatchNorm/LayerNorm 없음 | 각 인코더에 normalization 추가 |
| **Dropout 부족** | LSTM inter-layer만 0.1, 분류 헤드/CNN에 없음 | 적절한 dropout 추가 |

### 3.2 데이터 파이프라인 한계

| 한계 | 상세 | 개선 방향 |
|------|------|-----------|
| **심각한 데이터 누수** | step_size=1인 sliding window를 random split → 29/30 겹치는 윈도우가 train/test에 분산 | 시간 블록 단위 또는 세션 단위 분할 |
| **입력 정규화 없음** | 센서값 raw 사용 (온도/미세먼지/전류의 스케일이 매우 다름) | Z-score 또는 Min-Max 정규화 |
| **라벨 할당 방식** | 윈도우의 첫 번째 샘플 라벨만 사용 → 상태 전이 구간에서 부정확 | majority vote 또는 마지막 샘플 라벨 사용 |
| **DataLoader shuffle=False** | 학습 시 순차적 데이터 제공 → SGD 최적화에 불리 | shuffle=True |
| **데이터 증강 없음** | 열화상 이미지에 대한 augmentation 없음 | flip, rotation, noise injection 등 |

### 3.3 학습 파이프라인 한계

| 한계 | 개선 방향 |
|------|-----------|
| LR scheduler 없음 | CosineAnnealing, ReduceLROnPlateau 등 적용 |
| Early stopping 없음 | validation loss 기반 early stopping |
| 모델 체크포인팅 없음 | best model 저장 로직 추가 |
| Class weighting 없음 | 클래스 불균형 대응 (weighted CE 또는 Focal Loss) |
| Confusion matrix가 val+test 혼합 | 분리하여 별도 평가 |

---

## 4. 문서 분석 요약

### 4.1 AI모델 설명서
- **모델명:** mmTransformer (GitHub: decisionforce/mmTransformer)
- **목적:** OHT/AGV 이송장치 탄화(열화) 예측을 위한 온디바이스 AI
- **라이선스:** Apache 2.0

### 4.2 환경
- Python 3.7+, PyTorch 2.1.0+cu121
- Docker 배포 지원 (`docker44_1.tar`)
- 실행: `python runner.py` → 결과 `/logs`, `/model_results`

### 4.3 주요 dependencies
- torch 2.1.0+cu121, numpy 1.26.3, scikit-learn 1.5.2, pandas 2.2.3, matplotlib 3.9.2

---

## 5. 향후 연구 과제 (미결정 사항)

1. **아키텍처 선택:** Sensor LM 기반 vs MMTransformer 기반 → 비교 실험 필요
2. **3종 이상상태 매핑:** AI Hub의 4-class(0-3 열화도)를 우리의 3종(열적과부하/탄화연기/전기기계결함)으로 재정의하는 전략
3. **데이터 파이프라인 재설계:** 누수 없는 분할, 정규화, 증강 포함
4. **실시간 추론 최적화:** Jetson Orin Nano 배포를 위한 TensorRT/양자화 전략
