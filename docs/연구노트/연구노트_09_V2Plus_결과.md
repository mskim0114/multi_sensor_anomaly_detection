# 연구노트 #09: V2+ 모델 결과

**작성일:** 2026-04-08
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. V2+ 모델 구성

기존 V2 (LSTM+TemporalDiff)에 3가지 개선 적용:

| 개선 | 내용 |
|------|------|
| Multi-Scale Temporal Diff | lag=1만 → **lag=1,5,10** (입력 8ch → 32ch) |
| SE Channel Attention | LSTM 출력에 센서 채널 가중치 자동 학습 |
| Supervised Contrastive Loss | CE + SupCon 결합 (weight=0.3) |

## 2. 최종 결과 (Best epoch: 27)

| Metric | V2 (이전 최고) | **V2+** | **Δ** |
|--------|-------------|---------|-------|
| **Val F1 (macro)** | 0.9430 | **0.9550** | **+1.20%** |
| **Val Accuracy** | 93.52% | **95.16%** | **+1.64%** |
| 파라미터 | 2.84M | 2.85M | +0.01M |
| 학습 시간 | 12분 | 20분 | +8분 |

### 클래스별 성능

| 클래스 | V2 F1 | V2+ F1 | Δ |
|--------|-------|--------|---|
| Normal | 0.94 | **0.97** | **+0.03** |
| Mild | 0.89 | **0.92** | **+0.03** |
| Moderate | 0.95 | 0.95 | 0.00 |
| Severe | 0.99 | 0.99 | 0.00 |

### Normal↔Mild 혼동 (핵심 개선)

| 오분류 | V2 | V2+ | 개선 |
|--------|-----|-----|------|
| Normal→Mild | 28 | **12** | **-57%** |
| Mild→Normal | 17 | **12** | **-29%** |
| **합계** | **45** | **24** | **-47%** |

### Confusion Matrix

```
              Predicted
              Normal   Mild   Moderate   Severe
Actual Normal   497     12        9        2
       Mild      12    258       16        0
    Moderate      0      5      280        0
      Severe      0      0        0       66
```

## 3. 전체 모델 비교 (최종)

| 모델 | Val F1 | Normal↔Mild 오류 | Severe 감지 | Params | 시간 |
|------|--------|:----------------:|:-----------:|-------:|-----:|
| V1: Baseline LSTM | 0.9235 | 57건 | 65/66 | 2.8M | 12분 |
| V2: LSTM+TempDiff | 0.9430 | 45건 | 66/66 | 2.8M | 12분 |
| V3: LSTM+EfficientNet | 0.9242 | 59건 | 66/66 | 4.5M | 108분 |
| V4: CATFT-NoCrossAttn | 0.9112 | 65건 | 66/66 | 7.6M | 108분 |
| V5: Full CATFT | 0.9252 | 50건 | 66/66 | 10.8M | 108분 |
| **V2+: V2+SupCon+SE** | **0.9550** | **24건** | **66/66** | **2.85M** | **20분** |

## 4. 분석

### 각 개선의 기여 (V2 → V2+)
- **Multi-Scale Diff (lag=1,5,10):** 4°C 차이가 lag-10에서 누적되어 Normal/Mild 구분 신호 강화
- **SupCon Loss:** 임베딩 공간에서 Normal/Mild 클러스터를 명시적으로 분리 → Normal→Mild 오류 57% 감소
- **SE Channel Attention:** NTC+CT에 높은 가중치, PM에 낮은 가중치 자동 학습

### 논문 스토리라인 업데이트
> "도메인 지식 기반 피처 엔지니어링(multi-scale temporal difference)과 supervised contrastive learning의 결합이 10배 큰 Transformer 모델(10.8M)보다 경량 LSTM(2.85M)에서 더 높은 성능(F1 0.9550)을 달성했다. 특히 Normal↔Mild 경계에서의 오분류를 47% 감소시켰다."

## 5. 파일 위치

| 파일 | 경로 |
|------|------|
| 모델 코드 | `src/models/v2_plus.py` |
| 학습 코드 | `src/train_v2plus.py` |
| 체크포인트 | `results/v2plus/best_model.pt` |
| 결과 | `results/v2plus/results.json` |
