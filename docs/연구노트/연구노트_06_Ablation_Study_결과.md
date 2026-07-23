# 연구노트 #06: Ablation Study 결과

**작성일:** 2026-04-06
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 실험 목적

CATFT 모델의 각 컴포넌트가 성능에 미치는 기여도를 분석하기 위한 Ablation Study.

## 2. 실험 구성 (5개 모델)

| Variant | 모델 | Temporal Diff | EfficientNet | Transformer | Cross-Attn | Params |
|---------|------|:---:|:---:|:---:|:---:|---:|
| V1 | Baseline LSTM | | | | | 2.8M |
| V2 | LSTM + TempDiff | ✅ | | | | 2.8M |
| V3 | LSTM + EfficientNet | | ✅ | | | 4.5M |
| V4 | CATFT - NoCrossAttn | ✅ | ✅ | ✅ | | 7.6M |
| V5 | Full CATFT | ✅ | ✅ | ✅ | ✅ | 10.8M |

## 3. 최종 결과

| Variant | Val F1 | Val Acc | Severe 감지 | Best Epoch | 학습시간 |
|---------|--------|---------|------------|------------|---------|
| V1: Baseline LSTM | 0.9235 | 0.9188 | 65/66 | 18/20 | ~12분 |
| **V2: LSTM+TempDiff** | **0.9430** | **0.9352** | **66/66** | 19/20 | ~12분 |
| V3: LSTM+EfficientNet | 0.9242 | 0.9162 | 66/66 | 26/30 | ~1.8시간 |
| V4: CATFT-NoCrossAttn | 0.9112 | 0.9067 | 66/66 | 23/30 | ~1.8시간 |
| V5: Full CATFT | 0.9252 | 0.9213 | 66/66 | 26/30 | ~1.8시간 |

### 클래스별 F1 비교

| Variant | Normal | Mild | Moderate | Severe |
|---------|--------|------|----------|--------|
| V1: Baseline | 0.93 | 0.87 | 0.94 | 0.96 |
| **V2: +TempDiff** | **0.94** | **0.89** | **0.95** | **0.99** |
| V3: +EfficientNet | 0.93 | 0.85 | 0.94 | 0.98 |
| V4: -CrossAttn | 0.92 | 0.85 | 0.93 | 0.94 |
| V5: Full CATFT | 0.94 | 0.87 | 0.93 | 0.96 |

### Normal↔Mild 혼동 비교

| Variant | Normal→Mild | Mild→Normal | 합계 |
|---------|------------|------------|------|
| V1: Baseline | 39 | 18 | 57 |
| **V2: +TempDiff** | **28** | **17** | **45** |
| V3: +EfficientNet | 28 | 31 | 59 |
| V4: -CrossAttn | 49 | 16 | 65 |
| V5: Full CATFT | 36 | 14 | 50 |

## 4. 분석

### 4.1 각 컴포넌트 기여도

| 컴포넌트 | 기여도 (ΔF1) | 분석 |
|---------|-------------|------|
| **Temporal Diff** | **+1.95%** (V1→V2) | **가장 큰 기여.** 센서 변화율이 Normal↔Mild 구분에 결정적. 파라미터 증가 거의 없음 (+4K) |
| EfficientNet | +0.07% (V1→V3) | 미미한 기여. pretrained CNN이 열화상에서 추가 정보를 거의 추출하지 못함 |
| Transformer Encoder | -1.30% (V2→V4 추정) | **오히려 성능 하락.** 9,313개 윈도우로는 Transformer가 과적합 |
| Cross-Attention | +1.40% (V4→V5) | Cross-Attention이 Transformer의 과적합을 보상. 모달리티 간 정보 교환 효과 |

### 4.2 핵심 발견

1. **Temporal Difference가 핵심이다.** 단순히 `sensor[t] - sensor[t-1]`을 입력에 추가하는 것만으로 F1이 0.9235 → 0.9430 (+2%). 복잡한 아키텍처 변경 없이 가장 큰 성능 향상.

2. **모델 복잡도 ≠ 성능.** V2(2.8M)가 V5(10.8M)보다 F1이 +1.78% 높음. 데이터 규모(9,313 윈도우) 대비 Transformer가 과한 상태.

3. **EfficientNet은 효과 없음.** pretrained ImageNet 특징이 열화상 이미지(단일 채널, 온도 분포)에 잘 전이되지 않음. 3-layer CNN 베이스라인이 충분.

4. **Cross-Attention은 조건부 유효.** 단독으로는 Transformer 과적합을 유발하지만, cross-attention을 추가하면 모달리티 간 보완으로 과적합을 일부 해소.

### 4.3 논문 스토리라인

> "복잡한 Transformer 아키텍처보다, 도메인 지식에 기반한 간단한 피처 엔지니어링(temporal difference)이 더 효과적이다. 이는 제조 안전 분야에서 센서 변화율이 절대값보다 이상 상태 예측에 중요함을 시사한다."

## 5. 최종 추천 모델

**V2 (LSTM + Temporal Diff)** — 최고 성능, 최소 파라미터, 최단 학습 시간

| 항목 | 값 |
|------|-----|
| Val F1 (macro) | **0.9430** |
| Val Accuracy | **0.9352** |
| Severe 감지 | **66/66 (100%)** |
| Severe F1 | **0.99** |
| 파라미터 | 2.8M |
| 학습 시간 | ~12분 |

## 6. 파일 위치

| Variant | 결과 |
|---------|------|
| V1 | `/home/keti/factory_safety/results/baseline/` |
| V2 | `/home/keti/factory_safety/results/ablation_v2/` |
| V3 | `/home/keti/factory_safety/results/ablation_v3/` |
| V4 | `/home/keti/factory_safety/results/ablation_v4/` |
| V5 | `/home/keti/factory_safety/results/catft/` |
| 모델 코드 | `/home/keti/factory_safety/src/models/ablation_variants.py` |
| 학습 코드 | `/home/keti/factory_safety/src/train_ablation.py` |
