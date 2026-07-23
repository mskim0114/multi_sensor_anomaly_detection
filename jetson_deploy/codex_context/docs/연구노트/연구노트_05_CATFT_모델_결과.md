# 연구노트 #05: CATFT 모델 학습 결과

**작성일:** 2026-04-06
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 모델 개요

**CATFT (Cross-Attention Temporal Fusion Transformer)**

```
센서 (B,30,8) → Temporal Diff + Linear Proj → Transformer Encoder → (B,30,256)
                                                                        ↕ Cross-Attention (×2)
열화상 (B,30,120,160) → EfficientNet-B0 → Linear Proj → Transformer Encoder → (B,30,256)
                                                                        ↓
                                                              Concat + Mean Pool → MLP → 4-class
```

## 2. 실험 설정

| 항목 | 값 |
|------|-----|
| 모델 | CATFT |
| 파라미터 | 10,791,299 (trainable: 10,482,639) |
| GPU | Quadro RTX 6000 (GPU 1) |
| Epochs | 30 |
| Batch size | 8 |
| Optimizer | AdamW (lr=5e-4, weight_decay=0.01) |
| LR Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| Loss | CrossEntropyLoss (class weighted) |
| Gradient clipping | max_norm=1.0 |

## 3. 최종 결과 (Best epoch: 26)

### 베이스라인 비교

| Metric | Baseline (LSTM) | **CATFT** | Δ |
|--------|-----------------|-----------|---|
| Val Accuracy | 0.9188 | **0.9213** | +0.0025 |
| Val F1 (macro) | 0.9235 | **0.9252** | **+0.0016** |
| 파라미터 수 | 2.8M | 10.8M | +8M |
| 에폭당 시간 | 37초 | 215초 | +178초 |

### 클래스별 성능 비교

| 클래스 | Baseline F1 | CATFT F1 | Δ |
|--------|-------------|----------|---|
| Normal (0) | 0.93 | **0.94** | +0.01 |
| Mild (1) | 0.87 | 0.87 | 0.00 |
| Moderate (2) | 0.94 | 0.93 | -0.01 |
| **Severe (3)** | 0.96 | **0.96** | 0.00 |

### CATFT Confusion Matrix

```
              Predicted
              Normal  Mild  Moderate  Severe
Actual Normal   472    36      10       2
       Mild      14   258      14       0
    Moderate      0    11     270       4
      Severe      0     0       0      66
```

### 핵심 비교 (Normal↔Mild 혼동)

| 오분류 | Baseline | CATFT | 개선 |
|--------|----------|-------|------|
| Normal→Mild | 39 | **36** | -3 |
| Mild→Normal | 18 | **14** | -4 |
| **Severe 100% 감지** | 65/66 | **66/66** | +1 |

## 4. 분석

### 개선된 점
1. **Severe 완벽 감지**: 66개 중 66개 정확 (baseline은 65/66)
2. **Normal↔Mild 혼동 감소**: 오분류 57건 → 50건 (-12.3%)
3. **Normal Precision 향상**: 0.96 → 0.97
4. **전체 F1 소폭 개선**: 0.9235 → 0.9252

### 한계점
1. **수렴이 느림**: 30 에폭, ~1.8시간 (baseline은 20 에폭, 12.5분)
2. **학습 불안정성**: Epoch 8, 11에서 val F1 급락 후 회복 (EfficientNet fine-tuning 불안정)
3. **Mild F1 동일**: 0.87로 개선 없음 - Normal↔Mild 구분이 여전히 가장 어려운 과제
4. **파라미터 효율**: 4배 더 많은 파라미터로 소폭 개선 (+0.16%)

### 학습 곡선 특징
- Epoch 1-10: 느린 수렴 (EfficientNet 적응 기간)
- Epoch 11: val F1 0.65로 급락 (학습 불안정)
- Epoch 12: 0.91로 급반등 (backbone 적응 완료)
- Epoch 24-26: 최고 성능 도달 (LR이 충분히 낮아진 후)

## 5. 향후 개선 방향

1. **EfficientNet 더 많이 freeze**: 현재 5 layer → 7 layer freeze로 불안정성 감소
2. **Warmup scheduler 추가**: 초반 불안정 해소
3. **더 많은 에폭 / step_size=5**: 학습 데이터 늘려서 Transformer 수렴 돕기
4. **Contrastive loss 추가**: Normal↔Mild 경계를 명확히 분리
5. **Label smoothing**: 과적합 방지

## 6. 파일 위치

- 모델 체크포인트: `/home/keti/factory_safety/results/catft/best_model.pt`
- 결과 JSON: `/home/keti/factory_safety/results/catft/results.json`
- 모델 코드: `/home/keti/factory_safety/src/models/catft.py`
- 학습 코드: `/home/keti/factory_safety/src/train_catft.py`
