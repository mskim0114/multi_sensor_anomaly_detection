# 연구노트 #21: EXP-20260907-006 장비 그룹 K-fold — 장비 단위 불확실성 (2026-09-21)

- **연구 질문 및 hypothesis ID:** `AIHUB-H1` 계열, 논문 일반화 주장(CE-046). 실험 `EXP-20260907-006`.
  질문: 논문의 모든 수치가 같은 validation 장비 4대(agv17/18, oht17/18)에서 나왔다. V2+ 를 **처음 보는 장비**에 적용했을 때 F1 은 얼마이고,
  장비 간 편차는 seed 편차(0.0006~0.0018)와 비교해 얼마나 큰가. Severe 100 % 탐지는 다른 장비에서도 유지되는가.
- **시작 전 알고 있던 사실:** Training 32대(agv 16, oht 16, 303 session, 9,313 window), Validation 4대(39 session, 1,157 window). 같은 4대가
  모델·lag·구성 선택과 최종 보고에 반복 사용됐고 미사용 test 가 없다(CE-046 PARTIALLY CONFIRMED). 단일 split V2+ F1 0.9548(G2e)/0.9549(결정론).
- **이번 가설과 반증 조건:** 장비 단위 std 가 seed std 보다 크면 "0.9557 ± 0.0006" 은 일반화 불확실성을 과소 표현한다. held-out 장비에서
  Severe recall < 100 % 이면 "100 % detection" 은 validation 4대에 한정해야 한다.
- **데이터·annotation·split·코드·환경 버전:** AI Hub #71802 SSD 사본(G2d PASS). Training 32대만 K=4 장비 분리 fold 로 사용; Validation 4대는
  **frozen** (checkpoint 선택에 쓰지 않고 fold 모델당 1회 채점). 코드 `248a1a9` **clean**, `src/train_kfold.py`(self-test PASS, 1 epoch smoke).
  환경 `factory_training`, RTX 6000 ×2, **`--deterministic` 기본 on**(D-024, 경고 op 0).
- **실행 ID와 설정/로그/결과 위치:** `EXP-20260907-006/fold{0..3}` → `results/kfold_devices/fold{k}_of4_seed42/results.json`(서버 원본 + Jetson
  미러: fold map, 장비별 혼동행렬, history, provenance). 로그·집계 `~/review_runs/20260921_kfold/`(서버·Jetson, `aggregate_kfold.py/.txt`).
- **관측 결과 / 해석 / 결정:** §2~§5.
- **논문 후보에 미치는 영향:** PC0 — 장비 단위 불확실성 병기·Severe 100 % 한정 정정안(§5, O-113). **본문 미변경.**
- **다음 최소 실험과 통과 조건:** §7.

---

## 1. 설계

```
fold 배정   agv·oht 각각 id 순 정렬 후 i % 4 → fold 당 agv 4 + oht 4 (self-test: K=2/4/8 cover·balance PASS)
  fold 0  agv01 05 09 13 · oht01 05 09 13      fold 1  agv02 06 10 14 · oht02 06 10 14
  fold 2  agv03 07 11 15 · oht03 07 11 15      fold 3  agv04 08 12 16 · oht04 08 12 16
학습        24대 (≈6,975~7,009 window) → held-out 8대 (2,304~2,338 window) 로 checkpoint 선택(best held-out macro-F1)
            frozen val 4대(1,157) 는 epoch 마다 채점만 (선택에 불참)
프로토콜    G2e 와 동일: V2+ lags[1,5,10]+SE+SupCon 0.3, 30 epoch, AdamW 1e-3/wd 0.01, plateau(0.5, 3) on held-out loss, clip 1.0, FP32,
            class-weighted CE + weighted sampler(fold 학습셋 기준), seed 42 (초기 가중치 58f43109… = V2+ seed 42 와 동일)
정규화      DataConfig 의 고정 학습셋 상수(fold 간 공유 → 작은 통계 누출, 결과에 명시)
실행        4 fold 동시(GPU 당 2) 18:32~19:14 KST, fold 당 36~39분, 4 run 모두 exit 0, git_dirty False
```

## 2. 결과 — fold 단위

| fold | held-out 8대 | held-out F1 | acc | NM | best ep | frozen val F1 | frozen NM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | agv01/05/09/13, oht01/05/09/13 | 0.947488 | 0.9465 | 79 | 22 | 0.943005 | 41 |
| 1 | agv02/06/10/14, oht02/06/10/14 | 0.948953 | 0.9448 | 85 | 30 | 0.950049 | 30 |
| 2 | agv03/07/11/15, oht03/07/11/15 | 0.956120 | 0.9537 | 62 | 27 | 0.947247 | 32 |
| 3 | agv04/08/12/16, oht04/08/12/16 | 0.961247 | 0.9562 | 73 | 28 | 0.944129 | 37 |

```
fold 단위 held-out F1   mean 0.953452  std 0.006421  [0.947488, 0.961247]
frozen val F1 (24대 학습 모델 4개)   mean 0.946107  std 0.003182  [0.943005, 0.950049]
  vs 32대 전체 학습 단일 split: G2e 0.954825 · 결정론 0.954888 · 논문 0.9557 ± 0.0006
pooled held-out (9,313 window, 32대)  macro F1 0.953411 · acc 0.9503 · NM 299 · Severe recall 0.9902 (506/511)
per-class F1  Normal 0.9669 · Mild 0.8979 · Moderate 0.9606 · Severe 0.9883
```

## 3. 결과 — 장비 단위 (32대, held-out macro F1)

```
32대   mean 0.954642  std 0.014529  min 0.9276 (oht03)  max 0.9782 (agv16)
       bootstrap 95 % CI of mean (장비 재표집 10,000회, seed 0)  [0.949686, 0.959547]  half-width 0.0049
AGV 16 mean 0.960174  std 0.012855  CI [0.954116, 0.966329]
OHT 16 mean 0.949110  std 0.014343  CI [0.942353, 0.956009]
Severe 누락 5 window: oht05 1, oht09 1 (fold 0) · oht03 2 (fold 2) · agv08 1 (fold 3) — 전부 Moderate 로 예측
frozen val 장비별 (fold 모델 4개): agv17 0.969 ± 0.007 · agv18 0.890 ± 0.010 · oht17 0.952 ± 0.008 · oht18 0.960 ± 0.005
```

장비별 전체 표는 `aggregate_kfold.txt` §2.

## 4. 해석 — 관찰과 추론을 분리

**관찰.**
- 장비 간 std(0.0145)는 seed std(0.0006~0.0018)의 **8~24배**다. 평균의 95 % CI half-width 는 ±0.005 로 논문 표기 ±0.0006 의 8배.
- 처음 보는 장비에서의 평균 F1 0.9546 은 단일 split 값(0.9548)과 같지만, 장비 하나하나는 0.928~0.978 로 흩어진다. 최저 장비(oht03)와 최고
  장비(agv16)의 차이 0.051 은 논문에서 다루는 어떤 모델 간 차이(V2→V2+ 0.017)보다 크다.
- AGV 가 OHT 보다 평균 0.011 높다. OHT 의 Mild 오류가 더 많다(NM 장비별 표).
- **Severe 100 % 는 validation 4대에서만 성립**한다. held-out 32대 pooled 로는 506/511 (99.0 %), 5 window 가 Moderate 로 예측됐다.
- frozen val 4대에서 24대-학습 모델은 0.9461 ± 0.0032 로 32대-학습 모델(0.9548)보다 0.009 낮다. agv18 이 일관되게 어렵다(0.88~0.90).
- 4 fold 모두 결정론 모드, 초기 가중치 동일, 경고 op 0.

**추론(미검증).**
- 학습 장비 수(24 vs 32)가 frozen val 차이 0.009 의 원인일 수 있으나, fold 별 학습셋 구성 차이도 섞여 있어 분리하지 못했다.
- AGV/OHT 차이는 세션 길이(oht 12 vs agv 7 세션)·라벨 분포 차이와 관련될 수 있다. 이 실험은 원인을 다루지 않는다.
- Severe 누락 5건이 라벨 경계(Moderate↔Severe)의 애매함인지 모델 한계인지는 window 를 직접 봐야 한다.

**한계.** seed 1개(42) — fold 간 차이에 seed 효과가 섞이지 않도록 초기화는 같지만, 다른 seed 에서 순위가 바뀔 수 있다. 정규화 상수 공유(작은
누출). fold 당 학습 장비 24대이므로 "32대 학습 모델의 일반화" 를 직접 측정한 것은 아니다(약간 보수적 추정).

## 5. 결정과 정정안

- **D-025.** 논문 성능 수치에는 seed 불확실성과 **장비 단위 불확실성을 병기**한다. 근거 실행은 EXP-006. 논문 본문 교체 문안은 아래에 준비하되
  적용은 사용자 판단(**O-113**).
- 정정안 (a) 초록·§6.1·결론의 "0.9557 ± 0.0006" 옆에: "device-grouped 4-fold on the 32 training machines: held-out macro F1 0.9546, 95 % CI
  [0.9497, 0.9595] over machines, per-machine range 0.928–0.978 (single seed)". (b) 결론 `:575` "achieves 100% detection of safety-critical
  Severe degradation states" → "detects every Severe window of the four validation machines (66/66) and 506 of 511 Severe windows across the 32
  training machines under device-grouped cross-validation (99.0 %)". (c) §6.x 한계에 CE-046 문구 갱신: 장비 단위 CI 를 측정했으며 seed std 는 일반화
  불확실성의 하한이다. (d) `:442` Severe 문단에 (b) 의 수치 추가.
- claim_evidence: CE-046 갱신(장비 단위 CI 측정 완료), **CE-048** 신설(Severe 100 % 일반화 한정).

## 6. 하지 않은 것

- 논문 본문 수정(O-113), 다른 seed 의 K-fold, 32대 학습 모델의 장비별 채점(validation 4대 외 장비는 학습에 포함돼 불가), Severe 누락 5 window 검토,
  AGV/OHT 차이 원인 분석.

## 7. 다음 최소 실험과 통과 조건

| 후보 | 통과 조건 |
|---|---|
| K-fold seed 2개 추가 (8 run, ~80분) | 장비 순위·AGV/OHT 차이 부호 유지 여부 |
| Severe 누락 5 window 육안·라벨 검토 (학습 아님) | 라벨 경계 문제인지 판정 |
| O-113 결정 후 논문 반영 | — |
