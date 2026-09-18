# 연구노트 #19: EXP-20260907-003 2³ ablation, seed 42 동일 조건 재실행, bench 데이터 시작 (2026-09-18)

- **연구 질문 및 hypothesis ID:** `AIHUB-H1`(multi-scale diff 효과), `AIHUB-H2`(SupCon 의 Normal–Mild 오류 감소).
  실험 `EXP-20260907-003`(2³ factorial), `EXP-20260918-002`(seed 42 동일 조건 재실행).
  질문: (a) 논문 Table 8 의 "세 요소는 단독으로 무효·부정적이고 결합 시 시너지" 주장이 예산·seed·초기화를 통제하면
  유지되는가, (b) seed 가 초기 가중치를 넘어 학습 결과까지 재현하는가.
- **시작 전 알고 있던 사실:** Table 8 은 V2/V2a/V2b/V2c 20 epoch, V2+ 30 epoch, 단일 seed, 초기화 미통제(연구노트 #16 §3.3,
  CE-010 CONFOUNDED). 2요소 조합 3개는 실행된 적이 없다. G2e(#18)에서 seed 별 초기 가중치 hash 가 기록됐지만
  같은 seed 반복은 하지 않았다.
- **이번 가설과 반증 조건:** H-a: 요소별 주효과가 seed 간 spread 보다 크면 "단독 무효" 는 기각된다. H-b: 3요소
  interaction 이 seed 간 spread 이내면 "시너지" 주장은 근거를 잃는다. H-c: 동일 seed 재실행에서 초기 hash 는 같아야
  하며(다르면 F03 수정 실패), 최종 F1 이 다르면 학습 경로 비결정성이 있다.
- **데이터·annotation·split·코드·환경 버전:** AI Hub #71802 SSD 작업 사본(G2d PASS), 공급자 split(장비 32/4), 라벨
  plurality. 코드 EXP-003 `4fac5bc` **clean**(24 run 모두), 재실행 `3920ef1` clean(src 는 `be16342` 와 동일).
  환경 `factory_training`(torch 2.6.0+cu124, cuDNN 90100), `keti-Precision-7920-Tower`, Quadro RTX 6000 ×2.
- **실행 ID와 설정/로그/결과 위치:** `EXP-20260907-003/<condition>/seed{42,123,456}` →
  `results/factorial_ablation/<condition>_seed<N>/results.json`(서버 원본 + Jetson 미러). 로그·lane 로그·분석
  `~/review_runs/20260918_exp003/`(서버·Jetson). 재실행 `results/v2plus_g2e_seed42_rep/`, 로그
  `~/review_runs/20260918_g2e_rep/`. bench 트라이얼 `dataset/_smoke/normal_20260918T*` (§7).
- **관측 결과 / 해석 / 결정:** §2~§5.
- **논문 후보에 미치는 영향:** PC0 Table 8·§7.3 시너지 서술·H1/H2 판정. **본문은 이번에 변경하지 않았다**(§6 정정안, O-111).
- **다음 최소 실험과 통과 조건:** §8.

---

## 1. 설계와 실행 조건

```
요소       multiscale  lags [1,5,10] (on) / [1] (off)
           se          SEBlock(hidden 128, r=8) on sensor encoder output / Identity
           supcon      (1-0.3)·CE + 0.3·SupCon(τ=0.07) / CE only
조건       2³ = 8, seed 42/123/456 → 24 run
공통       30 epoch · AdamW(lr 1e-3, wd 0.01) · ReduceLROnPlateau(min, 0.5, patience 3) on val_loss
           grad clip 1.0 · batch 16 · FP32 학습·평가 · best val macro-F1 checkpoint · early stopping 없음
           class-weighted CE + weighted sampler (기존과 동일, F14b 이중 보정 그대로)
           classifier 앞 dropout 0.1 을 **모든 조건에** 적용 (legacy V2/V2a/V2b 는 없었음 — §4 주의)
러너       src/train_factorial_ablation.py  (커밋 78db0c9) · 큐 scripts/run_exp003_queue.sh (4fac5bc)
검증       --self-test PASS 4/4: ms0_se0 ≡ LSTMWithTemporalDiff(V2), ms1_se0 ≡ V2MultiScaleOnly(V2a),
           ms0_se1 ≡ V2SEOnly(V2b), ms1_se1 ≡ V2Plus — state_dict key·shape 동일, 같은 seed 에서 초기 가중치
           bit 동일, legacy 가중치 로드 후 logit 차 0.0. 파라미터 2,837,508 / 2,845,700 / 2,841,748 / 2,849,940
실행       4 lane (GPU 당 2) · 07:49~12:07 UTC (16:49~21:07 KST) · run 당 28~53분(mean 39분, GPU 공유 때문)
           24 run 모두 exit 0 · git_dirty False · 초기 가중치 hash 는 같은 구조·seed 의 sc0/sc1 쌍에서 동일 (12/12)
           ms1_se1 seed 42 의 초기 hash = G2e·재실행 seed 42 (`58f43109…`) → 팩토리얼 V2+ 조건 = 기존 V2+ 초기화
```

## 2. 결과 — 조건별 (mean ± std, n = 3 seed)

| 조건 | 설명 | F1 mean | std | NM mean | best ep | legacy Table 8 (단일 seed) |
|---|---|---:|---:|---:|---:|---|
| ms0_se0_sc0 | **V2** (전부 off) | 0.939404 | 0.001330 | 50.0 | 27.7 | V2 0.9430 / NM 45 (20 ep) |
| ms1_se0_sc0 | +multiscale | 0.949808 | 0.001751 | 35.3 | 29.0 | V2a 0.9432 / 45 (20 ep) |
| ms0_se1_sc0 | +SE | 0.940165 | 0.000948 | 50.3 | 28.3 | V2b 0.9319 / 53 (20 ep) |
| ms0_se0_sc1 | +SupCon | 0.941242 | 0.003538 | 45.3 | 28.3 | V2c 0.9422 / 46 (20 ep) |
| ms1_se1_sc0 | multiscale+SE | 0.947690 | 0.003290 | 41.3 | 26.7 | (없음) |
| ms1_se0_sc1 | multiscale+SupCon | 0.954697 | 0.000827 | 30.0 | 26.7 | (없음) |
| ms0_se1_sc1 | SE+SupCon | 0.941481 | 0.001317 | 45.0 | 28.0 | (없음) |
| ms1_se1_sc1 | **V2+** (전부 on) | 0.956327 | 0.001268 | 30.7 | 28.3 | V2+ 0.9550 / 24 (30 ep) |

seed 별 24 값, LR 감소 epoch, 20 epoch 시점 최고값, 초기 hash 는 `compare_exp003.txt` §1.

## 3. 요소 효과 (coded ±1 contrast, F1 기준; ± 는 같은 contrast 의 seed 간 std)

| 효과 | 평균 변화 | seed std | seed 별 | 판정 |
|---|---:|---:|---|---|
| multiscale | **+0.011558** | 0.000799 | +0.0108 / +0.0115 / +0.0124 | std 의 14배. 3 seed 모두 양 |
| supcon | **+0.004170** | 0.000637 | +0.0037 / +0.0049 / +0.0039 | std 의 6.5배. 3 seed 모두 양 |
| se | +0.000128 | 0.001137 | −0.0009 / −0.0000 / +0.0013 | 검출 불가 |
| multiscale×supcon | +0.002593 | 0.002012 | +0.0008 / +0.0048 / +0.0022 | 3 seed 모두 양이나 std 와 같은 크기 |
| multiscale×se | −0.000372 | 0.002249 | +0.0022 / −0.0016 / −0.0017 | 없음 |
| se×supcon | +0.000807 | 0.001495 | −0.0009 / +0.0015 / +0.0018 | 없음 |
| 3-way | +0.001068 | 0.000539 | +0.0006 / +0.0009 / +0.0017 | 3 seed 모두 양, 작음 |

```
V2 → V2+                    +0.016924   (Table 8: +0.011992, 20 vs 30 epoch 혼재)
단독 이득 합 (ms+se+sc)      +0.013003
결합 − 단독합 (interaction)  +0.003920   ← 전체 이득의 23 %. 나머지 77 % 는 가산적
V2 → +multiscale 만          +0.010404   ← 전체 이득의 61 %
multiscale+SupCon → V2+      +0.001630   (seed 별 +0.0010 / +0.0008 / +0.0031; SE 의 추가 기여)
```

Normal↔Mild 오류 (3 seed 평균):

```
V2 50.0 → +SupCon 45.3 (−9 %)      V2 50.0 → +multiscale 35.3 (−29 %)
+multiscale 35.3 → +SupCon 30.0 (−15 %)     V2 50.0 → V2+ 30.7 (−39 %; seed 별 43→33, 54→30, 53→29)
```

## 4. 해석 — 관찰과 추론을 분리

**관찰.**
- multiscale 이 지배 요소다. 단독으로 V2→V2+ 이득의 61 % 를 내고, NM 오류를 29 % 줄인다. 3 seed 에서 부호·크기가 일정하다.
- SupCon 은 두 번째 요소다(+0.0042). multiscale 위에서 더 크다(+0.0048/+0.0072/+0.0027 vs 단독 +0.0044/−0.0005/+0.0016).
- SE 는 F1 주효과가 seed 잡음 이내다. 다만 multiscale+SupCon 위에 얹으면 3 seed 모두 +0.0008~+0.0031 이고, 이 조건에서
  best epoch 가 늦어지지 않는다.
- "단독으로 무효·부정적" 은 **기각**된다(H-a). 논문 V2b 의 −1.11 % 는 단일 seed·20 epoch 조건의 산물이었고, 30 epoch·3 seed
  에서는 V2 와 같다(+0.0008 ± 0.0011).
- "결합 시너지" 는 **크게 축소**된다(H-b). interaction 총량은 +0.0039 로 전체의 23 % 이고, 그중 multiscale×SupCon 이 대부분이나
  seed std 와 같은 크기다. 3-way 는 3 seed 모두 양이지만 +0.001 수준이다. n=3 으로 "있다/없다" 를 판정하지 않는다.
- multiscale+SupCon 두 요소(0.954697 ± 0.000827)가 V2+ (0.956327 ± 0.001268)와 0.0016 차이다.
- 재실행(§5)에서 seed 는 초기 가중치를 bit 단위로 재현했지만 최종 F1 은 달랐다(H-c 후반 성립).

**추론(미검증).**
- multiscale 이 큰 이유는 입력 차원이 16→32 로 늘어 LSTM 이 5·10 tick 변화율을 직접 받기 때문일 수 있다. 이는 lag 민감도
  (Table 13) 와 함께 볼 문제이며 이 실험은 lag 집합을 비교하지 않았다.
- SE 의 작은 추가 기여가 실제인지(3/3 양)는 seed 를 늘려야 알 수 있다. 파라미터 +4,240 의 비용은 무시할 수준이다.
- SupCon 이 multiscale 위에서 더 큰 것은 embedding 이 더 구조화된 뒤 contrastive 항이 효과를 내는 것과 부합하지만, 이 데이터로
  기제를 말할 수는 없다.

**경쟁 설명과 한계.**
- **legacy 와의 직접 비교는 불가.** 이번 V2/V2a/V2b 는 classifier 앞 dropout 0.1 을 적용했고 legacy 는 하지 않았다(의도적으로
  통일). 따라서 legacy V2 0.9430 (20 ep) 이 이번 V2 mean 0.9394 (30 ep) 보다 높은 것을 "예산이 길어서 나빠졌다" 로 읽을 수 없다.
  팩토리얼 내부 비교만 유효하다.
- n=3 seed, validation 장비 4대(CE-046). 불확실성은 seed std 만 반영했고 장비 단위 불확실성은 EXP-006 이후.
- 비결정성(§5) 때문에 같은 seed 의 반복도 ±0.001 수준으로 흔들린다. 3-way(+0.0011)·SE 주효과(+0.0001)는 이 잡음과 같은 크기다.
- 예산은 모두 30 epoch 로 같지만 plateau scheduler 의 LR 감소 시점은 run 마다 다르다(`compare_exp003.txt`). "동일 schedule" 이
  아니라 "동일 scheduler 규칙" 이다.

## 5. EXP-20260918-002 — seed 42 동일 조건 재실행

```
명령   python -m src.train_v2plus --seed 42 --gpu 0 --results-dir results/v2plus_g2e_seed42_rep   (3920ef1 clean, src ≡ be16342)
시간   07:38:47 → 08:19:54 UTC (41분; 뒤 20분은 GPU 0 을 EXP-003 두 run 과 공유)
```

| 항목 | G2e seed 42 | 재실행 | |
|---|---|---|---|
| 초기 가중치 sha256 | `58f43109f4e5…` | `58f43109f4e5…` | **동일** |
| macro F1 | 0.954825 | 0.954040 | Δ −0.000785 |
| acc | 0.9490 | 0.9473 | |
| best epoch | 24 | 24 | 동일 |
| NM | 29 | 33 | |
| LR 감소 epoch | 13/18/22/26/30 | 10/14/18/22/26/30 | 경로 분기 |
| epoch 별 val F1 일치 | — | 30 중 0 | epoch 1 부터 다름 (0.8568 vs 0.8540) |

seed 는 초기화를 재현한다(F03 수정 확인). 학습 경로는 첫 epoch 부터 갈린다. 코드는 `torch.use_deterministic_algorithms` 나
`cudnn.deterministic` 을 켜지 않으므로 cuDNN LSTM/conv backward 의 비결정성이 원인으로 **추정**되며, 확인하지 않았다. Δ F1
−0.0008 은 G2e seed 간 std 0.0018 이내다. **재현성 주장은 "초기 가중치 재현 + 결과 분포 재현" 까지로 한정**한다.

## 6. 결정과 정정안

- **D-021.** Table 8 의 "단독 무효·결합 시너지" 해석은 철회 대상이다. ablation 근거는 EXP-003 으로 교체한다. 논문 본문(Table 8,
  §7.3 시너지 문단, Table 15 H1/H2)의 **교체 문안은 아래에 준비하되 적용은 사용자 판단**(O-111). D-020 과 같은 논리다.
- Table 8 교체안(요지): 8 조건 × 3 seed mean ± std 표(§2) + 효과표(§3). 각주: 30 epoch 동일 예산, 동일 scheduler 규칙, seed 통제
  초기화, classifier dropout 통일, n=3, 장비 4대 validation.
- §7.3 교체안(요지): "multi-scale difference 가 이득의 약 60 % 를 단독으로 설명하고 SupCon 이 그 위에서 추가 이득을 낸다. SE 의
  주효과는 seed 잡음 이내이며 세 요소 결합에서만 작은 추가 이득(+0.0016)이 관측된다. 요소 간 interaction 은 전체 이득의 약 1/4 이고
  n=3 에서 판정하지 않는다." 'cascading mechanism' 서술은 삭제.
- H1: "Supported" 유지 가능하나 근거를 EXP-003 주효과(+0.0116 ± 0.0008)로 교체. H2: SupCon 단독 NM 감소 −9 %, 결합 −39 %.
  "SupCon 이 NM 을 ≥40 % 줄인다" 는 **NOT SUPPORTED** 로 확정(감소의 대부분은 multiscale). CE-010/CE-011 갱신.

## 7. bench 데이터 시작 (Jetson, 현장 데이터 아님)

프로토콜 §15 에 규약을 적었다. 사무실 책상 데이터는 이상 탐지 학습에 쓰지 않고 장비 검증용으로만 쓴다.

```
normal_20260918T082429Z  bench_soak 8h 시도 #1 — tick ~200 에서 xcal 구도 전환을 위해 SIGINT (aborted, 부분 보존, timing_report 없음)
normal_20260918T082944Z  bench_thermal_xcal 1/3  360 tick completed, NTC 360/360
normal_20260918T083557Z  bench_thermal_xcal 2/3  completed
normal_20260918T084207Z  bench_thermal_xcal 3/3  completed
normal_20260918T120817Z  bench_soak 8h 시도 #2 — 21:08 KST 시작, xcal 구도 유지, 05:08 KST 종료 예정
```

xcal 관측(`~/review_runs/20260918_bench/xcal_analysis.txt`): NTC 31.4~32.9 °C 로 18분 동안 1.2 °C 만 변함. 프레임 최고온 영역
(y≈25, x≈49·70 두 열점) 41~43 °C, 프레임 max 45~48 °C, 프레임 중앙 36~39 °C. 최고온 ROI − NTC = +9.0~+9.9 °C 로 일정, 상관 r ≈ 0.
**해석:** 온도 변화가 없어 기울기를 추정할 수 없고, NTC 가 열점이 아닌 저온부에 있거나 접촉이 약하다. **보정값으로 쓰지 않는다.**
유효한 xcal 조건: (1) 프레임 내 NTC 위치 표시(트라이얼 초반 손가락으로 지시 또는 위치 진술), (2) 표면 온도가 오르내리는 구간
(부하 수동 on/off 직후 각 360 s). 가열·부하 제어 코드는 만들지 않는다.

## 8. 하지 않은 것 / 다음 최소 실험

- 논문 본문 수정(O-111), ONNX/ModelAdapter, 현장 학습, cuDNN 결정론 모드 실험, lag 집합 비교.
- 다음 후보: (1) `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic` 로 seed 42 재실행 1회 — 통과 조건: epoch 별
  val F1 30/30 일치(가능 여부 자체가 결과). (2) SE 추가 기여(+0.0016) 확인용 seed 3개 추가(ms1_se0_sc1 vs ms1_se1_sc1, 6 run,
  약 1시간). (3) `EXP-20260907-006` 장비 K-fold. (4) bench_thermal_xcal 유효 조건 재실행, bench_soak #2 timing_report 판독.
