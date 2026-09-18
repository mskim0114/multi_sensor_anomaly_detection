# 연구노트 #18: G2e — V2+ 3-seed 재실행 (2026-09-18)

- **연구 질문 및 hypothesis ID:** `AIHUB-H1` 계열. 실험 `EXP-20260907-002`. 질문: F03(seed 순서)·F04(TimesNet
  주기식)·F14a(eval autocast) 수정과 전용 서버 환경에서 V2+ 를 다시 돌리면 (a) 재현 가능한 실행 기록이 남는가,
  (b) 결과 분포가 기존 3-seed 와 어떤 관계인가.
- **시작 전 알고 있던 사실:** 기존 3-seed F1 0.955000 / 0.9560237 / 0.9559546 (mean 0.955659, std 0.000467),
  NM 24 / 29 / 33. 이 실행들은 seed 가 모델 생성 뒤에 설정되어 초기 가중치가 통제되지 않았고(F03), git commit·
  환경 기록이 없다(연구노트 #16 §3, §10).
- **이번 가설과 반증 조건:** 가설 없음(재현·기록 확보가 목적). 반증 조건: 실행 실패, provenance 누락, 결과가 기존
  분포와 극단적으로 다름(예: mean 차이가 기존 std 의 수 배).
- **데이터·annotation·split·코드·환경 버전:** AI Hub #71802 SSD 작업 사본(G2d PASS, 연구노트 #17 §7),
  공급자 Training/Validation split(장비 32/4), 라벨 plurality. 코드 `be16342` **clean**. 환경
  `factory_training`(torch 2.6.0+cu124, cuDNN 90100), `keti-Precision-7920-Tower`, Quadro RTX 6000.
- **실행 ID와 설정/로그/결과 위치:** `EXP-20260907-002/seed{42,123,456}`.
  결과 `results/v2plus_g2e_seed{N}/` (서버 원본 + Jetson 미러 results.json), 로그·비교 스크립트
  `~/review_runs/20260918_g2e/` (서버·Jetson 양쪽).
- **관측 결과 / 해석 / 결정:** 아래.
- **논문 후보에 미치는 영향:** PC0. 논문 수치는 대체하지 않는다(D-020).
- **다음 최소 실험과 통과 조건:** §5.

---

## 1. 실행 조건

```
공통 인자   --epochs 30 --lr 1e-3 --batch-size 16 --hidden-dim 128 --num-layers 3 --lags 1,5,10
            --supcon-weight 0.3 --temperature 0.07     (--amp / --fast 미사용 = 학습·평가 모두 FP32)
seed 42     GPU0  12:16:56 -> 12:37:15 KST  (20분 19초)
seed 123    GPU1  12:16:56 -> 12:35:46 KST  (18분 50초)   42 와 동시
seed 456    GPU0  12:37:22 -> 12:57:43 KST  (20분 21초)   42 종료 후 순차
출력         results/v2plus_g2e_seed{N}/   -- 논문 run 사본 results/v2plus 는 건드리지 않음 (--results-dir 신설, be16342)
```

`results.json` 에 처음으로 provenance 가 기록됐다: `git_commit be16342`, `git_dirty False`, hostname, python 3.12.13,
torch 2.6.0+cu124, cudnn 90100, GPU 이름, 실행 명령, 시작·종료 UTC, 그리고 **초기 가중치 sha256**.

| seed | 초기 가중치 sha256 (앞 12자) |
|---|---|
| 42 | `58f43109f4e5` |
| 123 | `d1d818434d15` |
| 456 | `b9b1fb9f5d19` |

세 값이 서로 다르고 seed 로 결정되므로 F03 수정이 의도대로 작동한다. 같은 seed 로 다시 돌리면 같은 hash 가
나와야 한다 — **이번엔 그 재실행까지는 하지 않았다**(다음 단계 후보).

## 2. 결과

| seed | 집합 | macro F1 | acc | best epoch | NM (0↔1) | F1@≤20 | LR 감소 epoch |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | legacy | 0.955000 | 0.9516 | 27 | 24 | — | 13 / 25 / 29 |
| 42 | **g2e** | 0.954825 | 0.9490 | 24 | 29 | 0.946804 | 13 / 18 / 22 / 26 / 30 |
| 123 | legacy | 0.956024 | 0.9499 | 30 | 29 | — | (기록 없음) |
| 123 | **g2e** | 0.952084 | 0.9447 | 29 | 35 | 0.945957 | 8 / 16 / 23 / 27 |
| 456 | legacy | 0.955955 | 0.9490 | 30 | 33 | — | (기록 없음) |
| 456 | **g2e** | 0.956400 | 0.9499 | 26 | 31 | 0.944092 | 8 / 16 / 20 |

```
legacy  F1 mean 0.955659  std 0.000467  [0.955000, 0.956024]    NM mean 28.67  std 3.68
g2e     F1 mean 0.954436  std 0.001783  [0.952084, 0.956400]    NM mean 31.67  std 2.49
Δ mean  F1 −0.001223      NM +3.00                               (n = 3 vs 3)
```

## 3. 해석 — 관찰과 추론을 분리

**관찰.**
- 세 run 모두 exit 0, 오류 0, 30 epoch 완주, provenance 완비. 각 약 20분.
- g2e 의 best epoch 는 24 / 29 / 26 으로 **셋 다 30 미만**이다. legacy 는 123·456 이 best@30(미수렴)이었다.
- 20 epoch 시점 최고값은 0.9441~0.9468 이고 최종 best 는 그보다 +0.006~+0.012 높다.
- plateau scheduler 의 LR 감소 시점이 seed 마다 다르다(첫 감소 8 또는 13 epoch). 연구노트 #16 §3.3 2차 정정에서
  "scheduler 발동 시점은 예산이 아니라 학습 경로에 달려 있다" 고 쓴 것이 실측으로 확인됐다.
- g2e 의 seed 간 spread(std 0.0018)가 legacy(0.0005)보다 약 4배 넓다.

**추론(미검증).**
- Δ mean F1 −0.0012 는 g2e std 의 0.7배, legacy std 의 2.6배다. n=3 이라 유의성을 말할 수 없다. 두 집합은
  초기화 통제 방식·seed 전달 경로가 다르므로 "같은 실험의 반복"이 아니다.
- legacy 의 좁은 spread 는 우연일 수도, `--all` 순차 실행에서 초기화가 이전 실험 RNG 상태에 연쇄되어
  실제로는 덜 독립적이었기 때문일 수도 있다(F03). 어느 쪽인지 이 데이터로 판정할 수 없다.
- 20 epoch 대비 +0.006~+0.012 의 후반 이득은 Table 8 예산 교란 논의(#16 §3.3)에 참고 수치가 되지만,
  V2 를 30 epoch 돌린 대조가 없으므로 **여전히 분해 불가**다.

**경쟁 설명 및 반례.** eval 정밀도 차이(F14a)는 없다 — legacy 3회(2026-04)는 autocast 도입 전이라 FP32 평가였고
g2e 도 FP32 다. 데이터 차이도 없다 — 같은 released subset, G2d 로 파일 단위 무결성 확인. 남는 차이는 초기화
경로와 torch/cuDNN 버전(legacy 의 버전은 기록이 없어 불명)이다.

## 4. 결정

- **논문 수치(0.9557 ± 0.0006)를 g2e 수치로 대체하지 않는다**(D-020). 두 집합을 나란히 보고한다.
- g2e 는 "재현 가능한 기록을 가진 V2+ 실행" 으로서 이후 비교(2³ ablation `EXP-003`, thermal 인코더 `EXP-005`,
  K-fold `EXP-006`)의 **기준 실행**이 된다. legacy 는 기록이 없어 기준으로 쓸 수 없다.
- claim_evidence CE-043(seed 재현성)의 근거를 갱신한다. 본문 `:360` 은 이미 "three independent initialisations"
  로 정정돼 있어 추가 변경 없음.

## 5. 다음 최소 실험과 통과 조건

| 후보 | 통과 조건 |
|---|---|
| 동일 seed 재실행 1회 (bit-level 재현 확인) | 초기 가중치 sha256 동일 + 최종 F1 동일(또는 cuDNN 비결정성 범위 내) |
| `EXP-20260907-003` 2³ ablation, 동일 예산·scheduler·3 seed | interaction 추정 가능 여부 판정 |
| `EXP-20260907-006` 장비 그룹 K-fold | 장비 단위 신뢰구간 |

## 6. 하지 않은 것

- 논문 본문 수정, ONNX export, TensorRT, ModelAdapter, 현장 데이터 학습.
- 같은 seed 반복 실행(재현성의 직접 증명).
- legacy 실행 환경(torch 버전 등) 복원 — 기록이 없다.
