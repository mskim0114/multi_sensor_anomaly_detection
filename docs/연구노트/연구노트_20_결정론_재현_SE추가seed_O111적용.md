# 연구노트 #20: 결정론 모드 bit-level 재현, SE 추가 seed, O-111 논문 적용, bench_soak #2 실패 (2026-09-21)

- **연구 질문 및 hypothesis ID:** `AIHUB-H1` 계열. 실험 `EXP-20260921-001`(결정론 모드 seed 42 × 2), `EXP-20260921-002`(SE 추가 seed 6 run).
  질문: (a) 결정론 옵션을 켜면 같은 seed 가 학습 결과까지 bit 단위로 재현하는가(연구노트 #19 §5 N17 의 후속),
  (b) EXP-003 에서 3 seed 모두 양이던 SE 의 추가 기여(+0.0016)가 seed 를 6개로 늘려도 부호를 유지하는가.
- **시작 전 알고 있던 사실:** seed 는 초기 가중치를 재현하지만 학습 경로는 epoch 1 부터 갈렸다(N17). 코드는 cuDNN 결정론 옵션을
  켜지 않았다. SE 추가 기여는 3 seed +0.0010/+0.0008/+0.0031 로 잡음(±0.001)과 같은 크기였다(#19 §4).
- **이번 가설과 반증 조건:** (a) 두 run 의 epoch 별 metric 30/30 일치 + 최종 가중치 hash 동일이면 성립. 결정론 kernel 이 없는 op 가
  있으면 경고가 기록되고 불일치가 나야 한다. (b) 통과 조건은 사용자 승인 시 명시: 6 seed 에서 부호 유지.
- **데이터·annotation·split·코드·환경 버전:** AI Hub #71802 SSD 사본(G2d PASS), 공급자 split. 코드 `38dc5c3` **clean**
  (`--deterministic` 추가, 큐 SEEDS/CONDS override). 환경 `factory_training`, `keti-Precision-7920-Tower`, Quadro RTX 6000 ×2.
- **실행 ID와 설정/로그/결과 위치:** `results/v2plus_det_seed42_{a,b}/`, 로그 `~/review_runs/20260921_det/`;
  `results/factorial_ablation/ms1_se{0,1}_sc1_seed{7,2026,31415}/`, 로그·분석 `~/review_runs/20260921_se_seeds/`(서버·Jetson).
- **관측 결과 / 해석 / 결정:** §1~§4.
- **논문 후보에 미치는 영향:** PC0 — O-111 을 적용해 Table 8·§6.2.2·§7.3·Table 15 H1/H2·초록·기여·결론 문장을 교체했다(§3).
- **다음 최소 실험과 통과 조건:** §6.

---

## 1. EXP-20260921-001 — 결정론 모드 seed 42 × 2 (GPU 0 / GPU 1 동시)

```
명령   python -m src.train_v2plus --seed 42 --gpu {0|1} --deterministic --results-dir results/v2plus_det_seed42_{a|b}
설정   cudnn.deterministic=True · cudnn.benchmark=False · use_deterministic_algorithms(True, warn_only=True) · CUBLAS_WORKSPACE_CONFIG=:4096:8
       비결정 kernel 경고는 results.json["deterministic"]["non_deterministic_ops_warned"] 에 수집 → 두 run 모두 [] (경고 0건)
시간   09:49 → 10:33 KST (a) / 09:50 → 10:31 (b); SE run 과 GPU 공유로 epoch 87~93 s
```

| 항목 | run a (GPU 0) | run b (GPU 1) | |
|---|---|---|---|
| 초기 가중치 sha256 | `58f43109f4e5…` | `58f43109f4e5…` | 동일 (= G2e·재실행 seed 42) |
| history 7종 × 30 epoch (train/val loss·acc·F1, lr) | | | **210/210 일치** |
| macro F1 / acc | 0.954888 / 0.948142 | 0.954888 / 0.948142 | 동일 |
| best epoch / NM | 30 / 32 | 30 / 32 | 동일 |
| 혼동행렬 | | | 동일 |
| best_model.pt `model_state_dict` sha256 | `92199a127d43…` | `92199a127d43…` | **동일** |
| optimizer state 텐서 | | | 전부 `torch.equal` |
| LR 감소 epoch | 15/20/24/28 | 15/20/24/28 | 동일 |

**관찰.** 결정론 옵션만으로 서로 다른 GPU 에서 epoch 단위·가중치 단위 bit-level 재현이 성립한다. 비결정 kernel 경고가 없었으므로
V2+ 의 op(LSTM, Conv2d, MaxPool2d backward, Linear, SupCon 의 matmul/exp)은 이 torch 2.6.0+cu124 / cuDNN 90100 조합에서 모두
결정론 구현을 가진다. 비결정 모드 결과(G2e 0.954825 best@24, 재실행 0.954040 best@24)와 결정론 결과(0.954888 best@30)의 차이는
±0.001 이내다. **추론(미검증):** 결정론 모드의 속도 비용은 이번에 측정하지 않았다(GPU 를 다른 run 과 공유했기 때문).

**해석.** N17 은 "seed 는 초기화만 재현" 에서 "**기본 모드에서는** 초기화만 재현, `--deterministic` 에서는 결과까지 재현" 으로 갱신한다.
재현성 주장이 필요한 실행(논문 재현, 비교 기준 실행)은 `--deterministic` 으로 돌리고 results.json 의 `deterministic` 블록을 인용한다(D-024).
이전 실행(G2e, EXP-003)은 비결정 모드였으므로 "같은 seed 로 같은 수치" 를 주장하지 않는다 — 분포 재현까지만.

## 2. EXP-20260921-002 — SE 추가 기여, seed 3개 추가 (7, 2026, 31415)

조건 `ms1_se0_sc1`(multiscale+SupCon) vs `ms1_se1_sc1`(V2+) 만 6 run. 팩토리얼 러너 동일(`38dc5c3`; `train_factorial_ablation.py` 는
`4fac5bc` 이후 변경 없음). 비결정 모드(EXP-003 과 같은 조건으로 맞추기 위해).

| seed | ms1_se0_sc1 | ms1_se1_sc1 | ΔF1 (SE 추가) | NM se0→se1 | 출처 |
|---:|---:|---:|---:|---|---|
| 42 | 0.954021 | 0.955027 | +0.001006 | 29→33 | EXP-003 |
| 123 | 0.955620 | 0.956395 | +0.000775 | 31→30 | EXP-003 |
| 456 | 0.954450 | 0.957560 | +0.003110 | 30→29 | EXP-003 |
| 7 | 0.954400 | 0.957544 | +0.003144 | 30→28 | 신규 |
| 2026 | 0.954098 | 0.957600 | +0.003502 | 36→26 | 신규 |
| 31415 | 0.957621 | 0.954906 | **−0.002715** | 29→31 | 신규 |

```
n=6   mean +0.001470   std 0.002359   양 5/6   [−0.002715, +0.003502]
6-seed mean   ms1_se0_sc1 0.955035 (std 0.001392)   ms1_se1_sc1 0.956505 (std 0.001276)
```

**관찰.** 통과 조건(6 seed 부호 유지)은 **실패**했다. seed 31415 에서 SE 없는 쪽이 0.9576 으로 6 run 중 최고였다.
**해석.** SE 의 추가 기여는 "3 seed 모두 양" 이 우연일 수 있음을 보여준다. 평균 +0.0015 는 seed std 0.0024 보다 작다. **SE 기여는 확립되지
않았다.** 논문 §7.3 은 "6 seed 에서 +0.15 ± 0.24 pp, 5/6 양, 기여를 확립하지 못함" 으로 고쳤다. **경쟁 설명:** 비결정 모드 잡음(±0.001)이
개별 Δ 에 섞여 있다. 결정론 모드로 6 seed 를 다시 돌려도 seed 간 spread 자체는 남으므로 결론이 바뀔 가능성은 낮지만 확인하지 않았다.

## 3. O-111 적용 — 논문 본문 교체 (D-023)

`paper_draft.md` 에서 교체한 곳: 초록의 ablation 문장, 기여 bullet(SupCon), §6.2.2 도입문, **Table 8 전체**(8 조건 × 3 seed mean ± std,
ΔF1 pp, NM), Table 8 각주(주효과·interaction·V2+ 헤드라인 수치 미교체 명시), 한계 문단(PENDING 마커 제거), Figure 4 캡션, "However, when
combined…" 문단 → NM 경로 서술, Table 15 H1(Supported, 근거 교체)·H2(**Not supported as stated**), H1/H2 후속 문단, §7.3 제목·본문
("Synergy" → "Contribution of the Combined Components", cascading mechanism 삭제, SE 6-seed 결과·결정론 재현 note 추가), 결론 두 문장.
`+1.20%`·`synergy` 주장·`EXP-20260907-003 PENDING` 마커는 0건 남았다. **V2+ 헤드라인 0.9557 ± 0.0006 은 그대로**(D-020).
diff 는 `git show <이번 커밋> -- docs/논문/paper_draft.md`.

## 4. bench_soak #2 — 27분 만에 FAILED (Jetson, 09-18 밤)

```
trial   dataset/_smoke/normal_20260918T120817Z   planned 28800 s → 1627 tick 에서 종료 (status failed)
원인    first_error: scd30.poll_and_read → I2cTransceiveError "I2C transceive failed: [Errno 121] Remote I/O error"
        12:35:29.945 UTC, 직전 tick 1626 의 SCD30 은 ok (fresh, 735.8 ppm). 단발 NACK 로 추정 (미확인)
동작    collector 는 첫 오류에서 멈춘다(stopped_on_error=True, CLAUDE.md §2 "첫 오류를 보존해 보고" 설계). 부분 파일 보존.
1627 tick 통계   missed 0 · period p95 1000.4 ms · |jitter| p95 0.56 ms / max 0.85 ms · tick work mean 732 ms / max 823 ms
        FLIR age p95 119 ms / max 1798 ms (stale tick 13 → 30 s window 54 중 9 invalid, 정책대로)
        SPS30 fresh 1543/1627 · SCD30 fresh 774 (2.1 s 주기) · CT1 431 샘플/burst 861.7 SPS clipping 0 · writer drop 0
```

**관찰.** 파이프라인 자체(tick·writer·FLIR·CT1)는 27분 동안 결함이 없었다. 종료는 context 센서(SCD30, 모델 입력 아님) 의 I2C NACK 1회다.
**해석과 미결(O-112).** 8시간 soak 는 이 정책으로는 완주가 어렵다. 선택지는 (a) 정책 유지 — 오류의 빈도·조건을 먼저 세어야 하므로 soak
를 여러 번 돌려 first_error 를 수집, (b) context 센서 오류는 tick 을 `error` 로 기록하고 계속 진행(REQUIRED 센서만 fatal) — 수집기 정책
변경이므로 사용자 결정. 어느 쪽도 자동 retry/workaround 는 아니다. **bench_soak 는 오늘 재시작하지 않았다**(정책 결정 전).

## 5. 하지 않은 것

- 결정론 모드 속도 측정, 결정론 모드로 EXP-003 재실행, SE 6 seed 결정론 재실행.
- bench_soak 재시작, 수집기 오류 정책 변경, bench_thermal_xcal 유효 조건 실행(사용자 물리 조작 필요).
- ONNX/ModelAdapter, 현장 학습.

## 6. 다음 최소 실험과 통과 조건

| 후보 | 통과 조건 |
|---|---|
| O-112 결정 후 bench_soak 8h 재실행 | 완주 + first_error 기록(있으면) |
| bench_thermal_xcal 유효 조건 (NTC 위치 표시 + 부하 on/off 직후 각 360 s) | NTC 변화 ≥ 5 °C 구간에서 ROI–NTC 기울기 추정 가능 |
| `EXP-20260907-006` 장비 K-fold | 장비 단위 신뢰구간 |
| 결정론 모드 속도 측정 1 run (단독 GPU) | epoch 시간 vs 비결정 40 s |
