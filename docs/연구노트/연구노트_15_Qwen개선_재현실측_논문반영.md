# 연구노트 #15: Qwen AI 개선 검증 · 재현 실측 · 논문 pending 반영

**작성일:** 2026-08-06
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 배경

이 세션의 목표는 세 가지:
1. **GitHub 저장소** (`mskim0114/multi_sensor_anomaly_detection`) 를 PC 측에 연결
2. 다른 컴퓨터 (Jetson 아님, 사용자가 별도 환경에서 Qwen AI Agent 사용) 에서 만들어진 **성능 최적화 커밋 2개 검토**
3. 연구노트 #14 에서 정리한 **논문 pending update #2 (11 지점) 반영**

---

## 2. GitHub 저장소 연결 · Race 상황

### 2.1 원격 저장소 초기 상태

원격에는 Jetson 측에서 push 한 초기 4개 커밋만 있음:

```
dfff2fa docs: document GitHub SSH auth setup
0834b1b docs: define first-phase sensor strategy
2c10c42 docs: add Jetson setup guide
2564d0e Initial commit
```

Jetson repo 에 이미 있는 것:
- 소스 코드 전체 (src/, configs/)
- Jetson 측에서 새로 만든 문서 (`docs/센서_보유목록.md`, `docs/SPS30_Jetson_I2C_연결.md`, `docs/NTC10K_ADS1115_Jetson_연결.md`, `docs/Jetson_Orin_Nano_초기세팅_가이드.md`, `docs/이상상태_시나리오_및_데이터수집전략.md`)
- Jetson 측에서 편집한 스크립트 (`jetson_deploy/scripts/01~05` 편집본 + 신규 `06_capture_purethermal.py`, `07_read_sps30.py`, `08_read_ntc_ads1115.py`)

PC 에만 있는 것:
- 논문 draft (paper_draft.md)
- 연구노트 (11편)
- 참고문서 (3편)
- 실험 결과 보고서

### 2.2 SSH 세팅 (신규)

이 PC 에는 SSH 키가 없어 새로 생성:
- 알고리즘: ed25519
- 라벨: `keti-Precision-7920-Tower`
- 사용자가 GitHub Settings 에 추가
- `ssh -T git@github.com` → "Hi mskim0114!" 확인

Git config:
- user.name: `mskim0114`
- user.email: `myeongseopkim0114@gmail.com` (GitHub 계정 verified email 로 맞춤)

### 2.3 실수 · 복구 (교훈)

첫 시도에서 `git checkout origin/main -- .` 을 실행하면서 **세션 편집본 4개 파일이 origin/main 버전으로 덮어씌워지는 사고** 발생. 사라진 것:
- `docs/논문/paper_draft.md` (603 → 570 lines)
- `docs/참고문서/실제 센서 및 엣지 보드 구매 사양서.md` (약 349 → 208 lines)
- `docs/참고문서/AI Hub 데이터셋 심층 분석 보고서.md` (약 256 → 40 lines)
- `docs/연구노트/연구노트_02_원본데이터셋_분석.md` (283 → 132 lines)

**원인**: 이 파일들이 origin/main 에 tracked 상태였는데 "PC-only untracked" 로 오판. `git checkout origin/main -- .` 은 tracked 파일을 강제 덮어씀. 편집본은 `git add/commit` 안 되어 있어서 reflog 로도 복구 불가.

**복구**: 세션 대화의 Edit/Write 이력 + `project_paper_pending_updates.md` 메모리를 근거로 4개 파일 전부 수동 재복원 (약 20분).

**재발 방지 규칙 (자체 정한 것)**:
- 파일 편집 전 `git ls-files <path>` 로 tracked 여부 반드시 확인
- 편집한 파일이 있는 상태에서 `git checkout` / `git reset --hard` 계열 실행 전 stash/commit 으로 편집 보존
- `git status` 로 편집 목록 사용자에게 표시 후 확인 받고 실행

---

## 3. Qwen AI 개선 커밋 분석

원격에 새로 push 된 커밋 2개 (다른 컴퓨터에서 Qwen 이 저자):

```
6542bd7 docs: 성능 최적화 상세 문서 추가                                (Qwen)
344ffdf perf: 데이터 로딩 병렬화 및 Mixed Precision Training           (Qwen)
```

### 3.1 변경 5건 심층 판정

| # | 파일 | 변경 | 개선 유효성 | 안전성 |
|:-:|:---|:---|:---:|:---:|
| 1 | `session_index.py` | Label 병렬 로딩 (ThreadPool 8) | ✅ | ⚠️ fallback=0 위험 |
| 2 | `dataset.py` | LRU cache (maxsize=128) | 🔴 **거의 무효** | ✅ |
| 3 | `config.py` | num_workers 4→8, prefetch 2→4 | ✅ | ✅ |
| 4 | `train_v2plus.py` | AMP + TF32 + cuDNN benchmark | ✅ 속도 | 🔴 **재현성 위협** |
| 5 | `.gitignore` | 확장 | — | ✅ |

### 3.2 심각한 이슈 3가지

**Issue 1 — LRU cache maxsize=128 은 무효**
- 데이터셋 파일 수 ~90,000 vs cache 128 슬롯 → hit rate 극히 낮음
- `num_workers=8` 이면 각 worker 프로세스마다 독립 캐시 (lru_cache 는 프로세스 스코프)
- Qwen 문서의 "파일 읽기 90k→1" 주장은 근거 없음

**Issue 2 — TF32/cudnn.benchmark 를 import 시점에 강제 활성화 → 논문 재현성 위협**
- `cudnn.benchmark=True` = 비결정론적
- CLI 플래그 없이 항상 켜져서 논문의 3-seed 통계 (F1 std=0.0006) 흔들 우려

**Issue 3 — Label 로딩 실패 시 조용히 fallback=0**
- 파일 손상 시 경고만 남기고 label=0 (정상) 로 학습 → silent error
- 원본 코드는 exception → 학습 중단 → 문제 인지 가능

### 3.3 잘 한 것

- AMP CLI opt-in (`--amp` 플래그)
- `non_blocking=True` on `.to(device)`
- Gradient clipping 을 unscale 후에 (표준 pattern)
- num_workers 8 · prefetch 4 (서버 스펙 적절)
- 커밋 메시지 · 문서화 정성

---

## 4. 재현 실측 (Qwen 코드 그대로)

### 4.1 실행 조건

- 환경: RTX 6000 (GPU 1), Python 3.12, PyTorch 2.6.0+cu124
- 명령: `python -m src.train_v2plus --gpu 1` (Qwen 원본, TF32 자동 켜짐, `--amp` 없음)
- 기존 결과 백업: `results/v2plus_pre_qwen_backup_20260806_121834/`

### 4.2 결과

| 지표 | 논문 (seed=42) | 논문 3-seed 평균 | **재현 결과** | 판정 |
|:---|:---:|:---:|:---:|:---:|
| Val F1 (macro) | 0.9550 | 0.9557 ± 0.0006 | **0.9551** | ✅ 재현 성공 |
| Val Accuracy | 95.16% | 95.02 ± 0.13% | **94.81%** | ✅ |
| Severe recall | 100% | 100% | **100% (66/66)** | ✅ |
| Model params | 2,849,940 | 2,849,940 | **2,849,940** | ✅ |
| Normal↔Mild 오류 | 24 | 28.7 ± 4.5 | **33** | ✅ std 범위 근접 |
| 총 소요 | (미기록) | — | **18분** | Qwen 주장 45s epoch × 30 근접 |

**결론**: Qwen 변경이 정확도 재현에 지장 없음 확인.

---

## 5. 코드 수정 (옵션 B-full)

### 5.1 수정 4곳

**`src/data/dataset.py`** — LRU cache 롤백
```python
# 제거: @lru_cache(maxsize=128) on _load_single_sensor/_load_single_thermal
# 보존: __init__ 시점 self._all_labels = self.get_all_labels() (WeightedRandomSampler 에 유효)
# 문서 정정: .bin 파일은 실제로 .npy 형식 (magic bytes \x93NUMPY 확인, np.load(mmap_mode="r") 이 올바른 loader)
```

**`src/train_v2plus.py`** — TF32/benchmark opt-in + torch.amp API
```python
def enable_fast_math():  # 신규
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

parser.add_argument("--fast", action="store_true", ...)  # 신규 CLI
if args.fast: enable_fast_math()  # opt-in

# torch.cuda.amp.autocast() → torch.amp.autocast("cuda")   (2곳)
# torch.cuda.amp.GradScaler() → torch.amp.GradScaler("cuda")
```

**`src/data/session_index.py`** — fallback → raise
```python
if failures:
    summary = "; ".join(f"{bn}: {err}" for bn, err in failures[:5])
    raise RuntimeError(f"Failed to read {len(failures)} label file(s). First few: {summary}")
```

**`PERFORMANCE_OPTIMIZATION.md`** — 실측 반영 재작성
- 2024 → 2026 년도 정정
- LRU cache 롤백 명시
- `--fast` 플래그 안내
- 실측 벤치마크 (Qwen 이론적 추정치와 구분)
- 재현성 경고 (`--fast` 사용 금지 조건)
- 향후 개선 5개 후보

### 5.2 검증 실측 (수정 후, `--fast` 없이 fp32 순수)

| 지표 | Qwen 원본 (TF32 자동 켜짐) | **수정본 (fp32 순수, `--fast` off)** | 판정 |
|:---|:---:|:---:|:---:|
| Val F1 (macro) | 0.9551 | **0.9551** | 완전 동일 ✅ |
| Val Accuracy | 0.9481 | **0.9481** | 완전 동일 ✅ |
| Best epoch | 28 | 21 | seed 미고정으로 자연 변동 |
| Severe recall | 100% | 100% | ✅ |
| NM 오류 | 33 | 32 | 거의 동일 |
| 총 소요 | 18분 | 18분 43초 | 유의미한 차이 없음 |

**결론**: TF32/cudnn.benchmark 를 opt-in 으로 분리해도 F1 완전 재현. 옵션 B-full 수정 안전.

---

## 6. 논문 Pending #2 반영 (11 지점)

연구노트 #14 에서 정리된 pending update #2 의 11개 지점을 이번 세션에서 모두 반영. paper_draft.md 603 → **642 lines** (+39 lines).

### 6.1 반영 지점 요약

| # | Section | 변경 요지 |
|:-:|:---|:---|
| 1 | §3.1 Dataset Overview | 원본 124,263 vs subset 111,870, 원본 정상 99% → subset 리샘플링, 3-manufacturer 소개 |
| 2 | §3.2 Sensor Specifications | Table 2b 신설 (실사용 4종), CT 채널 역할 (입/출/모터1/모터2), 열화상 32×32 native + 15× upsampled |
| 3 | §3.3 Degradation State Definition | Table 3a 신설 (Z-score 3.5 + GT-30s + 관심/경고 임의 1:1 분할), Table 3 에 KO/EN 라벨 |
| 4 | §3.5 Problem Definition | 100ms → 1s aggregation, Danger recall = 30초 개입 마진 |
| 5 | §6.1 Table 6 | **AI Hub MMTransformer baseline (F1 0.9109) 최상단 추가** + cross-attention 91-93% 상한 서술 |
| 6 | §6.3 Class-Level | Attention↔Warning 근본 하한 서술 |
| 7 | §6.5 Thermal | 낮은 기여도의 진짜 원인 = 32×32 native 상한 |
| 8 | §6.9 H3 evidence | MMTransformer 추가 |
| 9 | §7.1 Discussion | Cross-attention 3개 (MMTransformer/V4/V5) 모두 ~91-93% 상한 → 정량적 논거 강화 |
| 10 | §7.5 Limitations | 5개 소절 재구성 (dataset scope / unused fields / cross-facility / field deployment / pretraining) |
| 11 | §7.5 Future Work | Sensor LM · self-supervised pretraining 방향 IJCAI 확장으로 명시 |

### 6.2 §6.1 Table 6 갱신 (가장 큰 임팩트)

```
Model                             F1        Params
AI Hub reference MMTransformer   0.9109    n/a       ← 신규 추가
TimesNet                         0.9189    1.55M
PatchTST                         0.9311    1.36M
V1: Baseline LSTM                0.9235    2.83M
V2: + TempDiff                   0.9430    2.84M
V3: + EfficientNet               0.9242    4.52M
V4: CATFT - CrossAttn            0.9112    7.63M
V5: Full CATFT                   0.9252   10.79M
V2+ (Proposed)                   0.9557    2.85M    ★
```

3개 Cross-Attention 계열 (MMTransformer 0.9109 / V4 0.9112 / V5 0.9252) 이 모두 91-93% 상한 → V2+ (0.9557) 가 3-5 %p 뛰어넘음.

---

## 7. Git 커밋 3개 · Push 성공

**`f147b40`** — `fix: address optimization issues from Qwen changes (LRU rollback, TF32 opt-in, torch.amp API, error handling)`
- 4 files, +309/-296
- src/data/dataset.py, src/data/session_index.py, src/train_v2plus.py, PERFORMANCE_OPTIMIZATION.md

**`d8ef81a`** — `docs: sync paper/reference/notes with actual AI Hub schema and add explicit RQ/H`
- 4 files, +751/-250
- paper_draft.md (RQ/H 및 §6.9 신규), 참고문서 2, 연구노트_02 (§0/§0-2 addendum)
- 연구노트 #14 에서 정리한 발견 대부분 여기 포함

**`44aad95`** — `paper: reflect AI Hub schema/guideline findings across §3, §6, §7`
- 1 file, +69/-30
- paper_draft.md 만: §3/6/7 대규모 갱신 (11 지점)

**원격 push 성공**: `d8ef81a..44aad95` (race 없음, fast-forward).

---

## 8. 부수 성과

### 8.1 재현 실험 인프라

- 재현 실측 로그: `logs/reproducibility_test.log`, `logs/reproducibility_test_v2.log`
- Backup: `results/v2plus_pre_qwen_backup_20260806_121834/`, `results/v2plus_afterQwen_20260806_125555/`

### 8.2 사용자 보유 센서 판정 · 신규 구매 확정

병행으로 사용자 보유 센서 판정 완료 (참고문서 갱신):
- NTC 10K B=3950 프로브 → 그대로 사용
- Sensirion SPS30 → 그대로 사용 (AI Hub Sharp 대비 우수)
- FLIR Lepton 3.5 + PureThermal → 그대로 사용 (32×32 → 160×120 상회)
- DHS20P400A CL420 → 24V PSU 부담으로 교체 확정 → **YHDC SCT-024-000 (400A 패시브)** 신규 구매

Bias 회로 (33Ω × 2 병렬 = 16.5Ω burden + 10kΩ 분압 + 10µF) 는 사용자 보유 부품으로 완결.

### 8.3 로드맵 · Sensor LM 방향

포스터 착수 전 사용자가 물어봄:
- Q: "우리 연구 탄탄해? 학술적 의미 있어?"
- A: MDPI 급 탄탄, IJCAI top 은 확장 필요. Applied AI · 산업 안전 empirical contribution.

- Q: "Sensor LM 처럼 다양한 셋으로 정의할 수 있나?"
- A: 현재 draft 에는 over-scope (label 근본 하한 존재), IJCAI 확장판 방향으로 §7.5 에 명시. Multi-view / self-supervised 는 데이터셋 확장 후 유효.

---

## 9. 파일 위치

| 파일 | 경로 |
|------|------|
| Qwen 변경 원본 | git commit `344ffdf`, `6542bd7` |
| 재현 실측 로그 1 | `logs/reproducibility_test.log` |
| 재현 실측 로그 2 (수정 후) | `logs/reproducibility_test_v2.log` |
| 결과 백업 (Qwen 원본) | `results/v2plus_pre_qwen_backup_20260806_121834/` |
| 결과 백업 (수정 후) | `results/v2plus_afterQwen_20260806_125555/` |
| 코드 수정 커밋 | `f147b40` |
| Docs 갱신 커밋 | `d8ef81a` |
| Paper §3/6/7 갱신 커밋 | `44aad95` |
| Pending updates 메모리 | `~/.claude/projects/-home-keti/memory/project_paper_pending_updates.md` (#2 완료 표시) |

---

## 10. 결론

- Qwen AI 개선 (`344ffdf`, `6542bd7`) 는 정확도에 해 없음 실측 확인 (F1=0.9551 재현).
- 하지만 **LRU cache 는 무효**, TF32/benchmark 는 재현성 위협, silent fallback 은 위험 → 옵션 B-full 로 정정.
- 정정 후에도 F1=0.9551 완전 동일 → 안전.
- 논문 pending #2 (11 지점) 모두 반영, +39 lines. **AI Hub MMTransformer baseline 추가는 논문 최강 empirical 근거**.
- Git 3 커밋 (fix + docs + paper) push 성공. Jetson 이나 다른 컴퓨터에서 pull 로 반영.
- 남은 pending: #1 (Jetson 실측 결과 도착 시 §6.8 Table 14 / §6.9 / §7.5 소폭 갱신).
- 다음 큰 목표: **포스터 draft 작성** (Q1~Q4 확정 후).
