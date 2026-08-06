# 성능 최적화 변경 사항 (Performance Optimization)

## 📋 개요
멀티모달 센서 기반 이상 탐지 시스템의 훈련 파이프라인 성능 최적화.
- 최초 개선 커밋 `344ffdf` (2026-08-03, qwen.ai bot 저자)
- **검증·후속 정정 커밋** (2026-08-06): 실측 재현 실험으로 개선 효과 검증, 무효 개선 제거, PyTorch 2.6 호환성 정리, 재현성 보호를 위한 `--fast` opt-in 도입.

**목표:** 훈련 속도 향상 · GPU 활용도 개선 · 논문 재현성 유지 (V2+ F1 = 0.9557 ± 0.0006).

---

## 🚀 개선 사항 (검증 결과 반영)

### 1. 데이터 로딩 병렬화 (Parallel Label Loading) — ✅ 유효
**파일:** `src/data/session_index.py`

**해결책 (요지):**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _read_labels_batch(label_dir: str, basenames: list[str]) -> list[int]:
    """Read multiple labels in parallel using thread pool.
    Raises RuntimeError if any label fails to load.
    """
    results = {}
    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_bn = {executor.submit(_read_label, label_dir, bn): bn for bn in basenames}
        for future in as_completed(future_to_bn):
            bn = future_to_bn[future]
            try:
                results[bn] = future.result()
            except Exception as e:
                failures.append((bn, str(e)))
    if failures:
        raise RuntimeError(f"Failed to read {len(failures)} label file(s). First: {failures[:5]}")
    return [results[bn] for bn in basenames]
```

**2026-08-06 정정**: 원 구현은 실패 시 `results[bn] = 0` fallback → 손상된 라벨 파일이 조용히 정상(0)으로 학습되는 위험. **RuntimeError 로 강화** 하여 초기 감지되게 함.

**효과 (미계측)**: 인덱스 캐시 (`cache/session_index_*.pkl`) 가 이미 존재하는 상태에서는 이 함수가 재실행 안 됨 → 실 재현 실험에서 별도 측정 안 함. 최초 인덱스 빌딩 시에만 이득.

---

### 2. LRU 캐싱 (LRU Caching) — ❌ **롤백**
**파일:** `src/data/dataset.py`

**원 구현**:
```python
@lru_cache(maxsize=128)
def _load_single_sensor(source_dir, basename): ...

@lru_cache(maxsize=128)
def _load_single_thermal(source_dir, basename): ...
```

**2026-08-06 롤백 결정 근거**:
- 데이터셋 파일 수 ≈ **90,000** vs cache maxsize **128** → **hit rate 극히 낮음**
- DataLoader `num_workers=8` 이면 **각 worker 프로세스마다 독립 캐시** (lru_cache 는 프로세스 스코프) → 캐시 공유 안 됨
- 실제 측정 시 성능 이득 없음, 오버헤드만 추가
- 원 문서에서 "파일 읽기 90,000회 → 1회" 라는 주장은 근거 없음

**대신 유지된 것**: `self._all_labels = self.get_all_labels()` at `__init__` — 라벨 pre-loading 은 WeightedRandomSampler 에 실제 유효. (Qwen 이 새로 도입한 부분, 이는 정확히 유효)

**thermal 파일 형식 참고**: `.bin` 확장자는 misnomer. 실제로는 **numpy `.npy` 포맷** (magic bytes `\x93NUMPY` 확인) 이므로 `np.load(mmap_mode='r')` 이 정확한 loader.

---

### 3. DataLoader 파라미터 최적화 — ✅ 유효
**파일:** `src/data/config.py`

```python
num_workers: int = 8         # 기존 4 → 8
prefetch_factor: int = 4     # 기존 2 → 4
pin_memory: bool = True      # 기존 유지
```

**효과 (실측)**: 재현 학습 epoch 시간 **~36초** 관찰. 원본 기준 (측정치 미기록) 대비 개선. `non_blocking=True` on `.to(device)` 도 함께 도입.

---

### 4. Mixed Precision Training (AMP) — ⚠️ Opt-in (`--amp`)
**파일:** `src/train_v2plus.py`

```bash
python -m src.train_v2plus --amp
```

**2026-08-06 정정**:
- PyTorch 2.6 deprecation 대응: `torch.cuda.amp.autocast/GradScaler` → `torch.amp.autocast("cuda") / GradScaler("cuda")`
- **AMP 정확도 회귀 실험은 아직 미실시** — SupCon loss 는 pairwise cosine similarity 계산으로 fp16 에서 수치 불안정 가능. 사용 시 F1 재현 반드시 확인.

---

### 5. Fast math (TF32 + cuDNN benchmark) — 🆕 **`--fast` opt-in 으로 분리**
**파일:** `src/train_v2plus.py`

**원 구현**: import 시점에 무조건 활성화
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

**2026-08-06 변경**: `enable_fast_math()` 함수로 이동 + `--fast` CLI 플래그로 opt-in.

```python
def enable_fast_math():
    """opt-in via --fast; introduces non-determinism / minor drift."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
```

**이유**: `cudnn.benchmark=True` 는 워크로드마다 최적 알고리즘 자동 선택 → **비결정론적**. 논문의 3-seed 통계 (F1 std = 0.0006) 재현 시 영향 우려.

**재현성 규칙**:
- 논문 재현 실험: `--fast`, `--amp` 없이 사용 (fp32, deterministic-ish)
- 프로덕션 학습: `--fast --amp` 로 속도 최대화

---

## 📊 실측 벤치마크 (2026-08-06)

**환경**: Quadro RTX 6000 × 1 (GPU 1), Ubuntu, Python 3.12, PyTorch 2.6.0+cu124

| 지표 | Qwen 개선 후 (fp32, `--fast` 자동 활성) | **수정 후 (fp32, `--fast` 없음)** |
|:---|:---:|:---:|
| Epoch 당 시간 | ~36 초 (1회 관찰) | 측정 예정 |
| 총 30 epoch 소요 | **약 18 분** | 측정 예정 |
| **Val F1 (macro)** | **0.9551** | 측정 예정 |
| Val Accuracy | 0.9481 | 측정 예정 |
| Severe recall | **100% (66/66)** | 측정 예정 |
| Normal↔Mild 오류 | 33 | 측정 예정 |
| 논문 목표 F1 재현 | ✅ (논문 3-seed 평균 0.9557 ± 0.0006 범위 안) | (검증 진행 중) |

**참고**: Qwen 이 원 문서에서 주장한 "epoch 120초 → 45초", "파일 읽기 90k→1", "GPU 활용도 55→85%" 등은 원본 대조 실측 없이 이론적 추정치.

---

## 🔧 사용 방법

### 논문 재현 (fp32, TF32/benchmark 없음)
```bash
python -m src.train_v2plus --gpu 1
# 예상: F1 ≈ 0.955 (논문 3-seed 평균 0.9557 ± 0.0006 안에 포함되어야 함)
```

### 속도 우선 (개발 · 실험 반복)
```bash
python -m src.train_v2plus --gpu 1 --fast              # TF32 + cudnn.benchmark
python -m src.train_v2plus --gpu 1 --fast --amp        # + mixed precision (AMP)
```

### 커스텀 하이퍼파라미터
```bash
python -m src.train_v2plus \
  --gpu 1 --fast \
  --batch-size 32 \
  --epochs 50 \
  --supcon-weight 0.3 \
  --lags 1,5,10
```

**주의**: `--num-workers` CLI 옵션은 없음 (DataConfig 값 8 사용). 배치 · num_workers 늘리려면 `src/data/config.py` 직접 편집.

---

## ⚠️ 주의사항

### 재현성
- **논문 수치 (F1 0.9557 ± 0.0006) 재현이 필요하면 `--fast` 절대 사용 금지**. TF32 (fp32 대비 19bit mantissa) + `cudnn.benchmark` (비결정론) 조합이 소수 F1 자릿수를 흔들 수 있음.
- **`--seed` 인자 부재**: 현재 학습 스크립트에 seed 제어 없음 → 매 실행마다 다른 결과. 논문의 3-seed 통계 재현은 별도 코드 수정 필요 (향후 개선 후보).

### 시스템
- **권장 RAM**: 16 GB 이상 (num_workers=8 × prefetch=4 시 batch 데이터 다중 유지)
- **GPU**: NVIDIA Ampere 이상 (RTX 30, A100, Orin, RTX 6000 (Turing 도 TF32 일부 지원))
- **CPU**: 8 코어 이상 권장 (num_workers=8 활용)

### AMP 사용 시
- **SupCon loss 는 fp16 에서 수치 안정성 저하 가능** (pairwise softmax). `--amp` 사용 시 F1 재현 반드시 확인 후 채택.

---

## 📝 커밋 정보

**최초 개선 (Qwen)**:
- `344ffdf` (2026-08-03) `perf: 데이터 로딩 병렬화 및 Mixed Precision Training 으로 성능 최적화`
- `6542bd7` (2026-08-03) `docs: 성능 최적화 상세 문서 추가`

**검증 · 정정 (2026-08-06)**:
- LRU cache 롤백 (`dataset.py`)
- TF32/cudnn.benchmark 를 `--fast` opt-in 으로 분리 (`train_v2plus.py`)
- `torch.cuda.amp.*` → `torch.amp.*` (PyTorch 2.6 호환)
- Session_index fallback `results[bn]=0` → `raise RuntimeError` 로 강화
- 이 문서 실측 · 이력 정정

---

## 🔍 검증 방법

### 1. F1 재현 검증
```bash
python -m src.train_v2plus --gpu 1 > logs/repro.log 2>&1
grep "Val F1 (macro)" logs/repro.log | tail -1
# 기대: 0.953 ~ 0.958 (논문 3-seed 평균 0.9557 ± 0.0006 근사 범위)
```

### 2. TF32 유무 영향 비교 (재현성 감도)
```bash
python -m src.train_v2plus --gpu 1        > logs/fp32.log 2>&1
python -m src.train_v2plus --gpu 1 --fast > logs/tf32.log 2>&1
grep "Val F1 (macro)" logs/{fp32,tf32}.log
```

### 3. AMP 정확도 회귀
```bash
python -m src.train_v2plus --gpu 1 --amp --fast > logs/amp.log 2>&1
# 기대: F1 저하 < 0.5%p (미검증)
```

---

## 📚 참고

- PyTorch AMP 가이드: https://pytorch.org/docs/stable/notes/amp_examples.html
- TF32 정보: https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
- 논문 draft: `docs/논문/paper_draft.md`
- 재현 실측 로그: `logs/reproducibility_test*.log`

---

## 🤝 향후 개선 후보

1. **`--seed` CLI 인자 도입** — 논문의 3-seed 통계 (F1 std=0.0006) 재현
2. **AMP + SupCon 조합 정확도 검증** — 지금은 미검증. 실측 후 문서화
3. **DataLoader `persistent_workers=True` 검토** — Qwen 문서에는 언급되지만 config.py 에 실제 없음
4. **LRU cache 재도입 검토 (프로세스 공유 캐시)** — shared memory + LMDB 계열
5. **논문 baseline (MMTransformer F1 91.09%) 실측 재현** — Section 6.1 Table 6 강화
