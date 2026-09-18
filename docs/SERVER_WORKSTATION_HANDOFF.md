# SERVER WORKSTATION HANDOFF

**SERVER-TRAINING 환경에서 이 저장소를 처음 여는 agent 가 가장 먼저 읽는 문서다.**

이 문서는 이전 Jetson 세션의 대화 맥락을 모르는 상태에서도 서버 작업을 안전하게 이어갈 수
있도록 작성됐다. 여기에 적힌 것은 전부 Jetson 하드웨어에서 실측·검증된 사실이거나, 아직
확정되지 않았다는 사실 자체다. **추측은 들어 있지 않다.**

```
작성 시점 HEAD   8248e4a5367a44f3b1a3a72b0dad3c5119481a98
branch           feature/jetson-sensor-integration
milestone tag    jetson-dataset-v1-ready-2026 -> 8bc5e88   (이동 금지)
작성 환경        Jetson Orin Nano 8GB (keti-kms, aarch64, L4T R36.5.0)
```

---

## 1. 하나의 저장소, 두 개의 실행 프로파일

저장소는 하나다: `multi_sensor_anomaly_detection`. 그러나 실행 프로파일은 둘이며 섞지 않는다.

| | JETSON-RUNTIME | SERVER-TRAINING |
|---|---|---|
| 코드 영역 | `jetson_deploy/` | `src/` |
| venv | `$HOME/venvs/factory_runtime` | `$HOME/venvs/factory_training` |
| 아키텍처 | aarch64 (L4T/JetPack) | x86_64 Linux workstation |
| 판별 | `/etc/nv_tegra_release` 존재 | 그 파일이 없는 Linux |

### JETSON-RUNTIME 의 책임

- physical sensor acquisition
- trial dataset generation
- local raw storage / buffering
- **real-time edge inference**
- ONNX / TensorRT runtime

### SERVER-TRAINING 의 책임

- raw dataset import
- preprocessing
- dataset versioning
- training
- evaluation
- batch re-inference
- model export (ONNX)

**서버는 real-time field inference node 가 아니다.** 현장 실시간 판정은 Jetson 이 로컬에서
독립적으로 수행한다. 서버를 실시간 판정 경로에 넣지 않는다. 근거는
[DATA_PLATFORM_ARCHITECTURE.md](DATA_PLATFORM_ARCHITECTURE.md).

환경 판별 규칙은 새로 만들지 않는다. [../CLAUDE.md](../CLAUDE.md) 와
[../AGENTS.md](../AGENTS.md) 의 기존 규칙을 그대로 쓴다.

---

## 2. 현재 상태

### DONE

- Jetson sensor bring-up (ADS1115+NTC, CT1, SPS30, SCD30, BME680, FLIR Lepton)
- sensor diagnostics 및 SPI/I2C 배선 확정
- synchronized 1 Hz multi-sensor collector
- data quality control (per-tick / per-window)
- reproducible trial runner (preflight gate 포함)
- **development normal baseline collection** (5 trial, §4)
- **30-second offline window builder** (`src/data/window_builder.py`)

```
latest preprocessing commit   8248e4a5367a44f3b1a3a72b0dad3c5119481a98
milestone tag                 jetson-dataset-v1-ready-2026 -> 8bc5e88
```

### NOT DONE / NOT STARTED

- official German field dataset
- real anomaly dataset collection (`overload` / `thermal_abnormal` / `dust`)
- new model training on Jetson-collected data
- **model schema 결정** (§6)
- ModelAdapter (실측 데이터 → legacy 8-channel tensor)
- CT2 / CT3 / CT4 physical acquisition
- partner platform uploader
- production data API
- automatic model deployment

`docs/SERVER_ENVIRONMENT.md` 는 아직 `PENDING SERVER ENVIRONMENT AUDIT` 이다. 이 문서의
§9 audit 이 그 문서를 채우는 첫 작업이다.

---

## 3. Raw data 는 GitHub 에 없다

**이 항목을 먼저 이해해야 한다. clone 만으로는 데이터가 생기지 않는다.**

GitHub 저장소에 있는 것:

```
code (jetson_deploy/, src/)
docs/
docs/manifests/*.sha256          <- checksum 만, 데이터 아님
```

GitHub 저장소에 **없는** 것:

```
dataset/     raw trial 디렉터리      (.gitignore: /dataset/)
processed/   생성된 window 출력      (.gitignore: /processed/)
```

따라서 서버 작업에는 **두 개의 소스**가 필요하다.

| | 소스 | 획득 방법 |
|---|---|---|
| A | GitHub repository | `git clone` / `git pull` |
| B | raw 5 baseline trial | Jetson 에서 별도 전달 |

### raw transfer 방식

**canonical 전송 방식을 아직 확정하지 않는다.** 특정 network path 를 가정하지 않는다.
실제 workstation / network 상황에 맞는 것을 고른다.

```
rsync
scp
controlled external storage
그 밖의 통제된 전송 수단
```

**raw file 내용을 변환해서 전달하지 않는다.** 압축 컨테이너로 감싸는 것은 무해하지만,
파일 내용 자체를 재인코딩·재압축·정규화·병합하면 §8 checksum 검증이 깨지고 데이터
provenance 가 끊긴다. 전달 후 §8 을 반드시 통과시킨다.

`processed/` 는 전달하지 않는다. 서버에서 §10 으로 재생성하고 Jetson 결과와 대조하는 것이
바로 첫 서버 milestone 이다.

**AI Hub legacy dataset 은 또 다른 소스다.** 위 두 소스(GitHub, development baseline raw)만으로는
기존 연구를 재현할 수 없다. AI Hub 자산의 현재 상태와 전송·검증 절차는 **§13** 에 있다.
**2026-09-17 확인: 서버에는 이미 온전한 사본이 있다**(§13-6). Jetson 에서 zip 을 다시 보낼 필요는 없다.

**서버의 프로젝트 데이터는 HDD 에 있고 원래 경로는 심볼릭 링크다** (2026-09-17 이관, 연구노트 #17).

```
/home/keti/factory_safety/data/aihub  ->  /mnt/data-hdd/keti_data/factory_safety/aihub
```
코드가 참조하는 경로는 바뀌지 않았다. 새 데이터도 루트 디스크가 아니라 `/mnt/data-hdd/keti_data/` 또는
`/mnt/data-ssd/keti_data/` 아래에 둔다.

---

## 4. 2026 development baseline (5 trials)

```
normal_20260904T074717Z
normal_20260904T075323Z
normal_20260904T075930Z
normal_20260904T080536Z
normal_20260904T081141Z
```

**분류: development / pilot data.** official German field dataset 이 **아니다.**
`test_mode = true` 로 수집되어 `dataset/_smoke/` 에 있다.

| 항목 | 값 |
|---|---|
| scenario | `normal` |
| severity_level | `0` |
| trial 당 duration | 360 s |
| trial 수 | 5 |
| 총 duration | 1800 s (30.0 min) |
| snapshots | 1800 |
| missed master ticks | **0** |
| master period | mean 1000.001 ms, \|jitter\| max 0.93 ms |
| sensor profile | `jetson_factory_v1_2026` (SGP30 disabled) |
| 수집 시 git commit | `5570ab3c4398c4b90f9149252ac97aa393841404` |
| raw 총 용량 | 42,799,477 bytes (42.80 MB), 85 files |

**주의:** `dataset/_smoke/` 에는 이 5개 외에 `normal` scenario id 를 공유하는 짧은
orchestration test 가 더 있다 (`normal_20260903T070431Z`, `dust_20260903T070522Z`).
**scenario_id 는 배치 식별자가 아니다.** baseline 배치는 `operator_note` 가
`"Korea development baseline"` 으로 시작하는 trial 로 식별한다. §10 CLI 가 그 필터를 쓴다.

### expected preprocessing result

```
trials             5
snapshots          1800
30-second windows  60      (non-overlapping, stride 30 ticks)
valid              49
invalid            11      (18.3 %)
structural errors  0
invalid reason     flir_stale  (그 밖의 사유 없음)
```

Jetson 에서 검증된 교차 확인 결과:

- trial 별 window valid/invalid 가 collector 의 `timing_report.quality` 와 **5/5 일치**
- per-tick quality 재계산이 수집 당시 기록과 **1800/1800 일치**
- thermal frame 이 원본 chunk 와 **1800/1800 비트 단위 일치**
- FLIR min/mean/max 재계산 오차 **0.0 °C**
- `dataset/` 97 파일 SHA-256 빌드 전후 동일 (raw mutation 없음)

---

## 5. 서버 agent 가 반드시 알아야 할 관측 사실

### A. FLIR FFC — invalidity 는 random missingness 가 아니다

Lepton 의 automatic FFC (flat-field correction) 셔터가 약 180 s 주기로 스트림을 수백 ms ~
2 s 멈춘다. 그 tick 은 `flir_age_ms > 500` 으로 thermal invalid 가 되고, 그 tick 을 포함한
30-tick window 전체가 invalid 가 된다.

실측 (5 trial, 60 window):

```
stale tick sequence      28~31 과 211~214 두 군데에만 몰림
trial 내 invalid window   window 0: 2/5    window 1: 4/5    window 7: 5/5
```

FFC 위상이 camera open 기준으로 반복되므로 **손실이 trial 시간축에 균등하게 흩어지지 않고
같은 구간에 계통적으로 몰린다.** window 7 은 5 trial 전부 손실됐다.

- 18.3 % 를 **무작위 결측으로 가정하고 모델링하거나 phase 배치를 정하지 않는다**
- **현재 quality policy 를 서버에서 임의로 변경하지 않는다**
- FFC manual mode 전환, post-FFC exclusion rule 추가, interpolation, stale frame 대체
  **모두 금지**

같은 baseline 에서 FLIR `max_c 50.66 °C` / `min_c 15.23 °C` outlier 가 관측됐다. 해당 tick
자체는 `age < 500 ms` 로 valid 였고, 인접한 FFC stale tick 때문에 같은 window 가 invalid 가
되어 결과적으로 제외됐다. **이 값을 새 automatic threshold 로 쓰지 않는다.**

정책 구현: `jetson_deploy/sensors/snapshot.py` 의 `tick_quality()` / `window_quality()`.
window builder 는 이 정책을 재구현하지 않고 해당 파일을 직접 로드해 호출한다.

### B. CT1 — 현재 baseline 은 no-load noise floor 다

ADS1115 PGA ±2.048 V, 16-bit → 1 LSB = 62.5 µV. 실측:

```
no-load differential   약 ±1~2 LSB
vrms                   약 0.46~0.61 LSB   (2.90e-05 ~ 3.78e-05 V)
current_a_nominal      약 0.0196 A
clipping               0 tick
```

vrms 가 1 LSB 의 절반이다. 따라서 **`ct1_current_a_nominal ≈ 0.0196 A` 는 측정된 전류가
아니다.** analog + ADC noise floor 를 CT nominal 환산식에 통과시킨 값이며,
`std ≈ 1.5e-04 A` 도 실제 전류 변동이 아니다.

다음에 사용 금지:

- normal current threshold
- calibration value
- actual robot operating current
- "정상 운전 전류" 로 문서화

known-load 또는 실제 robot 운전 CT 데이터가 확보되기 전까지 이 채널은
**CT1 no-load / noise-floor baseline** 으로만 취급한다. hardware gain / PGA / burden 저항은
변경하지 않는다.

### C. CT2 / CT3 / CT4 — physical sensor 가 없다

front-end 자체가 존재하지 않는다. collector 는 `status = "disabled"`,
`reason = "no physical CT connected"` 로 기록하고 수치 배열을 만들지 않는다.

**절대 금지:**

```
zero fill
CT1 replication
interpolation
synthetic values
```

window builder 는 `channel_availability` 에 `available: false` 로만 기록한다. 8-channel 을
맞추는 것은 이 계층의 책임이 아니다 (§6).

### D. BME680 — context sensor

현재 model input 이 **아니다.** temperature / humidity / pressure / gas 를 계속 저장하지만
모델 입력 벡터에 넣지 않는다.

gas 저항은 trial 시작 직후 heater stabilization 영향을 보인다 (trial 1 첫 3 tick:
31,040 → 38,691 → 49,278 Ω). **확정된 warm-up 시간 근거가 없으므로 기준을 발명하지 않았다.**
window metadata 에 `bme680_gas_warmup_present` 를 first-window context 로만 표시하고,
**window invalidation 에 사용하지 않는다.**

### E. SCD30 — transient polling error 2건

baseline 전체에서 polling error 2건 (trial 1, trial 5 각 1건).

```
consecutive error          없음
age_ms > 4500              없음  (관측 최대 2327 ms)
snapshot stale/error       없음  (1800/1800 status = ok)
SPS30 동시 instability      없음  (SPS30 error 0)
```

**현재 window invalidation 조건이 아니다.** provenance 로만 기록한다. SCD30 은 native
0.5 Hz 이므로 `fresh = false` 로 값을 반복하는 것이 정상이며 (858/1800 = 47.7 % fresh),
보간하지 않는다.

---

## 6. Legacy model boundary

```
legacy expected scalar channels   ["NTC","PM1.0","PM2.5","PM10","CT1","CT2","CT3","CT4"]
                                  (src/data/config.py:SENSOR_CHANNELS)

현재 실측 model-relevant channels  ["NTC","PM1.0","PM2.5","PM10","CT1"]

legacy_model_ready = false
reason = ["CT2_missing", "CT3_missing", "CT4_missing"]
```

window builder 는 legacy 8-channel tensor 를 **만들지 않는다.** observed 5 channel 과 legacy
expected 8 channel 을 구분해 기록하고 거기서 멈춘다. `src/data/config.py` 는 읽기만 한다.

**서버 agent 는 reproduction (§10) 전에 기존 모델에 실측 데이터를 억지로 맞추지 않는다.**
model schema 변경 여부는 §11 의 별도 검토 단계에서 결정한다. **이 문서는 그 결정을 하지
않는다.**

참고: 학습된 모델은 AI Hub 데이터셋 기반이며 thermal 120×160, 30-tick window @ 1 Hz,
4-class 다. 채널을 바꾸면 `model_v2plus.onnx`, TensorRT engine, reference output, 논문 수치가
모두 무효가 된다.

---

## 7. Server first-run procedure (canonical)

**순서를 지킨다. 역순으로 하지 않는다.**

```
STEP 1   repository clone / pull
STEP 2   branch / commit 확인
STEP 3   실제 Linux workstation environment audit        (§9)
STEP 4   development baseline raw 5 trial 배치           (§3)
STEP 5   SHA-256 manifest verification                  (§8)
STEP 6   30-second preprocessing reproduction           (§10)
STEP 7   result cross-check
```

### STEP 1 — clone / pull

```bash
git clone git@github.com:mskim0114/multi_sensor_anomaly_detection.git
cd multi_sensor_anomaly_detection
git fetch --all --tags
git checkout feature/jetson-sensor-integration
```

기존 클론이 있으면 `git pull --ff-only` 만 쓴다. `git reset` / `git clean` / force checkout /
rebase 로 작업물을 날리지 않는다.

**2026-09-18 서버 실제 레이아웃.** 서버에는 clone 이 둘이다. 혼동하지 않는다.

```
/home/keti/projects/factory_safety     작업 clone. public 저장소 feature 브랜치 572cff8. Jetson 과 같은 경로
                                       remote origin = public, remote private = factory_safety (fetch 용)
  data/aihub/{models,docs}             -> HDD 아카이브 심볼릭 링크
  data/aihub/datasets/<zip들>          -> HDD 아카이브 심볼릭 링크
  data/aihub/datasets/extracted        -> /mnt/data-ssd/keti_data/factory_safety/aihub_extracted (SSD 작업 사본)
  dataset/_smoke                       -> HDD sensor_data import 의 smoke_trials (baseline 5 trial 포함)
  results/                             Jetson 논문 run 산출물 사본 (48 파일, 363 MB)
  processed/development_baseline_v1    G1-S 서버 실행 산출물

/home/keti/factory_safety              기존 clone. private main 5feabb5 + docs/특허 dirty 51. 건드리지 않는다.
                                       results/ 는 08-06 Qwen 이후 재실행분(논문 run 아님)
```

### STEP 2 — branch / commit 확인

```bash
git rev-parse --abbrev-ref HEAD          # feature/jetson-sensor-integration
git log --oneline --decorate -5
git rev-list -n1 jetson-dataset-v1-ready-2026    # 8bc5e88... 이어야 한다
git status --short                                # clean 이어야 한다
```

`jetson-dataset-v1-ready-2026` 태그는 `8bc5e88` 을 가리켜야 한다. **이 태그를 이동하거나
재생성하지 않는다.**

### STEP 7 — result cross-check

STEP 6 출력이 §4 의 expected result 와 완전히 일치해야 한다.

```
trials = 5   snapshots = 1800   windows = 60
valid = 49   invalid = 11       structural errors = 0
```

**첫 서버 milestone 은 학습이 아니라 이것이다:**

```
JETSON RAW -> SERVER COPY -> CHECKSUM PASS -> SAME 60 WINDOWS -> SAME 49 VALID / 11 INVALID
```

---

## 8. Raw baseline SHA-256 검증 (STEP 5)

manifest:

```
docs/manifests/development_baseline_20260904.sha256
```

- 대상: **위 5개 baseline trial 만.** `dataset/_smoke` 의 다른 trial 은 포함하지 않는다
- 85 checksum 줄, 대상 총 42,799,477 bytes
- 경로는 **dataset root 기준 상대 경로** (`_smoke/normal_.../experiment.json`).
  absolute Jetson path 는 들어 있지 않다
- `#` 로 시작하는 헤더 주석은 `sha256sum -c` 가 무시한다

검증:

```bash
cd dataset
sha256sum -c ../docs/manifests/development_baseline_20260904.sha256
```

Jetson 에서의 결과: `85 OK`, `FAILED 0`, exit 0.

### 실패 시 규칙

한 파일이라도 다음에 해당하면 **preprocessing 과 training 을 시작하지 않는다.**

```
missing
checksum mismatch
선택된 5 trial 안에 manifest 에 없는 extra file
```

전송을 다시 하고 원인을 보고한다. **checksum 을 다시 생성해서 맞추지 않는다.** manifest 는
Jetson 원본의 지문이며, 서버에서 재생성하는 것은 검증이 아니라 검증 회피다.

---

## 9. Server environment audit (STEP 3)

**2026-09-17 갱신.** read-only preflight 를 SSH 로 수행해 하드웨어·디스크·GPU·torch 환경을 실측했고
[SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) 에 기록했다(상태 `PARTIAL`). 요약:
2× RTX 6000(driver 580.178.04, `nvidia-smi` 정상) · 187 GiB · 루트 디스크 32 %(이관 후) ·
`monai_env` 에 torch 2.6.0+cu124 · **`$HOME/venvs/factory_training` 없음**.

**2026-09-18 추가 확인.** `monai_env` 에서 `src/` 가 요구하는 패키지 11개(torch 2.6.0+cu124,
torchvision 0.21.0, numpy 2.2.6, sklearn 1.8.0, matplotlib 3.10.8, seaborn 0.13.2, pyyaml 6.0.3,
onnx 1.20.0, onnxruntime 1.24.4, tqdm 4.67.1, pandas 2.3.3)가 전부 import 되고 CUDA 2장이 인식된다.
**설치 없이 학습이 가능하다.** 남은 것은 `requirements-server.txt` 작성과, 이 env 를 SERVER-TRAINING
으로 채택할지의 정책 결정(decisions O-108)이다. 결정 전에 패키지를 설치하지 않는다.
참고: numpy 가 2.2.6 이라 Jetson 의 1.26.4 고정과 다르다. 두 프로파일의 numpy 를 맞추지 않는다.

```
hostname / OS / kernel / 아키텍처
CPU / RAM
disk / filesystem / free space
GPU 모델·개수·VRAM
nvidia-smi
NVIDIA driver 버전
CUDA availability (nvcc, torch.version.cuda)
Python 버전
기존 $HOME/venvs/factory_training 존재 여부와 내용
pip package 목록
repository revision
```

절차와 수집 명령은 [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) 의 audit 섹션에 있다.

### audit 전 금지

```
pip install
apt install
driver install
CUDA 변경
venv 생성/삭제/recreate
```

**먼저 환경을 기록하고, 기존 requirements 와의 호환성을 검토한 뒤에 움직인다.**

canonical server venv 는 `$HOME/venvs/factory_training` 이다. **이미 존재하면 먼저 audit
한다. 삭제하거나 recreate 하지 않는다.** 저장소 내부에 `.venv` 를 만들지 않는다. 전체 정책은
[ENVIRONMENT_POLICY.md](ENVIRONMENT_POLICY.md).

### 알려진 미해결 사항

- `requirements-server.txt` 는 아직 비어 있다. audit 결과로 채운다
- 2026-09-10 국소 수정으로 `src/` 기본 입출력 경로는 현재 checkout 기준이 되었다.
  `DataConfig`/YAML 상대 데이터·캐시 경로도 checkout 기준이며, 외부 데이터는 절대경로로
  지정한다. 서버의 실제 경로·권한·학습 의존성 검증은 여전히 audit 대상이다.
  수정·검증 범위는 [LOCAL_REVIEW_20260910.md](LOCAL_REVIEW_20260910.md)를 참조한다
- `src/data/__init__.py` 가 `torch` 를 import 한다. Jetson 에는 torch 가 없어서 window
  builder 를 파일 경로로 로드하도록 작성했다. 서버에서는 `python3 -m src.data.scripts.build_windows`
  형식도 동작할 것으로 예상되지만 **검증되지 않았다.** package 구조를 바꾸지 않는다

---

## 10. 30-second preprocessing reproduction (STEP 6)

**§8 checksum PASS 후에만 실행한다.**

canonical command:

```bash
python3 src/data/scripts/build_windows.py \
    --scan-dir dataset/_smoke \
    --scenario normal \
    --operator-note-prefix "Korea development baseline" \
    --out processed/development_baseline_v1 \
    --expect-trials 5 \
    --expect-snapshots 1800 \
    --expect-windows 60 \
    --expect-valid 49 \
    --expect-invalid 11
```

서버 저장소에 repository policy 에 맞는 canonical runner 가 따로 있으면 그것을 써도 되지만
**semantic arguments 는 동일하게 유지한다.**

`--dry-run` 을 붙이면 아무것도 쓰지 않고 window 회계만 검증한다. 먼저 이것부터 돌리는 것을
권한다.

### 구현이 보장하는 것

- raw 는 read-only 로만 열리고 SHA-256 이 기록된다. raw 를 쓰거나 지우지 않는다
- quality policy 는 `jetson_deploy/sensors/snapshot.py` 를 직접 로드해 호출한다.
  offline 판정이 수집 당시 판정과 갈라질 수 없다
- 수집 당시 기록된 per-tick quality 와 재계산 결과를 대조한다. 불일치는 hard error 다
- thermal 은 snapshot 자신의 `thermal_chunk` / `thermal_index` 로만 주소를 지정한다.
  nearest-frame 대체와 interpolation 이 없다. frame 은 uint16 원본 그대로 저장되고
  섭씨 환산식(`celsius = raw/100 - 273.15`)만 기록된다
- 30 tick scalar 시계열을 전량 보존한다. 평균 한 값으로 축약하지 않는다
- window 회계가 collector report 또는 `--expect-*` 와 어긋나면 **아무것도 쓰지 않고 STOP**
  한다 (exit 3)

### 출력

```
processed/development_baseline_v1/
  manifest.json          배치 provenance, 정책, cross-check 결과
  windows.jsonl          60 record
  baseline_stats.json    valid window 기술 통계
  window_000000.npz ... window_000059.npz
```

Jetson 기준 출력 크기 약 37 MB. `processed/` 는 `.gitignore` 대상이며 **commit 하지 않는다.**

### PASS 기준

```
trials = 5   snapshots = 1800   windows = 60
valid = 49   invalid = 11       structural errors = 0
```

그리고 **preprocessing 전후 raw checksum 이 동일해야 한다.** §8 을 다시 돌려 확인한다.

### 2026-09-18 서버 실행 결과 — G1-S PASS

```
서버 clone 572cff8, python /home/keti/monai_env/bin/python (numpy 2.2.6)
STEP 5   LC_ALL=C sha256sum -c   85 OK, 0 FAILED           (서버 locale 이 한국어라 LC_ALL=C 없이는 'OK' 문자열 매칭이 안 됨)
STEP 6   --dry-run 후 정식 실행   trials 5 · snapshots 1800 · windows 60 · valid 49 · invalid 11 · 구조오류 0
         tick-quality 재계산 일치 · invalid 사유 flir_stale 11 · 출력 38.55 MB
STEP 7   전후 raw 85 OK · processed/ 는 git 무시
```

**Jetson 산출물과의 동등성은 배열 단위로 판정했다.** NPZ 60개 × 38 배열 = 2,280개 전부 동일,
`windows.jsonl` 60 레코드는 환경 종속 3필드(`source_trial_path`, `npz_sha256`, `npz_bytes` 중 sha 만 다름,
bytes 는 동일)를 제외하고 동일, `baseline_stats.json` 은 바이트 동일, manifest 의 totals·quality_policy·
channel_boundary 동일. **NPZ 파일 SHA-256 은 zip 엔트리 타임스탬프 때문에 달라지므로 파일 해시로
재현을 판정하지 않는다.** 연구노트 #17 §6.

### PASS 전 금지 (STOP boundary)

```
training
model schema modification
ONNX export
TensorRT
CT imputation
anomaly threshold generation
raw mutation
```

`baseline_stats.json` 의 수치는 **기술 통계이며 anomaly threshold 가 아니다.** threshold 로
전용하지 않는다.

---

## 11. Reproduction 이후 — SERVER MODEL INPUT REVIEW

reproduction 이 PASS 한 뒤의 별도 phase 다. 검토 대상:

- current training dataset schema
- `src/data/config.py` (`SENSOR_CHANNELS`, `SensorStats`, `ThermalStats`)
- scalar normalization (legacy mean/std 가 실측 분포와 맞는지)
- temporal alignment (window/step, label strategy)
- thermal branch input shape 와 정규화 범위
- CT2 / CT3 / CT4 의 origin — legacy 학습 데이터에서 이 채널이 무엇이었는지
- existing checkpoint assumptions
- legacy training data 와 실측 Jetson data 의 분포 차이

**그 이후에만** 다음 중 방향을 결정한다.

```
A. legacy 8-channel 유지
B. real 5-channel model
C. future multi-CT design
```

**이 handoff 문서는 이 결정을 하지 않는다.** 근거 없이 A/B/C 를 고르지 않는다.

---

## 12. 2027 field deployment context

2027 상반기에 독일 robot test site 에 Jetson 기반 sensor module 을 설치할 예정이다.

```
[독일 현장]
  Sensors -> Jetson Orin Nano -> local acquisition -> edge inference
                              -> partner-operated DB / data platform

[한국]
  partner platform -> remote access / export -> Korea Linux workstation
                   -> preprocessing -> training -> evaluation
                   -> ONNX artifact -> German Jetson
```

- Jetson 의 실시간 판정은 한국 워크스테이션이나 partner platform 의 network 가용성에
  의존하지 않는다
- 한국 워크스테이션은 독일 현장의 central server 가 아니다. research/training node 다
- **partner platform 사양은 아직 전부 TBD 다.** 제품명 / API specification / database type /
  hosting location / authentication / network topology 를 **추측하지 않는다**

canonical 문서:

- [DATA_PLATFORM_ARCHITECTURE.md](DATA_PLATFORM_ARCHITECTURE.md) — 3-node 아키텍처
- [CONSORTIUM_DATA_PLATFORM_QUESTIONS.md](CONSORTIUM_DATA_PLATFORM_QUESTIONS.md) — P01~P20
  partner 확인 질문 (전부 `Answer = TBD`, `Status = OPEN`)

---

## 13. AI Hub legacy dataset 자산 (재학습 blocker)

기존 AI Hub 연구(V1~V2+, 논문)를 재현·재학습하려면 이 절이 먼저 해소돼야 한다.
**§4의 development baseline 5 trial 과는 완전히 별개의 자료다.**

### 13-1. `extracted/` 의 정확한 상태 — 파일은 있고 내용이 비어 있다

"비어 있다"를 두 가지로 나눈다. **파일이 없는 것이 아니라, 파일이 존재하고 크기가 0바이트다.**
디렉터리 4개 모두 존재하며 파일 목록도 온전하다.

경로: `data/aihub/datasets/extracted/`  (2026-09-07 Jetson `keti-kms` 계측)

| split / 종류 | 확장자 | 파일 존재 | 0바이트 | 정상(>0) |
|---|---|---:|---:|---:|
| Training / 원천데이터 | csv | 99,476 | **99,476** | 0 |
| Training / 원천데이터 | bin | 99,476 | **99,476** | 0 |
| Training / 라벨링데이터 | json | 99,476 | **99,476** | 0 |
| Validation / 원천데이터 | csv | 12,394 | 12,391 | 3 |
| Validation / 원천데이터 | bin | 12,394 | 12,390 | 4 |
| Validation / 라벨링데이터 | json | 12,394 | 4,247 | **8,147** |

```
extracted 총 파일      459,873        (위 6종 336,510 + 그 밖의 파일)
정상(>0) 파일          15,621
extracted 총 용량      84 MB
누락(파일명 없음)      0 - 파일 목록은 zip 내용과 일치한다
```

**누락은 0건이다.** 파일명 스켈레톤은 완전하고 대부분의 내용만 0바이트다. 그래서 파일명 기반
분석(예: timestamp 정합 검사)은 가능했고 학습만 불가능하다.

발견 당시 `src/data/config.py` 및 학습·export 진입점은 `/home/keti/factory_safety/...` 에
하드코딩되어 있었다. 2026-09-10 국소 수정으로 checkout 기준 경로가 적용됐다.
이 수정은 데이터 복구 성공을 의미하지 않으며, 서버 audit 시 실제 경로를 확인한다(§9).

### 13-2. 압축 자산 — 존재·구조·내용을 분리해 검사했다

경로: `data/aihub/datasets/` 이하. 세 검사를 **분리**한다.

| 검사 | 방법 | 결과 |
|---|---|---|
| (a) 존재 | 파일 열거 | **684 / 684** |
| (b) 구조 무결성 | `zipfile.is_zipfile` + central directory 파싱 (`infolist`) | **684 / 684 OK** |
| (c) 내용 무결성 | 전 member CRC 검증 (`ZipFile.testzip`), 18.78 GB 압축해제 | **684 / 684 OK, 실패 0** (142 s) |

(b)가 통과해도 (c)가 통과한다는 뜻이 아니다. 두 결과를 각각 기록한다.

| 위치 | zip | 압축 크기 | member |
|---|---:|---:|---:|
| `Training/01.원천데이터` | 303 | 4.91 GB | 198,952 |
| `Training/02.라벨링데이터` | 303 | 0.08 GB | 99,476 |
| `Validation/01.원천데이터` | 38 | 0.62 GB | 24,788 |
| `Validation/02.라벨링데이터` | 38 | 0.01 GB | 12,394 |
| `Other/Other.zip` | 1 | 0.11 GB | 124,263 |
| `Sample.zip` (저장소 루트 `datasets/`) | 1 | 0.32 GB | 18,822 |
| **합계** | **684** | **6.05 GB** | **478,695** |

member 수가 `extracted/` 파일 수와 일치한다: Training 원천 198,952 = csv 99,476 + bin 99,476,
Training 라벨 99,476, Validation 원천 24,788 = 12,394 × 2, Validation 라벨 12,394.
비압축 크기 합계는 central directory 신고값 기준 **18.78 GB**다.

### 13-3. Manifest — 무엇을 증명하고 무엇을 증명하지 않는가

```
docs/manifests/aihub_zips.sha256      684 checksum + 29줄 주석
경로 기준                              dataset root = <repo>/data/aihub/datasets
검증                                   cd <repo>/data/aihub/datasets &&                                        sha256sum -c ../../../docs/manifests/aihub_zips.sha256
Jetson 결과                            684 OK, FAILED 0, exit 0
```

**증명한다:** 이 장비가 2026-09-07에 보유한 684개 zip의 바이트 지문. 서버로 전송한 뒤 같은
명령으로 재검증하면 **전송 중 손상·누락**을 판정할 수 있다.

**증명하지 않는다:**

- **공급자 원본과 동일하다는 것.** AI Hub가 배포한 체크섬을 확보하지 못했으므로 원본 대조가
  불가능하다. 이 manifest는 **현재 보유 파일의 지문**이며 출처 증명이 아니다.
- 압축 내용이 손상되지 않았다는 것 — 그것은 §13-2 (b)(c)의 별도 검사다.
- 데이터셋의 과학적 정합성 — §13-4의 계층 검증이 필요하다.

경로 683개에 공백이 포함되어 있다(`67.제조현장 이송장치의 ...`). `sha256sum -c`는 2-space
구분자 뒤 전체를 파일명으로 취급하므로 그대로 동작한다(Jetson에서 684 OK 확인).

### 13-4. 재추출 검증 기준 — 계층별로 분리한다

**상위 계층이 통과해도 하위 계층을 건너뛰지 않는다.** 각 계층의 결과를 따로 기록한다.

**L1 — 압축 무결성**

```
[L1-a] 684개 파일 존재
[L1-b] 684개 전부 is_zipfile + central directory 파싱
[L1-c] 684개 전부 CRC 검증(testzip) 통과
[L1-d] sha256sum -c 로 aihub_zips.sha256 684 OK   <- 전송 손상 판정
```
L1-d 실패는 전송 문제다. 재전송한다. **checksum 을 다시 생성해 맞추지 않는다.**

**L2 — 추출된 파일의 크기와 파싱 가능성**

```
[L2-a] 추출 파일 수가 member 수와 일치        Training csv/bin 각 99,476, json 99,476
                                              Validation csv/bin 각 12,394, json 12,394
[L2-b] 0바이트 파일 0건                        <- 현재 상태의 재발 여부
[L2-c] CSV 파싱: 헤더 1행 + 데이터 1행, 8개 필드,
       헤더가 NTC,PM1.0,PM2.5,PM10,CT1,CT2,CT3,CT4
[L2-d] BIN 파싱: np.load 성공, shape (120,160), dtype 확인
[L2-e] JSON 파싱: annotations[0].tagging[0].state 존재, meta_info.duration_time 존재
```
L2는 전수 검사를 권한다(파일당 수십~수백 KB). 표본 검사로 대체하면 **표본 수와 선정 방식을
기록**한다.

**L3 — CSV / BIN / JSON 대응**

```
[L3-a] basename 집합이 세 종류에서 완전히 일치 (차집합 양방향 0건)
[L3-b] 파일명 규약 준수: ^(agv|oht)\d+_\d{4}_\d{6}$
[L3-c] 파일명 timestamp 와 JSON meta_info.collection_date/collection_time 일치
[L3-d] 고아 파일 0건 (csv 있고 bin 없음 등)
```

**L4 — split / session / window 집계**

```
[L4-a] split 별 파일 수      Training 99,476 · Validation 12,394
[L4-b] 장비 집합             Training 32대(agv01-16, oht01-16) · Validation 4대(agv17,18, oht17,18)
                             교집합 0
[L4-c] session 수            gap_threshold=120 s 로 Training 303 · Validation 39
[L4-d] window 수             window 30 / step 10 으로 Training 9,313 · Validation 1,157
[L4-e] 세션 내 인접 간격      전 인접쌍 Δ=1 s, 30행 window span 29 s
```

**중요 — 숫자 일치만으로 복구 성공을 선언하지 않는다.**

- L4-c/L4-d 의 session·window 수 일치는 **원시 데이터 내용의 무결성을 입증하지 않는다.**
  세션 경계와 윈도 수는 CSV 파일명 및 시간 간격으로 정해져 CSV/BIN 내용이 비어 있어도
  그 수만 같을 수 있다. 다만 현재 `session_index.py`는 인덱스를 새로 만들 때 JSON 라벨을
  실제로 파싱하며, 라벨이 비어 있거나 잘못되면 실패한다. 기존 캐시의 생성 당시 데이터
  상태와 라벨 오염 여부는 확인되지 않았다. 이전 문서의 “현재 캐시가 0바이트 상태에서
  생성됐다”는 단정은 2026-09-10 재검토에서 철회했다.
- 따라서 **L4 통과는 L2 전수 통과를 전제로만 의미가 있다.** L2 없이 L4만 보고하는 것은
  복구 검증이 아니다.
- **실패했을 때 기대 숫자에 맞추려고 파일을 제외하지 않는다.** 예: L2-c 파싱 실패 파일을
  버려서 L4-d 를 1,157 에 맞추는 행위를 금지한다. 불일치는 불일치로 보고하고 원인을 조사한다.
- 기대값과 다른 수가 나오면 **그 수가 맞을 가능성도 함께 검토한다.** 이전 집계 자체가
  0바이트 상태에서 산출된 값이므로 절대 기준이 아니다.

### 13-5. 공식 공급자 자산의 결손

같은 0바이트 문제가 공급자 배포 코드에도 있다.

```
models/AI모델/1.모델소스코드/models/.../*.py            8개 전부 0바이트
models/AI모델/1.모델소스코드/trainers/train_manager.py   0바이트
models/AI모델/3.도커이미지/docker44_1.tar               Jetson: 17.9 GB 저장 / 내부 선언 56.8 GB -> 잘림
                                                        서버:  61 GB 완전본 존재 (2026-09-17 확인, 내부 미검증)
models/AI모델/2.AI학습모델파일/best_model.pth            11,339,490 bytes 정상
docs/AI모델_문서파일/**                                  정상 (readme, PDF, requirements)
```

그 결과 **공식 F1 averaging 방식과 보고 split 을 확정할 수 없다**(blocker B-6).
공식 아키텍처는 `best_model.pth` state_dict 로 복원했다 — Multimodal LSTM+CNN,
2,833,412 params, attention 텐서 0개. 우리 V1 과 동일하다(§6, 연구노트 #16 §3.1).

### 13-6. 서버에는 이미 온전한 사본이 있다 (2026-09-17 확인)

§13-1~13-4 는 **Jetson 사본** 기준이다. 서버 `/home/keti/factory_safety/data/aihub/` 를 확인한 결과
**0바이트 문제는 Jetson 사본에만 있었다.**

```
서버 extracted/   Training   csv 99,476 · bin 99,476 · json 99,476    0바이트 0
                  Validation csv 12,394 · bin 12,394 · json 12,394    0바이트 0
                  총 18 GB · mtime 2025-03-10 · 표본 csv 82 B, bin 153,728 B
서버 zip           684개 (Jetson 과 동일 구성, sha256 대조는 미실행)
서버 docker        61 GB 완전본
현재 위치          /mnt/data-hdd/keti_data/factory_safety/aihub (링크 경유, 연구노트 #17 §3)
```

**이것은 존재·크기 기준이다.** L2(파싱)·L3(대응)·L4(집계) 검증은 아직 하지 않았다. 숫자가 맞는다는
사실만으로 복구 완료를 선언하지 않는다(§13-4 원칙 그대로).

### 13-7. 서버 first-run 에 추가되는 단계 (2026-09-17 개정)

§7 의 STEP 1~7 은 development baseline 용이다. AI Hub 재학습에는 다음이 더 필요하다.
**서버 사본이 온전하므로 zip 전송·재추출 단계는 생략하고 검증으로 바로 간다.**

```
STEP 8   (생략) zip 전송 · 재추출 - 서버 사본 사용. Jetson zip 과의 sha256 대조는 선택
STEP 9   L1 검증  서버 zip 684개 구조·CRC (선택; extracted 를 쓰므로 필수 아님)
STEP 10  L2 검증  extracted 파일 수 · 0바이트 0건 · CSV 8필드 / BIN (120,160) / JSON state 파싱   <- 전수
STEP 11  L3 검증  basename 3종 대응 · 파일명 규약 · timestamp 일치 · 고아 0건
STEP 12  L4 검증  split · 장비 32/4 · session 303/39 · window 9,313/1,157 · Δ=1 s
STEP 13  server clone 을 작업 브랜치로 갱신 (현재 5feabb5 는 09-10 경로 수정 미포함) 후 session index 재생성
```

STEP 10 을 건너뛰고 STEP 12 만 통과했다고 재학습을 시작하지 않는다.
Jetson 사본(0바이트)은 복구 대상이 아니라 **폐기 또는 서버 사본으로 교체** 대상이다.

### 13-8. 학습 읽기는 SSD 작업 사본에서 한다 (2026-09-18)

`dataset.py` 는 window 마다 30개 tick 의 csv(82 B)+bin(153 KB) 파일을 개별로 읽는다. train 9,313
window 면 **1 epoch 에 558,780 파일 읽기**다. 무작위 소파일 읽기 실측:

```
HDD  /mnt/data-hdd   1,500쌍  29.6 s   ->     51 샘플/s,     8 MB/s   (page cache 일부 히트 가능)
SSD  /mnt/data-ssd   1,500쌍   0.11 s  -> 13,287 샘플/s, 2,044 MB/s   (rsync 직후라 cache 히트 가능)
```

두 수치 모두 냉각 상태가 아니지만 차이가 250배라 결론은 바뀌지 않는다. RAM 187 GiB 라 첫 epoch 뒤에는
전체가 캐시되지만 첫 epoch 와 캐시 축출 시 HDD 는 I/O 병목이다. 따라서:

```
HDD  /mnt/data-hdd/keti_data/factory_safety/aihub/           불변 아카이브 (zip · extracted · models · docs)
SSD  /mnt/data-ssd/keti_data/factory_safety/aihub_extracted/ 학습 읽기용 작업 사본 (459,873 파일 · 17,795,317,991 B · HDD 와 동일)
```

작업 clone 의 `data/aihub/datasets/extracted` 는 SSD 를 가리키고 나머지는 HDD 를 가리킨다.
SSD 사본이 손상되면 HDD 에서 다시 만든다. SSD 사본을 아카이브로 취급하지 않는다.

---

## 14. 관련 문서

| 문서 | 내용 |
|---|---|
| [../CLAUDE.md](../CLAUDE.md) | 환경 판별, 프로파일별 요약 |
| [../AGENTS.md](../AGENTS.md) | agent 강제 규칙 (충돌 시 우선) |
| [ENVIRONMENT_POLICY.md](ENVIRONMENT_POLICY.md) | venv / 패키지 정책 |
| [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) | 서버 audit (PENDING) |
| [JETSON_DATASET_PROTOCOL.md](JETSON_DATASET_PROTOCOL.md) | dataset 의미론, FFC 정책, CT1 no-load, sensor profile |
| [JETSON_SENSOR_COLLECTION.md](JETSON_SENSOR_COLLECTION.md) | 샘플링 주기, fresh/age_ms/stale 의미 |
| [JETSON_ENVIRONMENT.md](JETSON_ENVIRONMENT.md) | Jetson 실측 환경 |
| [DATA_PLATFORM_ARCHITECTURE.md](DATA_PLATFORM_ARCHITECTURE.md) | 2027 3-node 아키텍처 |
| [CONSORTIUM_DATA_PLATFORM_QUESTIONS.md](CONSORTIUM_DATA_PLATFORM_QUESTIONS.md) | partner 확인 질문 P01~P20 |
| [manifests/development_baseline_20260904.sha256](manifests/development_baseline_20260904.sha256) | development baseline raw 85 파일 지문 (§8) |
| [manifests/aihub_zips.sha256](manifests/aihub_zips.sha256) | AI Hub 압축 자산 684 파일 지문 (§13) |
| [RESEARCH_STATUS.md](RESEARCH_STATUS.md) | 연구 gate 현황·blocker |
| [연구노트 #16](연구노트/연구노트_16_외부검토_실측검증.md) | 외부 검토 실측 검증 상세 근거 |

---

## 15. 이 문서가 하지 않는 것

- 서버 하드웨어/소프트웨어 사양을 적지 않는다 (§9 audit 전까지 TBD)
- model schema 방향을 결정하지 않는다 (§11)
- raw transfer 방식을 확정하지 않는다 (§3)
- partner platform 사양을 가정하지 않는다 (§12)
- anomaly induction 절차를 정의하지 않는다 (consortium / robot owner 확인 필요)

확인되지 않은 것은 확인되지 않았다고 적혀 있다. 그 상태를 추측으로 덮지 않는다.
