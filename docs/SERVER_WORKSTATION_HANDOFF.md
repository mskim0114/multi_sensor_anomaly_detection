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

**Jetson 에서 서버 사양을 추측해 채우지 않았다.** `docs/SERVER_ENVIRONMENT.md` 는 의도적으로
`PENDING SERVER ENVIRONMENT AUDIT` 상태다. 서버 agent 의 첫 작업은 실제 workstation 에서
아래를 확인하고 그 문서를 채우는 것이다.

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
- `src/` 와 `configs/data_config.yaml` 이 `/home/keti/factory_safety/...` 를 하드코딩하고
  있으나 Jetson 의 실제 경로는 `/home/keti/projects/factory_safety` 였다. 서버에서도 같은
  문제가 나올 수 있다. **아직 수정하지 않았다** — audit 시 함께 확인한다
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

## 13. 관련 문서

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

---

## 14. 이 문서가 하지 않는 것

- 서버 하드웨어/소프트웨어 사양을 적지 않는다 (§9 audit 전까지 TBD)
- model schema 방향을 결정하지 않는다 (§11)
- raw transfer 방식을 확정하지 않는다 (§3)
- partner platform 사양을 가정하지 않는다 (§12)
- anomaly induction 절차를 정의하지 않는다 (consortium / robot owner 확인 필요)

확인되지 않은 것은 확인되지 않았다고 적혀 있다. 그 상태를 추측으로 덮지 않는다.
