# Data Platform Architecture (초안)

**상태: 초안. 코드 미구현. 이 문서는 설계 합의를 위한 것이며 구현을 지시하지 않는다.**

Jetson acquisition code 는 `jetson-dataset-v1-ready-2026` (`8bc5e88`) 에서 freeze 되어 있고
이 문서는 그 코드를 변경하지 않는다.

관련 문서: [`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) ·
[`JETSON_SENSOR_COLLECTION.md`](JETSON_SENSOR_COLLECTION.md) ·
[`ENVIRONMENT_POLICY.md`](ENVIRONMENT_POLICY.md)

**partner platform 확인 질문 목록**:
[`CONSORTIUM_DATA_PLATFORM_QUESTIONS.md`](CONSORTIUM_DATA_PLATFORM_QUESTIONS.md) —
이 문서의 모든 `TBD — consortium partner confirmation required` 항목은 거기에서 P01~P20 으로
추적한다.

---

## 1. 목적과 범위

이 문서는 **누가 어떤 데이터를 보유하고, 어떤 방향으로 흐르고, 어디서 추론하는지**를 고정한다.
특정 제품이나 API 를 확정하는 문서가 아니다.

2027년 상반기 국제공동과제 field deployment 를 전제로 하되, **아직 확정되지 않은 것은
확정하지 않는다.** 특히 consortium partner 가 운영할 data platform 의 사양은 전부 TBD 다.

---

## 2. Repository / execution profile

**GitHub repository 는 하나다.** 실행 역할은 그 안에서 분리된다.

```
git@github.com:mskim0114/multi_sensor_anomaly_detection.git
one repository, two execution profiles
```

| | SERVER-TRAINING | JETSON-RUNTIME |
|---|---|---|
| 코드 영역 | `src/` | `jetson_deploy/` |
| 환경 | `factory_training` | `factory_runtime` |
| 역할 | training · evaluation · dataset processing · model export | sensor acquisition · dataset collection · **edge inference** · ONNX/TensorRT runtime |

프로파일 판별과 설치 규칙은 [`ENVIRONMENT_POLICY.md`](ENVIRONMENT_POLICY.md) 를 정본으로 한다.

---

## 3. Deployment profile — 두 상황을 혼동하지 않는다

### DEVELOPMENT / 2026 (현재)

```
Korea Linux Workstation          개발 · dataset 처리 · 학습 · 평가
        +
Local Jetson development device  센서 수집 · trial 생성 · edge 추론 검증
```

현재는 **partner platform 이 존재하지 않는다.** trial 데이터는 Jetson 로컬
(`dataset/<scenario>/trial_NNN/`) 과 한국 workstation 사이에서만 다룬다.

### FIELD / 2027 (계획)

```
German Robot Test Site
  Jetson Orin Nano                 센서 수집 · trial 생성 · local spool · edge 추론
        |
        v  outbound upload
  Partner-operated DB / Data Platform      (consortium partner 운영)
        |
        v  authenticated remote access
  Korea Linux Training Workstation  dataset 처리 · 학습 · 평가 · batch 재추론 · ONNX export
```

**한국 workstation 은 독일 현장의 단일 central server 가 아니다.** 현장 데이터의 운영
저장소는 partner platform 이고, 한국 workstation 은 연구·학습용 노드다.

### 확정된 프로젝트 맥락

- 2027년 상반기, 독일 robot test site 에 Jetson Orin Nano 기반 sensor module 설치 예정
- robot / test site 는 **consortium partner 측이 운영**
- 현장에서 생성된 sensor·trial 데이터는 **그 업체가 운영하는 DB 또는 data platform** 에 업로드 예정
- 한국 연구팀은 설치 후 귀국하여 **그 platform 에 원격 접속**해 데이터를 확인할 예정

---

## 4. 세 노드의 역할

### A. Korea training workstation

현재 보유 중인 Linux workstation. `src/` + `factory_training` 이 이 역할이다.

```
development · dataset processing · training · evaluation ·
batch re-inference · model export · research analysis
```

### B. German field Jetson

2027년 상반기 독일 robot test site 설치 예정. `jetson_deploy/` + `factory_runtime` 이 이 역할이다.

```
sensor acquisition · trial dataset generation · local buffering · real-time edge inference
```

### C. Partner data platform

독일 현장 데이터를 업로드할 **consortium partner 운영** DB 또는 platform.

```
field trial data ingestion · storage · remote consortium access ·
data retrieval / export interface
```

**정확한 implementation 은 TBD 다.** 아래는 **현재 확정되지 않았으며 추측하지 않는다.**

```
TBD — consortium partner confirmation required

  제품명
  API specification
  database type
  hosting location
  authentication method
  network topology
```

---

## 5. Canonical 2027 data flow

```
[GERMAN TEST SITE]
  Sensors
    -> Jetson Orin Nano
    -> Collector
    -> Trial Dataset            (local, immutable)
    -> Edge Inference           (현장 실시간 판정)

  Jetson
    -> outbound upload
    -> Partner-operated DB / Data Platform

[KOREA]
  Partner DB / Data Platform
    -> authenticated remote access / API / export
    -> Korea Linux Workstation
    -> dataset processing
    -> training
    -> evaluation
    -> batch re-inference
    -> ONNX model export

[MODEL DELIVERY]
  Korea Workstation
    -> controlled model delivery
    -> German Jetson
    -> edge inference
```

방향은 두 개이며 **방향별로 credential 과 권한을 분리한다** (§13).

---

## 6. Inference role — edge 와 batch 를 혼동하지 않는다

```
GERMAN JETSON   real-time / online EDGE inference
  Sensors -> Collector -> model input window -> ONNX / TensorRT -> anomaly prediction
  현장 실시간 판정은 Jetson 의 책임이다.

KOREA WORKSTATION   batch inference / re-evaluation
  저장된 trial 재추론 · 새 모델과 기존 모델 비교 · validation/test evaluation ·
  논문 지표 산출 · regression evaluation
  이것은 현장 실시간 판정 경로가 아니다.
```

### Network dependency 규칙

**German Jetson 의 real-time anomaly inference 는 한국 workstation 이나 partner data
platform 의 network availability 에 의존하지 않는다.**

```
canonical      :  Sensors -> Jetson -> Edge inference
NOT canonical  :  Sensors -> Jetson -> remote server -> inference -> Jetson
```

network outage 가 발생해도 현장 edge inference 가 계속 가능한 구조를 유지한다. 원격 노드가
실시간 판정 경로에 들어가면 회선 단절이 곧 판정 중단이 된다.

---

## 7. Local data spool (Jetson) — 필수

**2027 field deployment 에서 network availability 를 항상 보장할 수 있다고 가정하지 않는다.**
따라서 Jetson architecture 에 logical local spool 을 정의한다.

```
trial completed
  -> immutable local trial directory
  -> upload attempt
  -> server checksum / integrity confirmation
  -> uploaded state
```

**upload 실패 시**

```
trial 삭제 금지
local spool 에 유지
retry 가능
```

**upload 성공을 확인하기 전에 local raw trial 을 제거하지 않는다.** 확인은 서버 측
checksum/integrity 응답으로 한다. "보냈다" 가 아니라 "받았고 무결하다" 가 기준이다.

> **현재 uploader code 는 구현하지 않는다.** spool 의 물리적 형태(상태 파일, 디렉터리 이동,
> DB) 는 partner platform 사양을 받은 뒤 정한다.

---

## 8. Upload unit

기존 원칙을 유지한다.

```
upload unit = completed trial directory
```

포함 파일:

```
experiment.json
metadata.json
scalars.csv
snapshots.jsonl
thermal_*.npz
timing_report.json
optional  ct_raw_*.npz
```

- **`trial_id` + 파일 SHA-256** 으로 향후 idempotency / integrity verification 이 가능하도록
  설계한다. 같은 내용을 다시 올려도 중복 저장되지 않아야 한다
- trial 은 완료 시점에 불변이 되므로 업로드는 **완료 후 1회**다. 실행 중 스트리밍은 범위가 아니다
- **`aborted` / `failed` trial 도 보존·업로드 가능하다.** 다만 official completed dataset 과는
  `status` 로 구분한다 ([`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) §12)

---

## 9. Partner platform integration — TBD

**partner platform 이 아직 정해지지 않았으므로 특정 REST endpoint, DB table, S3 bucket 을
만들어내지 않는다.** 대신 확인해야 할 integration requirement 만 정의한다.

아래는 전부 **`TBD — consortium partner confirmation required`** 다.

| # | 확인 항목 |
|---|---|
| 1 | upload mechanism |
| 2 | remote download / query mechanism |
| 3 | API 또는 SDK availability |
| 4 | authentication / credential mechanism |
| 5 | TLS / VPN / network requirement |
| 6 | maximum file / object size |
| 7 | trial directory / object representation |
| 8 | checksum support |
| 9 | duplicate / idempotent upload handling |
| 10 | partial upload recovery |
| 11 | retention policy |
| 12 | storage quota |
| 13 | data export capability |
| 14 | API rate limits |
| 15 | timestamp / timezone convention |
| 16 | partner-side backup responsibility |
| 17 | Korea-side access permission |
| 18 | model artifact delivery path |

이 항목들은 [`CONSORTIUM_DATA_PLATFORM_QUESTIONS.md`](CONSORTIUM_DATA_PLATFORM_QUESTIONS.md)
의 **P01~P20** 으로 추적한다. 답이 나오기 전에는 production upload / API code 를 만들지 않는다.

참고로 우리 쪽 데이터의 형태는 이미 고정되어 있으므로(§8, 아래 규모), 위 항목 중
6·7·8·9·12 는 우리 값으로 즉시 대조 가능하다.

### 데이터 규모 (실측 기반)

30분 수집 실측 43.8 MB (thermal + CT raw 포함) 기준 환산이다.

| 단위 | 크기 |
|---|---|
| 360초 trial 1건 | 약 **8.8 MB** |
| 그중 thermal NPZ | 약 87 % |
| trial 100건 | 약 0.9 GB |
| trial 500건 | 약 4.4 GB |

360초 trial 의 파일 구성: `experiment.json`, `metadata.json`, `scalars.csv`(360행),
`snapshots.jsonl`(360줄), `thermal_000000..000011.npz`(30프레임 × 12), `timing_report.json`,
선택적으로 `ct_raw_000000..000005.npz`.

**이 규모에서는 분산 스토리지 아키텍처가 필요하지 않다.** 이 숫자는 과설계를 막기 위한 것이다.

---

## 10. Korea training workstation

한국 Linux workstation 은 계속 **canonical training environment** 다.

```
SERVER-TRAINING profile
  src/
  $HOME/venvs/factory_training
```

역할:

```
partner platform 으로부터 확보한 dataset
  -> local research / training storage
  -> preprocessing
  -> dataset versioning
  -> training
  -> evaluation
  -> batch re-evaluation
  -> ONNX export
```

> **partner platform 의 production database 를 training process 가 직접 임의 수정하지 않는다.**
> 학습용으로 가져온 dataset 은 명확한 version / provenance 를 가진 **local research
> copy / cache** 로 관리한다.

`docs/SERVER_ENVIRONMENT.md` 는 현재 `PENDING SERVER ENVIRONMENT AUDIT` 상태다. 서버에서
직접 audit 한 뒤 채우며, Jetson 에서 관측한 버전을 서버 환경으로 가정하지 않는다.

---

## 11. Data authority / provenance

2027 field deployment 에서 개념적 역할 분리는 다음과 같다.

| 노드 | 데이터 권위 |
|---|---|
| **Partner platform** | field-uploaded **raw data 의 operational source** |
| **Korea workstation** | research / training **copy** + processed dataset + model artifacts |

### 불변성과 annotation 분리

- **raw data 는 immutable 원칙을 유지한다**
- **annotation 은 raw file 을 수정하지 않고 별도 annotation / version layer 에 둔다**

이후 annotation 단계에서 채워질 값

```
observed_anomaly_onset_tick
observed_recovery_tick
observed_response        (ct / thermal / pm 실측 반응)
state_label
```

은 원본 `experiment.json` 을 수정해서 넣지 않는다. 별도 레코드로 저장하고 작성자·시각·버전을
남긴다. **어느 값이 수집 시점 사실이고 어느 값이 사후 해석인지 항상 구분되어야 한다.**

### 추적 가능하게 유지할 것

```
source platform
trial_id
source checksum
imported_at
dataset_version
annotation_version
git_commit
```

이것이 있으면 "이 학습에 쓰인 데이터가 어느 platform 의 어느 trial 에서 언제 가져온
무엇인가" 를 복원할 수 있다.

---

## 12. Model lifecycle

```
German Jetson
  -> raw trial
  -> partner platform
  -> Korea workstation
  -> dataset version
  -> training
  -> evaluation
  -> model version
  -> ONNX
  -> German Jetson
  -> edge inference
```

### Model artifact 정책

- **ONNX 를 우선 canonical portable model artifact 로 사용한다**
- **TensorRT serialized engine 은 runtime environment 에 민감하다.** server 에서 빌드한
  engine 이 Jetson 에서 실행된다고 가정하지 않는다
- Jetson-target TensorRT build / deployment policy 는 **Phase 2 에서 실제 환경으로 검증한 뒤
  결정한다**

이 판단의 근거: 현재 `jetson_deploy/model/model_v2plus_fp16.trt` 는 이 보드에서 빌드된
것이고, 같은 보드에서 ORT TensorRT EP 가 첫 `run()` 에 SIGSEGV 를 낸 사례가 기록되어 있다
([`JETSON_ENVIRONMENT.md`](JETSON_ENVIRONMENT.md) §10). engine 이식성을 낙관적으로 가정하면
현장에서 곤란해진다.

### Model version provenance

model artifact 는 아래 provenance 를 연결할 수 있도록 architecture 를 정의한다.
**지금 구현하지 않는다.**

```
model_id
model_version
git_commit
dataset_version
training_config
metrics
created_at
artifact_sha256
```

Jetson 측에서는 향후 runtime metadata 에 아래를 기록할 수 있도록 확장 여지를 둔다.

```
deployed_model_id
deployed_model_version
artifact_sha256
```

**현재 Jetson code 는 수정하지 않는다.** 실제 automatic deployment / update mechanism 도
구현하지 않으며 Phase 2 architecture 항목으로만 정의한다. 현재 Jetson 의 모델 배치는 수동이며
`jetson_deploy/model/` 의 파일이 그 상태다.

---

## 13. Security boundary

**Jetson 을 public Internet 에 직접 노출하지 않는다.**

```
기본 방향
  Jetson  -> outbound upload           -> partner platform
  Korea   -> authenticated remote access -> partner platform

사용하지 않는 구조
  remote user -> Jetson direct inbound
```

- Jetson 에 포트 포워딩, DDNS, 원격 접속 노출을 설정하지 않는다
- Jetson 은 outbound 만 사용한다. 원격에서 Jetson 으로 접속하지 않는다

### Credential 분리

**하나의 credential 에 모든 권한을 주지 않는다.**

```
Jetson              upload permission            (조회·모델 다운로드 불가)
Korea researchers   read / export permission
model deployment    separate authenticated mechanism
```

partner platform 의 authentication mechanism 자체는 **TBD** 이므로(§9-4), 위는 권한 분리
원칙이며 구체적 구현 방식은 사양 확인 후 정한다.

---

## 14. External access 와 우리 자체 tool 의 위치

**2027 German deployment 에서는 partner platform 이 이미 API / viewer 를 제공할 수 있다.**
따라서 다음을 원칙으로 한다.

- partner platform 의 existing viewer / API 가 충분하면 **재사용한다**
- **부족한 기능만** 별도 research tool / API 로 보완한다
- **동일 기능을 중복 구현하지 않는다**

우리 자체 FastAPI / Web viewer 는 **production 필수 구성요소가 아니다.**

```
우리 자체 API / viewer 의 위치
  candidate / fallback / development option
```

**partner interface 를 확인하기 전에는 production implementation 을 시작하지 않는다.**

### 필요해질 경우의 후보 형태 (구현 아님)

2026 개발 단계에서 로컬 데이터를 보기 위해, 또는 partner platform 이 특정 기능을 제공하지
않을 때의 **후보** 다.

```
GET /api/v1/health
GET /api/v1/trials                      목록 + 필터 (test_mode 기본 제외)
GET /api/v1/trials/{trial_id}
GET /api/v1/trials/{trial_id}/metadata  metadata + sensor_manifest + sensor_profile
GET /api/v1/trials/{trial_id}/scalars   시계열 스칼라 (360행, 다운샘플링 불필요)
GET /api/v1/trials/{trial_id}/thermal   프레임 인덱스 / 단일 프레임 / 원본 NPZ
GET /api/v1/trials/{trial_id}/files     파일 목록 및 다운로드
```

- `scenario_id` · `phenomenon` · `display_name` 을 함께 제공한다 (§16)
- 응답에 `quality` 요약과 `protocol_compliant` 를 포함해, **FFC 로 무효화된 window 를 모르고
  학습에 쓰는 일**이 없게 한다
- thermal 은 NPZ 를 그대로 내리면 웹에서 다루기 어렵다. 인덱스 / 단일 프레임(°C 변환은
  `raw/100 - 273.15`) / 원본 다운로드를 분리한다

viewer 후보 화면도 최소로 둔다: trial 목록 · trial 상세 · sensor time-series · thermal viewer.
집계 dashboard 와 실시간 스트리밍은 범위가 아니다.

### 연구용 local storage model (Korea workstation)

**이것은 partner platform 의 저장 구조가 아니라 우리 연구 copy 의 구조다.**

```
경량 DB (SQLite 또는 PostgreSQL)   metadata / search index
   trial, sensor_unit, quality, file, import record, annotation, dataset_version

파일시스템 (또는 S3 호환)           raw file — 원본 형식 그대로
```

`snapshots.jsonl` 의 tick 단위 레코드를 관계형 DB 에 넣지 않는다. 목록·조회·다운로드에
필요하지 않다. 시계열 질의가 실제로 필요해지면 그때 별도로 검토한다.

---

## 15. Phase 와 구현 경계

```
Phase 1   dataset 확보 경로 정리 · metadata/index · provenance ·
          (필요 시) 로컬 조회 도구
Phase 2   dataset versioning · training jobs · evaluation ·
          model registry · model artifact export / delivery
Phase 3   batch re-inference jobs · model comparison ·
          prediction result management
```

Phase 1 이 끝나고 실제 dataset 이 쌓인 뒤에 Phase 2 를 시작한다. Phase 2 없이 Phase 3 를
먼저 하지 않는다.

> **German Jetson 의 real-time edge inference 는 Phase 3 서버 기능이 아니다.** 이미 별도로
> 존재하는 edge runtime 역할이며(`jetson_deploy/` + ONNX/TensorRT), 서버 phase 진행과
> 무관하게 동작한다.

### 이번 단계에서 구현하지 않는 것

```
upload agent 구현
FastAPI 구현
DB schema 구현
web viewer 구현
training 변경
Jetson collector 변경
model artifact 변경
trial 실행
```

**partner platform specification 을 받기 전에 production upload / API code 를 만들지 않는다.**

---

## 16. Scenario 식별자 — 결정 완료 (2026-09-04)

**코드의 `scenario_id` 를 변경하지 않는다.** canonical machine ID 는 다음이며, trial runner /
디렉터리 구조 / `experiment.json` / milestone `jetson-dataset-v1-ready-2026` 을 다시 바꾸지
않는다.

```
normal   overload   thermal_abnormal   dust
```

문서·API·viewer 는 phenomenon 과 display name 을 쓴다. 정본은
[`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) §1 이다.

| scenario_id (machine) | phenomenon | display (EN / KO) | primary | secondary |
|---|---|---|---|---|
| `normal` | — | Normal / 정상 | — | — |
| `overload` | `load_abnormality` | Load Abnormality / 부하 이상 | CT | NTC, FLIR |
| `thermal_abnormal` | `thermal_abnormality` | Thermal Abnormality / 열 이상 | FLIR, NTC | CT |
| `dust` | `particulate_abnormality` | Particulate Matter Abnormality / 입자상 물질 이상 | SPS30 | — |

> **`overload` 라는 scenario_id 가 실제 정격 초과 운전을 의미한다고 해석하지 않는다.**
> stable dataset identifier 일 뿐이다.

API 응답은 세 값을 함께 제공한다.

```json
{ "scenario_id": "overload",
  "phenomenon": "load_abnormality",
  "display_name": { "en": "Load Abnormality", "ko": "부하 이상" } }
```

anomaly induction procedure 와 robot 관련 항목은 전부
`TBD — consortium / robot owner confirmation required` 다
([`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) §1).

---

## 17. Open questions

### HIGH PRIORITY

추적은 [`CONSORTIUM_DATA_PLATFORM_QUESTIONS.md`](CONSORTIUM_DATA_PLATFORM_QUESTIONS.md) 에서
한다 (P01 · P02 · P03 · P09 · P10 이 먼저 확인할 다섯 가지).

| # | 항목 | 상태 |
|---|---|---|
| 1 | partner platform specification | **TBD — consortium partner confirmation required** |
| 2 | German test site outbound network availability | **TBD** |
| 3 | partner API / upload method | **TBD** |
| 4 | Korea remote access method | **TBD** |
| 5 | data retention / export policy | **TBD** |
| 6 | model delivery mechanism to German Jetson | **TBD** |
| 7 | raw data backup responsibility | **TBD** |
| 8 | platform 이 programmatic training-data export 를 허용하는지 | **TBD** |

2번과 8번은 특히 답에 따라 architecture 가 달라진다. outbound 회선이 없으면 §7 의 spool 은
"업로드 대기" 가 아니라 "오프라인 반출" 설계가 되고, 8번이 불가하면 §10 의 학습 경로 자체를
다시 짜야 한다.

### Data governance

독일 현장 데이터에 **사람을 식별할 수 있는 영상·정보가 포함되는 경우**의 data governance /
access policy 는 consortium 확인 사항으로 둔다.

**현재 FLIR 사용 자체가 personal data 를 포함한다고 단정하지 않는다.** 120×160 열화상의
식별 가능성, 현장에 사람이 등장하는지, 독일·EU 규정 적용 여부는 확인이 필요한 별개 사안이다.
확인 전에 "포함된다" 또는 "포함되지 않는다" 로 문서에 적지 않는다.

### 기타

| 항목 | 상태 |
|---|---|
| anomaly induction procedure (3종) | **TBD — 컨소시엄 협의** |
| robot payload / trajectory / speed / operating limit | **TBD — robot owner** |
| sensor mounting position, robot cell / fence 변경 | **TBD — robot owner** |
| heating method / location, particulate generation method | **TBD — 컨소시엄 협의** |
| 기관별 공개 범위 / 라이선스 | **TBD — 컨소시엄 협의** |
| Korea workstation 연구 copy 의 백업 정책 | **TBD** |
