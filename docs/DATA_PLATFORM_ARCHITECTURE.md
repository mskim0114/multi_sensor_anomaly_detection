# Data Platform Architecture (V1 초안)

**상태: 초안. 코드 미구현. 이 문서는 설계 합의를 위한 것이며 구현을 지시하지 않는다.**

Jetson acquisition code 는 `jetson-dataset-v1-ready-2026` (`8bc5e88`) 에서 freeze 되어 있고
이 문서의 내용은 그 코드를 변경하지 않는다.

관련 문서: [`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) ·
[`JETSON_SENSOR_COLLECTION.md`](JETSON_SENSOR_COLLECTION.md)

---

## 1. 목적

수집한 trial 데이터를 **컨소시엄 참여기관이 안전하게 조회**할 수 있게 하고, 동시에
**Jetson 을 외부에 노출하지 않는 것**이 목적이다.

```
Sensors -> Jetson Collector -> Dataset -> Central Data Storage -> REST API -> Web Viewer
```

V1 은 **read-only** 다. 외부기관이 데이터를 쓰거나 지우거나 trial 을 실행할 수 없다.

### 전제 (현재 확인된 사실만)

- fenced robot test area, 약 2평
- robot arm 1대 설치
- **robot 은 컨소시엄 참여기관 소유·운영 설비다**
- 이 환경에서 Jetson multi-sensor acquisition system 을 사용한다

**TBD — consortium / robot owner confirmation required**: robot payload, trajectory, speed,
operating limit, sensor mounting location, robot cell modification, fence modification,
heating location, particulate generation method. 이 문서는 이 항목들을 가정하지 않는다.

### 2026 anomaly scope (physical phenomenon 수준)

**machine ID 와 표시 이름을 분리한다.** `scenario_id` 는 안정적인 데이터셋 식별자이며
변경하지 않는다. 문서·API·Web viewer 는 phenomenon 과 display name 을 쓴다.

| scenario_id (machine) | phenomenon | display (EN / KO) | primary | secondary |
|---|---|---|---|---|
| `normal` | — | Normal / 정상 | — | — |
| `overload` | `load_abnormality` | Load Abnormality / 부하 이상 | CT | NTC, FLIR |
| `thermal_abnormal` | `thermal_abnormality` | Thermal Abnormality / 열 이상 | FLIR, NTC | CT |
| `dust` | `particulate_abnormality` | Particulate Matter Abnormality / 입자상 물질 이상 | SPS30 | — |

> **`overload` 라는 scenario_id 가 실제 정격 초과 운전을 의미한다고 해석하지 않는다.**
> stable dataset identifier 일 뿐이다.

API 응답은 세 값을 함께 제공한다. 그래야 외부기관이 machine ID 로 필터링하면서 화면에는
읽을 수 있는 이름을 보여줄 수 있다.

```json
{ "scenario_id": "overload",
  "phenomenon": "load_abnormality",
  "display_name": { "en": "Load Abnormality", "ko": "부하 이상" } }
```

구체적인 anomaly induction procedure 는 컨소시엄 협의 후 결정한다. 정의와 TBD 목록은
[`JETSON_DATASET_PROTOCOL.md`](JETSON_DATASET_PROTOCOL.md) §1 을 정본으로 한다.

> 기존 문서에 기술된 손난로, 온수병, 히터 등의 이상상태 유발 방식은 **예시이며 현재
> 확정된 현장 실험 절차가 아니다.** 실제 intervention 방법은 로봇 소유·운영기관 및
> 컨소시엄 협의 후 결정한다.

---

## 2. Jetson → central server data flow

**Jetson 은 언제나 outbound 로만 통신한다.** 중앙 서버가 Jetson 으로 접속하지 않는다.

```
[robot test area]                          [central data server]
  Jetson                                     ingest endpoint (HTTPS)
   trial 종료                                     |
   status = completed | aborted | failed          |
        |                                          |
   upload agent  --- HTTPS POST (outbound) ------->|  검증 -> object store + DB index
        |                                          |
   업로드 성공 기록                                 |
```

### 업로드 단위와 시점

trial 은 완료 시점에 **불변**이 된다. 따라서 업로드 단위는 **trial 디렉터리 전체**이며,
`experiment.json` 의 `status` 가 종료 상태(`completed` / `aborted` / `failed`)가 된 뒤에
한 번 올린다. 실행 중 스트리밍은 V1 범위가 아니다.

- **멱등성**: `trial_id` + `scenario_id` + 파일 `sha256` 로 판단한다. 같은 내용을 다시
  올려도 중복 저장하지 않는다
- **무결성**: 업로드 시 파일별 `sha256` 을 함께 보내고 서버가 재계산해 대조한다
- **재시도**: 네트워크 실패 시 Jetson 로컬 `dataset/` 은 그대로 남아 있으므로 재시도로 족하다.
  Jetson 은 업로드 성공 여부를 로컬에 기록하고 미전송 trial 을 다시 시도한다
- **부분 trial 도 올린다**: `aborted` / `failed` trial 을 버리지 않는다
  (`JETSON_DATASET_PROTOCOL.md` §12 partial data policy). 다만 official completed count 에는
  포함하지 않으며 API 는 `status` 로 구분해 제공한다

### 데이터 규모 (실측 기반)

30분 수집 실측 43.8 MB (thermal + CT raw 포함) 기준으로 환산한다.

| 단위 | 크기 |
|---|---|
| 360초 trial 1건 | 약 **8.8 MB** |
| 그중 thermal NPZ | 약 87 % |
| trial 100건 | 약 0.9 GB |
| trial 500건 | 약 4.4 GB |

360초 trial 의 파일 구성: `experiment.json`, `metadata.json`, `scalars.csv`(360행),
`snapshots.jsonl`(360줄), `thermal_000000..000011.npz`(30프레임 × 12), `timing_report.json`,
선택적으로 `ct_raw_000000..000005.npz`.

**V1 규모에서는 특별한 스토리지 기술이 필요하지 않다.** 이 숫자는 과설계를 막기 위한 것이다.
**V1 에서 분산 스토리지 아키텍처를 도입하지 않는다. single central server 로 시작한다.**

---

## 3. Storage model

**raw sensor data 를 relational DB 에 넣지 않는다.**

```
PostgreSQL  (또는 동등한 경량 DB)      metadata / search index
   trial, sensor_manifest, quality, phase_plan, upload record, annotation

Object store (filesystem 또는 S3-compatible)   raw file
   dataset/<scenario>/<trial_id>/<원본 파일 그대로>
```

현재 dataset format 을 **변형 없이 그대로 수용**한다. 서버가 파일을 재가공해 저장하지 않는다.
원본을 그대로 보관하고, 조회 편의를 위한 파생물은 필요할 때 생성한다.

### DB 에 색인할 것 (조회·필터에 필요한 것만)

```
trial          trial_id, scenario_id, severity_level, status, protocol_compliant,
               test_mode, started_at, completed_at, planned_duration_s,
               git_commit, sensor_profile_name, upload_time, storage_prefix
sensor_unit    trial_id, sensor, bus, address, serial, firmware, chip_id
quality        trial_id, total/valid/invalid ticks, window 통계, invalid_reason_counts,
               missed_ticks, writer drop/error, 센서별 error count
file           trial_id, filename, bytes, sha256, content_type
```

`snapshots.jsonl` 의 tick 단위 레코드는 DB 에 넣지 않는다. 360 tick × trial 수만큼의 행을
관계형 DB 에 넣는 것은 V1 목적(목록·조회·다운로드)에 필요하지 않다. 시계열 질의가 실제로
필요해지면 그때 별도로 검토한다.

### 불변성과 annotation 분리 — 중요

**업로드된 raw 는 불변으로 다룬다.** 이후 annotation 단계에서 채워질

```
observed_anomaly_onset_tick
observed_recovery_tick
observed_response (ct/thermal/pm 실측 반응)
state_label
```

는 **원본 `experiment.json` 을 수정해서 넣지 않는다.** DB 의 별도 annotation 레코드로 저장하고
작성자·시각·버전을 남긴다. API 는 원본과 annotation 을 합쳐 보여줄 수 있지만, 어느 값이 수집
시점 사실이고 어느 값이 사후 해석인지 항상 구분되어야 한다.

이것을 지키지 않으면 나중에 "이 값이 측정된 것인가 판단된 것인가" 를 복원할 수 없다.

---

## 4. REST API (V1, read-only)

```
GET /api/v1/health

GET /api/v1/trials                          목록 + 필터/페이지네이션
GET /api/v1/trials/{trial_id}               요약

GET /api/v1/trials/{trial_id}/metadata      metadata.json + sensor_manifest + sensor_profile
GET /api/v1/trials/{trial_id}/scalars       시계열 스칼라
GET /api/v1/trials/{trial_id}/thermal       thermal 프레임 접근
GET /api/v1/trials/{trial_id}/files         파일 목록 및 다운로드
```

### 설계 노트

- **`/trials`** 필터: `scenario_id`, `severity_level`, `status`, `protocol_compliant`,
  `test_mode`, 기간. **기본적으로 `test_mode=true` 는 제외**한다. smoke 데이터가 조회 결과에
  섞이면 안 된다
- **`/scalars`** 는 `scalars.csv` 를 JSON 으로 제공한다. 360행이므로 V1 에서 다운샘플링이
  필요 없다. 컬럼 선택(`?fields=ct1_vrms,ntc_temperature_c`)만 지원하면 충분하다
- **`/thermal`** 은 NPZ 를 그대로 내리면 웹에서 쓰기 어렵다. 세 가지를 분리한다
  ```
  GET .../thermal                 프레임 인덱스 (chunk, index, tick, min/max/mean °C)
  GET .../thermal/{tick}          해당 tick 1프레임 (PNG 또는 JSON 배열)
  GET .../files/thermal_000000.npz  원본 그대로 다운로드
  ```
  °C 변환은 `raw/100 - 273.15` 이며 서버가 수행한다
- **`/files`** 는 파일 목록(이름·크기·sha256)과 개별 다운로드를 제공한다. trial 전체를
  묶어 내려받는 archive 는 편의 기능이며 V1 필수는 아니다
- 모든 응답에 `quality` 요약과 `protocol_compliant` 를 포함해, 사용자가 **FFC 로 무효화된
  window 를 모르고 학습에 쓰는 일**이 없게 한다

---

## 5. Authentication

- **API key 또는 JWT.** V1 은 API key 로 시작해도 충분하며, 조직·사용자 구분이 필요해지면
  JWT 로 확장한다
- **외부기관별 credential 분리.** 기관 단위로 발급·폐기·회전이 가능해야 한다.
  한 기관의 키를 폐기해도 다른 기관이 영향받지 않아야 한다
- 최소 권한: V1 credential 은 **read-only**. 쓰기 권한은 Jetson upload agent 전용의 별도
  credential 로 분리하고, 그 credential 로는 조회 API 를 쓸 수 없게 한다
- 접근 로그: 어느 기관이 어느 trial 을 언제 조회했는지 남긴다
- **TBD**: 기관별로 볼 수 있는 trial 범위를 제한할 필요가 있는지 (전체 공개인지, scenario
  단위인지) — 컨소시엄 협의 필요

---

## 6. Web viewer (V1)

과도한 dashboard 를 만들지 않는다. 화면 4개.

1. **Dataset / Trial list** — scenario·severity·status·기간 필터, `test_mode` 기본 제외
2. **Trial detail** — experiment metadata, sensor manifest(serial 포함), phase plan,
   quality summary, 파일 목록
3. **Sensor time-series viewer** — `scalars.csv` 기반. NTC / CT1 / PM / CO2 / BME680.
   phase 경계와 invalid tick 을 시간축에 표시
4. **Thermal viewer** — tick 슬라이더로 프레임 이동, min/max/mean °C 표시

집계 통계, 사용자 정의 대시보드, 실시간 스트리밍은 V1 범위가 아니다.

---

## 7. External consortium access

```
컨소시엄 기관 --- HTTPS ---> 중앙 데이터 서버 API / Web viewer
```

- 외부기관은 **중앙 서버만** 접근한다. Jetson 주소를 알 필요도, 알아서도 안 된다
- 기관별 credential 로 인증하고 접근 로그를 남긴다
- V1 은 read-only 이므로 외부기관이 데이터를 수정·삭제하거나 trial 을 실행할 수 없다
- **TBD**: 공개 범위, 라이선스/이용 조건, 개인정보 해당 여부 — 컨소시엄 협의 필요

---

## 8. Security boundary

**가장 중요한 규칙: Jetson device 를 public Internet 에 직접 노출하지 않는다.**

```
  robot test area (신뢰 구역)          |   중앙 서버 (DMZ/클라우드)      |  외부기관
  ----------------------------------- | ------------------------------ | -----------
  Jetson                              |  HTTPS ingest (인증 필요)       |
   - inbound port 개방 없음            |  HTTPS API (인증 필요)          |  read-only
   - outbound HTTPS 만 사용            |  object store (비공개)          |  credential
   - 외부에서 접속 불가                 |  DB (비공개)                    |
```

- Jetson 에 포트 포워딩, DDNS, 원격 접속 노출을 설정하지 않는다
- Jetson 의 upload credential 은 **쓰기 전용**이며 조회 API 에 쓸 수 없다. 유출되어도
  데이터 열람으로 이어지지 않는다
- 중앙 서버는 TLS 종단, 인증, 접근 로그를 담당한다
- object store 는 서버를 통해서만 접근한다. 버킷/디렉터리를 공개하지 않는다
- **TBD**: 사내망/전용망 여부, 방화벽 정책, 기관 IP 제한 가능 여부

---

## 9. Deployment options

| 옵션 | 구성 | 적합한 경우 |
|---|---|---|
| **A. 단일 서버** | 1 VM 에 API + PostgreSQL + 파일시스템 + 정적 viewer | V1 규모(수 GB)에 충분. 가장 단순 |
| **B. 관리형 분리** | 관리형 PostgreSQL + S3 호환 object storage + API 컨테이너 | 운영 부담을 낮추고 확장 여지를 둘 때 |
| **C. 온프레미스** | 기관 내부 서버 + NAS | 데이터 반출 제약이 있을 때 |

V1 데이터 규모가 작으므로 **A 로 시작해 필요 시 B 로 옮기는 것**을 권한다. 파일 접근을
object-store 추상화 뒤에 두면 A → B 이전이 저렴해진다.

**TBD**: 호스팅 주체, 데이터 보관 위치 제약, 백업 정책 — 컨소시엄 협의 필요.

---

## 10. 서버의 세 가지 용도와 구현 순서

서버는 최종적으로 아래 세 용도를 모두 지원할 수 있도록 **architecture 만 열어둔다.**
구현은 단계적으로 한다.

| 용도 | 내용 |
|---|---|
| **A. Dataset storage / external viewing** | trial 보관, 색인, read-only API, Web viewer, 컨소시엄 접근 |
| **B. Server-side training** | 서버에 축적된 dataset 으로 학습 job 실행, 산출 모델·지표 보관 |
| **C. Server-side batch / online inference** | 저장된 trial 에 대한 batch 추론, 또는 온라인 추론 API |

열어둔다는 것의 의미: object store 를 추상화 뒤에 두고, trial 을 불변 단위로 다루며,
DB 에 학습·추론 산출물을 붙일 자리를 남긴다는 뜻이다. **B/C 를 위한 코드를 지금 만들지
않는다.**

```
Phase 1   storage + metadata + read-only API + viewer      <- 현재 계획 범위
Phase 2   training job
Phase 3   inference job / API
```

Phase 1 이 끝나고 실제 dataset 이 쌓인 뒤에 Phase 2 를 시작한다. Phase 2 없이 Phase 3 를
먼저 하지 않는다.

---

## 11. MVP scope (Phase 1)

**포함**

- Jetson upload agent (outbound, 완료된 trial 디렉터리 업로드, sha256 검증, 재시도)
- ingest endpoint + 무결성 검증 + 멱등 저장
- DB 색인: trial / sensor_unit / quality / file
- read-only API: `health`, `trials`, `trials/{id}`, `metadata`, `scalars`, `thermal`, `files`
- API key 인증, 기관별 credential 분리, 접근 로그
- Web viewer 4화면
- `test_mode` trial 기본 제외

**미포함 (V1 아님)**

- 실시간 스트리밍, 원격 trial 실행/제어
- annotation 편집 UI (스키마는 §3 에서 미리 분리해 두되 구현은 이후)
- 모델 추론·학습 파이프라인 연동
- 사용자 정의 dashboard, 집계 분석
- 외부기관 쓰기 권한

---

## 12. 해소해야 할 항목

| 항목 | 상태 |
|---|---|
| anomaly induction procedure (3종) | **TBD — 컨소시엄 협의** |
| robot payload / trajectory / speed / operating limit | **TBD — robot owner** |
| sensor mounting location, robot cell / fence 변경 | **TBD — robot owner** |
| heating location, particulate generation method | **TBD — 컨소시엄 협의** |
| scenario_id 명칭 | **결정 완료** — 아래 |
| 기관별 공개 범위 / 라이선스 | **TBD — 컨소시엄 협의** |
| 호스팅 주체 / 보관 위치 / 백업 | **TBD** |

### scenario_id 명칭 — 결정 완료 (2026-09-04)

**코드의 `scenario_id` 를 변경하지 않는다.** canonical machine ID 는 다음이며, trial runner /
디렉터리 구조 / `experiment.json` / milestone `jetson-dataset-v1-ready-2026` 를 다시 바꾸지
않는다.

```
normal   overload   thermal_abnormal   dust
```

연구 문서와 API 에서는 별도의 phenomenon / display name 을 쓴다 (§1). 두 층을 분리했으므로
표기를 바꾸기 위해 이미 수집된 데이터의 식별자를 건드릴 필요가 없다.
