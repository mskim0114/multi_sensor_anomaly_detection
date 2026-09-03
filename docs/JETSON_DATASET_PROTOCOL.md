# 이상상태 데이터 수집 프로토콜 (canonical)

**상태: 프로토콜 결정 확정 (2026-09-03). 수집 스크립트 `12_run_trial.py` 는 아직 미구현.**

이 문서는 실측 데이터를 모으기 *전에* 형식을 고정하기 위한 결정 기록이다. 잘못된 형식으로
모은 실측 데이터는 다시 만들 수 없으므로, 수집 시작 전에 여기 정의된 온톨로지를 따른다.

수집 계층 자체는 [`JETSON_SENSOR_COLLECTION.md`](JETSON_SENSOR_COLLECTION.md) 를 본다.

---

## 1. 세 축은 서로 독립이다

`phase`, `scenario_id`, `severity_level` 은 **서로 다른 축이며 어느 것도 학습 label 과
동일시하지 않는다.**

### phase — 실험 진행 구간

```
baseline    개입 전 관측 구간
anomaly     개입이 적용되는 구간
recovery    개입을 멈춘 뒤의 관측 구간
```

`phase` 는 **실험 절차의 서술**이다. 그 자체로 상태 label 이 아니다.

### scenario_id — 유발한 고장 유형

```
normal              개입 없음 (정상 운전 기준선 수집)
overload            과부하
thermal_abnormal    냉각 이상 / 국부 과열
dust                분진
```

### severity_level — 데이터셋 독립 ontology

```
0   normal
1   mild
2   moderate
3   severe
```

**이 값을 현재 모델의 4-class 출력과 결합하지 않는다.** 데이터셋은 모델보다 오래 살아야
하고, 모델은 재학습·교체될 수 있다. severity_level 은 데이터셋 자신의 척도다. 모델 class 로의
사상(mapping)은 학습 파이프라인 쪽 책임이며 이 문서의 범위가 아니다.

---

## 2. recovery 를 자동으로 normal 로 라벨링하지 않는다

`baseline` 구간은 `state_label = normal` 로 사용할 수 있다.

**`recovery` 전체를 자동으로 `normal` 로 라벨링하는 것은 금지한다.** recovery 초기에는
다음이 남아 있을 수 있다.

- 잔열 (thermal mass 가 식는 데 걸리는 시간)
- 잔류 분진 (공기 중 입자가 가라앉는 데 걸리는 시간)
- 전기적 settling (부하 제거 후 전류 안정화)

따라서 `recovery` 는 기본값으로 **`transition` / unlabeled 영역으로 보존**하고,
**실제 정상 복귀 기준을 만족한 이후 구간만** `state_label = normal` 로 확정한다.
그 기준의 정의와 판정은 annotation 단계에서 수행하며, 수집 시점에 추정해서 채우지 않는다.

---

## 3. intervention time 과 observed anomaly time 을 분리한다

**개입을 시작한 시각과 센서에 이상이 관측된 시각은 다르다.** 열은 늦게 오고, 분진은 확산에
시간이 걸리고, 전류는 즉시 반응한다. 이 둘을 하나의 필드로 합치면 나중에 분리할 수 없다.

`experiment.json` 에 최소 다음 네 필드를 둔다. 단위는 **master tick sequence** 다.

| 필드 | 의미 | 수집 시점 |
|---|---|---|
| `intervention_start_tick` | 조작자가 개입을 시작한 tick | 수집 시 기록 |
| `intervention_end_tick` | 조작자가 개입을 멈춘 tick | 수집 시 기록 |
| `observed_anomaly_onset_tick` | 센서에서 이상이 실제로 관측되기 시작한 tick | **null 허용** |
| `observed_recovery_tick` | 센서가 정상 범위로 복귀한 tick | **null 허용** |

`observed_*` 는 수집 시 `null` 로 두고 **annotation 단계에서 채운다.** 수집 중에 자동으로
추정해 넣지 않는다.

---

## 4. canonical trial 구성

```
baseline    90 s      tick   0 ..  89
anomaly    180 s      tick  90 .. 269
recovery    90 s      tick 270 .. 359
────────────────────────────────
total      360 s      tick   0 .. 359
```

master tick 은 1 Hz 이므로 tick 번호가 곧 경과 초다.

목적은 **30초 window 기준 trial 당 12 window 확보**다. FLIR automatic FFC 로 일부 window 가
invalid 되어도 충분한 유효 데이터가 남도록 잡은 길이다 (관측된 FFC 손실률은 30초 window
6개당 약 1개).

---

## 5. FLIR FFC 정책

**automatic FFC 를 유지한다.** 지금 manual mode 로 바꾸지 않는다.

```
frame_age_ms > 500                        ->  tick thermal invalid
30초 window 안에 thermal invalid tick 포함 ->  window quality invalid
```

- **raw data 는 삭제하지 않는다.** window 가 invalid 라는 것은 학습에서 제외한다는 뜻이며
  원본 기록은 분석용으로 남는다
- **stale frame 을 복제하거나 보간해서 정상 데이터처럼 만들지 않는다**
- FFC 손실이 실제로 dataset throughput 의 bottleneck 이 될 때 manual/scheduled FFC 를
  별도로 검토한다. 그 시점까지는 이 정책을 바꾸지 않는다

구현은 `jetson_deploy/sensors/snapshot.py` 의 `tick_quality()` / `window_quality()` 다.

---

## 6. 디렉터리 구조

```
dataset/
├── normal/
│   ├── trial_001/
│   └── trial_002/
├── overload/
│   └── trial_001/
├── thermal_abnormal/
│   └── trial_001/
└── dust/
    └── trial_001/
```

각 trial 디렉터리 내용:

```
experiment.json      실험 메타데이터 (이 문서가 정의하는 것)
metadata.json        collector 실행 설정
scalars.csv          tick 당 1행 평면 뷰
snapshots.jsonl      tick 당 1줄 전체 스냅샷
thermal_*.npz        30프레임 단위 열화상
timing_report.json   타이밍/품질 통계
```

`dataset/` 은 실측 raw 이므로 **Git 에 커밋하지 않는다** (`.gitignore` 대상).

---

## 7. `experiment.json` 필드

```
scenario_id                    normal | overload | thermal_abnormal | dust
trial_id                       trial_001 형식
severity_level                 0 | 1 | 2 | 3      (§1 ontology)
phase_ticks                    {baseline:[0,89], anomaly:[90,269], recovery:[270,359]}
intervention_start_tick        조작자 개입 시작
intervention_end_tick          조작자 개입 종료
observed_anomaly_onset_tick    null 허용, annotation 에서 채움
observed_recovery_tick         null 허용, annotation 에서 채움
start_time / end_time          UTC
operator_note                  자유 서술
equipment_condition            장비 상태 서술
collector_run_id               collector metadata 의 run_id
protocol                       {baseline_s:90, anomaly_s:180, recovery_s:90, total_s:360}
schema_version / git_commit_sha
quality_summary                유효/무효 window 수 (수집 직후 자동 기록)
```

**`state_label` 은 이 파일에 자동으로 기록하지 않는다.** phase 와 severity_level 로부터
label 을 유도하는 것은 annotation 단계의 명시적 작업이다 (§2 참조).

---

## 8. 센서 unit 편향 규칙

bring-up 중 SGP30 이 교체되어 serial 이 `000001B9391C` → `000001665DBF` 로 바뀐 사례가
있었다. **물리 센서 unit 이 scenario 와 상관되면 모델이 고장이 아니라 센서 개체를 학습한다.**

- 공식 anomaly dataset 은 **가능한 한 동일한 physical sensor set** 으로
  `normal` / `overload` / `thermal_abnormal` / `dust` 를 모두 수집한다
- 센서를 교체하면 **serial / identity 를 반드시 기록한다.** collector 가 실행마다
  `metadata.json` 의 `sensor_manifest` 에 자동 기록한다 (§9)
- **특정 scenario 와 특정 sensor unit 이 1:1 로 대응하지 않게 한다.** 예: `dust` trial 만
  새 SGP30 으로 수집하는 상황을 만들지 않는다
- 여러 unit 을 쓸 수밖에 없으면 **scenario 별로 균형 있게 배치**한다
- train / validation / test split 시 **physical sensor identity 에 의한 leakage** 를
  고려한다. 같은 unit 의 trial 이 train 과 test 로 갈리면 성능이 낙관적으로 편향될 수 있고,
  반대로 unit 이 scenario 와 완전히 겹치면 분류기가 unit 을 구분하는 것으로 충분해진다

---

## 9. 센서 identity 기록

collector 가 실행마다 `metadata.json` 의 `sensor_manifest` 에 **실제로 읽은 값만** 남긴다.

```
sensor_manifest.<sensor> = { bus, address, serial?, firmware?, chip_id?, ... }
```

**없는 serial 을 만들어내지 않는다.** 장치가 unique serial 을 제공하지 않으면 그 키는
아예 나타나지 않는다 (ADS1115 가 그런 경우다).

역할 분리:

| 위치 | 의미 |
|---|---|
| `metadata.json` 의 `sensor_manifest` | **run 시작 당시의 physical inventory** |
| `timing_report.json` 의 `sgp30.sessions[]` | **runtime initialization history** (session 별 `iaq_init` 기록) |

---

## 11. Preflight inventory (official trial 시작 조건)

**공식 dataset trial 은 collector 가 실행 중 센서 복구를 기다리는 방식으로 시작하지 않는다.**
trial 시작 전에 preflight inventory 를 수행한다.

official full-sensor trial 에서 기대되는 canonical 센서:

| 센서 | 확인 대상 |
|---|---|
| ADS1115 / NTC / CT1 | `/dev/i2c-7` `0x48` 응답 |
| SPS30 | `/dev/i2c-1` `0x69` + serial |
| SCD30 | `/dev/i2c-1` `0x61` + serial |
| SGP30 | `/dev/i2c-7` `0x58` + serial |
| BME680 | `/dev/i2c-7` `0x77` + chip_id / variant_id |
| FLIR | `/dev/video0` + USB serial |

각 장치가 **기대 주소에서 기대 identity 로 응답하는지 확인한 뒤** acquisition 을 시작한다.
unique identity 를 제공하는 장치는 그 identity 를 experiment metadata 에 고정한다.

**SGP30 처럼 run 시작 시 identity 확보가 실패하면 official trial 은 기본적으로 시작을
거부한다.** 이는 `12_run_trial.py` 의 gate 로 구현할 예정이며, **이번 커밋에는 구현되지
않았다 — 정책만 기록한다.**

`sensor_manifest` 는 run 시작 시점 inventory 이므로, 실행 중간에 복귀한 센서의 identity 는
manifest 에 남지 않고 `timing_report.sgp30.sessions[].serial` 에만 남는다(§9). official trial
에서는 중간 복귀에 의존하지 않는다.

SGP30 의 현재 하드웨어 상태는 `JETSON_ENVIRONMENT.md` §19 를 본다
(`HARDWARE_STABILITY = UNRESOLVED`).

---

## 12. 유지되는 원칙

- **CT2 / CT3 / CT4 는 `disabled`** 로 기록한다. 0 으로 채우거나 CT1 을 복제하지 않는다.
  모델이 요구하는 8채널을 맞추는 것은 이후 ModelAdapter 의 책임이다
- **SGP30 / SCD30 / BME680 은 계속 저장한다.** 올해 모델 V1 에 쓰이지 않아도, 내년에 4~6번째
  이상상태를 정의할 때 이미 장기 축적된 정상/이상 환경 데이터가 남아 있게 된다.
  수집 계층을 모델 입력과 분리한 가장 큰 이득이 이것이다
- collector 는 **관측 사실만 기록한다.** 추정·보간·정규화·라벨 유도를 하지 않는다
