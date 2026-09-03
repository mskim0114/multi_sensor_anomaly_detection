# Jetson 다중 센서 동기 수집 (SensorCollector)

모든 센서를 **모델의 시간축인 1 Hz** 로 한 스냅샷에 모으는 raw 수집 계층이다.

```bash
./jetson_deploy/run_python.sh jetson_deploy/scripts/11_collect_sensors.py --duration 120
```

이 계층은 **추론을 하지 않고 모델 입력 벡터를 바꾸지 않는다.** 모델 입력은
`src/data/config.py` 의 `[NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4]` 로 고정이며,
SGP30 / SCD30 / BME680 은 context 센서다.

---

## 1. 시간축

모델은 1 Hz × 30 tick 을 한 window 로 소비한다. 따라서 master tick 은 **정확히 1.0 초**다.

스케줄링은 **절대 monotonic deadline** (`start + n × 1.0 s`) 으로 한다.
`time.sleep(1.0)` 반복은 지연이 누적되므로 쓰지 않는다. 한 tick 의 작업이 다음 deadline 을
넘기면 지나간 deadline 을 **missed 로 계수**하고 스케줄 자체는 유지한다.

각 스냅샷에 기록되는 항목:

```
sequence  target_monotonic_ns  actual_monotonic_ns  timestamp_utc
tick_jitter_ms   tick_work_ms
```

---

## 2. 센서별 sampling cadence

| 센서 | 물리 위치 | Native 획득 | 스냅샷 rate | 모델 사용 |
|---|---|---|---|---|
| **NTC** | ADS1115 A2, `/dev/i2c-7` `0x48` | 1 Hz (tick 당 single-shot 1회) | 1 Hz | **YES** (temperature) |
| **SPS30** | `/dev/i2c-1` `0x69` | ~1 Hz (device data-ready) | 1 Hz | **YES** (PM1.0 / PM2.5 / PM10) |
| **CT1** | ADS1115 AIN0−AIN1, `/dev/i2c-7` `0x48` | 860 SPS × 0.5 s burst | 1 Hz (RMS 1개) | **YES** |
| **FLIR Lepton** | PureThermal USB (UVC) | continuous ~8.7 fps | 1 Hz (최근 프레임 1장) | **YES** (120×160) |
| SGP30 | `/dev/i2c-7` `0x58` | **1 Hz strict** | 1 Hz | NO — context |
| BME680 | `/dev/i2c-7` `0x77` | 1 Hz | 1 Hz | NO — context |
| SCD30 | `/dev/i2c-1` `0x61` | **0.5 Hz** (2 s native) | 1 Hz | NO — context |
| CT2 / CT3 / CT4 | 없음 | — | `disabled` | 스키마상 존재 |

**CT2–CT4 는 물리 front-end 가 없다.** `status="disabled"`, `values={}` 로 기록하며
**0 으로 채우거나 CT1 을 복제하지 않는다.** 모델 연결은 이 계층의 범위가 아니다.

---

## 3. `fresh` / `age_ms` / `stale` 의미

센서마다 native rate 가 다르므로, master tick 보다 느린 센서는 **같은 값을 연속 스냅샷에
반복하는 것이 정상**이다. 그 반복은 다음과 같이 표현한다.

| 필드 | 의미 |
|---|---|
| `fresh: true` | 이 tick 에서 **새 측정값**을 처음 사용했다 |
| `fresh: false` | 직전과 **같은 측정값**을 재사용했다. 센서가 아직 새 값을 내놓지 않았다 |
| `age_ms` | 그 측정값이 **획득된 시점부터 이 tick 까지 경과 시간** |
| `stale` | `age_ms` 가 임계를 넘었다 (SCD30 4.5 s, FLIR 500 ms) |

**값이 같다는 이유로 fresh 를 판단하지 않는다.** 환경이 안정되면 동일한 값이 연속으로
나오는 것이 정상이다. 판정 근거는 오직 **device 의 data-ready / 새 측정 시퀀스**다.

SCD30 실제 예 (soak test 발췌):

```
seq 4  fresh=False  age=1925 ms  CO2=752.298
seq 5  fresh=True   age= 800 ms  CO2=755.772
seq 6  fresh=False  age=1800 ms  CO2=755.772
seq 7  fresh=True   age= 665 ms  CO2=756.858
```

`status` 값은 `ok` / `warming_up` / `stale` / `error` / `disabled` 다섯 가지뿐이다.

### 3-1. SGP30 `warming_up` — 값이 아니라 session 경과시간

Sensirion SGP30 은 **성공한 `iaq_init` 이후 첫 15초가 initialization phase** 이고, 그 구간
`measure_iaq` 는 고정값 `eCO2 = 400 ppm` / `TVOC = 0 ppb` 를 반환한다. 그 구간에도 1초
cadence 는 유지된다.

**판정은 오직 경과시간으로 한다.**

```
session_elapsed_s = (now - session_init_monotonic) / 1e9
warming_up   if session_elapsed_s < 15.0
ok           otherwise
```

**값을 판정에 쓰지 않는다.** 15초 이후 실제 측정값이 다시 `400/0` 이 되어도 `ok` 를 유지한다.
초기 구현은 `eCO2 == 400 and TVOC == 0` 로 판정했는데, 라이브 값이 그 쌍에 걸릴 때마다
`warming_up ↔ ok` 가 진동했다. 실측 90초 run 에서 15초 이후 `400/0` 인 tick 이 70개 있었고
전부 `ok` 로 기록되는 것을 확인했다.

`warming_up → ok` 전이는 **initialization session 당 정확히 1회**다. `session_elapsed_s` 는
단조 증가하므로 같은 session 안에서 되돌아가는 것이 구조적으로 불가능하다.

통신 실패 후 **실제 재초기화에 성공하면 새 session** 이 열려 15초 phase 가 정상적으로 다시
시작한다. 진동과 구분할 수 있도록 스냅샷과 timing report 에 다음을 남긴다.

```
session_id            session 번호 (재초기화마다 증가)
session_init_sequence 그 session 의 iaq_init 이 성공한 tick
session_elapsed_s     그 session 시작부터의 경과시간
initialisation_count  전체 실행에서의 iaq_init 성공 횟수
sessions[]            session 별 init tick / monotonic / serial
```

---

## 4. 버스 중재

동일 I2C 버스의 동시 접근을 피한다. 각 버스에 소유자가 하나씩 있다.

```
/dev/i2c-7   master tick 스레드가 직렬로 접근
             T+0    SGP30 iaq_measure      <- 1 Hz strict, 최우선
             다음   BME680 4개 값
             다음   CT burst ~0.5 s (AIN0-AIN1 continuous, 860 SPS)
             다음   NTC 1회 (A2 single-shot) -> CT 설정 복원

/dev/i2c-1   전용 백그라운드 스레드, 250 ms 주기로 data-ready polling
             SPS30 / SCD30. 1 Hz 로 읽지 않고 4 Hz 로 polling 하는 이유는
             age_ms 를 정직하게 만들기 위해서다 (새 SCD30 샘플을 최대 250 ms 안에 포착)

USB          FLIR 은 I2C 스케줄과 무관한 별도 스레드. 카메라는 한 번만 열고
             최신 프레임을 유지. tick 마다 open/close 하지 않는다
```

**ADS1115 는 collector 내부에 소유자가 단 하나다.** 하나의 ADC 가 CT 와 NTC 를
time-multiplex 하므로, 이 collector 를 돌리는 동안 `scripts/08`(NTC) 이나
`scripts/09`(CT) 를 **동시에 실행하면 양쪽 모두 깨진다.**

SGP30 을 tick 맨 앞에 두는 이유는 다음 1 초 deadline 을 CT burst 가 침범하지 않게
하기 위해서다. 실측 tick 작업 시간은 평균 788 ms 로 1 초 예산 안에 들어온다.

---

## 5. CT 측정

tick 마다 AIN0−AIN1 hardware differential 을 PGA ±2.048 V / 860 SPS / config `0x04E3`
로 약 0.5 초 캡처한다 (실측 **431 samples @ 861.6 SPS**). 50/60 Hz 를 여러 주기 포함한다.

```
offset = mean(vdiff)
vac    = vdiff - offset
Vrms   = sqrt(mean(vac^2))
I      = Vrms / 0.68 * 400
```

결과 필드: `sample_count` `capture_duration_ms` `actual_sample_rate` `vdiff_mean`
`vdiff_min` `vdiff_max` `vrms` `current_a_nominal` `clipping`.

전류는 **`current_a_nominal`** 이다. known-current transfer calibration 이 아직 완료되지
않았으므로 `calibrated_current` 라고 부르지 않는다.

NTC 는 CT burst 직후 A2 single-ended 로 전환해 1회 측정하고(첫 변환 discard),
곧바로 CT 설정을 복원한다. 검증된 식을 그대로 쓴다.

```
R_NTC = 10000 * Vout / (3.3 - Vout)     B=3950  R0=10000  T0=298.15K
```

---

## 6. FLIR 프레임

PureThermal 스트림은 **160×122 GRAY16** 이고 **마지막 2행(120–121)이 telemetry** 다
(이 보드에서 실측 확인: 해당 행은 0 과 58744 같은 비온도 값). 앞의 120행만 사용하며
그것이 모델이 기대하는 **120×160** 이다. 온도 변환은 `raw/100 − 273.15`.

프레임은 30장 단위 NPZ 로 저장하고 스냅샷에는 참조만 남긴다. **압축/쓰기는 acquisition
critical path 밖에서 수행한다** — bounded queue 하나와 background writer 스레드 하나.
inline 압축은 30 tick 마다 tick_work 를 +176 ms 밀어올려 1000 ms 예산의 여유를 18 ms 까지
줄였다(1800 tick soak 실측). queue 는 bounded 이며, writer 가 따라오지 못하면 조용히 버리지
않고 `storage.degraded_events` 와 `dropped_chunks` 로 기록한다. 종료 시 queue drain →
flush → writer join 을 완료한다.

```
thermal_000000.npz   frames (30,120,160) uint16 + sequences
스냅샷: "thermal_chunk": "thermal_000000.npz", "thermal_index": 7
```

Lepton 은 주기적으로 **FFC(flat-field correction)** 셔터 동작 때문에 잠깐 프레임을
멈춘다. 이때 `age_ms` 가 커지고 `status=stale` 로 기록되지만 **collector 는 죽지 않는다.**
120초 soak 에서 2/120 tick 이 여기에 해당했다 (708 ms, 1713 ms).

---

## 7. 실패 정책

센서 하나가 실패해도 전체 수집을 중단하지 않는다. 해당 센서만 `status="error"` 로
기록하고 `consecutive_errors` 를 센다. SGP30/BME680 은 10 tick 마다 재초기화를 시도하므로
**센서가 돌아오면 자동으로 `ok` 로 복귀**한다.

**startup fatal 은 셋뿐이다**: ADS1115 접근 불가, 출력 디렉터리 생성 불가, 스키마 초기화 실패.
나머지는 degraded 수집으로 계속한다.

`Ctrl+C` 는 현재 tick 을 마치고 센서를 정리한 뒤 파일을 flush 하고 **exit 130** 으로 끝난다.

---

## 8. 출력

```
jetson_deploy/results/sensor_collection/<RUN_ID>/     RUN_ID = UTC 타임스탬프
    metadata.json        실행 설정, git SHA, 버스 맵, native rate, 모델 채널 명시
    scalars.csv          tick 당 1행 평면 뷰 (빠른 확인/플롯용)
    snapshots.jsonl      tick 당 1줄 전체 스냅샷
    thermal_000000.npz   30프레임 단위 (120,160) uint16
    ct_raw_000000.npz    --save-ct-raw 일 때만, 60 tick 단위 원시 파형
    timing_report.json   아래 통계
```

`jetson_deploy/results/` 는 `.gitignore` 대상이다.

### CLI

```
--duration N        수집 초. 0 이면 Ctrl+C 까지 연속
--out-dir PATH
--save-ct-raw       CT 원시 파형 NPZ 보존 (기본 off, scalar RMS 는 항상 저장)
--no-thermal-save   thermal 프레임 저장 안 함 (scalar 는 계속 기록)
--ct-burst SEC      tick 당 CT 캡처 길이 (기본 0.5)
--thermal-device    기본 /dev/video0
```

---

## 9. 실측 timing (2026-09-03, 120초 soak)

```
master tick   snapshots 120/120,  missed 0
              period ms  mean 1000.002  p50 999.998  p95 1000.127  max 1000.25
              abs jitter ms  p95 0.311  max 0.538
              tick work ms   mean 787.9  max 975.5

SGP30         120 measurements,  interval mean 999.378 ms  p95 1000.139  max 1000.485
              >1.1 s 위반 0건,  >1.5 s 위반 0건
              warming_up 20 tick (eCO2 400 / TVOC 0) 후 ok 로 전이
SPS30         fresh 97/120,  age mean 654.7 ms  max 1064.2 ms,  error 0
SCD30         fresh 58 / non-fresh 62,  fresh interval mean 2120.9 ms  max 2654.8 ms
              age max 2116.9 ms,  stale 0 tick,  error 0
BME680        120/120 ok
NTC           120/120 ok
CT1           120 bursts,  samples/burst mean 431.0 (428–431)
              actual SPS mean 861.58 (855.7–861.7),  clipping 0
FLIR          120/120 tick 에 프레임 있음,  age mean 74.9 ms  p95 115.9 ms  max 1713.0 ms
              frames 1045 (~8.7 fps),  shape failure 0,  stale 2 tick (FFC)
```

---

## 10. 배선 변경 절차 (canonical)

**40-pin I2C / SPI 센서 배선 변경과 커넥터 재장착은 반드시 아래 순서로 한다.**

```
1. collector 종료
2. Jetson shutdown          (sudo shutdown -h now)
3. external power 제거       (어댑터 분리, 전원 LED 소등 확인)
4. sensor connector 변경
5. 배선 / 방향 확인          (실크 첫 핀 위치를 눈으로)
6. power on
```

**live I2C hot-plug / reseat 를 canonical procedure 로 사용하지 않는다.**

근거: 활선 상태에서 SGP30 커넥터를 만졌을 때 같은 `/dev/i2c-7` 의 ADS1115 에
`OSError Errno 121` transient 가 실제로 발생했다 (1800 tick soak, tick 75). ADS1115 는
모델 입력 채널(NTC, CT1)을 담당하므로 그 순간의 데이터가 오염된다.

한 번의 hot-plug 에서 문제가 없었다는 사실은 **hot-plug 가 안전하다는 뜻이 아니다.**
90초 test 에서 재장착 후 ADS1115 오류가 0건이었지만, 그것은 재현된 안전성이 아니라
한 번의 관측일 뿐이다.

---

## 11. 주의

- 이 collector 가 도는 동안 `scripts/08` / `scripts/09` 를 동시에 실행하지 않는다
- 센서 커넥터 접점이 이 프로젝트의 반복 실패 원인이었다. 수집 전에 이 스크립트를 5초만
  돌려 스냅샷의 `status` 로 전 센서를 확인한다. **I2C 전 주소 bare-read 스캔은 쓰지 않는다** —
  Sensirion 장치(SGP30/SPS30/SCD30)에는 프로토콜 위반이고 상태를 망가뜨릴 수 있다
  (`JETSON_SPI_BME680_SETUP.md` §10-7)

  ```bash
  ./jetson_deploy/run_python.sh jetson_deploy/scripts/11_collect_sensors.py --duration 5
  ```
- BME680 온도는 gas 히터 자체 발열로 실온보다 높다. 모델 온도 채널은 NTC 다
- SGP30 `warming_up` 은 값이 아니라 **initialization session 경과시간**으로 판정한다 — §3-1
- **SGP30 은 I2C 버스에서 간헐적으로 완전히 사라지는 현상이 반복 관측되었다**
  (`HARDWARE_STABILITY = UNRESOLVED`, `JETSON_ENVIRONMENT.md` §19). 부재 시 collector 는
  degraded mode 로 계속 동작하고 serial 을 만들어내지 않는다. **공식 dataset 수집을 이
  상태에서 조용히 시작하지 않는다** — `JETSON_DATASET_PROTOCOL.md` §10

관련 문서: [`JETSON_ENVIRONMENT.md`](JETSON_ENVIRONMENT.md),
[`JETSON_SPI_BME680_SETUP.md`](JETSON_SPI_BME680_SETUP.md), [`../AGENTS.md`](../AGENTS.md)
