# 동기 센서 수집 구현·검증 — 2026-09-17

## 구현 판단

현재 30초 모델 입력창에 맞춰 공통 기록 주기 1 Hz와 열화상 1장/초 저장을 유지한다.
카메라는 연속으로 읽으며 최신 원본 uint16 프레임을 선택한다. CT는 ADC 860 SPS 설정으로
0.5초 burst를 수집하고 RMS를 tick에 기록한다. SCD30은 native 주기를 유지하고 fresh/age로
중복 관측을 구분한다. 소프트웨어 시간축 정렬이며 동시 하드웨어 trigger를 뜻하지 않는다.
1초 미만의 사건을 포착하는 데 충분한지는 실제 사건 데이터로 추가 검증해야 한다.

## 변경

- `jetson_deploy/collect.sh start|stop|status|run`: 전용 runtime의 Python 관리 CLI.
  백그라운드 실행은 SSH 종료 후에도 유지되며 `stop`은 SIGTERM 뒤 저장·검증 완료를 기다린다.
- `sensors/runtime.py`: checkout과 출력 경로에 독립적인 사용자별 flock,
  PID/시작시간/부팅 ID 및 pidfd 기반 종료. 개별 진단 06~10과 수집기 11/12가 잠금을 공유한다.
- `sensors/collector.py`: 최초 오류 기록·정지, 자동 retry 제거, 부분 초기화 정리,
  bounded/idempotent 종료, 원자적 NPZ 저장, CT/NTC 측정 구간과 tick/수신시각 기록.
- `scripts/11_collect_sensors.py`: 기본 SGP30 비활성 v1 프로파일, 필수 환경 metadata,
  시작부터 최종 검증까지 신호 처리와 cleanup. 저장 또는 수집 실패는 exit 1.
- `sensors/validation.py`: JSONL/CSV sequence와 행 수, collector 기록 수,
  thermal NPZ dtype·shape·index·저장 sequence를 최종 검사한다.
- 기존 schema 2, 모델 채널, 500 ms 품질 기준을 유지한다. stale 원본도 해당 tick sequence로
  저장하여 기존 `src/data/window_builder.py`와 호환된다. 이전 프레임을 fresh로 위장하지 않는다.

기본 수집에는 NTC, CT1, SPS30, SCD30, BME680, FLIR가 포함된다. SGP30은 기존 하드웨어
불안정 상태 때문에 기본 비활성이고 `--enable-sgp30`으로 명시적으로 선택할 수 있다.
CT2~4는 연결이 없어 disabled로 남긴다. 이 CLI는 추론이나 공식 anomaly 라벨을 생성하지 않는다.

## 교차 검토

Spark 호출은 현재 ChatGPT 계정에서 해당 모델이 지원되지 않아 실패했다. 대체 탐색 에이전트가
수집 경로를 조사했고 Sol이 설계·회귀 위험을 검토하고 collector 변경을 담당했다.
별도 작업자가 진단 CLI 잠금과 시작/종료 경쟁 조건을 검토했다. 메인 에이전트가 결과를 통합했다.

검토 중 발견한 시작 중 stop 차단, stopping 상태의 시작 성공 오판, 최종 파일 검증 도중
신호 핸들러 조기 해제, 저장 행 수 누락 검증을 수정하고 회귀 테스트로 확인했다.

## 테스트

추가한 하드웨어 독립 테스트 22개 통과:

- collector 13개: 최초 오류·errno, retry 방지, 부분 startup cleanup, timeout 소유권,
  원자적 부분 청크, thermal tick sequence/fresh, CT/NTC timestamp, age.
- 제어/파일 검사 7개: 실제 자식 프로세스와 신호를 이용한 시작·중복·종료,
  PID 재사용 방지, stale 상태, 인자 검증, CSV/NPZ 누락 및 검증 중 SIGTERM.
- 시작 경쟁 2개: starting→stopping→failed 및 handler 설치 전 stop.

기존 `12_run_trial.py --self-test`와 `git diff --check`도 통과했다.
최종 전체 unittest는 50개 중 **46 PASS, 4 SKIP**이며, skip은 Jetson runtime에
PyTorch가 없는 서버 학습 관련 조건부 테스트다.
전체 unittest 및 개별 실행 로그는 `/home/keti/review_runs/20260917_sensor_collection/`에 보관한다.

## 실제 Jetson 실행

2026-09-17 UTC 01:59:47 시작, 02:00:36 정상 종료. 60초 상한으로 시작한 뒤 `stop`을 호출했다.
센서나 OS의 영구 설정·배선·보정값 변경은 하지 않았다. dirty checkout의 기능 검증이며
논문용 성능 측정이 아니다.

```bash
./jetson_deploy/collect.sh start --duration 60 --save-ct-raw \
  --out-dir /home/keti/review_runs/20260917_sensor_collection/live
./jetson_deploy/collect.sh stop
```

| 확인 항목 | 결과 |
|---|---|
| 최종 상태 / 관리 stop 종료 코드 | stopped / 0; 실제 수집 프로세스 종료 확인 |
| JSONL / CSV / 열화상 참조 | 각 45, sequence 0~44 일치 |
| 열화상 원본 | uint16, 120×160, 30장 + 종료 시 15장 |
| CT 원시 파형 | 45 burst, 코드 19,395개; lengths 합·sequence 일치 |
| 누락 tick / 통신 오류 / 저장 오류 | 모두 0 |
| 평균 master tick 간격 | 약 1초 (1000.004 ms) |
| FLIR stale | 2 tick, 최대 age 1774.17 ms; 원본 보존 및 invalid 표시 |
| 30초 품질 window | 1개 평가, stale을 포함하여 invalid 1개 |
| SCD30 새 값 / 이전 값 관측 | 22 / 23 tick |
| 종료 후 수집 프로세스 | 실행 중 아님 |

**파일 저장 성공과 학습 품질 통과는 별개다.** 카메라 지연의 원인이 FFC인지 이 실행에서
직접 검출한 것은 아니다. 전체 모델/이상 탐지 정확도나 장시간 무중단 운용은 이 시험으로
입증하지 않는다. 기존 데이터셋이나 서버 전처리 코드는 변경하지 않았다.

원본 결과:
`/home/keti/review_runs/20260917_sensor_collection/live/20260917T015947.494403Z/`
(`metadata.json`, `timing_report.json`, `control.json`, 원시 파일).
추가 배열 검사는 상위 폴더의 `live-validation.json`에 기록했다.

일상 사용법은 [JETSON_SENSOR_COLLECTION.md](JETSON_SENSOR_COLLECTION.md)의 첫 절을 따른다.
