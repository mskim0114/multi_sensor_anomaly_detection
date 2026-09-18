# Jetson 로컬 센서 대시보드

Jetson 브라우저에서 센서 수집을 시작·중지하고 실측값과 열화상을 확인한다.
Python 표준 라이브러리와 기존 NumPy만 사용하며 별도 패키지 설치가 필요 없다.

## 실행

```bash
cd /home/keti/projects/factory_safety
./jetson_deploy/collect.sh dashboard
```

터미널에 출력되는 **토큰이 포함된 전체 주소**를 Jetson 브라우저에 붙여 넣는다.
기본 주소는 `http://127.0.0.1:8765/#token=...`이다. 화면에서 **수집 시작**을 누르고,
끝낼 때 **중지**를 누른다. 서버 실행만으로 센서가 켜지지는 않는다.
다른 포트는 `--port 8766`으로 지정한다. 기존 `collect.sh start`로 시작한 수집도 표시된다.

기본은 중지할 때까지 연속 수집이며 기존 CLI와 동일하게 CT ADC 개별 코드 저장은 선택이다.
열화상 원본은 기본 1장/초 저장한다. 예를 들어 CT ADC 원본까지 60초간 수집하려면:

```bash
./jetson_deploy/collect.sh dashboard --duration 60 --save-ct-raw
```

기본 원본 저장 경로는 아래와 같고 `--out-dir PATH`로 바꿀 수 있다.

```text
/home/keti/projects/factory_safety/jetson_deploy/results/sensor_collection/<UTC run ID>/
```

원본은 기존 `snapshots.jsonl`, `scalars.csv`, `thermal_*.npz`, 실행 메타데이터·검증 보고서로 남는다.
이 기능은 서버 HDD 자동 업로드를 설정하지 않는다. 브라우저·대시보드 종료는 원본 수집을
중지하지 않는다. 화면 없이 종료할 때는 다음 명령을 사용한다.

```bash
./jetson_deploy/collect.sh stop
./jetson_deploy/collect.sh status
```

중지 후 부분 청크 저장과 검증이 끝날 때까지 기다린다. 화면의 중지 상태와
최종 보고서의 `validation.passed`를 확인한다. foreground `run`이나 trial 수집은 관측만
가능하며 원래 실행한 터미널에서 종료한다.

## 화면 의미

- 열화상과 NTC 온도·CT1 전류·PM2.5·CO₂의 최근 120개 수신 스냅샷 추이.
- PM1/PM4/PM10, 온습도·기압·가스 저항, CT 샘플 수·클리핑, 저장 위치와 수집 상태.
- 센서 `status`, `fresh`, `age_ms`, 프레임 번호와 tick 지터를 그대로 표시한다.
  SCD30은 약 2초 측정 주기이므로 이전 값 유지는 정상일 수 있다.
- SGP30은 기본 비활성, CT2~4는 미연결이다. 누락값은 `—`로 표시하며 그래프를 보간하지 않는다.
- 수집 종료·연결 끊김·3초 이상 미리보기 지연은 마지막 값을 유지하면서 명확히 표시한다.
  센서 통신 오류의 첫 원인도 표시하며 하드웨어 자동 재시도는 하지 않는다.
- 열화상 색상은 프레임별 최소·최대 자동 범위이다. `raw / 100 - 273.15` 환산값이며
  TLinear 설정·절대온도 정확도 검증을 완료했다는 뜻은 아니다. CT는 기존 공칭 환산값이다.

수집 설정 표시는 **이 대시보드에서 다음 시작 요청에 사용할 설정**이다.
이미 다른 CLI로 시작된 run의 설정은 해당 run의 `metadata.json`을 확인한다.
미리보기는 위험 판정이나 모델 추론 결과가 아니며 원본 실측 관측 화면이다.

## 센서를 켜지 않고 기록 확인

```bash
./jetson_deploy/collect.sh dashboard --port 8766 --replay /절대/경로/run_directory
```

처음 최대 120개 스냅샷을 초당 하나씩 반복 표시한다. **기록 재생** 배너가 표시되고
시작·중지 버튼은 비활성이다. 저장된 raw/sequence를 확인하며 프레임이 없으면 합성하지 않는다.

## 수집과 화면의 분리

수집기가 실제 tick에 선택한 동일 프레임 참조와 스냅샷을 크기 1 큐에 비차단으로 전달한다.
별도 worker가 RAM 파일시스템의 작은 캐시를 원자 교체하며, HTTP 서버는 이 캐시만 읽는다.
원본 파일의 10 tick flush·30장 열화상 청크 완성을 기다리지 않아도 화면에 표시된다.
UI·worker가 느리면 중간 미리보기만 버리고 원본 수집은 유지한다. 오류는 한 번 기록하고
해당 미리보기를 중지한다. `timing_report.json.preview`에 발행 수·버린 수·오류가 남는다.

캐시는 `/run/user/<uid>/factory-safety-preview` 또는 `/dev/shm/factory-safety-preview-<uid>`의
사용자 전용 폴더이며 영구 데이터가 아니다. 최근 run 캐시만 제한적으로 유지한다.
`dashboard-<port>.json`에는 실행 중인 대시보드의 접속 주소가 있고 종료 시 제거한다.

HTTP는 `127.0.0.1`에만 열리며 API 토큰과 Host/Origin 검사를 사용한다. 연구실 다른 PC나
외부망으로 공개하는 서버는 아니다. 포트를 외부 인터페이스에 노출하도록 바꾸지 않는다.

검증: `tests/test_dashboard.py`는 하드웨어 없이 미리보기 실패 격리·프레임 정합성·HTTP
제어 보호·비동기 시작/정지를 검증한다. 실제 기능 점검 기록은
`/home/keti/review_runs/20260917_dashboard/`에 보관한다. 수정 중 코드로 얻은 데이터는
개발 검증용이며 논문용 성능/정확도 결과로 인용하지 않는다.
