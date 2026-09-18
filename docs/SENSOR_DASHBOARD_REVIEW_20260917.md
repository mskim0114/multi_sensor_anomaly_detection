# Jetson 센서 대시보드 기능 검증 — 2026-09-17

범위: 로컬 브라우저의 수집 시작·중지, 센서 값·열화상 미리보기, 기존 원본 저장의 유지.
수정 중인 `git_dirty=true` 코드의 기능 점검이며 연구 성능/정확도 결과가 아니다.

## 구현 및 검토

경로 탐색에서 JSONL의 10 tick flush와 열화상의 30장 청크 저장을 확인했다.
화면은 저장 완료를 기다리지 않고 collector가 선택한 동일 tick/frame을 크기 1 큐로 받아
별도 worker가 RAM 캐시에 발행한다. HTTP 프로세스는 센서 장치를 열지 않는다.
순간적인 UI 지연은 미리보기만 생략하며, 실패는 latch하고 원본 수집을 유지한다.

Sol 검토에서 발견한 사항을 반영했다.

- 기존 수집과 동일하게 CT raw 저장은 `--save-ct-raw` opt-in.
- 첫 미리보기 누락과 갱신 지연을 실행 중 3초 timeout으로 표시.
- 시작/중지 helper를 새 session과 regular temporary log FD로 분리하여 대시보드 종료 시
  SIGINT 전파나 BrokenPipe가 수집 시작을 취소하는 경합을 방지.
- `failure=null`이어도 `first_error`의 operation·예외·message·errno·마지막 성공 작업을 표시.
  제어 실패의 상세 message와 기록 재생의 sensor.error도 유지.
- HTTP는 loopback 한정, API token·Host·Origin 검사, 고정 action/정적 경로만 허용.

## 자동·브라우저 검증

JETSON-RUNTIME 래퍼로 실행했다. 패키지 설치와 장비 설정 변경은 하지 않았다.

| 검증 | 결과 |
|---|---|
| `test_dashboard.py` | 19개 PASS |
| `test_sensor_collector.py` | 13개 PASS |
| `test_collection*.py` | 9개 PASS |
| `node --check .../app.js` / `git diff --check` | PASS |
| Chromium 실제 렌더링 | 저장 기록 재생·실시간 실측 화면 확인 |

Dashboard 테스트는 임시 캐시와 mock collector/subprocess만 사용한다. 프레임 순서·endianness,
큐 제한, 저장 실패·초기화 실패·offer 실패의 원본 격리, stale/PID 재사용,
HTTP 인증·경로·JSON 제약, replay 제어 차단, 비동기 Start/Stop 및 helper 분리를 검사한다.
프런트 오류 메시지는 Node DOM 모사로 추가 확인했다.

## 실제 장비 검증

대시보드 HTTP API로 Start 요청 후 35개 tick을 관측하고 Stop을 요청했다.
60초 자동 종료 제한도 설정했으며 다른 수집이 없는 상태에서만 실행했다.

```text
run: /home/keti/review_runs/20260917_dashboard/live/20260917T081148.466246Z
evidence: /home/keti/review_runs/20260917_dashboard/live_verification.json
screenshots: /home/keti/review_runs/20260917_dashboard/live.png, replay.png
```

| 항목 | 결과 |
|---|---|
| 상태 전이 | stopped → starting → running → stopping → stopped |
| JSONL / CSV / 저장 열화상 참조 | 각각 35개, validation PASS |
| 화면 열화상과 저장 NPZ | 35개 모두 sequence·frame_sequence 일치, raw bytes SHA256 일치 |
| 최종 preview 발행 / drop / 오류 | 35 / 0 / 없음 |
| 수집 missed tick / 원본 chunk drop / write error | 0 / 0 / 0 |
| 첫 센서 오류 / 종료 오류 | 없음 / 없음 |
| 해당 시험의 thermal-invalid tick | 0 |
| Stop 후 프로세스 / 실시간 표시 | 종료 / false |

35개 tick과 열화상 2개 청크, 명시적으로 켠 CT raw 1개 청크가 남았다.
미리보기와 화면은 원본 센서 값을 보정하지 않는다. 이 시험이 장시간 운전이나
열화상 절대온도 정확도 검증을 대신하지 않는다.

시험 수집과 시험용 HTTP 서버는 모두 종료했다. 자동 서버 업로드나 부팅 자동 실행은
추가하지 않았다. 사용법은 [로컬 대시보드 안내](JETSON_SENSOR_DASHBOARD.md)를 따른다.
