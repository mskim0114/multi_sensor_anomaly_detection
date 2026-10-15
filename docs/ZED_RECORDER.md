# ZED 2i 보조 기록기 (zed-v1)

작성 2026-10-13. 코드: `jetson_deploy/zedcap/`, `jetson_deploy/scripts/14_zed_recorder.py`.

센서 수집과 **같은 run 폴더에, 같은 tick 기준으로** ZED 2i 영상과 깊이를 저장한다.
- 수집기(`11_collect_sensors.py`, `12_run_trial.py`, `sensors/`)는 한 줄도 바꾸지 않았다.
- ZED 기록기는 별도 프로세스다. 수집기는 ZED 를 모르고, ZED 가 죽거나 느려져도 1 Hz 수집에 영향이 없다.

## 1. 구조

| 항목 | 내용 |
|---|---|
| 프로세스 | user 서비스 `factory-zed` (`jetson_deploy/zedcap/factory-zed.service`) |
| Python | `~/venvs/zed_runtime` (pyzed 5.5 · numpy 2.2.6 · Pillow). pyzed 가 numpy ≥ 2 를 요구해서 `factory_runtime`(numpy 1.26.4 고정)과 분리했다 |
| 켜고 끄기 | `~/.config/factory_safety/zed.json` 의 `enabled`. 기록기가 1초마다 다시 읽는다 |
| 언제 기록하나 | 스위치가 켜져 있고 **수집(free run 또는 trial)이 실행 중일 때만** 카메라를 연다. 그 밖에는 카메라를 닫아 둔다 |
| 어느 run 인가 | 수집기의 live preview 가 쓰는 `active.json`(run 경로 + 프로세스 identity)을 읽는다. 수집기 프로세스가 살아 있고, run 이 설정된 roots 안에 있고, 최종 status 가 아직 없을 때만 따라간다 |
| 자원 상한 | MemoryMax 2.5 GB(넘으면 ZED 만 종료), Nice 5, CPUWeight 50 |

### 켜고 끄는 방법
- Jetson: `python3 jetson_deploy/scripts/14_zed_recorder.py on|off|status`
- 웹: [장치 현황] 카드의 **ZED 켜기 / ZED 끄기** (운영자, 장치의 `allow_remote_control` 필요)
  - hub 명령 `zed_on`/`zed_off` 는 `zed.json` 의 `enabled` 만 바꾼다. 수집기는 건드리지 않는다
- 수집 도중에 바꿔도 된다. 끄면 현재 ZED 세션이 정상 종료되고, 다시 켜면 같은 run 에 새 세션이 시작된다

### 오류 처리 (저장소 규칙: 첫 오류 보존, 숨은 재시도 없음)
- 카메라가 열리지 않거나 5초 이상 프레임이 오지 않으면, 그 오류를 `zed.json` 과 상태에 남기고 그 run 에서는 다시 시도하지 않는다.
- 다시 시도하려면 스위치를 껐다 켜거나 새 수집을 시작한다.

## 2. 저장 형식 (`<run>/zed/`)

| 파일 | 내용 |
|---|---|
| `zed.json` | schema `zed-v1`, 카메라(모델·S/N·펌웨어·해상도·좌우 내부 파라미터·baseline), 설정, tick 시계, 세션 목록(시작·종료·종료 이유·프레임 수·누락 tick·오류), `status`(`recording`/`complete`), 기록기 identity |
| `frames.csv` | 저장된 프레임마다 한 줄: session, segment, index, **tick**, frame_monotonic_ns, image_timestamp_ns, offset_ms, IMU(가속도·자이로·자세 쿼터니언·온도) |
| `seg_SS_NNNNN_left.npz` | `left_jpeg`(uint8, JPEG 을 이어 붙임) + `left_offsets`(int64, N+1). 압축 없이 저장해 hub 가 한 장만 바로 읽는다 |
| `seg_SS_NNNNN_depth.npz` | **`depth_mode` 가 `NONE` 이 아닐 때만.** `depth_mm` uint16 (N, 360, 640), deflate 압축. **0 = 측정 없음**(가림·너무 가까움), **65535 = 측정 범위 밖**(너무 멂) |

- **2026-10-15 부터 기본값은 `depth_mode: "NONE"`(왼쪽 RGB 만)이다.** 깊이는 계산도 저장도 하지 않고, `zed.json` 의 `depth_encoding` 과 `files.depth` 가 `null` 이다. 깊이가 다시 필요하면 `~/.config/factory_safety/zed.json` 에 `"depth_mode": "PERFORMANCE"` 를 넣는다. 그 이전 run(예: `dataset/normal/trial_001`)에는 깊이 파일이 있다.

- 세그먼트 하나는 60 프레임(1분)이다. 배경 스레드가 tmp + rename 으로 쓰고, 다 쓴 다음에만 `frames.csv` 에 줄을 추가한다. 그래서 index 의 모든 줄은 완전한 파일을 가리킨다.
- 기록기가 비정상 종료되면 아직 쓰지 못한 세그먼트(최대 몇 분)를 잃는다. `*.partial` 파일이 남을 수 있다. 다음 세션이 이전 세션을 `interrupted` 로 표시한다.
- 왼쪽 영상은 1280×720 JPEG(품질 85, 약 78 KB)이다. 깊이를 켜면 깊이 맵은 640×360 으로 줄여 저장한다(약 180 KB).
  - 내부 파라미터는 캡처 해상도 기준이므로, 깊이 맵에 쓸 때는 fx·cx 에 가로 비율, fy·cy 에 세로 비율을 곱한다.
- 용량: RGB 만이면 약 **80 KB/s ≈ 0.28 GB/시간**이다. 깊이까지 켜면 약 255 KB/s ≈ 0.92 GB/시간이다(2026-10-15 `normal/trial_001` 실측).

### tick 정렬
- 수집기는 tick n 을 CLOCK_MONOTONIC 의 `t0 + n·1 s` 에 찍는다. t0 는 `snapshots.jsonl` 첫 줄의 `target_monotonic_ns` 이고, 기록기는 이 파일을 읽기만 한다.
- ZED 는 15 fps 로 받는다. 각 tick 목표 시각에서 **반 프레임(+5 ms) 이내**에 찍힌 첫 프레임을 그 tick 으로 저장한다.
  - 프레임 시각은 SDK 의 영상 시각(realtime)을 monotonic 으로 바꾼 값이다.
- 해당 프레임이 없는 tick 은 `missed_ticks` 로 세고 채우지 않는다.
- 카메라를 여는 데 약 3~4초 걸려서 **run 의 처음 몇 tick(실측 0~3)에는 ZED 프레임이 없다.**

## 3. hub / 웹

- **[장치 현황]**: ZED 상태를 보여 준다(켜짐/꺼짐, 대기/기록 중/오류, run, 프레임 수). ZED 켜기/끄기 버튼이 있다.
- **[실시간]**: ZED 썸네일(320×180)을 보여 준다. heartbeat 로 약 30초마다 갱신되며, 크기가 96 KB 를 넘거나 JPEG 이 아니면 버린다.
- **[수신 데이터] 상세**: ZED 프레임 수와 세션을 보여 준다. 그래프를 클릭하면 그 tick 에 가장 가까운 저장 프레임이 보인다(`GET /api/v1/unit/zed?device&unit&tick`). 라벨링할 때 실제로 무슨 일이 있었는지 영상으로 확인하는 용도다.
- 업로드: uplink 가 `zed/zed.json` 이 `complete` 가 되기 전(또는 기록기가 죽었다고 증명되기 전)에는 그 unit 을 올리지 않는다.

## 4. 검증 기록 (2026-10-13, Jetson Orin Nano Super, ZED 2i S/N 38116610)

| 시험 | 결과 |
|---|---|
| 수집기와 동시 실행, ZED HD720 30 fps + 깊이 매 프레임 (3분) | 수집 300/300 tick, tick 오차 p95 0.2 ms, tick 처리 시간 변화 없음, ZED 30.0 fps·메모리 341 MB·실패 0, GPU 평균 50 % |
| 기록기 end-to-end, 실제 수집 150 s (15 fps) | ZED 147 프레임(tick 4–150), **누락 0**, \|offset\| 평균 28 ms·최대 32 ms, 세그먼트 3개 |
| 같은 시험의 첫 버전(세그먼트를 grab 루프에서 저장) | tick 64 누락. 1분치 깊이 압축(약 1 s)이 grab 을 막았다. 배경 쓰기 스레드로 고친 뒤 누락 0 |
| 파일 호환 | numpy 2 로 쓴 npz 를 수집 환경(numpy 1.26.4, Pillow 9)에서 읽어 JPEG·깊이 복원 확인 |
| 단위 테스트 | `tests/test_zed_recorder.py` 16개(가짜 카메라), `tests/test_edge_hub.py` ZED 4개 |

주의:
- Jetson 화면에서 ZED GUI 도구(ZED_Explorer, ZED_Depth_Viewer 등)를 쓰지 않는다. 2026-10-13 에 메모리를 다 써서 화면이 멈췄고 강제 재부팅했다. 확인은 웹 썸네일로 한다.
- ZED 는 일반 컬러 카메라라 사람이 찍힌다. 저장 범위와 보관·공유 방침은 별도로 정한다.
