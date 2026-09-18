# Jetson Orin Nano 수집·추론 검증 패키지

이 디렉터리는 실제 센서 수집, trial 기록, ONNX 추론 검증을 담당한다.
현장 데이터와 기존 모델을 연결하는 ModelAdapter 및 이상 사건 데이터는 아직 준비되지 않았다.
`04_realtime_pipeline.py`는 저장된 검증 입력으로 만든 합성 스트림의 처리 시간을 측정한다.

현재 작업 기준은 [루트 AGENTS.md](../AGENTS.md),
[환경 정책](../docs/ENVIRONMENT_POLICY.md),
[연구 상태](../docs/RESEARCH_STATUS.md),
[서버 이관 절차](../docs/SERVER_WORKSTATION_HANDOFF.md)다.
`codex_context/`는 2026-05-22 기록 보관용 스냅샷이며 현재 실행 지침으로 사용하지 않는다.

| 경로 | 역할 |
|---|---|
| `run_python.sh`, `check_environment.py` | 전용 가상환경 실행, 의존성·장치·실행 provider 목록 점검 |
| `scripts/01_check_environment.py` | 과거 명령 호환용 진입점. 위 래퍼와 현재 checker로 위임 |
| `scripts/02`~`05` | 지연시간, PC 예측 일치, 합성 스트림, 결과 요약 |
| `scripts/06`~`10` | 열화상·SPS30·NTC·CT·BME680 개별 진단 |
| `scripts/11_collect_sensors.py`, `sensors/` | 1 Hz 동기 원시 데이터 수집 |
| `collect.sh`, `collect.py` | 백그라운드 수집 start / stop / status, foreground run, dashboard |
| `dashboard.py`, `dashboard_web/` | [로컬 열화상·센서 대시보드](../docs/JETSON_SENSOR_DASHBOARD.md), 수집 시작·중지 |
| `scripts/12_run_trial.py` | 사전 점검 및 실험 메타데이터가 포함된 trial 수집 |
| `model/`, `reference/` | 기존 ONNX/TensorRT 파일과 기준 입력 |
| `results/` | 측정·검증 기록. Git 추적 대상에서 제외 |
| `codex_context/` | 과거 문서·코드 스냅샷 |

Jetson에서는 저장소 루트에서 실행한다. 아래 경로는 이 장비의 위치이며 다른 장비에서는
실제 checkout 경로를 사용한다.

```bash
cd /home/keti/projects/factory_safety
./jetson_deploy/run_python.sh jetson_deploy/check_environment.py
```

래퍼는 `$HOME/venvs/factory_runtime`과 `PYTHONNOUSERSITE=1`을 적용한다.
새 장비의 환경 생성은 [환경 정책](../docs/ENVIRONMENT_POLICY.md)의 `setup_jetson_env.sh` 절차를 따른다.
학습용 PyTorch 환경과 JetPack 제공 CUDA/TensorRT/OpenCV 패키지를 혼합하지 않는다.

환경 점검은 모듈 import, 장치 노드·권한, ONNX Runtime provider 목록을 검사한다.
**센서 실측이나 모델 추론은 실행하지 않는다.** provider가 목록에 있다는 사실만으로
그 provider의 실제 추론이 성공한다고 판단하지 않는다.

아래 명령은 기존 모델·reference의 별도 추론 검증 절차다. 실행하면 `results/` 기록을 갱신한다.
새 모델 학습·export 및 현장 모델 입력 결정은 서버 이관 문서의 선행 조건을 따른다.

```bash
# 기본 벤치마크: 사용 가능한 CUDA와 CPU. TensorRT EP는 기본 제외.
./jetson_deploy/run_python.sh jetson_deploy/scripts/02_benchmark_latency.py --runs 200

# 기준 샘플과 PC 예측 일치 검증
./jetson_deploy/run_python.sh jetson_deploy/scripts/03_verify_accuracy.py --small --provider cuda

# 저장된 reference로 구성한 합성 스트림
./jetson_deploy/run_python.sh jetson_deploy/scripts/04_realtime_pipeline.py --n 300 --stride 10 --provider cuda

# 저장된 결과 통합
./jetson_deploy/run_python.sh jetson_deploy/scripts/05_summary.py
```

ONNX Runtime의 TensorRT EP는 현재 모델 첫 추론에서 SIGSEGV가 기록되어 있다.
독립 `trtexec` 실행과 ONNX Runtime TensorRT EP 실행은 구분한다.
기존 측정 결과와 해당 문제의 범위는 [JETSON_ENVIRONMENT.md](../docs/JETSON_ENVIRONMENT.md)를 참고한다.
모델·provider·입력·정밀도에 따른 출력 차이를 기록하며 예측 일치율을 현장 정확도로 해석하지 않는다.

센서 수집은 [JETSON_SENSOR_COLLECTION.md](../docs/JETSON_SENSOR_COLLECTION.md),
trial 규약은 [JETSON_DATASET_PROTOCOL.md](../docs/JETSON_DATASET_PROTOCOL.md)를 따른다.

```bash
./jetson_deploy/collect.sh start
./jetson_deploy/collect.sh status
./jetson_deploy/collect.sh stop
```

기본값은 공통 1 Hz 기록·열화상 1장/초 저장·SGP30 비활성화다. 시작한 수집은 SSH 연결이
끝나도 유지되며 `stop`은 마지막 파일 저장과 검증이 끝날 때까지 기다린다.

현재 개발용 baseline의 관측 스칼라는 NTC·PM 3채널·CT1이며 CT2~CT4는 없다.
모델 입력으로 연결하기 전에 채널·단위·정규화·품질 처리 계약을 확정해야 한다.

서버로 이관할 때 코드와 raw 데이터는 각각 전달한다. Git clone만으로 `dataset/`,
`processed/`, 모델 및 reference가 모두 준비되지 않으므로
[SERVER_WORKSTATION_HANDOFF.md](../docs/SERVER_WORKSTATION_HANDOFF.md)의 자산·checksum 절차를 따른다.
현재 후속 작업은 서버 환경 audit, 데이터 복구·재현, 실센서 입력 설계 및 모델 연결 순서다.
