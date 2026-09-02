# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) and [docs/ENVIRONMENT_POLICY.md](docs/ENVIRONMENT_POLICY.md) before
modifying this repository.**

`AGENTS.md` 의 규칙이 이 파일과 충돌하면 `AGENTS.md` 가 우선한다. 아래는 Claude Code 세션에서
가장 자주 필요한 요약과 판별 절차다.

---

## 1. 먼저 어느 환경인지 판별한다

절대 추측하지 않는다. 자동 설치 전에 반드시 확인한다.

```bash
uname -m
cat /etc/nv_tegra_release 2>/dev/null && echo "JETSON" || echo "NOT JETSON"
```

| 판별 결과 | 프로파일 | 작업 코드 | venv |
|---|---|---|---|
| `/etc/nv_tegra_release` 존재 (aarch64) | **JETSON-RUNTIME** | `jetson_deploy/` | `$HOME/venvs/factory_runtime` |
| 그 외 (x86_64 학습 서버) | **SERVER-TRAINING** | `src/` | `$HOME/venvs/factory_training` |
| 불명확 | — | — | **설치하지 말고 사용자에게 확인** |

## 2. JETSON-RUNTIME 에서 작업할 때

Python 실행은 런처를 쓴다. plain `python3` 를 쓰지 않는다.

```bash
./jetson_deploy/run_python.sh jetson_deploy/scripts/07_read_sps30.py --i2c-port /dev/i2c-1
./jetson_deploy/run_python.sh jetson_deploy/check_environment.py
```

환경을 처음 만들 때:

```bash
./jetson_deploy/setup_jetson_env.sh
```

핵심 규칙 (전체는 AGENTS.md):
- 항상 `PYTHONNOUSERSITE=1` — `$HOME/.local` 패키지에 의존하지 않는다
- `sudo pip` 금지, system Python 직접 pip install 금지
- `numpy` 는 `1.26.4` 고정 (2.x는 JetPack OpenCV/pandas/PIL 을 ABI로 깨뜨린다)
- `tensorrt`, `cv2`, `Jetson.GPIO`, CUDA 는 JetPack 제공 — requirements에 넣지 않는다
- I2C 버스: Pin 3/5 = `/dev/i2c-7` (400 kHz), Pin 27/28 = `/dev/i2c-1` (100 kHz)
- device-tree / pinmux / jetson-io / reboot 자동 실행 금지
- 40-pin 확장 헤더 기능 설정은 **공식 `jetson-io.py` 메뉴 경로가 FIRST CHOICE**다
  (*Save pin changes* → *Save and reboot*). 공식 절차와 공식 loopback 이 모두 실패하기
  전에는 DTBO/DTS 를 수동 생성/수정하지 않는다 — `docs/JETSON_SPI_BME680_SETUP.md`
- `i2cset` / `i2cdump` / `devmem` 금지
- 센서 오류 시 자동 retry나 workaround를 만들지 말고 첫 오류를 보존해 보고한다
- 측정값에 calibration / offset correction / threshold 를 자동 적용하지 않는다

## 3. SERVER-TRAINING 에서 작업할 때

`docs/SERVER_ENVIRONMENT.md` 는 현재 **PENDING SERVER ENVIRONMENT AUDIT** 상태다.
Jetson에서 수집한 버전을 서버 환경으로 가정하지 않는다. 서버에서 직접 audit한 뒤 채운다.

- venv: `$HOME/venvs/factory_training`
- requirements: `requirements-server.txt` (아직 비어 있음)
- Jetson 센서 패키지를 이 환경에 설치하지 않는다

## 4. 저장소 상태에서 알고 있어야 할 것

- **`.gitignore` 의 "Project-specific rules" 섹션을 삭제하면 안 된다.** 삭제되면 약 26 GB의
  `data/`, `results/`, `jetson_deploy/reference/*.npz`, `*.tar` 이 추적 대상으로 노출된다.
  2026-08-31에 실제로 발생한 회귀다
- `src/` 와 `configs/data_config.yaml` 이 `/home/keti/factory_safety/...` 를 하드코딩하고 있으나
  실제 경로는 `/home/keti/projects/factory_safety` 다. 아직 수정되지 않았다
- 학습된 모델 입력은 8채널 `[NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4]` + thermal 120×160 으로
  고정이다. 새 센서(SGP30/SCD30/BME680)를 모델 입력에 넣으려면 ONNX/TensorRT 엔진과 논문 수치가
  전부 무효가 된다. 수집 계층과 모델 입력 계층을 분리한다
- `main` 에 직접 작업하지 않는다. commit / push 는 사용자 승인 후에만 한다

## 5. 작업 스타일

- 실측으로 확인할 수 있는 것을 추측으로 대체하지 않는다. 버스 번호, 주소, 버전, 경로는 확인한다
- 파괴적/비가역 작업 전에 확인을 받는다
- 실패는 숨기지 않고 그대로 보고한다. 재실행으로 덮지 않는다
- 검증하지 않은 것을 검증했다고 쓰지 않는다
