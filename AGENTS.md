# AGENTS.md — AI agent 작업 규칙

이 저장소를 수정하기 전에 이 파일과 [docs/ENVIRONMENT_POLICY.md](docs/ENVIRONMENT_POLICY.md)를 읽는다.
규칙은 권고가 아니라 **강제**다.

관련 문서: [docs/JETSON_ENVIRONMENT.md](docs/JETSON_ENVIRONMENT.md) (실측값),
[docs/SERVER_ENVIRONMENT.md](docs/SERVER_ENVIRONMENT.md)

> `jetson_deploy/codex_context/AGENTS.md` 는 별개 파일이다. 그것은 2026-05-22 스냅샷 기준의
> Codex 전용 컨텍스트이며 일부 내용이 오래되었다. 충돌하면 **이 루트 파일이 우선한다.**

---

## 1. 코드 영역과 환경

| 디렉터리 | 프로파일 | venv |
|---|---|---|
| `src/` | SERVER-TRAINING (PyTorch 학습) | `$HOME/venvs/factory_training` |
| `jetson_deploy/` | JETSON-RUNTIME (센서 + 추론) | `$HOME/venvs/factory_runtime` |

두 영역의 의존성을 섞지 않는다. Jetson 센서 패키지를 학습 환경에, PyTorch를 Jetson 런타임에
설치하지 않는다.

**환경이 불명확하면 자동 설치하지 말고 먼저 판별한다.**

```bash
uname -m                       # aarch64 -> Jetson 가능성
cat /etc/nv_tegra_release      # 존재하면 Jetson 확정
```

## 2. Python 패키지 규칙

**금지**
```
sudo pip / sudo pip3 / sudo python -m pip
/usr/bin/python3 에 직접 pip install
pip install --upgrade 로 전체 무분별 upgrade
JetPack 제공 CUDA / TensorRT / Jetson.GPIO / OpenCV 를 PyPI 패키지로 덮어쓰기
$HOME/.local (user-site) 패키지에 의존
저장소 내부에 .venv 생성
venv 를 Git 에 commit
```

**Jetson runtime 실행은 항상 `PYTHONNOUSERSITE=1`**

```bash
./jetson_deploy/run_python.sh <script.py> [args...]
# 또는
env PYTHONNOUSERSITE=1 "$HOME/venvs/factory_runtime/bin/python" <script.py>
```

`pip` 호출에도 적용한다. 없으면 pip이 `~/.local` 패키지를 `Requirement already satisfied` 로
인식해 venv에 설치하지 않는다.

## 3. requirements 변경 규칙

`jetson_deploy/requirements-jetson.txt` = 직접 의존성만
`jetson_deploy/constraints-jetson.txt` = 하드웨어에서 검증된 버전 제약

**버전을 추가하거나 올리기 전에 반드시 실제 하드웨어 또는 최소한 import 검증을 통과해야 한다.**
검증하지 않은 pin은 pin이 없는 것보다 나쁘다.

- `pip freeze` 결과를 requirements로 복사하지 않는다
- JetPack/APT 제공 패키지(`tensorrt`, `cv2`, `Jetson.GPIO`, CUDA)를 requirements에 넣지 않는다
- 설치 전에 resolver가 무엇을 할지 먼저 확인한다 (`pip download` 로 미리보기 가능)
- 시스템/JetPack 패키지를 변경하려는 dependency resolution이 나오면 **중단하고 보고한다**
- `numpy` 는 `1.26.4` 고정. 2.x로 올리면 JetPack OpenCV/pandas/PIL 이 ABI로 깨진다

## 4. 센서 / 하드웨어 규칙

**버스와 주소를 임의로 바꾸지 않는다.** 확정된 매핑:

| 물리 핀 | Linux device | 클럭 | 장치 |
|---|---|---|---|
| Pin 3/5 | `/dev/i2c-7` | 400 kHz | ADS1115 `0x48`, SGP30 `0x58` |
| Pin 27/28 | `/dev/i2c-1` | 100 kHz | SPS30 `0x69`, SCD30 `0x61` |

I2C bus 번호를 하드코딩으로 추측하지 않는다. 위 표 또는 CLI 옵션을 쓴다.

**확장 헤더(40-pin) 기능 설정 — 공식 경로만 사용한다.**

Jetson expansion-header configuration must use the official `jetson-io.py`
*Save pin changes* / *Save and reboot to reconfigure pins* workflow first.
Do not manually generate or patch DTBO/DTS unless the official workflow has
demonstrably failed.

FIRST CHOICE (사용자가 대화형으로 실행):
```
sudo /opt/nvidia/jetson-io/jetson-io.py
  -> Configure Jetson 40pin Header
  -> Configure header pins manually
  -> spi1 (19,21,23,24,26)
  -> Back -> Save pin changes -> Save and reboot to reconfigure pins
```

일반 설정 경로로 쓰지 않는다: `config-by-function.py -o dt`, *Export as Device-Tree
Overlay*, 수동 DTS/DTBO 편집, tristate 직접 수정, devmem write, pinmux 저수준 분석부터
시작하기. 저수준 분석은 **공식 Jetson-IO 절차와 공식 loopback 이 모두 실패한 경우에만**
수행한다. 근거는 `docs/JETSON_SPI_BME680_SETUP.md`.

**진단 순서** — 아래 순서를 지키고 역순으로 하지 않는다.
```
L1 전원/GND/배선  ->  L2 공식 Jetson 설정 도구  ->  L3 공식 device node
->  L4 공식/표준 loopback·bus test  ->  L5 실제 sensor chip-id
->  L6 high-level Python driver  ->  L7 그 이후에만 pinmux / DT / register debugging
```

**절대 금지**
```
device-tree 수정 / pinmux 수정 / /opt/nvidia/jetson-io 자동 실행
i2cset / i2cdump / devmem / GPIO register 직접 쓰기
reboot
JetPack / CUDA / TensorRT 변경
센서 factory reset / calibration 값 변경 / EEPROM write / persistent setting 변경
임의 register sweep
```

**허용**: 데이터시트 규격의 휘발성 protocol command (measurement start/stop, 레지스터 read,
ADS1115 conversion config write 등)

**오류 처리**: 통신 오류가 나면 자동 retry loop를 만들거나 workaround를 구현하지 않는다.
첫 오류를 그대로 보존하고 (operation, exception class, errno, 발생 시점, 그 전에 성공한 operation)
보고하고 멈춘다.

**측정값 보정 금지**: calibration coefficient 생성, offset correction, noise subtraction,
threshold 적용을 측정 스크립트가 자동으로 하지 않는다. 예상과 다른 값이 나와도 그대로 보고한다.

**공유 자원**: 하나의 ADS1115는 단일 MUX ADC다. CT(A0-A1) continuous 측정과 NTC(A2) 측정을
동시에 할 수 없다. `/dev/i2c-1` 은 SPS30과 SCD30이 공유하므로 순차 접근만 한다.

**CT**: 검증된 설계 범위는 0 ~ 400 A RMS (SCT024TS 정격). ADC FSR 계산으로 나오는 더 큰 값을
측정 범위나 overload capability로 표현하지 않는다.

## 5. 실험 재현성

모든 실험/수집 출력의 metadata에 최소 다음을 기록한다.

```
git_commit_sha, git_dirty, timestamp(ISO 8601 + TZ), hostname,
jetpack_l4t, python_version, environment_profile, sensor_driver_versions
```

센서별로 해당하는 것: ADS1115 address/PGA/data rate/MUX, CT burden·ratio,
SCD30 serial, SPS30 serial, I2C bus path와 클럭.

`git_dirty == true` 상태의 결과는 논문/보고서 수치로 인용하지 않는다.

## 6. Git 규칙

- `main` 에 직접 작업하지 않는다. feature branch를 쓴다
- `git reset` / `git clean` / force checkout / rebase 로 작업물을 날리지 않는다
- 원격 동기화는 `git fetch` 후 fast-forward 가능 여부를 확인하고 `git pull --ff-only` 만 쓴다
- **`.gitignore` 의 프로젝트 전용 섹션을 삭제하지 않는다.** 그 섹션이 없으면 약 26 GB의
  데이터셋/모델/tarball 이 추적 대상으로 노출된다 (2026-08-31에 실제 발생)
- commit / push 는 사용자 승인 후에만 한다
