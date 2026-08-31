# 환경 정책 (Environment Policy)

이 저장소는 하나의 Git repository를 쓰지만 **런타임 환경은 두 개로 완전히 분리**한다.
어느 쪽에서 작업하는지 먼저 판별하고, 해당 프로파일의 규칙만 적용한다.

작성 2026-08-31. 근거가 되는 실측은 [JETSON_ENVIRONMENT.md](JETSON_ENVIRONMENT.md) 참조.

---

## 1. 두 개의 환경 프로파일

| | SERVER-TRAINING | JETSON-RUNTIME |
|---|---|---|
| 목적 | PC/서버 학습, 모델 개발, 논문 실험, ONNX export | 실제 센서 acquisition, ONNX/TensorRT 추론, realtime pipeline |
| 코드 영역 | `src/` | `jetson_deploy/` |
| 주요 의존성 | PyTorch, CUDA 학습 스택 | onnxruntime-gpu, TensorRT, 센서 드라이버, Jetson GPIO/I2C/SPI |
| venv 경로 | `$HOME/venvs/factory_training` | `$HOME/venvs/factory_runtime` |
| requirements | `requirements-server.txt` | `jetson_deploy/requirements-jetson.txt` + `constraints-jetson.txt` |
| 상태 | **PENDING SERVER ENVIRONMENT AUDIT** | 구축 완료 (2026-08-31) |

**교차 오염 금지**: Jetson 센서 패키지를 SERVER 환경에 강제로 설치하지 않고, PyTorch 학습 스택을
JETSON 런타임에 설치하지 않는다. 두 환경은 서로의 requirements를 참조하지 않는다.

### JETSON-LAB (참조용 보존 환경)

`$HOME/venvs/factory_sensors` 는 2026-08-31에 SPS30/SCD30 최초 검증을 통과한 **lab/reference 환경**이다.

- 삭제하지 않는다
- 패키지를 uninstall 하지 않는다
- **production runtime으로 사용하지 않는다** (user-site 의존이 남아 있음)
- `factory_runtime` 이 충분히 검증될 때까지 대조군으로 보존한다

---

## 2. venv 생성 규칙

### JETSON-RUNTIME

```bash
python3 -m venv --system-site-packages "$HOME/venvs/factory_runtime"
```

**`--system-site-packages` 를 쓰는 이유**: JetPack이 제공하는 네이티브 패키지는 PyPI로 재현할 수 없다.

| 패키지 | 버전 | 출처 | PyPI 대체 가능? |
|---|---|---|---|
| `tensorrt` | 10.3.0 | APT `tensorrt` | 불가 |
| `cv2` (OpenCV) | 4.8.0 CUDA 빌드 | JetPack | 불가 (PyPI 휠은 CPU 전용) |
| `Jetson.GPIO` | 2.1.9 | APT `python3-jetson-gpio` | 덮어쓰면 안 됨 |
| CUDA runtime | 12.6 | JetPack | 불가 |

이들을 venv에서 쓰려면 시스템 site-packages를 상속해야 한다. 격리를 완전히 하면 GPU 스택을 잃는다.

### 실행 시 `PYTHONNOUSERSITE=1` 필수

```bash
env PYTHONNOUSERSITE=1 "$HOME/venvs/factory_runtime/bin/python" ...
```

**이유**: `--system-site-packages` 는 시스템 패키지뿐 아니라 **`$HOME/.local/lib/python3.10/site-packages`
(user-site)도 상속**한다. 이 디렉터리는 누가 언제 무엇을 `pip install --user` 했는지 추적되지 않는
비관리 영역이며, Git으로 재현할 수 없다.

실측된 피해 사례 (2026-08-31):

```
user-site 활성 :  numpy 2.2.6 (~/.local) 이 APT numpy 1.21.5 를 가림
                 -> cv2 import 실패: "ImportError: numpy.core.multiarray failed to import"
                    (JetPack OpenCV 4.8.0 은 numpy 1.x C ABI 로 컴파일됨)

PYTHONNOUSERSITE=1 : cv2 4.8.0 정상 import
                     단, onnxruntime / pyserial / sensirion 전부 사라짐
                     (이들이 user-site 에만 있었기 때문)
```

즉 `PYTHONNOUSERSITE=1` 은 **문제를 드러내는 장치**다. 이 상태에서 필요한 패키지를 venv 안에
명시적으로 설치하면, requirements 파일이 실제 런타임과 일치하게 된다.

**pip 실행에도 반드시 적용한다.** `PYTHONNOUSERSITE=1` 없이 설치하면 pip이 user-site의 패키지를
`Requirement already satisfied` 로 인식해 venv에 넣지 않는다. 실측: `factory_sensors` 에 SCD30을
설치할 때 의존성 6개가 모두 `~/.local` 재사용으로 처리되어, venv가 SCD30 하나만 소유하게 되었다.

---

## 3. 절대 금지 사항

```
sudo pip / sudo pip3 / sudo python -m pip
system Python(/usr/bin/python3)에 직접 pip install
pip install --upgrade 로 전체 무분별 upgrade
JetPack 제공 CUDA/TensorRT/Jetson.GPIO/OpenCV 를 PyPI 패키지로 덮어쓰기
$HOME/.local 패키지에 암묵적으로 의존
저장소 내부에 .venv 생성
venv 디렉터리를 Git에 commit
```

`.gitignore` 가 `.venv/`, `venv/`, `env/`, `ENV/` 를 차단하지만, 애초에 venv는 `$HOME/venvs/` 에만 만든다.

---

## 4. JetPack/system dependency 와 pip dependency 구분

**pip requirements에 넣지 말 것** (JETSON_ENVIRONMENT.md에 문서로만 기록):
`tensorrt`, `cv2`, `Jetson.GPIO`, CUDA runtime, `pycuda`

**pip requirements에 넣을 것**: 그 외 프로젝트가 직접 import하는 패키지

### 파일 역할 구분

| 파일 | 역할 |
|---|---|
| `jetson_deploy/requirements-jetson.txt` | 프로젝트가 **직접** import하는 패키지만 |
| `jetson_deploy/constraints-jetson.txt` | 실제 하드웨어에서 **검증된** 버전 제약 (전이 의존 포함) |

`pip freeze` 전체 결과를 requirements로 복사하지 않는다. 검증하지 않은 pin은 pin이 없는 것보다
나쁘다 — 동작하던 해석을 하드 실패로 바꾼다.

### onnxruntime-gpu 특수 사례

`onnxruntime-gpu` 는 **PyPI에 aarch64 배포판이 없다** (`from versions: none`).
jetson-ai-lab 인덱스(`https://pypi.jetson-ai-lab.io/jp6/cu126`)에서 받아야 하며,
`requirements-jetson.txt` 에 `--extra-index-url` 로 명시되어 있다.

### numpy 버전 고정 근거

`numpy==1.26.4` (1.x 마지막 릴리스)로 고정한다. 두 제약이 이 버전을 강제한다.

- `onnxruntime-gpu 1.24.0` 이 `numpy>=1.21.6` 요구 → APT numpy 1.21.5는 미달
- JetPack OpenCV 4.8.0 / APT pandas 1.3.5 / APT PIL 9.0.1 이 numpy **1.x C ABI**로 컴파일 → numpy 2.x는 이들을 깨뜨림

검증 완료: numpy 1.26.4 + onnxruntime-gpu 1.24.0 (TensorRT/CUDA/CPU EP 전부) + cv2 4.8.0 동시 동작.
2.x로 올리려면 cv2 재확인과 `03_verify_accuracy.py --small` 재현 검증이 선행되어야 한다.

---

## 5. 환경 재현 절차

### JETSON-RUNTIME

```bash
git clone git@github.com:mskim0114/multi_sensor_anomaly_detection.git
cd multi_sensor_anomaly_detection
./jetson_deploy/setup_jetson_env.sh
```

스크립트가 수행하는 것: 전제 검증 → venv 생성 → `PYTHONNOUSERSITE=1` 로 설치 →
provenance 검증(user-site에서 로드되는 모듈이 있으면 실패) → import smoke test.

apt 자동 설치, JetPack/CUDA/TensorRT/device-tree 변경, reboot은 하지 않는다.
부족한 시스템 패키지가 있으면 설치 명령만 안내하고 중단한다.

이후 모든 Python 실행:

```bash
./jetson_deploy/run_python.sh jetson_deploy/scripts/07_read_sps30.py --i2c-port /dev/i2c-1
```

환경 점검:

```bash
./jetson_deploy/run_python.sh jetson_deploy/check_environment.py
```

### SERVER-TRAINING

**PENDING SERVER ENVIRONMENT AUDIT.** [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) 참조.
Jetson에서 서버 의존성 버전을 추측하지 않는다.

---

## 6. 실험 재현성 규칙

모든 실험/수집 세션의 metadata에 최소 다음을 기록한다.

**환경 정보**
```
git_commit_sha        git rev-parse HEAD
git_dirty             작업 트리에 uncommitted 변경이 있는지
timestamp             ISO 8601, 타임존 포함
hostname
jetpack_l4t           /etc/nv_tegra_release
python_version
environment_profile   "JETSON-RUNTIME" | "SERVER-TRAINING"
sensor_driver_versions
```

**센서별 설정** (해당하는 것)
```
ADS1115  I2C address, PGA FSR, data rate, MUX 설정
CT       burden resistance, CT ratio (primary/secondary)
SCD30    serial number, firmware version
SPS30    serial number, product type, firmware version
I2C      bus device path 및 클럭 주파수
```

`git_dirty == true` 인 상태의 결과는 논문/보고서 수치로 인용하지 않는다.

**측정값 보정 금지**: baseline correction, calibration coefficient, threshold는 별도로 검증된
절차를 통해서만 도입한다. 측정 스크립트가 값을 자동 보정하거나 분류하지 않는다.
