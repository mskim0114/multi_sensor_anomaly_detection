# Jetson Orin Nano 초기 세팅 가이드

이 문서는 새 Jetson Orin Nano 보드를 받을 때마다 제조공장 멀티모달 센서 이상상태 예측 프로젝트를 같은 상태로 재현하기 위한 기준 절차다.

목표:

- Jetson OS/JetPack/CUDA/TensorRT/Python 환경을 확인한다.
- ONNX/TensorRT 추론 검증 스크립트가 GPU로 통과하게 만든다.
- PureThermal 열화상 카메라와 SPS30 미세먼지 센서를 실제로 읽는다.
- 현장 전용 데이터셋이 없다는 전제를 두고, 센서 설치 후 정상 데이터 수집과 라벨링 절차를 시작한다.
- 1차 설치에서는 센서를 추가하지 않고 기존 모델 입력 구조와 맞는 열화상, NTC, 미세먼지, 전류 센서를 우선한다.
- 다음 보드에서도 같은 과정을 반복할 수 있도록 GitHub에 문서와 스크립트를 남긴다.

## 기준 링크

- NVIDIA Jetson Orin Nano Developer Kit User Guide: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/index.html
- NVIDIA JetPack: https://developer.nvidia.com/embedded/jetpack
- NVIDIA Jetson downloads: https://developer.nvidia.com/embedded/downloads
- Sensirion SPS30 Python driver: https://github.com/Sensirion/python-i2c-sps30
- Sensirion SPS30 I2C docs: https://sensirion.github.io/python-i2c-sps30/

## 현재 성공 기준 환경

2026-07-23 현재 이 보드에서 성공한 기준값이다.

```text
Board compatible: nvidia,p3768-0000+p3767-0005-super
L4T: R36.5.0
Python: 3.10.12
CUDA toolkit: 12.6.11
TensorRT: 10.3.0
ONNX Runtime GPU: 1.24.0
ONNX Runtime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
Power mode: MAXN_SUPER
```

확인 명령:

```bash
cat /etc/nv_tegra_release
python3 --version
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
nvpmodel -q
```

## 1. OS/JetPack 설치

새 보드는 NVIDIA 공식 문서의 최신 Jetson Orin Nano Developer Kit 설치 절차를 따른다.

주의:

- 새 보드/새 펌웨어는 최신 JetPack 이미지로 바로 설치 가능할 수 있다.
- 기존 펌웨어 보드는 NVIDIA 안내에 따라 firmware/UEFI 업데이트가 먼저 필요할 수 있다.
- 우리 프로젝트에서 현재 검증된 계열은 JetPack 6.x, L4T R36.x, Python 3.10 환경이다.

초기 부팅 후 기본 확인:

```bash
cat /etc/nv_tegra_release
uname -a
python3 --version
df -h
free -h
```

## 2. 기본 패키지 설치

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs \
  python3-pip python3-venv python3-dev \
  i2c-tools v4l-utils \
  gstreamer1.0-tools ffmpeg \
  build-essential pkg-config
```

센서/카메라 장치 권한을 위해 사용자를 그룹에 추가한다.

```bash
sudo usermod -aG i2c,video,gpio,render,dialout $USER
```

그룹 변경 후에는 로그아웃/로그인 또는 재부팅이 필요하다.

확인:

```bash
groups
ls -la /dev/i2c-1 /dev/i2c-7
```

## 3. 전원/성능 모드

현재 보드는 `MAXN_SUPER`에서 검증했다.

```bash
nvpmodel -q
sudo jetson_clocks --show
```

필요하면 Jetson 모델별로 가능한 전원 모드를 먼저 확인한 뒤 설정한다. 보드/JetPack 버전에 따라 mode 번호가 달라질 수 있으므로 무조건 숫자를 고정하지 않는다.

```bash
sudo nvpmodel -q --verbose
```

## 4. 프로젝트 가져오기

GitHub 저장소가 준비되면 새 보드에서는 clone으로 시작한다.

```bash
mkdir -p ~/projects
cd ~/projects
git clone <factory_safety_repo_url> factory_safety
cd factory_safety
```

GitHub 저장소가 아직 없고 tar 파일로 받았을 때:

```bash
mkdir -p ~/projects/factory_safety
tar xf ~/Downloads/factory_safety.tar -C ~/projects/factory_safety --strip-components=1
cd ~/projects/factory_safety
```

대용량 데이터와 모델 파일은 GitHub 일반 commit에 넣지 않는다. 필요하면 Git LFS, release artifact, 사내 NAS, USB 복사 중 하나로 관리한다.

## 5. Python 패키지

현재 Jetson에서 성공한 주요 패키지:

```text
onnxruntime-gpu==1.24.0
numpy==2.2.6
sensirion-i2c-sps30==1.0.0
sensirion-i2c-driver==1.0.2
sensirion-driver-adapters==2.3.1
tensorrt==10.3.0
```

설치:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --user numpy onnxruntime-gpu sensirion-i2c-sps30
```

OpenCV는 Jetson 시스템 패키지와 `numpy` ABI가 충돌할 수 있다. PureThermal 캡처 스크립트는 OpenCV 없이 동작하도록 작성되어 있으므로, OpenCV가 필요한 작업을 시작하기 전 아래 확인을 먼저 한다.

```bash
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## 6. ONNX/GPU 검증

```bash
cd ~/projects/factory_safety/jetson_deploy
python3 scripts/01_check_environment.py
python3 scripts/02_benchmark_latency.py --runs 200
python3 scripts/03_verify_accuracy.py --small --provider cuda
python3 scripts/04_realtime_pipeline.py --n 300 --stride 10 --provider cuda
python3 scripts/05_summary.py
```

기대:

- `CUDAExecutionProvider` 또는 `TensorrtExecutionProvider`가 보여야 한다.
- 정확도 검증 match rate는 99% 이상이어야 한다.
- 실시간 파이프라인이 1초 주기보다 충분히 빠르게 돌아야 한다.

TensorRT 엔진이 필요할 때:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model/model_v2plus.onnx \
  --saveEngine=model/model_v2plus_fp16.trt \
  --fp16 \
  --minShapes=sensor:1x30x8,thermal:1x30x120x160 \
  --optShapes=sensor:1x30x8,thermal:1x30x120x160 \
  --maxShapes=sensor:1x30x8,thermal:1x30x120x160
```

## 7. PureThermal Mini Pro 확인

정상 연결 시 USB/UVC 장치로 보인다.

```bash
lsusb
v4l2-ctl --list-devices
gst-device-monitor-1.0 Video/Source
```

현재 성공한 장치 특징:

```text
USB: 1e4e:0100 Cubeternet WebCam
Device: GroupGets PureThermal (fw:v1.3.0)
Format: GRAY16_LE 160x120 9fps
```

캡처:

```bash
cd ~/projects/factory_safety/jetson_deploy
python3 scripts/06_capture_purethermal.py --count 1
```

기대 출력:

- `results/camera/*_gray16.raw`
- `results/camera/*_preview.png`
- `results/camera/*_celsius.csv`
- `results/camera/*_metadata.json`

## 8. SPS30 미세먼지 센서 확인

SPS30은 I2C standard mode `100 kbit/s`까지만 지원한다. 현재 Jetson Orin Nano Super DevKit에서는 pin 27/28의 `/dev/i2c-1`이 100 kHz라 이쪽을 사용한다.

배선:

```text
SPS30 VDD -> Jetson pin 2 또는 pin 4, 5V
SPS30 SDA -> Jetson pin 27, /dev/i2c-1
SPS30 SCL -> Jetson pin 28, /dev/i2c-1
SPS30 SEL -> Jetson GND
SPS30 GND -> Jetson GND
```

주의:

- `SEL`은 전원 인가 시점부터 GND에 연결되어 있어야 I2C 모드로 시작한다.
- SDA/SCL 풀업은 3.3V 기준이어야 한다. 5V로 풀업하지 않는다.
- 필요하면 SDA/SCL 각각 10kΩ pull-up을 3.3V에 추가한다.
- 테스트 중에는 배선을 짧게 둔다.

확인:

```bash
sudo i2cdetect -r -y 1
```

정상이라면 `0x69`가 보인다.

측정:

```bash
cd ~/projects/factory_safety/jetson_deploy
python3 scripts/07_read_sps30.py --samples 10 --interval 1.0
```

현재 성공 예:

```text
serial_number: E95C50BEF297082A
product_type: 00080000
firmware_version: (2, 3)
device_status: 0
```

결과 파일:

```text
results/sps30/*.csv
results/sps30/*.json
```

## 9. ADS1115, NTC, 전류 센서 계획

1차 설치 범위는 기존 모델 입력 구조와 맞는 센서로 제한한다. NTC는 ADS1115를 통해 읽는다. 전류 센서는 아직 전체 실측 검증 전 단계다.

- ADS1115는 Jetson I2C에 연결하고 3.3V 구동을 우선한다.
- ADS1115 2개는 주소를 분리한다. 예: `0x48`, `0x49`.
- NTC 10K 3950은 10kΩ 기준 저항 전압분배 후 ADS1115로 읽는다.
- DHS20P400A-CL420 전류 센서는 20~30V 전원이 필요하다. 보통 24V DC 전원을 준비한다.
- DHS20P400A-CL420의 4-20mA 출력은 shunt resistor로 전압 변환 후 ADS1115로 읽는다.
- 전류 센서용 shunt는 100Ω 정밀 저항을 우선 구매/사용한다.

상세 메모:

- `docs/센서_보유목록.md`
- `docs/SPS30_Jetson_I2C_연결.md`
- `docs/NTC10K_ADS1115_Jetson_연결.md`
- `docs/이상상태_시나리오_및_데이터수집전략.md`
- `docs/Jetson_Orin_Nano_40pin_pinmap.md`

## 10. 새 보드 인수 테스트 체크리스트

새 Jetson 세팅이 끝났다고 판단하려면 아래가 모두 통과해야 한다.

```text
[ ] cat /etc/nv_tegra_release 기록
[ ] nvpmodel -q 기록
[ ] python3 scripts/01_check_environment.py 통과
[ ] onnxruntime providers에 CUDA 또는 TensorRT 표시
[ ] python3 scripts/02_benchmark_latency.py --runs 200 통과
[ ] python3 scripts/03_verify_accuracy.py --small --provider cuda 통과
[ ] PureThermal이 /dev/video*로 보임
[ ] python3 scripts/06_capture_purethermal.py 성공
[ ] SPS30이 sudo i2cdetect -r -y 1 에서 0x69로 보임
[ ] python3 scripts/07_read_sps30.py 성공
[ ] ADS1115가 sudo i2cdetect -r -y 1 에서 0x48로 보임
[ ] python3 scripts/08_read_ntc_ads1115.py 성공
[ ] 결과 JSON/CSV를 results/ 아래에 저장
[ ] 현장 데이터셋이 없으므로 정상 운전 로그 수집 시작
[ ] BME680, SGP30, SCD30은 1차 설치에서 제외
[ ] 세팅 중 변경사항을 docs/오류_및_해결_로그.md 또는 관련 센서 문서에 기록
```

## 11. GitHub 관리 규칙

GitHub에는 코드, 문서, 작은 설정 파일만 올린다.

올릴 것:

```text
README.md
docs/
configs/
src/
jetson_deploy/README.md
jetson_deploy/scripts/
.gitignore
```

기본적으로 올리지 않을 것:

```text
data/
cache/
results/
jetson_deploy/results/
jetson_deploy/reference/*.npz
jetson_deploy/model/*.onnx
jetson_deploy/model/*.trt
*.tar
*.tar.gz
```

대용량 모델/데이터가 필요하면 Git LFS 또는 GitHub Release artifact로 따로 관리한다.

초기 GitHub 업로드 예:

```bash
cd ~/projects/factory_safety
git init
git branch -M main
git add README.md docs configs src jetson_deploy/README.md jetson_deploy/scripts .gitignore
git commit -m "docs: add Jetson setup guide"
git remote add origin <factory_safety_repo_url>
git push -u origin main
```

이미 저장소가 있을 때:

```bash
git status
git add docs/Jetson_Orin_Nano_초기세팅_가이드.md \
        docs/Jetson_Orin_Nano_40pin_pinmap.md \
        docs/SPS30_Jetson_I2C_연결.md \
        jetson_deploy/scripts/07_read_sps30.py \
        .gitignore
git commit -m "docs: document Jetson sensor setup"
git push
```

## 12. GitHub 인증 방식

Jetson에서 매번 GitHub token을 입력하지 않도록 SSH key 방식을 기본으로 사용한다. Personal Access Token은 채팅이나 문서에 남기지 않는다. 토큰이 노출되면 즉시 GitHub에서 revoke하고 새로 발급한다.

새 Jetson에서 SSH key를 만드는 예:

```bash
ssh-keygen -t ed25519 -C "jetson-orin-nano"
cat ~/.ssh/id_ed25519.pub
```

`cat`으로 출력된 공개키를 GitHub 웹에서 등록한다.

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

등록 후 프로젝트 remote를 SSH 주소로 바꾼다.

```bash
cd ~/projects/factory_safety
git remote set-url origin git@github.com:mskim0114/multi_sensor_anomaly_detection.git
ssh -T git@github.com
git push
```

여러 Jetson 보드에 같은 개인 token을 복사해서 쓰지 않는다. 여러 대에 배포할 때는 보드별 SSH key를 만들고, 필요하면 저장소 단위 deploy key 또는 별도 machine account를 사용한다.
