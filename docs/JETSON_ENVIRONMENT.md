# Jetson 환경 실측 기록

이 문서는 **실제 장비에서 read-only로 수집한 값**만 담는다. 추정치를 넣지 않는다.
정책은 [ENVIRONMENT_POLICY.md](ENVIRONMENT_POLICY.md) 참조.

- 수집 일자: **2026-08-31**
- 수집 호스트: `keti-kms`

---

## 1. 하드웨어 / OS

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 보드 | Jetson Orin Nano 8GB **Super** DevKit | `TNSPEC 3767-300-0005-W.1-1-0-jetson-orin-nano-devkit-super-` |
| Tegra chip | `TEGRA_CHIPID 0x23` (Orin) | `/etc/nv_boot_control.conf` |
| L4T | **R36 REVISION 5.0** (JetPack 6.x) | `/etc/nv_tegra_release` |
| L4T 빌드 | `GCID 43688277`, 2026-01-16, `KERNEL_VARIANT: oot` | 〃 |
| 커널 | `5.15.185-tegra`, aarch64, SMP PREEMPT | `uname -a` |
| OS | Ubuntu **22.04.5 LTS** (jammy) | `/etc/os-release` |
| 전력 모드 | `MAXN_SUPER` (mode 2) | `nvpmodel -q` |
| 부팅 스토리지 | `mmcblk0` (234 GB, 39% 사용) | `/etc/nv_boot_control.conf`, `df` |

`dpkg-query -W 'nvidia-jetpack*'` → **해당 패키지 없음** (메타패키지 미설치. 개별 구성요소로 설치됨)

## 2. CUDA / TensorRT

| 항목 | 버전 | 패키지 |
|---|---|---|
| CUDA | **12.6** (`V12.6.68`) | `cuda-*-12-6` (`cuda-cccl-12-6 12.6.37-1`, `cuda-compiler-12-6 12.6.11-1`, …) |
| TensorRT | **10.3.0** | APT `tensorrt 10.3.0.30-1+cuda12.5` |
| TensorRT python | 10.3.0 | `/usr/lib/python3.10/dist-packages/tensorrt` |
| ONNX parser | 10.3.0 | `libnvonnxparsers10 10.3.0.30-1+cuda12.5` |

## 3. Python

| 항목 | 값 |
|---|---|
| 시스템 Python | **3.10.12** (`/usr/bin/python3`) |
| JETSON-RUNTIME venv | `$HOME/venvs/factory_runtime` (`--system-site-packages`) |
| JETSON-LAB venv (보존) | `$HOME/venvs/factory_sensors` |

### JetPack/APT 제공 Python 패키지 (pip requirements에 넣지 말 것)

| 모듈 | 버전 | 제공 패키지 | 위치 |
|---|---|---|---|
| `tensorrt` | 10.3.0 | `tensorrt` | `/usr/lib/python3.10/dist-packages` |
| `cv2` | **4.8.0** (CUDA 빌드) | JetPack OpenCV | `/usr/lib/python3.10/dist-packages` |
| `Jetson.GPIO` | 2.1.9 | `python3-jetson-gpio 2.1.9ubuntu1` | `/usr/lib/python3/dist-packages` |
| `numpy` | 1.21.5 | `python3-numpy 1:1.21.5-1ubuntu22.04.1` | 〃 |
| `yaml` | 5.4.1 | `python3-yaml 5.4.1-1ubuntu1` | 〃 |
| `PIL` | 9.0.1 | `python3-pil 9.0.1-1ubuntu0.3` | 〃 |
| `matplotlib` | 3.5.1 | `python3-matplotlib 3.5.1-2build1` | 〃 |
| `pandas` | 1.3.5 | `python3-pandas 1.3.5+dfsg-3` | 〃 |

`cuda` / `pycuda` Python 바인딩은 **설치되어 있지 않다**. 현재 코드가 사용하지 않는다.

### factory_runtime venv 소유 패키지 (2026-08-31 설치)

```
numpy                          1.26.4
onnxruntime-gpu                1.24.0      <- jetson-ai-lab 인덱스
sensirion-i2c-sps30            1.0.0
sensirion-i2c-scd30            1.1.1
sensirion-i2c-driver           1.0.2
sensirion-driver-adapters      2.3.1
sensirion-driver-support-types 1.2.1
sensirion-shdlc-driver         0.1.5
sensirion-shdlc-sensorbridge   0.2.0
pyserial                       3.5
intelhex                       2.3.0
flatbuffers                    25.12.19
```

`PYTHONNOUSERSITE=1` 상태에서 **user-site(`~/.local`)에서 로드되는 모듈 0건** 확인.

ONNX Runtime execution providers 실측:
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 4. I2C 매핑

device-tree와 실측 검출로 확정했다. **버스 번호를 추측하지 말고 이 표를 따른다.**

| 40핀 물리 핀 | Linux device | DT 노드 | 클럭 | 용도 |
|---|---|---|---|---|
| Pin 3 (SDA) / Pin 5 (SCL) | **`/dev/i2c-7`** | `c250000.i2c` | **400 kHz** | MAIN: ADS1115 `0x48`, SGP30 `0x58` |
| Pin 27 (SDA) / Pin 28 (SCL) | **`/dev/i2c-1`** | `c240000.i2c` | **100 kHz** | SLOW: SPS30 `0x69`, SCD30 `0x61` |

존재하는 모든 I2C 어댑터:
```
i2c-0  3160000.i2c              i2c-1  c240000.i2c   (40핀 pin 27/28)
i2c-2  3180000.i2c              i2c-4  Tegra BPMP I2C adapter
i2c-5  31b0000.i2c              i2c-7  c250000.i2c   (40핀 pin 3/5)
i2c-9  NVIDIA SOC i2c adapter 0 (display)
```

**`/dev/i2c-1` 의 온보드 장치** — 커널이 점유 중이며 정상이다. `i2cdetect` 에서 `UU` 로 보인다.
```
fusb301@0x25      USB-C 포트 컨트롤러
power-sensor@0x40 INA3221 전력 모니터
```

### ⚠️ `i2cdetect` 사용 시 주의

Tegra I2C 어댑터는 **SMBus Quick Command를 지원하지 않는다** (`i2cdetect -F` 로 확인).
따라서 `i2cdetect -y N` 의 기본 auto 모드는 **0x30–0x37 과 0x50–0x5F 만 실제로 probe**하고
나머지 주소는 건드리지 않는다.

```
Warning: Can't use SMBus Quick Write command, will skip some addresses
```

즉 `0x48`, `0x61`, `0x69` 는 스캔 표에서 빈칸으로 보이지만 **"없음"이 아니라 "검사되지 않음"** 이다.
주소 존재 확인은 실제 드라이버로 정식 레지스터 read를 수행해서 판정한다.

## 5. SPI 상태 및 헤더 매핑 (2026-08-31 증명)

```
/dev/spidev0.0  /dev/spidev0.1   <- 3210000.spi (spi_master spi0), DT status=okay
/dev/spidev1.0  /dev/spidev1.1   <- 3230000.spi (spi_master spi1), DT status=okay
```

Jetson-IO 수동 설정 없이 노드가 이미 존재한다. device-tree 자식 노드는 네 개 모두
`compatible = tegra-spidev`, `spi-max-frequency = 50000000`, `reg` = CS 인덱스(0 또는 1)다.

### 물리 핀 19/21/23/24 → `/dev/spidev0.0` (독립 근거 4개)

| # | 근거 | 내용 |
|---|---|---|
| 1 | **sysfs** | `/dev/spidev0.0` → `/sys/devices/platform/bus@0/3210000.spi/spi_master/spi0/spi0.0`, DT 자식 `spi@0`의 `reg`(CS) = 0 |
| 2 | **Blinka board 정의** (`adafruit_blinka/board/nvidia/jetson_orin_nx.py`) | `D11=Z03(SCK)`, `D10=Z05(MOSI)`, `D9=Z04(MISO)`, `D8=Z06(CE0)`. Blinka의 D 번호는 RPi BCM 규약이며 BCM11/10/9/8 = 물리 핀 23/19/21/24 |
| 3 | **Blinka spiPorts** (`microcontroller/tegra/t234/pin.py:131`) | `spiPorts = ((0, Z03, Z05, Z04),)` — T234의 SPI 포트는 **id 0 하나뿐**이고 그 SCK/MOSI/MISO가 `board.SCK/MOSI/MISO`와 일치. Blinka는 port id N을 `/dev/spidev<N>.<CS>`로 연다 → **port 0 → `/dev/spidev0.x`**, CE0 → **`/dev/spidev0.0`** |
| 4 | **저장소 핀맵 문서** (`Jetson_Orin_Nano_40pin_pinmap.md`, 이 보드에서 작성) | 핀 19=MOSI, 21=MISO, 23=SCK, 24=CS0, 26=CS1 — 한 컨트롤러에 CS 2개 = `spidev0.0`/`spidev0.1` 구조와 일치 |

**명명 규칙 차이 주의 (모순 아님)**: 저장소 핀맵 문서는 이 컨트롤러를 `SPI0`으로,
Blinka/NVIDIA Tegra 레지스터 규약은 `SPI1`(base `0x3210000`)로 부른다. 네 근거 모두
**물리 핀 기능과 컨트롤러 정체에는 일치**하며, 라벨만 0-based / 1-based로 다르다.

**결론**: BME680의 CS가 물리 핀 24에 연결되어 있으므로 대상 장치는 **`/dev/spidev0.0`** 이다.
다른 `spidev` 노드를 순차 probing 하지 않는다.

**미검증 사항**: 노드 존재와 매핑은 증명되었으나, 해당 핀이 실제로 SPI 기능으로
pinmux 되어 있는지는 SPI 트랜잭션 없이는 확인할 수 없다. BME680 드라이버는 아직
설치하지 않았고 측정도 하지 않았다.

## 6. USB 열화상

```
lsusb  : Bus 001 Device 006: ID 1e4e:0100 Cubeternet WebCam
by-id  : usb-GroupGets_PureThermal__fw:v1.3.0__0003001d-5101-3133-3332-373300000000
device : /dev/video0, /dev/video1
driver : uvcvideo
format : GRAY16_LE 160x120 @9fps
```

`1e4e:0100` 이 "Cubeternet WebCam" 으로 표시되는 것은 PureThermal 보드의 알려진 USB descriptor다.
FLIR Lepton 3.5 + PureThermal Mini Pro, USB-to-JST-SR 어댑터로 직결. 만능기판 I2C/SPI 회로와 무관.

## 7. 장치 권한

```
/dev/i2c-*    root:i2c   crw-rw----
/dev/spidev*  root:gpio  crw-rw----
/dev/video*   root:video crw-rw----+
```

사용자 `keti` 는 `i2c`(116), `gpio`(999), `video`(44) 그룹에 모두 속해 있어 **sudo 없이 접근 가능**하다.

```
uid=1000(keti) gid=1000(keti) groups=...,44(video),116(i2c),995(docker),999(gpio)
```

새 보드에서는 다음이 필요하다 (재로그인 필수):
```bash
sudo usermod -aG i2c,gpio,video "$USER"
```

---

## 8. 검증된 센서 상태 (2026-08-31)

| 센서 | Interface | Bus / Device | Address | 상태 | 마지막 실측값 |
|---|---|---|---|---|---|
| ADS1115 | I2C | `/dev/i2c-7` | `0x48` | **PASS** | — |
| NTC 10K B3950 | ADS1115 A2 | — | — | **PASS** | 24.05 °C (R 10,430 Ω) |
| SCT024TS CT front-end | ADS1115 AIN0−AIN1 (하드웨어 differential) | `/dev/i2c-7` | `0x48` | **front-end PASS** / known-load calibration **PENDING** | zero-current baseline 확보 |
| SPS30 | I2C | `/dev/i2c-1` | `0x69` | **PASS** | PM1.0 5.23 / PM2.5 6.77 / PM10 8.29 µg/m³ |
| SCD30 | I2C | `/dev/i2c-1` | `0x61` | **PASS** | CO2 702 ppm / 28.67 °C / 48.0 % |
| FLIR Lepton 3.5 | USB UVC | `/dev/video0` | — | **PASS** | 27.34–35.93 °C, mean 29.07 |
| SGP30 | I2C | `/dev/i2c-7` | `0x58` | **PASS** (communication PASS / dynamic response YES) | eCO2 413 ppm / TVOC 4 ppb (65샘플, 68.4 s) |
| BME680 | SPI | **`/dev/spidev0.0`** | — | **FAIL — `spi1` 활성화 후에도 무응답** (pinmux 원인 배제됨) | chip ID `0x00` (기대 `0x61`) |

### 센서 식별 정보

| 센서 | serial / 식별자 | firmware |
|---|---|---|
| SPS30 | `E95C50BEF297082A` (product type `00080000`) | `(2, 3)` |
| SCD30 | `3115957-3117121-204041148` | `(3, 66)` |
| SGP30 | `000001B9391C` (48-bit) | feature set 검증 통과 |
| BME680 | chip ID `0x0` 반환 (BME680 은 `0x61`) | 통신 미확립 |
| PureThermal | `0003001d-5101-3133-3332-3733...` | `v1.3.0` |

### CT front-end 회로 및 zero-current baseline

```
CT k  -> ADS1115 A0
CT l  -> ADS1115 A1 + VBIAS (3.3V - 10k - VBIAS - 10k - GND)
0.68 Ω / 5W burden resistor across A0-A1
YHDC SCT024TS  400 A / 1 A
```

ADS1115 설정: MUX = differential AIN0-AIN1, PGA = ±2.048 V, MODE = continuous, DR = 860 SPS
(Config register `0x04E3`)

스케일: `I_primary_rms = Vrms × 588.2352941176471` (= Vrms / 0.68 × 400)
**검증된 설계 측정 범위: 0 ~ 400 A RMS** (CT 정격). ±2.048 V PGA는 400 A 정격에서 ADC
headroom을 확보하기 위한 선택이며, 측정 범위를 정격 이상으로 확장하지 않는다.

zero-current baseline (CT window에 도체 없음, 3초 × 10회, 총 25,810 샘플):

| 항목 | 값 |
|---|---|
| AC RMS noise floor | **33.750 ± 0.693 µV** (median 33.904, P95 34.552, P99 34.631) |
| 등가 1차 전류 | **0.019853 ± 0.000408 A** |
| 차동 DC 오프셋 | +1.535 ± 0.743 µV |
| raw code 분포 | 전 샘플이 −2 … +2 코드(∓125 µV) 내 |
| 실측 sample rate | 859.97 ~ 860.00 SPS |

이 값은 **실제 전류가 아니라 측정계의 noise floor**다. baseline correction이나
current threshold로 사용하지 않는다.

### 관찰된 특이점

- **A0/A1 common-mode 드리프트**: 40초 동안 1.7052 ~ 1.7192 V (약 14 mV, 3.3V 레일의 0.4%)로
  불규칙하게 움직였다. 하드웨어 differential MUX가 common-mode를 제거하므로 차동 측정값에는
  영향이 없었다 (상관 없음 확인).
- **single-ended 차분 사용 금지 근거**: A0/A1을 각각 single-ended로 읽어 뺀 값은
  −0.083 ~ +0.917 mV로 흩어졌는데(최대 0.54 A 상당), 하드웨어 differential 측정값은
  0.3 ~ 3.0 µV였다. 약 300배 차이. 반드시 하드웨어 MUX를 쓴다.
- **SCD30 온도**: NTC 24.05 °C 대비 28.67 °C로 약 4.6 °C 높다. SCD30 자체 발열의 알려진 특성이며
  `set_temperature_offset` 은 변경하지 않았다.
- **SCD30 clock stretching**: 최대 150 ms 요구 사양이지만 100 kHz 버스(`/dev/i2c-1`)에서
  timeout / NACK / CRC / Remote I/O error **0건**. 400 kHz MAIN 버스에서는 규격 초과이므로 쓰지 않는다.

---

## 9. 추론 런타임 실측 (2026-08-31, factory_runtime)

### numpy 1.26.4 회귀 검증 — PASS

`03_verify_accuracy.py --small --provider cuda` 를 기존 결과와 동일 provider로 실행해
numpy만 단일 변수로 비교했다 (`auto` 는 TensorRT 를 선택해 교란변수가 된다).

| 필드 | numpy 2.2.6 (기존) | numpy 1.26.4 (신규) | 동일 |
|---|---|---|---|
| provider | CUDAExecutionProvider | CUDAExecutionProvider | ✓ |
| n_samples | 100 | 100 | ✓ |
| match_rate_vs_pc_onnx | 1.0 | 1.0 | ✓ |
| accuracy | 0.96 | 0.96 | ✓ |
| **macro_f1** | **0.9601262674881273** | **0.9601262674881273** | ✓ |
| logit_max_abs_diff | 0.001041412353515625 | 0.001041412353515625 | ✓ |
| logit_mean_abs_diff | 0.0002856228675227612 | 0.0002856228675227612 | ✓ |
| per-class F1 (4개) | 전부 동일 | 전부 동일 | ✓ |
| confusion_matrix | `[[23,1,1,0],[0,23,2,0],[0,0,25,0],[0,0,0,25]]` | 동일 | ✓ |

**예측과 모든 지표가 비트 단위로 동일하다.** 타이밍만 14.78 → 15.46 ms/sample로
달라졌는데 이는 측정 노이즈이며 비교 대상이 아니다.

백업: `/tmp/jetson_accuracy_small.before_numpy126.json`, `/tmp/jetson_accuracy_small.numpy126_cuda.json`

### Execution provider 실측 — available / configured / actually used 구분

| Provider | available | configured (session 등록) | inference 실제 성공 | steady 지연 |
|---|---|---|---|---|
| CPUExecutionProvider | ✓ | ✓ | **✓ 성공** | 46.45 ms/sample |
| CUDAExecutionProvider | ✓ | ✓ | **✓ 성공** | 16.27 ms/sample |
| TensorrtExecutionProvider | ✓ | ✓ (3.42 s) | **✗ Segmentation fault** | — |

### ⚠️ TensorrtExecutionProvider Segmentation fault

`model_v2plus.onnx` 로 TensorRT EP session을 만들면 **생성은 성공**하고
`sess.get_providers()` 에 세 EP가 모두 등록되지만, **첫 `sess.run()` 에서
SIGSEGV(exit 139, core dumped)** 로 프로세스가 죽는다. **2/2 재현.**

죽기 직전 TensorRT 경고:
```
[RemoveDeadLayers] Input Tensor thermal is unused or used only at compile-time,
                   but is not being removed.
ModelImporter.cpp:797: Make sure output /Shape_22_output_0 has Int64 binding.  (외 다수)
```

**실무상 영향**: `02_benchmark_latency.py`, `03_verify_accuracy.py`,
`04_realtime_pipeline.py` 의 `--provider auto` 는 TensorRT를 최우선 선택한다.
따라서 **`auto` 로 실행하면 segfault 한다.** 검증이 끝날 때까지 `--provider cuda` 를
명시해서 사용한다.

이는 README 의 "TensorRT FP16 < 2 ms" 목표와 `STATE.md` 의 미완료 항목 ⓶가 왜
아직 측정되지 않았는지에 대한 근거 자료이기도 하다. 원인 규명 전까지 모델 재export,
TensorRT 엔진 재빌드, threshold 변경은 하지 않는다.

CUDA vs CPU logit 최대 절대차 `0.00058174` — 정상적인 부동소수점 차이 범위.

---

## 10. TensorRT EP 원인 분리 (2026-08-31)

`TensorrtExecutionProvider` 의 SIGSEGV 원인을 소스·모델·환경을 변경하지 않고 계층별로 분리했다.

### 10-1. 런타임 fingerprint

| 항목 | 값 |
|---|---|
| onnxruntime | 1.24.0 (wheel sha256 `d980b934…ce0950`) |
| numpy | 1.26.4 |
| TensorRT (python) | 10.3.0 |
| TensorRT (apt) | `10.3.0.30-1+cuda12.5` |
| libnvinfer | `/usr/lib/aarch64-linux-gnu/libnvinfer.so.10.3.0` (2024-09-15, 232 MB) |
| CUDA | 12.6 (`nvcc V12.6.68`) |
| cuDNN | `libcudnn9-cuda-12 9.3.0.75-1` |
| L4T / kernel | R36.5.0 / 5.15.185-tegra |

TensorRT 는 CUDA **12.5** 빌드이고 시스템 CUDA 는 **12.6** 이다 (JetPack 6.x 표준 조합).

### 10-2. 동적 링크 — 문제 없음

`ldd` 결과 **unresolved(`not found`) 0개**:

| 바이너리 | not found |
|---|---|
| `libonnxruntime_providers_tensorrt.so` | 0 |
| `libonnxruntime_providers_cuda.so` | 0 |
| `libonnxruntime_providers_shared.so` | 0 |
| `onnxruntime_pybind11_state.so` | 0 |

`libnvinfer.so.10`, `libnvonnxparser.so.10`, `libcudnn.so.9`, `libcublas.so.12`,
`libcudla.so.1`, `libnvdla_*` 전부 정상 해석. **라이브러리 누락 문제가 아니다.**

### 10-3. crash 지점 (gdb backtrace)

`coredumpctl` 은 미설치(`core_pattern` 은 apport, `ulimit -c 0`)이나 `gdb` 가 이미 설치되어
있어 live backtrace 를 얻었다. dmesg/journalctl 은 권한으로 확인 불가.

```
#0  ?? () from /lib/aarch64-linux-gnu/libnvinfer.so.10          <- SIGSEGV
#1-#4  (libnvinfer.so.10 내부)
#5  onnxruntime::TensorrtExecutionProvider::CreateNodeComputeInfoFromGraph(...)
      ::{lambda(void*, OrtApi const*, OrtKernelContext*)#3}::operator()
      from libonnxruntime_providers_tensorrt.so
#7  onnxruntime::FunctionKernel::Compute
#8  onnxruntime::ExecuteKernel
#9  onnxruntime::LaunchKernelStep::Execute
#14 onnxruntime::ExecuteThePlan
#18 onnxruntime::InferenceSession::Run
```

주 스레드만 `libnvinfer` 안에 있고 나머지 11개 스레드는 전부 futex 대기 → ORT 스레드 경쟁이 아니다.
즉 **ORT TRT EP 의 추론 시점 compute lambda 가 TensorRT 를 호출하는 지점에서 TensorRT 가 죽는다.**

### 10-4. 그래프 분할이 핵심 단서

verbose 로그(`log_severity_level=0`)에서 확인된 것:

```
registered providers : ['TensorrtExecutionProvider', 'CPUExecutionProvider']
input  sensor  ['batch', 30, 8]        <- dynamic batch
input  thermal ['batch', 30, 120, 160] <- dynamic batch
Node(s) placed on [CPUExecutionProvider]. Number of nodes: 3
  ScatterND (/ScatterND), ScatterND (/ScatterND_1), ScatterND (/ScatterND_2)
+ MemcpyToHost / MemcpyFromHost 노드 다수 삽입
Session successfully initialized.
[RemoveDeadLayers] Input Tensor thermal is unused or used only at compile-time,
                   but is not being removed.
stream 0 activate notification 0,1,2 / stream 1 wait on Notification 2,1,0
<SIGSEGV>
```

TensorRT 가 `ScatterND` 3개를 처리하지 못해 그래프가 **TRT 서브그래프 + CPU 노드로 분할**되고,
cross-device memcpy 와 multi-stream 동기화가 삽입된다. crash 는 그 실행 경로에서 발생한다.

`[RemoveDeadLayers] Input Tensor thermal is unused...` 경고는 crash 직전에 관측된
**정황 기록**이다. 이것을 근거로 "thermal 분기가 잘못 처리된다"고 원인을 확정하지 않는다.
확정할 수 있는 것은 아래 §10-6 의 failure boundary 까지이며, **정확한 root cause 는 미해결**이다.

### 10-5. trtexec 독립 검증 — TensorRT 자체는 정상

`/usr/src/tensorrt/bin/trtexec` (이미 설치됨, TensorRT v100300). input shape 는 모델 메타데이터에서
확보한 값만 사용(`sensor:1x30x8`, `thermal:1x30x120x160`). TRT 10.x 는 `--buildOnly` 가
`--skipInference` 로 변경되어 있다.

| 단계 | 결과 |
|---|---|
| **A. ONNX parse + engine build** (`--skipInference`) | **PASSED** — Engine built in 24.54 s |
| **B/C. build + 실제 inference** (FP32) | **PASSED** — Throughput 204.4 qps, GPU compute mean **4.88 ms** (median 4.47, p99 9.51) |
| **B/C. build + 실제 inference** (`--fp16`) | **PASSED** — Throughput 300.0 qps, GPU compute mean **3.33 ms** (median 3.05) |

`trtexec` 는 **그래프 전체를 TensorRT 하나로** 실행한다. 동일 ONNX 가 parse·build·inference 전부
성공하므로 **TensorRT 10.3.0 과 이 모델에는 문제가 없다.**

### 10-6. 결론: 가장 가능성 높은 failure layer

```
모델(ONNX)                  정상  (trtexec parse/build/inference 전부 PASS)
TensorRT 10.3.0             정상  (동일 모델 FP32 4.88 ms / FP16 3.33 ms 실행)
동적 라이브러리 링크          정상  (not found 0개)
ONNX Runtime CUDA EP        정상  (16.27 ms/sample, bit-identical 정확도)
ONNX Runtime CPU EP         정상  (46.45 ms/sample)
--------------------------------------------------------------------------
ONNX Runtime TensorRT EP    FAIL  <- 분할 서브그래프 실행 시 libnvinfer 내부 SIGSEGV
  (ScatterND 3개가 CPU 로 빠지면서 생기는 TRT/CPU 혼합 실행 + dynamic shape 경로)
```

정확한 표현: **failure is isolated to the ORT TensorRT EP partition/execution
integration path; the exact root cause remains unresolved.**

**운영 규칙**: `TensorRT = BROKEN/EXPERIMENTAL`, `CUDA = VERIFIED`.
`auto` provider 는 CUDA 를 우선 선택하며 TensorRT 를 후보에 넣지 않는다 (§12 참조).

TensorRT provider option (`trt_min_subgraph_size`, `trt_max_partition_iterations` 등)으로
production 코드를 실험하지 않았다. 그 실험은 별도 승인 사항이다.

참고: FP16 3.33 ms 는 README 의 "TensorRT FP16 < 2 ms" 목표에 도달하지 못한다.
다만 CUDA EP 의 16.27 ms 보다는 크게 빠르므로, EP 통합 문제가 해결되면 얻을 이득은 있다.

### 10-7. check_environment.py 의미 정정

`check_environment.py` 가 `FAILURES: none` 을 출력하면서도 TensorRT 추론이 죽는 모순을
없애기 위해, provider 절의 출력 의미를 명확히 했다 (checker 는 계속 read-only/lightweight 유지,
모델을 실행하지 않는다).

```
CPU/CUDA/TensorRT provider available   PASS
CPU/CUDA/TensorRT actual inference     NOT TESTED BY THIS CHECKER
+ TensorRT KNOWN ISSUE 안내 및 이 문서 참조
```

**실제 추론 검증은 `03_verify_accuracy.py --small --provider cuda` 또는 provider diagnostic 으로 수행한다.**

---

## 11. BME680 SPI 실측 (2026-08-31) — communication FAIL

드라이버 `adafruit-circuitpython-bme680 3.7.16` 을 `factory_runtime` 에만 설치했다.
설치로 인해 numpy / onnxruntime / Jetson.GPIO / cv2 / TensorRT / Sensirion / 기존 Adafruit
버전은 **하나도 변경되지 않았다** (resolver 결과가 모두 설치본과 동일 버전).

Blinka 가 실제로 여는 장치를 코드로 확인: `_SPI(portid)` → `spi.SPI(device=(portid, 0))`
→ `spiPorts` 의 port 0 → **`/dev/spidev0.0`**. 증명된 매핑과 일치한다.

```
spiPorts : ((0, GP47_SPI1_CLK, GP49_SPI1_MOSI, GP48_SPI1_MISO),)
board.SCK/MOSI/MISO/CE0 : GP47_SPI1_CLK / GP49_SPI1_MOSI / GP48_SPI1_MISO / GP50_SPI1_CS0_N
```

**결과**: `board.SPI()` 와 `digitalio.DigitalInOut(board.CE0)` 는 성공했으나
칩 식별에서 실패했다.

```
RuntimeError: Failed to find BME680! Chip ID 0x0      (BME680 은 0x61 을 반환해야 함)
completed_before : ['board.SPI()', 'digitalio.DigitalInOut(board.CE0)']
```

SPI 전송 자체는 OSError 없이 완료되었고 **MISO 가 계속 0 을 반환**한다. 자동 retry, 다른
spidev 노드 probing, CS 변경은 하지 않았다.

`0xFF` 가 아니라 `0x00` 이라는 점이 단서다. 미연결 MISO 는 보통 풀업으로 `0xFF` 로 읽히므로,
MISO 가 어딘가에 로우로 묶여 있거나 클럭/CS 가 센서에 도달하지 못하는 쪽에 가깝다.

**확인 후보 (순서대로, 아직 아무것도 변경하지 않음)**

1. **SPI pinmux 미라우팅** — 이것이 유일하게 확보하지 못한 근거다. `/dev/spidev0.0` 노드 존재와
   4개 매핑 근거는 확인했지만, 물리 핀 19/21/23/24 가 실제로 SPI 기능으로 pinmux 되어 있는지는
   `/sys/kernel/debug/pinctrl` 가 root 전용이라 확인할 수 없었다. 라우팅되지 않았다면
   SCLK/MOSI 가 센서에 도달하지 않고 MISO 는 0 으로 읽힌다.
2. **CS 경합** — Blinka 가 `/dev/spidev0.0`(하드웨어 CS0 = 핀 24)을 열면서 동시에
   `digitalio` 로 같은 핀 24 를 GPIO 출력으로 잡는다. 커널 CS 와 GPIO CS 가 같은 핀을 다툰다.
3. **MISO 배선** — BME680 의 SDO/MISO 가 물리 핀 21 에 실제로 연결되어 있는지
4. **브레이크아웃 핀 순서** — 보드 표기(MISO/SCLK/CS/MOSI/GND/VCC)와 실제 실크스크린이 일치하는지.
   MOSI/MISO 가 바뀌면 정확히 이 증상이 난다
5. **전원** — BME680 VCC(핀 17) 에서 GND 기준 3.15~3.45 V 인지 측정
6. **SPI 모드 선택** — 일부 BME680 브레이크아웃은 I2C 기본이며 SPI 사용 시 별도 조건이 필요하다

**현재 상태**: SPI mapping **PASS**, hardware communication **FAIL**.
`adafruit-circuitpython-bme680` 은 실측 PASS 전이므로 `requirements-jetson.txt` /
`constraints-jetson.txt` 에 **추가하지 않았다**. 따라서 현재 `factory_runtime` 에는
requirements 에 없는 패키지가 하나 존재한다 (아래 재현성 주의 참조).

---

## 12. Provider 정책 (2026-08-31 적용)

`jetson_deploy/scripts/02_benchmark_latency.py`, `03_verify_accuracy.py`,
`04_realtime_pipeline.py` 의 provider 선택을 다음과 같이 변경했다.
(03/04 는 동일한 `pick_provider` 를 각각 중복 보유하며 공통 helper 는 없다. 같은 패치를 양쪽에 적용했다.)

```
AUTO 우선순위 : CUDAExecutionProvider -> CPUExecutionProvider
                TensorrtExecutionProvider 는 AUTO 후보에 포함하지 않는다.

명시 요청     : --provider tensorrt 는 계속 지원하되 경고를 출력한다.
                요청 자체를 막지는 않는다.
```

변경 이유: segfault 는 파이썬에서 잡을 수 없다. 기존 `auto` 는 TensorRT 를 최우선
선택했으므로 기본 실행이 프로세스 전체를 죽였다.

`02_benchmark_latency.py` 는 `--provider` 옵션이 없고 사용 가능한 provider 를 모두
벤치마킹하는 구조였다. 기본 후보에서 TensorRT 를 제거하고 `--include-tensorrt` opt-in
플래그를 추가했다. 기존 코드의 `except Exception` 은 segfault 를 잡을 수 없어
CUDA/CPU 결과까지 함께 사라졌다.

검증 (`--provider auto`):

| 항목 | 값 |
|---|---|
| auto 가 선택한 provider | **CUDAExecutionProvider** |
| n_samples | 100 |
| accuracy | 0.96 |
| macro_f1 | **0.9601262674881273** |
| match_rate_vs_pc_onnx | 1.0 |
| per-class F1 / confusion matrix | 기존과 전부 동일 |

`pick_provider` 단위 확인 (session 생성 없이): `auto`→CUDA, `cuda`→CUDA, `cpu`→CPU,
`tensorrt`→TensorRT + 경고 출력. TensorRT 추론은 재실행하지 않았다.

---

## 13. BME680 RAW hardware-CS SPI 재검증 (2026-08-31) — FAIL

Adafruit CircuitPython 경로가 Linux 하드웨어 CE0 와 `DigitalInOut(board.CE0)` 를 동시에
사용하는 구조일 가능성이 있었으므로, **software-CS layer 를 완전히 우회한** RAW 테스트를
먼저 수행했다.

사용하지 않은 것: `board.SPI()`, `busio.SPI()`, `digitalio.DigitalInOut()`,
`adafruit_bme680.Adafruit_BME680_SPI`, `adafruit_bus_device.SPIDevice`.

backend 는 이미 설치된 **`Adafruit_PureIO.spi.SPI`** 를 사용했다 (`spidev` 모듈은 미설치이며
설치하지 않았다). 이 backend 는 `/dev/spidev<bus>.<dev>` 를 직접 열고 `SPI_IOC_MESSAGE`
ioctl 로 full-duplex 전송하며, CS 는 커널이 관리한다 (`no_cs = False`).

커널에서 읽어온 실제 설정:
```
device        /dev/spidev0.0     (bus 0, CS 0 = 물리 핀 24, 하드웨어 CS)
max_speed_hz  100000
bits_per_word 8
mode          0     (phase False, polarity False)
lsb_first     False (MSB first)
cs_high       False   no_cs False   three_wire False   loop False
```

BME680 SPI 레지스터는 7비트 주소 + bit7 = read 플래그다.
page 0 chip-id 레지스터 `0x50` → read control byte `0x50 | 0x80 = 0xD0`.

| # | TX | RX | chip-id 후보 |
|---|---|---|---|
| 1 | `[0xD0, 0x00]` | `[0x00, 0x00]` | `0x00` |
| 2 | `[0xD0, 0x00]` | `[0x00, 0x00]` | `0x00` |
| 3 | `[0xD0, 0x00]` | `[0x00, 0x00]` | `0x00` |

3회 모두 안정적으로 `0x00`. 기대값은 `0x61`.

```
BME680_RAW_SPI = FAIL
```

**중요**: `0x00` 이라는 값만으로 "MISO 가 GND 에 short" 라고 판정하지 않는다.
floating MISO 의 읽힘값은 플랫폼과 풀 상태에 따라 `0x00` 또는 `0xFF` 로 달라질 수 있다.

**해석**: RAW 하드웨어 CS 경로도 동일하게 실패하므로 **Adafruit software-CS 경합은
단독 원인이 아니다.** 소프트웨어 계층 변경을 중단한다.

미검증으로 남은 항목:
- 물리 핀 19/21/23/24 의 실제 pinmux (`/sys/kernel/debug/pinctrl` 은 root 전용, sudo 미사용)
- 앞선 Blinka 실행이 핀 24 를 GPIO 로 남겼는지 (`/sys/class/gpio` 접근 불가로 확인 불가.
  단 CS 가 고정되어도 MOSI/SCLK/MISO 는 영향받지 않으므로 유일한 원인은 아니다)

### 필요한 사용자 물리 점검

**전원 OFF 상태에서 도통 확인**
```
BME680 VCC  <-> Jetson Pin 17
BME680 GND  <-> Jetson Pin 20
BME680 MOSI <-> Jetson Pin 19
BME680 MISO <-> Jetson Pin 21
BME680 SCLK <-> Jetson Pin 23
BME680 CS   <-> Jetson Pin 24
```

**전원 ON 상태에서 전압만 측정**
```
BME680 VCC - GND 실제 DC 전압 (3.15 ~ 3.45 V 기대)
```

전원이 인가된 상태에서 도통 측정이나 배선 흔들기는 하지 않는다.

### 패키지 정책

`adafruit-circuitpython-bme680 3.7.16` 은 `factory_runtime` 에 설치되어 있으나
**diagnostic-only installed package** 이며 canonical 하지 않다.
이번 디버깅 동안 uninstall 하지 않는다. `requirements-jetson.txt` 에는
**hardware PASS + 최종 software transport PASS** 가 모두 확인된 뒤에만 추가한다.

따라서 `setup_jetson_env.sh` 로 새로 만든 venv 에는 이 패키지가 없다. 의도된 차이다.

---

## 14. BME680 page-0 명시 선택 재검증 (2026-09-02) — mode 0 · mode 3 모두 FAIL

§13 의 RAW 테스트는 read 형식(`[0xD0, 0x00]`)은 맞았으나 **SPI memory page 를 명시적으로
page 0 으로 선택하지 않았다.** Bosch BME680 은 register access 전에 page 선택이 필요하고
Adafruit 드라이버도 이를 수행한다. 배선을 변경하기 전에 이 절차를 포함해 다시 측정했다.

Bosch BME680 SPI memory map:
```
status register SPI address = 0x73,  spi_mem_page = bit 4
page 0 선택      : bit4 = 0
chip id (page 0) : SPI address 0x50
read  control byte : 0x50 | 0x80 = 0xD0     (bit7 = 1 -> read)
write control byte : 0x73                   (bit7 = 0 -> write)
expected chip id   : 0x61
```

각 iteration 마다 **분리된 두 SPI transaction** 을 정확한 순서로 수행했다.
고정 3회 diagnostic 이며 retry loop 가 아니다. 다른 register write 없음, soft reset 없음,
속도 변경 없음. Adafruit CS abstraction 은 전부 우회 (backend = `Adafruit_PureIO.spi.SPI`,
`/dev/spidev0.0`, 커널 하드웨어 CS0).

### mode 0 (CPOL=0, CPHA=0)

커널 readback: `max_speed_hz=100000, bits_per_word=8, mode=0, phase=False, polarity=False, lsb_first=False, cs_high=False, no_cs=False`

| iter | 1. page0 select TX → RX | 2. chip id read TX → RX | chip id |
|---|---|---|---|
| 1 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |
| 2 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |
| 3 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |

**FAIL** (stable `0x00`, 기대 `0x61`)

### mode 3 (CPOL=1, CPHA=1)

Bosch BME680 은 mode 0 과 mode 3 만 공식 지원한다. 그 외 mode 는 시도하지 않았고 속도도 그대로 두었다.

커널 readback: `mode=3, phase=True, polarity=True` (나머지 동일)

| iter | 1. page0 select TX → RX | 2. chip id read TX → RX | chip id |
|---|---|---|---|
| 1 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |
| 2 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |
| 3 | `[0x73 0x00]` → `[0x00 0x00]` | `[0xD0 0x00]` → `[0x00 0x00]` | `0x00` |

**FAIL** (stable `0x00`)

### pinmux 확인 — UNVERIFIABLE (debugfs 로는 판정 불가)

Blinka 매핑을 소스에서 재확인했다 (추측 아님):

```
adafruit_blinka/board/nvidia/jetson_orin_nx.py
    D11 = pin.Z03   SCK/SCLK      D10 = pin.Z05   MOSI
    D9  = pin.Z04   MISO          D8  = pin.Z06   CE0
adafruit_blinka/microcontroller/tegra/t234/pin.py
    Z03 = Pin("GP47_SPI1_CLK")    Z05 = Pin("GP49_SPI1_MOSI")
    Z04 = Pin("GP48_SPI1_MISO")   Z06 = Pin("GP50_SPI1_CS0_N")
    spiPorts = ((0, Z03, Z05, Z04),)
```

BCM 규약상 D11/D10/D9/D8 = 물리 핀 23/19/21/24 이다.

| 물리 핀 | 신호 | Blinka pin | 판정 |
|---|---|---|---|
| 19 | MOSI | `Z05` `GP49_SPI1_MOSI` | **UNKNOWN** |
| 21 | MISO | `Z04` `GP48_SPI1_MISO` | **UNKNOWN** |
| 23 | SCK | `Z03` `GP47_SPI1_CLK` | **UNKNOWN** |
| 24 | CS0 | `Z06` `GP50_SPI1_CS0_N` | **UNKNOWN** |

판정할 수 없었던 이유:

1. `/sys/kernel/debug/pinctrl` 은 root 전용이라 사용자가 직접 read-only sudo 로 조회했다.
   조회는 성공했으나 **결과가 판정 근거가 되지 못했다** (아래 반증 참조).
   debugfs 를 새로 mount 하지 않았고 `pinmux-select` 등 write 인터페이스는 건드리지 않았다.
2. runtime device-tree 에도 헤더 SPI 핀의 pinmux 정보가 없다. 확인한 내용:
   - `spi@3210000` 노드에 `pinctrl-*` 속성이 **없다**
   - `pinmux@2430000` 의 자식은 `eqos_rx_enable/disable`, `pex_rst_c{4,5,6,7,10}_in` 뿐이다
   - `pinmux@c300000` 은 자식 노드가 없다
   - DT 전체에 `pz3`~`pz6` 문자열도 `nvidia,function` 속성도 존재하지 않는다

   즉 Orin 계열은 40핀 pinmux 를 MB1/부트로더 단계에서 적용하며 runtime DT 에 노출하지 않는다.
   `jetson-io` 가 reboot 을 요구하는 이유와 같다.

### 현재 확정 상태

```
BME680 = RAW SPI FAIL (mode 0, mode 3) / PINMUX UNVERIFIED
```

`HARDWARE PATH BLOCKED` 로도, `SPI HEADER PINMUX NOT CONFIGURED` 로도 아직 판정할 수 없다.
두 판정 모두 pinmux 확인을 전제로 하는데 그 확인이 불가능했기 때문이다.

확정된 것과 배제된 것:

| 항목 | 상태 |
|---|---|
| `/dev/spidev0.0` open 및 ioctl 전송 | 성공 (오류 없음) |
| 커널 SPI 설정 적용 (mode/speed/bits/CS) | readback 으로 확인 |
| logical SPI mapping (핀 19/21/23/24 → `/dev/spidev0.0`) | 근거 4개 일치 |
| Adafruit software-CS 경합이 단독 원인 | **배제됨** — RAW 하드웨어 CS 경로도 동일 실패 |
| SPI mode 불일치가 원인 | **배제됨** — mode 0·3 모두 동일 실패 |
| SPI memory page 미선택이 원인 | **배제됨** — page 0 명시 선택 후에도 동일 실패 |
| 물리 핀 pinmux 기능 | **UNVERIFIED** |
| 배선 / 센서 응답 | **UNVERIFIED** |

`chip ID 0x00` 을 근거로 "MISO 가 GND 에 short" 같은 원인을 적지 않는다.
floating MISO 의 읽힘값은 플랫폼과 풀 상태에 따라 달라진다.

### 다음에 필요한 것 (둘 다 아직 수행하지 않음)

**(a) pinmux 확인** — 사용자가 직접 실행해야 하는 read-only sudo 명령:
```bash
sudo ls -la /sys/kernel/debug/pinctrl
sudo cat  /sys/kernel/debug/pinctrl/*/pinmux-pins  | grep -iE 'pz3|pz4|pz5|pz6|spi1'
sudo cat  /sys/kernel/debug/pinctrl/*/pinconf-pins | grep -iE 'pz3|pz4|pz5|pz6'
sudo grep -iE 'spi1|pz[3-6]' /sys/kernel/debug/pinctrl/*/pins
```
mount, devmem, jetson-io, device-tree 수정, pinmux write, GPIO export/write, reboot 은 하지 않는다.

**(b) 물리 점검** — 전원 OFF 상태에서 end-to-end 도통:
```
BME680 VCC  <-> Jetson Pin 17      BME680 GND  <-> Jetson Pin 20
BME680 MOSI <-> Jetson Pin 19      BME680 MISO <-> Jetson Pin 21
BME680 SCLK <-> Jetson Pin 23      BME680 CS   <-> Jetson Pin 24
```
그리고 breakout board 에 **실제 인쇄된 핀 이름과 순서**를 다시 확인해 기록.

전원 ON 상태에서는 **BME680 모듈 자체의 VCC-GND DC 전압만** 측정한다.
전원 인가 상태에서 도통 측정이나 배선 흔들기는 하지 않는다.

### pinmux debugfs 실측과 반증 (2026-09-02)

사용자가 read-only sudo 로 `/sys/kernel/debug/pinctrl` 을 조회했다.
(`sudo cat /path/*` 형태는 실패했는데, glob 을 확장하는 것은 sudo 가 아니라 사용자 셸이고
`/sys/kernel/debug` 가 `drwx------ root root` 여서 확장되지 못하기 때문이다. `sudo sh -c '...'`
로 root 셸 안에서 확장시켜 해결했다.)

BME680 이 쓰는 네 핀의 상태:

```
pin 133 (SPI1_SCK_PZ3):  (MUX UNCLAIMED) (GPIO UNCLAIMED)     물리 핀 23
pin 134 (SPI1_MISO_PZ4): (MUX UNCLAIMED) (GPIO UNCLAIMED)     물리 핀 21
pin 135 (SPI1_MOSI_PZ5): (MUX UNCLAIMED) (GPIO UNCLAIMED)     물리 핀 19
pin 136 (SPI1_CS0_PZ6):  (MUX UNCLAIMED) (GPIO UNCLAIMED)     물리 핀 24
```

`spi1` function 과 그룹은 정의되어 있다:
`function 38: spi1, groups = [ spi1_cs0_pz6 spi1_miso_pz4 spi1_sck_pz3 spi1_cs1_pz7 spi1_mosi_pz5 ]`

**여기서 "SPI 로 mux 되지 않았다" 고 결론내리려 했으나, 반증 테스트가 그 가설을 뒤집었다.**

대조군: 이 보드에서 **실제로 통신 중인** I2C 핀들의 상태를 같은 인터페이스로 확인했다.

```
pin 54 (GEN1_I2C_SCL_PI3): (MUX UNCLAIMED) (GPIO UNCLAIMED)
pin 55 (GEN1_I2C_SDA_PI4): (MUX UNCLAIMED) (GPIO UNCLAIMED)
pin 19 (GEN2_I2C_SCL_PCC7): (MUX UNCLAIMED) (GPIO UNCLAIMED)
pin 20 (GEN2_I2C_SDA_PDD0): (MUX UNCLAIMED) (GPIO UNCLAIMED)
```

전체 집계: `pinmux-pins` 203줄 (2430000.pinmux 169 + c300000.pinmux 34) 중
**claim 된 핀은 단 하나도 없다** (헤더 2줄을 제외하면 `MUX UNCLAIMED` 가 아닌 줄이 0개).

SGP30(0x58), ADS1115(0x48), SPS30(0x69), SCD30(0x61) 이 모두 정상 통신하는 상태에서도
그 핀들이 `MUX UNCLAIMED` 로 보인다. 따라서 이 플랫폼에서 **`MUX UNCLAIMED` 는 pad 라우팅
여부에 대해 아무 정보도 담지 않는다.** Tegra 는 pinmux 를 MB1 부트로더 단계에서 적용하고
Linux pinctrl 은 아무것도 claim 하지 않기 때문이다.

**결론: `SPI HEADER PINMUX NOT CONFIGURED` 가설 철회.**
debugfs pinctrl 인터페이스로는 이 질문에 답할 수 없다.

| 물리 핀 | 신호 | Blinka pin | Tegra pin | 판정 |
|---|---|---|---|---|
| 19 | MOSI | `Z05` | `pin 135 SPI1_MOSI_PZ5` | **UNVERIFIABLE** (이 인터페이스로 판정 불가) |
| 21 | MISO | `Z04` | `pin 134 SPI1_MISO_PZ4` | **UNVERIFIABLE** |
| 23 | SCK | `Z03` | `pin 133 SPI1_SCK_PZ3` | **UNVERIFIABLE** |
| 24 | CS0 | `Z06` | `pin 136 SPI1_CS0_PZ6` | **UNVERIFIABLE** |

### 최종 상태

```
BME680 = RAW SPI FAIL (mode 0, mode 3) / PINMUX UNVERIFIABLE
```

`HARDWARE PATH BLOCKED` 도 `SPI HEADER PINMUX NOT CONFIGURED` 도 주장하지 않는다.
전자는 pinmux 가 SPI 임이 확인되어야 하고 후자는 SPI 가 아님이 확인되어야 하는데,
둘 다 확인할 수 없었다.

남은 미검증 항목은 **물리 배선** 과 **부트 단계 pinmux 설정** 두 가지이며,
소프트웨어로는 더 좁힐 수 없다.

---

## 15. 40핀 헤더 `spi1` function 조회 (2026-09-02) — `NOT_ENABLED`

> **정정 (2026-09-02).** 이 섹션은 원래 "BME680 원인 확정" 으로 작성되었다. 그 판정은
> **§16 에서 반증되었다** — `spi1` 을 활성화한 뒤에도 BME680 은 동일하게 무응답이었다.
> 아래 내용은 조회 당시의 사실 기록으로만 유효하다. 최종 canonical 절차는 §17 과
> `docs/JETSON_SPI_BME680_SETUP.md` 를 본다.

debugfs `pinmux-pins` 로는 판정이 불가능했으므로(§14) NVIDIA 공식 Jetson-IO 의
**read-only 조회 기능**으로 확인했다. 설정 변경은 하지 않았다.

도구는 `Board()` 생성자의 `fio.is_rw(bootdir)` 사전 검사 때문에 `/boot` 에 대한 read/write
권한을 요구한다. 따라서 조회에도 root 가 필요하다 (조회 자체가 쓰기를 하는 것은 아니다.
`config-by-pin.py` 는 `print` 만 하고, `config-by-function.py -l` 은 출력 후 `sys.exit(0)`
하며 변경 코드는 `-o` 또는 function 인자를 줄 때만 도달한다).

### 조회 결과

`config-by-pin.py -p <pin>`:

```
Pin 19 : unused
Pin 21 : unused
Pin 23 : unused
Pin 24 : unused
```

`config-by-pin.py` 전체 출력 중 40핀 헤더에서 기능이 배정된 핀은 다음뿐이다:

```
  3: i2c8     5: i2c8          <- MAIN I2C  (실측: /dev/i2c-7)
  8: uarta   10: uarta
 27: i2c2    28: i2c2          <- SLOW I2C  (실측: /dev/i2c-1)
```
그 외 `unused` 또는 전원/GND. 핀 19/21/23/24/26 은 전부 `unused`.

`config-by-function.py -l enabled`:

```
Header 1 [default]: Jetson 40pin Header
  Enabled functions (pins):
   1. i2c2 (27,28)
   2. i2c8 (3,5)
   3. uarta (8,10)
```

`config-by-function.py -l all`:

```
Header 1 [default]: Jetson 40pin Header
  Supported functions (pins):
   1. aud (7)                    7. pwm1 (15)
   2. extperiph3_clk (29)        8. pwm5 (33)
   3. extperiph4_clk (31)        9. pwm7 (32)
   4. i2c2 (27,28)              10. spi1 (19,21,23,24,26)     <-- 지원되지만 미활성
   5. i8c8 (3,5)                11. spi3 (13,16,18,22,37)
   6. i2s2 (12,35,38,40)        12. uarta (8,10)
                                13. uarta-cts/rts (11,36)
```

### 판정

```
JETSON_IO_SPI_HEADER_CONFIG = NOT_ENABLED
BME680 = BLOCKED (40-pin header spi1 function not enabled)
```

`spi1` 은 **지원 목록에 있고 그 핀 집합이 `(19,21,23,24,26)` 으로 이 프로젝트의 배선과 정확히
일치**한다 (19 MOSI / 21 MISO / 23 SCK / 24 CS0 / 26 CS1). 그러나 enabled 목록에 없고
네 핀 모두 `unused` 다. 즉 **패드가 헤더로 라우팅되어 있지 않다.**

이 시점에는 이것이 `chip ID 0x00` 을 설명한다고 판정했다. **그 판정은 §16 에서 반증되었다** —
`spi1` 활성화 후에도 동일하게 실패했으므로 `NOT_ENABLED` 는 원인이 아니었다(필요조건이었을
뿐이다). `/dev/spidev0.0` 이 열리고 ioctl 이 성공하는 것은 spidev 가 컨트롤러 레벨에서
동작하기 때문이며 패드 라우팅 여부와 무관하다 — 이 부분은 여전히 유효하다.

**BME680 배선 문제로 판정하지 않는다.** 배선은 여전히 미검증이지만, 현재 상태에서는
배선이 완벽해도 동일하게 실패한다.

### function label 주의

Jetson-IO 의 function 이름은 **carrier-board function label** 이며 Linux `/dev/i2c-N`,
`/dev/spidevX.Y` 번호와 대응하지 않는다. 실제로 Jetson-IO 는 핀 3/5 를 `i2c8`, 핀 27/28 을
`i2c2` 로 부르지만, 이 보드에서 실측 확인된 Linux 매핑은 핀 3/5 → `/dev/i2c-7`,
핀 27/28 → `/dev/i2c-1` 이다 (센서 실제 응답으로 확인, §4). 숫자 label 만으로 다른 버스라고
판정하지 않는다. **물리 핀 집합으로 판단해야 한다** — `spi1 (19,21,23,24,26)` 이 우리 SPI 다.

### §14 의 debugfs 관측에 대한 정정된 위치

debugfs 의 `(MUX UNCLAIMED)` 는 이 플랫폼에서 **판정 근거가 될 수 없다** (동작 중인 I2C 핀도
동일하게 표시됨, §14). 결과적으로 결론 방향은 맞았지만 그 근거로는 아무것도 증명할 수 없었다.
**실제 판정 근거는 Jetson-IO 조회이며 debugfs 관측이 아니다.**

### 다음 단계 (별도 승인 필요, 이번에 수행하지 않음)

`spi1` 을 활성화하려면 Jetson-IO 로 DTBO 를 생성하고 **reboot** 이 필요하다.
이번 단계에서는 조회만 했고 다음을 전부 하지 않았다:
Jetson-IO 대화형 실행, `config-by-function.py -o dt|dtbo`, DTBO 생성, device-tree 변경,
pinmux write, reboot, 배선 변경, MOSI-MISO 점퍼, BME680 재측정.

활성화 시 고려사항:
- 핀 19/21/23/24/26 은 현재 전부 `unused` 이므로 다른 기능과 충돌하지 않는다
- 이미 활성화된 `i2c2 (27,28)`, `i2c8 (3,5)`, `uarta (8,10)` 는 핀이 겹치지 않는다.
  즉 SPS30/SCD30(핀 27/28) 과 ADS1115/SGP30(핀 3/5) 는 영향받지 않아야 한다
- reboot 이 필요하므로 진행 시점을 사용자가 정해야 한다
- 활성화 후 `/dev/spidev*` 노드 번호가 바뀔 수 있으므로 재확인이 필요하다

---

## 16. 40핀 헤더 `spi1` 활성화 및 재부팅 후 검증 (2026-09-02)

### Jetson-IO 변경 내역

```
적용 명령 : config-by-function.py -o dt 1="i2c2 i2c8 uarta spi1"     exit 0
생성 DTBO : /boot/jetson-io-hdr40-user-custom.dtbo   (2126 bytes)
rollback  : /boot/extlinux/extlinux.conf.pre_spi1_20260902_145420  (892 bytes, pristine)
            /boot/extlinux/extlinux.conf.jetson-io-backup          (892 bytes, Jetson-IO 자체 백업)
```

`extlinux.conf` 변경은 두 곳뿐이다: `DEFAULT primary` → `DEFAULT JetsonIO`, 그리고
`LABEL JetsonIO` 블록 추가 (`FDT`, `OVERLAYS` 포함). `LABEL primary` fallback 은 온전하며
kernel/initrd 는 변경되지 않았다.

> `extlinux.conf.pre_spi1_20260902_145922` (1322 bytes) 는 pristine 이 아니다 —
> 이미 JetsonIO 항목이 포함된 상태의 백업이다. 롤백에는 `_145420` 을 쓴다.

### 재부팅 후 확인 — `spi1 = ENABLED`

부팅 정상 (uptime 4분, kernel `5.15.185-tegra` 동일, `/proc/cmdline` 의 rootfs/console 인자 동일).
`DEFAULT JetsonIO` 로 부팅되었고 오버레이가 적용되었다.

**근거는 라이브 device-tree 다** (sudo 불필요, Jetson-IO 조회보다 직접적):
재부팅 전 `pinmux@2430000` 의 자식은 `eqos_*`, `pex_rst_*` 뿐이었으나, 이제
`exp-header-pinmux`, `pinctrl-0`, `pinctrl-names` 가 존재한다.

```
/proc/device-tree/bus@0/pinmux@2430000/exp-header-pinmux/
  hdr40-pin19   pins=spi1_mosi_pz5   function=spi1   gpio-mode=0 tristate=1 enable-input=1
  hdr40-pin21   pins=spi1_miso_pz4   function=spi1   gpio-mode=0 tristate=1 enable-input=1
  hdr40-pin23   pins=spi1_sck_pz3    function=spi1   gpio-mode=0 tristate=1 enable-input=1
  hdr40-pin24   pins=spi1_cs0_pz6    function=spi1   gpio-mode=0 tristate=1 enable-input=1
  hdr40-pin26   pins=spi1_cs1_pz7    function=spi1   gpio-mode=0 tristate=1 enable-input=1
```
AON 쪽(`pinmux@c300000`)에도 `exp-header-pinmux` 가 추가되었다 (오버레이의 fragment@1).

| 물리 핀 | 신호 | function | 판정 |
|---|---|---|---|
| 19 | MOSI | `spi1` | **ENABLED** |
| 21 | MISO | `spi1` | **ENABLED** |
| 23 | SCK | `spi1` | **ENABLED** |
| 24 | CS0 | `spi1` | **ENABLED** |
| 26 | CS1 | `spi1` | **ENABLED** |

### SPI 노드 매핑 — 재부팅 전과 동일

```
/dev/spidev0.0, 0.1  ->  3210000.spi (spi_master spi0)   <- 핀 19/21/23/24/26
/dev/spidev1.0, 1.1  ->  3230000.spi (spi_master spi1)
```
노드 번호가 바뀌지 않았으므로 `/dev/spidev0.0` (CS0 = 핀 24) 을 계속 사용한다.

### 기존 기능 보존 — 전부 유지

I2C 장치 노드 목록 변화 없음 (`/dev/i2c-{0,1,2,4,5,7,9}`).
핀 3/5 = `i2c8`, 핀 27/28 = `i2c2`, 핀 8/10 = `uarta` 유지.

**센서 회귀 (재부팅 후, 순차 실행)**

| 센서 | 버스 | 결과 | 실측값 |
|---|---|---|---|
| ADS1115 / NTC | `/dev/i2c-7` `0x48` A2 | **PASS** | 24.69 °C (R 10,139 Ω), exit 0 |
| SGP30 | `/dev/i2c-7` `0x58` | **PASS** | serial `000001B9391C`, 8샘플 오류 0건 |
| SPS30 | `/dev/i2c-1` `0x69` | **PASS** | serial `E95C50BEF297082A`, fw `(2,3)`, status 0 |
| SCD30 | `/dev/i2c-1` `0x61` | **PASS** | serial `3115957-3117121-204041148`, fw `(3,66)` |

SGP30 은 부팅 8초 후 측정이라 `eCO2 400 / TVOC 0` 고정값이었다 — warm-up 구간이며 실패가 아니다.
SCD30 첫 샘플 `CO2 0.00 ppm` 은 `start_periodic_measurement` 직후의 알려진 거동이고,
2·3번째 샘플에서 631 / 781 ppm 으로 정상화되었다.
SPS30 과 SCD30 은 공유 버스이므로 순차 실행했고, 사이에 버스 점유 프로세스가 없음을 확인했다.

FLIR / CT 는 SPI1 DTBO 와 독립적이므로 이번 회귀에서 재검증하지 않았다.

### BME680 — `spi1` 활성화 후에도 **FAIL**

`spi1` 이 확실히 활성화된 상태에서 동일한 RAW hardware-CS 진단을 재실행했다.
배선은 변경하지 않았다.

커널 readback: `mode=0, max_speed_hz=100000, bits_per_word=8, lsb_first=False, cs_high=False, no_cs=False`

| iter | 1. page0 select | 2. chip id read | chip id |
|---|---|---|---|
| 1 | TX `[0x73 0x00]` → RX `[0x00 0x00]` | TX `[0xD0 0x00]` → RX `[0x00 0x00]` | `0x00` |
| 2 | TX `[0x73 0x00]` → RX `[0x00 0x00]` | TX `[0xD0 0x00]` → RX `[0x00 0x00]` | `0x00` |
| 3 | TX `[0x73 0x00]` → RX `[0x00 0x00]` | TX `[0xD0 0x00]` → RX `[0x00 0x00]` | `0x00` |

```
BME680_RAW_SPI = FAIL   (stable 0x00, 기대 0x61)
```

RAW 가 실패했으므로 Adafruit high-level test 는 수행하지 않았고,
`adafruit-circuitpython-bme680` 의 requirements 승격도 하지 않았다 (diagnostic-only 유지).
mode 3 재시도, 배선 변경, retry loop 모두 하지 않았다.

### 원인 후보 갱신 — pinmux 가설 배제

> **정정 (2026-09-03).** 이 시점의 후보 목록은 §17 에서 해소되었다. SPI1·SPI3 컨트롤러는
> 모두 정상으로 확인되었고, BME680 모듈 손상이 현재 가장 유력한 원인이다.
> 중간에 "SPI1 선두비트 소실 결함" 으로 판정한 적이 있으나 **점퍼 접촉 불량 오진**이었다
> (`JETSON_SPI_BME680_SETUP.md` §10-6). 아래는 당시의 후보 기록이다.

```
BME680 = FAIL (spi1 ENABLED 상태에서도 무응답)
```

| 가설 | 상태 |
|---|---|
| Adafruit software-CS 경합 | 배제 (§13) |
| SPI mode 불일치 (0/3) | 배제 (§14) |
| SPI memory page 미선택 | 배제 (§14) |
| spidev 노드 / 논리 매핑 오류 | 배제 |
| **40핀 헤더 `spi1` 미활성** | **배제 — 활성화 후에도 동일 실패** |
| 물리 배선 / 브레이크아웃 핀 순서 | **제거 (§17)** — 모듈 자리 SDI-SDO 점퍼로 배선 왕복 loopback PASS |
| 센서 전원 | **제거 (§17)** — 모듈 VCC-GND 전압 정상 |
| BME680 모듈 자체 불량 | **제거 (§17)** — BME680 ×2 + BMP388 ×1 이 동일 실패 |
| 패드 tristate 설정 | **반증됨 — 아래 참조** |

**tristate 가설과 그 반증 (2026-09-02).** 당시 관찰: `config-by-function.py -o dt` 로 생성된
오버레이가 5개 핀 모두에 `nvidia,tristate = <1>` 을 설정하고 있었고, tristate=1 은 패드를
high-Z 로 두므로 SPI master 출력(SCK/MOSI/CS)에는 부적합해 보였다.

**이 가설은 A/B 시험으로 반증되었다.** 출력 3핀(19/23/24)만 `tristate=0` 으로 바꾼 별도 오버레이를
만들어 부팅한 뒤 라이브 `pinconf-pins` 로 `tristate=0` 을 확인했으나, 물리 Pin19↔Pin21 loopback
결과는 **변화 없이 동일하게 실패**했다. 즉 tristate 는 root cause 가 아니었다.

이 수동 DTBO 경로 전체가 최단 경로가 아니었다. 최종적으로 문제를 해소한 것은 §17 의
**공식 `jetson-io.py` 메뉴 경로**다. 수동 DTBO/DTS/tristate 편집은 앞으로 일반 설정 경로로
쓰지 않는다 (`AGENTS.md` §4).

### 필요한 사용자 물리 점검

**전원 OFF 상태 — end-to-end 도통**
```
BME680 VCC  <-> Jetson Pin 17        BME680 GND  <-> Jetson Pin 20
BME680 MOSI <-> Jetson Pin 19        BME680 MISO <-> Jetson Pin 21
BME680 SCLK <-> Jetson Pin 23        BME680 CS   <-> Jetson Pin 24
```

**브레이크아웃 보드에 실제 인쇄된 핀 이름과 순서를 그대로 기록** — 알려주신 순서
(MISO/SCLK/CS/MOSI/GND/VCC)와 실크스크린이 다르면 MOSI↔MISO 가 바뀌어 정확히 이 증상이 난다.

**전원 ON 상태 — BME680 모듈 자체의 VCC-GND DC 전압만** 측정 (3.15 ~ 3.45 V 기대).
전원 인가 상태에서 도통 측정이나 배선 흔들기는 하지 않는다.

`0x00` 이라는 값만으로 MISO short 같은 원인을 단정하지 않는다.
floating MISO 의 읽힘값은 플랫폼과 풀 상태에 따라 달라진다.

---

## 17. SPI1 canonical setup 확정 (2026-09-02)

**상세 절차는 [`JETSON_SPI_BME680_SETUP.md`](JETSON_SPI_BME680_SETUP.md) 를 본다.**
여기에는 요약과 최종 상태만 기록한다.

### canonical 설정 경로

40핀 헤더 기능 설정은 **NVIDIA 공식 `jetson-io.py` 대화형 메뉴가 FIRST CHOICE** 다.

```
sudo /opt/nvidia/jetson-io/jetson-io.py
  -> Configure Jetson 40pin Header -> Configure header pins manually
  -> spi1 (19,21,23,24,26) -> Back -> Save pin changes
  -> Save and reboot to reconfigure pins
```

수동 DTBO/DTS/pinmux/tristate 편집은 **공식 절차와 공식 loopback 이 실제로 실패한 경우에만**
한다 (`AGENTS.md` §4). §15/§16 의 수동 DTBO 경로는 최단 경로가 아니었다.

### 확정된 매핑

```
Jetson header function : spi1        물리 핀 19 MOSI / 21 MISO / 23 SCK / 24 CS0 / 26 CS1
SPI controller         : spi@3210000
Linux master           : spi0
CS0 / CS1              : /dev/spidev0.0  /  /dev/spidev0.1
```

`spi1` 이라는 function 이름은 `/dev/spidev1.x` 를 뜻하지 않는다. `/dev/spidev1.x` 는
`spi@3230000` (SPI3, 물리 핀 13/16/18/22/37) 이다.

### 검증 결과 (2026-09-02)

| 항목 | 상태 |
|---|---|
| 공식 Jetson-IO `spi1` 설정 | **PASS** — 적용 DT 5핀 `func=spi1`, `tristate=0` |
| `/dev/spidev0.0` | **PASS** |
| **Pin19↔Pin21 물리 loopback (SPI1)** | **PASS** — 모든 선두비트 패턴, 100 kHz~5 MHz, 10/10 |
| **Pin37↔Pin22 물리 loopback (SPI3)** | **PASS** — 동일 조건 10/10 |
| SCK / MOSI / CS 실제 구동, CS assert | **PASS** — 전송 ON/OFF DC 전압차로 확인 |
| 실측 SPI 클럭 | 요청 100 kHz → **실제 약 3.12 MHz** (BME680 상한 10 MHz 이내) |
| **BME680 / BMP388 chip ID** | **FAIL** — 모듈 손상 추정 (아래) |

### SPI 컨트롤러 상태 (2026-09-03 최종)

```
SPI1  spi@3210000  /dev/spidev0.0   정상
SPI3  spi@3230000  /dev/spidev1.0   정상
```

**중간에 "SPI1 선두비트 소실 결함" 으로 판정했던 것은 오진이었다.** 원인은 loopback 점퍼의
접촉 불량이었고, 점퍼를 다시 꽂자 `TX D0 00 -> RX D0 00` 으로 10/10 정상이 되었다.
증상과 오진 과정, 재발 방지 절차는
[`JETSON_SPI_BME680_SETUP.md`](JETSON_SPI_BME680_SETUP.md) §10-6 에 실패 기록으로 남겼다.

### 센서 상태

```
BME680 (모듈 3개)   SPI · I2C 양쪽 모두 무응답 — 사망 추정, 신규 구매 진행
BMP388              SPI 무응답
ADS1115 0x48        정상 (i2c-7)
SCD30   0x61        정상 (i2c-1)
SPS30   0x69        정상 (i2c-1)
FLIR Lepton         정상 — PureThermal 보드, USB(UVC). SPI 와 무관
```

BME680 모듈들은 초기에 **역방향 삽입 이력**이 있다. 대조군(ADS1115)을 둔 I2C 생존 확인에서도
어떤 주소에도 나타나지 않았다. 절차는 `JETSON_SPI_BME680_SETUP.md` §10-7.

**loopback 검증 시 반드시 첫 바이트 MSB=1 패턴(`FF FF FF FF`, `D0 00`)을 포함해야 한다.**
`"HelloWorld..."`(첫 바이트 `0x48`)만으로는 선두비트 이상이 드러나지 않는다. 그리고 이상이
보이면 **컨트롤러를 의심하기 전에 점퍼를 다시 꽂고 재현부터 확인한다.**

### 이 플랫폼에서 통하지 않는 진단 방법 (기록)

- `spidev_test -l` (`SPI_LOOP`, 컨트롤러 내부 loopback) — Tegra 드라이버 미지원, `EINVAL`
- `spidev_test.c` master 브랜치 — 5.15 헤더에 없는 매크로로 컴파일 실패. **v5.15 태그**를 쓴다
- debugfs `MUX UNCLAIMED` — 정상 I2C 핀도 동일 표시. 판정 근거 불가 (§14)
- `gpioinfo` 의 input/output 표시 — SPI pad direction 근거로 사용 불가
- `/proc/cmdline` — 세 boot entry 가 동일하므로 부팅 entry 판별에 사용 불가.
  적용된 DT (`exp-header-pinmux`) 를 본다
