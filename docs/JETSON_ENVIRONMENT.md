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
| SGP30 | I2C | `/dev/i2c-7` | `0x58` | **communication PASS / dynamic response YES** | eCO2 413 ppm / TVOC 4 ppb (65샘플, 68.4 s) |
| BME680 | SPI | **`/dev/spidev0.0`** | — | **SPI mapping PASS / hardware communication FAIL** | chip ID `0x0` (기대 `0x61`) |

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
TensorRT 가 `thermal` 입력을 "사용되지 않음"으로 보고하는 것도 서브그래프 분할이
thermal 분기를 잘못 다루고 있음을 시사한다.

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

**운영 규칙**: `TensorRT = BROKEN/EXPERIMENTAL`, `CUDA = VERIFIED`.
모든 정상 추론은 `--provider cuda` 를 명시한다.

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
