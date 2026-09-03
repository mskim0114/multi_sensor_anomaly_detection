# Jetson Orin Nano — 40핀 SPI1 + BME680 설정 가이드

새 Jetson을 받았거나 SPI1 을 다시 설정해야 할 때 **이 문서 하나만 순서대로** 따라간다.
목표 소요시간 10~20분.

이 문서는 2026-09-02 실제 bring-up 에서 측정으로 확인된 내용만 담는다. 추측은 담지 않는다.
관련 규칙은 [`AGENTS.md`](../AGENTS.md) §4, 환경 전반은
[`JETSON_ENVIRONMENT.md`](JETSON_ENVIRONMENT.md) 를 본다.

---

## 0. 핵심 원칙 — JETSON-IO MENU FIRST

40핀 헤더 기능 설정은 **NVIDIA 공식 `jetson-io.py` 대화형 메뉴 경로를 FIRST CHOICE 로 사용한다.**

수동 DTBO/DTS/pinmux 편집은 **공식 절차와 공식 loopback 이 실제로 실패한 경우에만** 한다.
2026-09-02 작업에서 수동 DTBO 경로로 먼저 들어갔다가 몇 시간을 잃었고, 최종적으로 문제를
해소한 것은 공식 메뉴 경로였다. §11 을 반드시 읽는다.

**진단 순서** (역순으로 하지 않는다):

```
L1 전원 / GND / 배선
L2 공식 Jetson 설정 도구 (jetson-io.py)
L3 공식 device node (/dev/spidev0.0)
L4 공식/표준 loopback 또는 bus test (spidev_test)
L5 실제 sensor chip-id
L6 high-level Python driver
L7 그 이후에만 pinmux / DT / register debugging
```

---

## 1. 대상 Jetson 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin Nano 8GB Super DevKit (`3767-300-0005`) |
| L4T | R36.5.0 (JetPack 6.x) |
| OS | Ubuntu 22.04.5 |
| 커널 | 5.15.185-tegra |
| 전원 모드 | MAXN_SUPER |

확인:

```bash
cat /etc/nv_tegra_release
uname -r
```

### 핀 매핑 (물리 핀 번호 기준)

| 물리 핀 | 신호 | Tegra pad |
|---|---|---|
| Pin 19 | MOSI | `spi1_mosi_pz5` |
| Pin 21 | MISO | `spi1_miso_pz4` |
| Pin 23 | SCK | `spi1_sck_pz3` |
| Pin 24 | CS0 | `spi1_cs0_pz6` |
| Pin 26 | CS1 | `spi1_cs1_pz7` |
| Pin 17 | 3.3 V | — |
| Pin 20 / 25 / 6 / 39 | GND | — |

### Linux 매핑 — 여기서 반드시 헷갈린다

```
Jetson header function name : spi1
SPI controller              : spi@3210000
Linux master                : spi0
CS0                         : /dev/spidev0.0      <- BME680 은 여기
CS1                         : /dev/spidev0.1
```

**`spi1` 이라는 이름이 `/dev/spidev1.x` 를 뜻하지 않는다.**
Jetson-IO 의 function 이름은 carrier-board function label 이고, Tegra 하드웨어 인스턴스
번호(1-based)와 Linux master 번호(0-based)가 다르기 때문이다. `/dev/spidev1.x` 는
`spi@3230000` (SPI3) 이며 물리 핀 13/16/18/22/37 이다.

같은 함정이 I2C 에도 있다. Jetson-IO 는 핀 3/5 를 `i2c8`, 핀 27/28 을 `i2c2` 로 부르지만
실제 Linux 매핑은 핀 3/5 → `/dev/i2c-7`, 핀 27/28 → `/dev/i2c-1` 이다.
**숫자 label 로 판단하지 말고 물리 핀 집합으로 판단한다.**

---

## 2. Jetson-IO 로 SPI1 활성화 (공식 절차)

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

```
Configure Jetson 40pin Header
  -> Configure header pins manually
     -> [*] spi1 (19,21,23,24,26)      <- 스페이스바로 선택
     -> Back
  -> Save pin changes
  -> Save and reboot to reconfigure pins
```

`spi1 (19,21,23,24,26)` 의 핀 집합이 이 프로젝트 배선과 일치하는지 화면에서 확인한다.

**하지 않는다**: `config-by-function.py -o dt`, *Export as Device-Tree Overlay*.
§11 참조.

이 절차는 `/boot/jetson-io-hdr40-user-custom.dtbo` 를 생성하고 `/boot/extlinux/extlinux.conf` 에
`LABEL JetsonIO` entry 를 추가하며 `DEFAULT` 를 그 entry 로 바꾼다. 기존 `LABEL primary` 는
보존된다.

---

## 3. 재부팅

"Save and reboot" 이 재부팅한다. 별도로 할 필요 없다.

---

## 4. 설정이 실제로 적용되었는지 확인

### 4-1. 적용된 device-tree 확인 (가장 확실한 방법)

```bash
EH=/proc/device-tree/bus@0/pinmux@2430000/exp-header-pinmux
for g in "$EH"/*/; do
  [ -d "$g" ] || continue
  printf "%-13s %-16s func=%-6s tristate=%s\n" "$(basename $g)" \
    "$(tr -d '\0' < $g/nvidia,pins)" "$(tr -d '\0' < $g/nvidia,function)" \
    "$(od -An -tu4 -N4 --endian=big < $g/nvidia,tristate | tr -d ' ')"
done
```

기대 출력 (공식 메뉴 경로로 설정했을 때 실측값):

```
hdr40-pin19   spi1_mosi_pz5    func=spi1   tristate=0
hdr40-pin21   spi1_miso_pz4    func=spi1   tristate=0
hdr40-pin23   spi1_sck_pz3     func=spi1   tristate=0
hdr40-pin24   spi1_cs0_pz6     func=spi1   tristate=0
hdr40-pin26   spi1_cs1_pz7     func=spi1   tristate=0
```

자식 pin 노드가 **없으면** overlay 가 적용되지 않은 것이다 (예: `primary` entry 로 부팅).

### 4-2. device node 확인

```bash
ls -l /dev/spidev*
ls -d /sys/class/spi_master/*
```

기대:

```
/dev/spidev0.0  /dev/spidev0.1  /dev/spidev1.0  /dev/spidev1.1     (root:gpio, 660)
spi0 -> 3210000.spi     spi1 -> 3230000.spi
```

사용자가 `gpio` 그룹에 속해 있으면 sudo 없이 접근된다 (`id` 로 확인).

### 4-3. 어느 boot entry 로 부팅했는지 확인

`/proc/cmdline` 은 세 entry 가 동일하므로 **판별에 쓸 수 없다.** §4-1 의 device-tree 를 본다.

---

## 5. Pin19 ↔ Pin21 loopback (물리 점퍼)

BME680 을 연결하기 **전에** Jetson SPI1 데이터 경로 자체를 먼저 증명한다.

1. Jetson **전원 완전 차단** (어댑터 제거)
2. BME680 **분리** (6선 전부)
3. **Pin 19 ↔ Pin 21** 점퍼 연결
4. 멀티미터 도통 모드로 점퍼 확인 (삐 소리 = 연결됨)
5. 전원 ON

---

## 6. spidev_test 로 loopback 검증

### 6-1. 도구 준비 — 반드시 v5.15 태그 소스를 쓴다

```bash
wget -q -O /tmp/spidev_test.c \
  https://raw.githubusercontent.com/torvalds/linux/v5.15/tools/spi/spidev_test.c
sha256sum /tmp/spidev_test.c
#   87329094ea3010eae9e65ad7ea2d2309e044845942414aa0ca8f1a7095b80178
gcc -O2 -Wall /tmp/spidev_test.c -o /tmp/spidev_test
```

**master 브랜치 소스는 이 커널에서 컴파일되지 않는다.**
`SPI_RX_CPHA_FLIP`, `SPI_MOSI_IDLE_LOW` 가 5.15 계열 `/usr/include/linux/spi/spidev.h` 에 없다.
실행 커널과 같은 **v5.15 태그**를 쓴다. apt/pip 설치는 필요 없다 (`gcc` 는 기본 설치되어 있다).

### 6-2. MODE 0 loopback

```bash
/tmp/spidev_test -D /dev/spidev0.0 -s 500000 -v -p "HelloWorld123456789abcdef"
```

기대 (PASS):

```
spi mode: 0x0
bits per word: 8
max speed: 500000 Hz (500 kHz)
TX | 48 65 6C 6C 6F 57 6F 72 6C 64 31 32 33 34 35 36 37 38 39 61 62 63 64 65 66 ...
RX | 48 65 6C 6C 6F 57 6F 72 6C 64 31 32 33 34 35 36 37 38 39 61 62 63 64 65 66 ...
```

**판정: TX == RX 이면 PASS.** `-O` / `-H` 를 쓰지 않는다 (그러면 MODE 0 이 아니다).
2~3회 반복해서 재현되는지 확인한다.

### 반드시 선두 비트가 1인 패턴도 시험한다 (중요)

위 `"HelloWorld..."` 는 첫 바이트가 `0x48` (MSB=0) 이라 **§10-6 의 선두비트 소실 결함을
가려버린다.** 아래를 반드시 함께 실행한다.

```bash
/tmp/spidev_test -D /dev/spidev0.0 -s 100000 -v -p "\xff\xff\xff\xff"   # 기대 RX = FF FF FF FF
/tmp/spidev_test -D /dev/spidev0.0 -s 100000 -v -p "\x80\x00"             # 기대 RX = 80 00
/tmp/spidev_test -D /dev/spidev0.0 -s 100000 -v -p "\xd0\x00"             # 기대 RX = D0 00
```

`RX` 가 `3F FF FF FF` / `00 00` / `10 00` 으로 나오면 **선두비트 소실 결함**이다 (§10-6).
이 상태에서는 어떤 SPI 센서도 read 에 응답하지 않는다. 센서를 의심하기 전에 이것을 먼저 본다.

### 6-3. loopback PASS 가 증명하는 것과 하지 않는 것

**증명한다**: SPI 컨트롤러 transfer 경로, spidev/ioctl 경로, Pin19 MOSI 출력, Pin21 MISO 입력.

**증명하지 않는다**: Pin23 SCK 외부 파형, Pin24 CS 외부 파형, BME680 자체 정상.
loopback 은 CS 상태와 무관하게 통과한다. CS 검증은 §10-2 의 방법을 쓴다.

### 6-4. `-l` 옵션은 이 플랫폼에서 쓸 수 없다

```bash
/tmp/spidev_test -D /dev/spidev0.0 -l ...    # can't set spi mode: Invalid argument
```

Tegra SPI 드라이버가 `SPI_LOOP` (컨트롤러 내부 loopback) 를 지원하지 않는다.
"컨트롤러 내부 vs pad" 를 이 방법으로 분리하려는 시도는 하지 않는다.

---

## 7. BME680 배선

### 7-1. 반드시 먼저 — loopback 점퍼를 제거한다

**전원 차단 후 Pin19 ↔ Pin21 점퍼를 제거한다.**
점퍼는 MOSI 와 MISO 를 직접 단락시키므로, BME680 6선이 연결되어 있어도 점퍼가 신호를
가로채 센서 응답을 볼 수 없다. 2026-09-02 작업에서 실제로 여기서 시간을 잃었다.

제거 확인 (에코가 나오면 아직 붙어 있는 것):

```bash
/tmp/spidev_test -D /dev/spidev0.0 -s 500000 -v -p "\xa5\x5a\x0f\xf0\x33\xcc"
#   RX 가 TX 와 같으면 -> 점퍼/브릿지가 아직 있음
```

### 7-2. 배선

브레이크아웃 실크 예: `SDO  SCK  SDI  CS  GND  VCC`

| BME680 (데이터시트 이름) | 브레이크아웃 실크 | Jetson 물리 핀 |
|---|---|---|
| SDO (센서 출력) | `SDO` | **Pin 21** (MISO) |
| SCK | `SCK` | **Pin 23** |
| SDI (센서 입력) | `SDI` | **Pin 19** (MOSI) |
| CSB | `CS` | **Pin 24** |
| GND | `GND` | **Pin 20** |
| VDD/VDDIO | `VCC` | **Pin 17** (3.3 V) |

데이터시트 기준으로 **`SDI` 는 센서 입력(=MOSI), `SDO` 는 센서 출력(=MISO)** 이다.
보드가 `MISO/MOSI` 로 표기되어 있으면 `MISO → Pin21`, `MOSI → Pin19`.

**배선 변경은 항상 전원 완전 차단 후에 한다.** 활선 상태에서 신호선을 옮기면 순간 단락으로
패드가 손상될 수 있다.

> **⚠️ 역삽입 주의 — 이번 bring-up 에서 실제로 모듈 3개를 잃었다.**
> 6핀 모듈을 앞뒤 반대로 꽂으면 `VCC` 자리에 `MISO` 신호가, `GND` 자리에 `SCLK` 가 들어간다.
> 칩은 정규 전원 핀이 아니라 I/O 핀의 ESD 다이오드를 통해 전류가 흐르게 되고, 몇 초만에
> 파괴될 수 있다. **같은 소켓에 차례로 꽂은 모듈은 전부 같은 일을 당한다.**
> 꽂기 전에 실크 첫 핀 위치를 눈으로 확인하고, 전원을 넣은 직후 모듈이 따뜻해지는지
> 손끝으로 확인한다 (정상 BME680 은 상온이다).
>
> 6핀 모듈을 4핀 소켓(예: SGP30 자리)에 꽂는 것도 같은 사고를 일으킨다. 핀 수가 다르면
> 개별 점퍼로 연결한다.

### 7-3. 배선 검증 — 쌍으로 확인한다

"모든 선에서 삐 소리가 난다" 는 각 선이 어딘가에 연결되었다는 것만 증명한다.
**어느 핀이 어느 패드로 가는지**를 쌍으로 확인해야 한다. 전원 차단 후, 프로브 한쪽은
Jetson 헤더 핀, 다른 쪽은 **모듈 패드에 직접** 댄다.

```
Pin19 <-> 모듈 SDI     삐 나야 함      Pin19 <-> 모듈 SDO   삐 나면 안 됨
Pin21 <-> 모듈 SDO     삐 나야 함      Pin21 <-> 모듈 SDI   삐 나면 안 됨
Pin23 <-> 모듈 SCK     삐 나야 함
Pin24 <-> 모듈 CS      삐 나야 함
Pin17 <-> 모듈 VCC     삐 나야 함
Pin20 <-> 모듈 GND     삐 나야 함
모듈 VCC <-> 모듈 GND  삐 나면 안 됨 (전원 단락)
인접 패드끼리           삐 나면 안 됨 (납땜 브릿지)
```

전원 ON 상태에서 **모듈 자체의 VCC-GND DC 전압**을 재서 3.15 ~ 3.45 V 를 확인한다.
Jetson Pin17 에서 재는 것이 아니라 모듈 쪽에서 잰다.

---

## 8. chip ID 확인 — `0x61`

BME680 의 chip ID 레지스터는 **SPI 주소 `0x50`, page 0** 이고, read control byte 는
주소 7비트 + bit7=RW=1 → **`0xD0`** 이다. page 0 은 power-on 기본 상태이므로 별도 설정이
필요 없다 (데이터시트 §5.1, §5.3.1.5, §6.3).

```bash
/tmp/spidev_test -D /dev/spidev0.0 -s 100000 -v -p "\xd0\x00"
```

기대:

```
TX | D0 00
RX | ?? 61          <- 두 번째 byte 가 0x61
```

3회 반복해서 3회 모두 `0x61` 이면 `BME680_RAW_SPI = PASS`.
**mode sweep / speed sweep / page 변경을 하지 않는다.** 이유는 §10-1.

### SPI mode 는 신경쓰지 않아도 된다

데이터시트 §6.3:

> The SPI interface is compatible with SPI mode '00' (CPOL = CPHA = '0') and mode '11'
> (CPOL = CPHA = '1'). **The automatic selection between mode '00' and '11' is determined
> by the value of SCK after the CSB falling edge.**

BME680 이 CSB falling edge 시점의 SCK 레벨로 mode 를 스스로 판별한다.
**mode 불일치는 실패 원인이 될 수 없다.** mode 0 으로 고정하고 mode 를 바꿔가며 시도하지 않는다.

### 요청 속도는 그대로 적용되지 않는다

`-s 100000` 을 줘도 실측 클럭은 **약 3.12 MHz** 였다 (4000바이트 전송 20회 평균에서
프로세스 오버헤드를 빼서 산출). BME680 상한 10 MHz 이내이므로 문제는 없다.
`spi-max-frequency` DT 값은 50 MHz 다. 속도가 원인이라고 의심되면 추측하지 말고 아래처럼 실측한다.

```bash
python3 -c "open('/tmp/p2.bin','wb').write(b'\xaa'*2); open('/tmp/p4k.bin','wb').write(b'\xaa'*4000)"
# 각각 20회 실행 시간을 재서 차이 / 31984 bit 로 클럭 산출
```

---

## 9. high-level 측정 (chip ID PASS 이후에만)

`$HOME/venvs/factory_runtime` 의 Adafruit BME680 driver 로 10 samples 를 읽는다.
새 패키지를 설치하지 않는다.

```bash
./jetson_deploy/run_python.sh <BME680 diagnostic script>
```

temperature / humidity / pressure / gas resistance 가 연속적으로 정상 read 되고
exception / chip-id error / SPI error 가 없으면 `BME680_FINAL = PASS`.

> **현재 상태 (2026-09-03)**
> ```
> Jetson SPI1 (/dev/spidev0.0)   정상 — loopback 10/10, 모든 선두비트 패턴, 100 kHz~5 MHz
> Jetson SPI3 (/dev/spidev1.0)   정상 — 동일 조건 10/10
> BME680 (모듈 3개)               SPI · I2C 양쪽 모두 무응답 — 사망 추정
> BMP388                          SPI 무응답
> 다른 센서 (ADS1115/SCD30/SPS30) 정상 동작 확인
> ```
> BME680 모듈들은 초기에 **역방향으로 삽입된 이력**이 있다(§7-2 경고). 대조군을 둔 I2C 생존
> 확인(§10-7)에서도 응답이 없어 모듈 손상으로 판단하고 신규 구매로 진행한다.
> 진단 과정에서 한때 "SPI1 컨트롤러 결함" 으로 판정했으나 **점퍼 접촉 불량이었다** — §10-6.

---

## 10. Troubleshooting

### 10-1. 먼저 읽을 것

- **첫 오류를 보존한다.** 자동 retry loop 나 workaround 를 만들지 않는다.
- **추측으로 설정을 바꾸지 않는다.** mode/speed/page 를 무작위로 바꿔가며 시도하면 어떤
  가설도 제거되지 않는다. 한 번에 하나의 가설을 측정으로 제거한다.
- **debugfs `MUX UNCLAIMED` 는 이 플랫폼에서 판정 근거가 되지 않는다.** 정상 동작하는 I2C
  핀도 동일하게 표시된다 (Tegra 는 MB1/부트로더에서 pinmux 를 적용하고 Linux pinctrl 은
  아무것도 claim 하지 않는다).
- **`gpioinfo` 의 input/output 표시를 SPI pad direction 의 근거로 쓰지 않는다.**

### 10-2. pad 가 실제로 구동되는지 확인하는 방법 (오실로스코프 없이)

멀티미터로 판정할 수 있다. **핵심은 절대값이 아니라 전송 ON/OFF 의 차이**다.
떠 있는(high-Z) 핀은 멀티미터 입력저항 때문에 중간 전압으로 읽히므로, 한 번만 재면
"토글 중" 과 "떠 있음" 이 구분되지 않는다.

```bash
# 4000바이트 read 버스트를 연속 전송 (0xAA 는 bit7=1 이므로 read 명령 -> write 발생 안 함)
python3 -c "open('/tmp/spi_pattern.bin','wb').write(b'\xaa'*4000)"
rm -f /tmp/spi_busy.count
N=0; END=$((SECONDS+600))
while [ $SECONDS -lt $END ]; do
  /tmp/spidev_test -D /dev/spidev0.0 -s 100000 -i /tmp/spi_pattern.bin -o /dev/null >/dev/null 2>&1
  N=$((N+1)); echo "$N" > /tmp/spi_busy.count
done
```

**반드시 카운터(`/tmp/spi_busy.count`)가 증가하는지 확인한다.**
`pgrep -f <패턴>` 은 자기 자신의 명령줄을 매칭하는 오탐이 흔하다. 전송이 돌지 않는 상태에서
전압을 재면 완전히 틀린 결론이 나온다.

전송 ON / OFF 각각에서 DC 전압을 재고 비교한다 (검은 프로브는 Pin 6 또는 Pin 39 GND).

| 핀 | 정상 구동 (2026-09-02 실측) | 떠 있음 |
|---|---|---|
| Pin 24 CS | OFF **3.3 V** → ON **1.6 V** | ON/OFF 거의 동일 |
| Pin 23 SCK | OFF 3.4 V → ON 2.2 V | ON/OFF 거의 동일 |
| Pin 19 MOSI | OFF 1.0 V → ON 1.7 V | ON/OFF 거의 동일 |
| Pin 21 MISO | 센서가 구동하면 ON 에서 뚜렷히 하강 | ON/OFF 모두 3.3 V |

Pin 24 의 ON/OFF 차이가 **CS 가 실제로 assert 된다는 증거**다.
Pin 21 이 ON/OFF 모두 3.3 V 면 **센서가 SDO 를 전혀 구동하지 않는 것**이다.

### 10-3. 2026-09-02 에 측정으로 제거된 가설

| 가설 | 결과 | 제거 근거 |
|---|---|---|
| 40핀 `spi1` 미활성 | 제거 | 공식 Jetson-IO 설정 + §4-1 DT 확인 |
| pad tristate=1 | 제거 | 수동 A/B 오버레이로 tristate=0 을 확인해도 동일 실패. 공식 설정은 5핀 전부 tristate=0 |
| pad 미구동 | 제거 | §10-2 ON/OFF 전압차 |
| CS 미assert | 제거 | Pin24 3.3 V → 1.6 V |
| CS 극성 반전 (`spi-cs-high`) | 제거 | `/proc/device-tree/bus@0/spi@3210000/spi@{0,1}` 에 해당 property 없음 |
| MISO 입력 경로 고장 | 제거 | §6 loopback PASS 재현 (3/3) |
| MOSI-MISO 단락 | 제거 | 서로 다른 패턴 전송 시 에코 없음 |
| SPI mode 0/3 불일치 | 제거 | 데이터시트 §6.3 자동 판별 |
| SPI memory page 미선택 | 제거 | page 0 이 power-on 기본 (데이터시트 §5.1) |
| 클럭 과속 (>10 MHz) | 제거 | 실측 3.12 MHz |
| spidev 노드 / 논리 매핑 오류 | 제거 | §1 매핑 확인, loopback PASS |
| Adafruit software-CS 경합 | 제거 | 순수 C ioctl 도구로도 동일 |
| PureIO backend 구현 결함 | 제거 | upstream `spidev_test` 로도 동일 |
| 배선 끊김 | 제거 | 6쌍 도통 확인 |
| 모듈 전원 / 단락 | 제거 | 모듈 VCC-GND 전압 정상, 단락 없음 |
| 개별 모듈 불량 | **제거** | BME680 ×2 + BMP388 ×1, 각각 정규 프로토콜로 동일 실패 |
| 명령 형식 오류 | 제거 | BME680 `0xD0`(reg 0x50), BMP388 `0x80`+dummy byte 각각 데이터시트대로 |
| MOSI/MISO 배선 | 제거 | 모듈 자리에서 SDI-SDO 를 점퍼로 물려 배선 왕복 loopback PASS |
| 선두비트 소실 (§10-6) | **오진 — 점퍼 접촉 불량** | 점퍼 재삽입 후 10/10 정상. SPI1·SPI3 모두 정상 |
| **모듈 손상 (역삽입 이력)** | **현재 가장 유력** | SPI·I2C 양쪽 무응답, 대조군 정상, 모듈 3개 동일 |

### 10-4. `RX` 가 전부 `FF` 또는 전부 `00` 일 때

MISO 가 구동되지 않고 떠 있는 상태다. `FF` 와 `00` 중 어느 쪽으로 읽히는지는 플랫폼과
잔류 전하에 따라 달라지고 **세션마다 바뀔 수 있다.** 값 자체로 원인을 단정하지 않는다.

확인 순서: ① loopback 점퍼가 남아 있지 않은지 (§7-1) → ② 6쌍 도통 (§7-3) →
③ 모듈 VCC-GND 전압 → ④ §10-2 로 Pin21 ON/OFF 비교.

### 10-5. `RX` 출력이 비어 있을 때

**재부팅하면 `/tmp` 가 비워진다.** `/tmp/spidev_test` 바이너리가 사라져 명령이 실패하면
출력이 빈 문자열이 되고, 이를 "TX == RX (PASS)" 로 오판할 수 있다.
스크립트에 **빈 출력 검사를 반드시 넣는다.** 재부팅 후에는 §6-1 로 다시 컴파일한다.

### 10-6. 접촉 불량을 컨트롤러 결함으로 오진한 사례 (2026-09-02 ~ 09-03)

**이 절은 실패 기록이다.** 같은 함정을 반복하지 않기 위해 남긴다.

**관측된 증상**: loopback 에서 프레임 시작 시점의 연속된 1 구간이 최대 2비트까지 0 으로 떨어졌다.

| TX | RX | 소실 |
|---|---|---|
| `7F FF` / `00 FF` | 그대로 | 없음 (선두가 0) |
| `80 00` / `A5 5A` | `00 00` / `25 5A` | 1비트 |
| `C0 00` / `FF FF FF FF` | `00 00` / `3F FF FF FF` | 2비트 |
| **`D0 00`** | **`10 00`** | 2비트 |

`0xD0`(BME680 chip ID read)이 `0x10`(레지스터 0x10 write)으로 나가므로, 이 상태에서는 어떤 SPI
센서도 read 에 응답하지 않는다. 증상 자체는 실재했다.

**잘못된 결론**: 아래를 근거로 "SPI1 컨트롤러 고유 결함" 으로 단정했다.

- 배선 길이 무관 — 헤더 직결 짧은 점퍼에서도 재현
- 요청 속도 무관 — 50 kHz ~ 10 MHz 전부
- 전송 길이 무관 — 2 ~ 64 바이트, PIO/DMA 경로 모두
- 재현율 4/5 (1회만 우연히 정상) → "경계선 타이밍" 으로 해석
- 드라이버가 참조하는 `nvidia,tx-clk-tap-delay` / `nvidia,rx-clk-tap-delay` 가 DT 에 미설정

**반증**: SPI3 를 활성화해 비교하던 중 점퍼를 옮겼다가 **Pin19↔Pin21 에 다시 꽂았더니
결함이 완전히 사라졌다.** 같은 부팅, 같은 pinmux, 같은 도구인데 `TX D0 00 -> RX D0 00` 이 되고
`FF FF FF FF` 가 10/10 정상이 되었다. 원인은 **점퍼의 접촉 불량**이었다.

```
SPI1 (spi@3210000, /dev/spidev0.0)   정상 — 모든 선두비트 패턴, 100 kHz ~ 5 MHz, 10/10
SPI3 (spi@3230000, /dev/spidev1.0)   정상 — 동일 조건 10/10
```

**교훈 — loopback 이상을 컨트롤러 결함으로 판정하기 전에 반드시 할 것**

1. **점퍼를 뽑았다가 다시 꽂고 재현부터 확인한다.** 이것을 먼저 하지 않으면 아래 어떤 근거도
   의미가 없다. "여러 조건에서 재현된다" 는 사실은 접촉 불량을 배제하지 못한다 — 접점이
   그대로면 조건을 바꿔도 그대로 재현된다.
2. **다른 컨트롤러 인스턴스와 비교한다** (SPI1 ↔ SPI3). 단 **두 점퍼를 동시에 꽂고 같은 시점에**
   측정한다. 하나를 옮겨가며 비교하면 접점 상태가 함께 바뀌어 비교가 성립하지 않는다.
3. **DT/드라이버 파라미터는 마지막에 본다.** `nvidia,*-clk-tap-delay` 가 미설정인 것은
   이 플랫폼의 정상 상태이며 결함의 근거가 아니다.

**이 세션에서 접점이 원인이었던 다른 사례들** — 전부 같은 뿌리다.

- loopback 결과가 세션마다 PASS/FAIL 로 뒤집힘
- MISO 읽힘값이 `FF` 와 `00` 사이를 오감 (떠 있는 노드의 전형)
- SPI3 가 5/5 정상에서 0/10 으로 바뀜 (점퍼가 빠져 있었다)

### 10-7. 센서 생존 여부를 I2C 로 확인하는 방법

BME680 은 SPI 와 I2C 를 모두 지원한다. SPI 로 응답이 없을 때 **칩 자체가 살아 있는지** 판정할 수
있다. 이것은 SPI 문제를 우회하는 것이 아니라 사망 여부 판정이다.

**배선** (실크 `MISO SCLK CS MOSI GND VCC` 기준)

```
모듈 SCLK  ->  Pin 5   (SCL)
모듈 MOSI  ->  Pin 3   (SDA)        BME680 은 SDI 가 I2C SDA 겸용
모듈 CS    ->  3.3 V                I2C 모드 선택. 없으면 SPI 모드에 머물러 절대 응답 안 함
모듈 MISO  ->  GND                  주소 선택. GND=0x76, 3.3V=0x77
모듈 GND   ->  GND
모듈 VCC   ->  3.3 V
```

**반드시 지킬 것 세 가지**

1. **풀업이 있어야 한다.** Jetson 40핀은 I2C 풀업을 제공하지 않는다. 다른 I2C 센서 모듈이
   버스에 함께 붙어 있어야 통신이 성립한다. BME680 만 단독으로 붙이면 살아 있어도 `errno 121` 이다.
2. **대조군을 같은 버스에 둔다.** 예: ADS1115(`0x48`). 대조군이 응답하는데 BME680 만 무응답이어야
   비로소 판정이 성립한다.
3. **power-on-reset 을 한다.** 데이터시트 §6.1 — CSB 가 한 번이라도 LOW 로 내려간 뒤에는
   다음 POR 전까지 I2C 가 비활성이다. SPI 시험을 했다면 반드시 겪은 상태다.
   `reboot` 으로는 3.3 V 레일이 내려간다는 보장이 없으므로 **어댑터를 뽑고 10초 이상** 기다린다.

**스캔** (쓰기 없이 read 만 하므로 다른 장치의 상태를 바꾸지 않는다)

```bash
python3 -c "
import os
from fcntl import ioctl
found=[]
for a in range(0x08,0x78):
    fd=os.open('/dev/i2c-7',os.O_RDWR)
    try:
        ioctl(fd,0x0703,a); os.read(fd,1); found.append(a)
    except OSError: pass
    finally: os.close(fd)
print(' '.join('0x%02X'%a for a in found))
"
```

`0x76` 에서 chip_id `0x61` 이 나오면 칩 생존이다. 대조군 정상 + 올바른 배선 + POR 을 모두
만족한 상태에서 무응답이면 **사망으로 판정한다.**

**주의**: `i2cdetect` 는 이 플랫폼에서 신뢰할 수 없다. Tegra I2C 어댑터가 SMBus Quick Command 를
지원하지 않아 대부분의 주소를 probe 조차 하지 않는다 (`i2cdetect -F` 로 확인 가능).
위처럼 직접 read 로 확인한다.
## 11. 하지 말아야 할 접근

2026-09-02 작업에서 실제로 시간을 잃은 경로들이다.

| 하지 않는다 | 이유 |
|---|---|
| `config-by-function.py -o dt` 로 overlay 직접 생성 | 일반 설정 경로가 아니다. 공식 메뉴 경로와 생성 결과가 달랐다 (아래 참조) |
| *Export as Device-Tree Overlay* 를 일반 설정용으로 사용 | 같은 이유 |
| 수동 DTS/DTBO 편집 | tristate 를 손으로 바꿔도 문제가 해소되지 않았다 |
| `tristate` 직접 수정 | 위와 같음. root cause 가 아니었다 |
| `devmem` write / pinmux register 직접 write | 위험하고, 문제 해결에 기여하지 않았다 |
| pinmux / GPIO 저수준 분석부터 시작 | L7 을 L2 보다 먼저 하면 안 된다 |
| `/boot` 에 custom DTBO 를 여러 개 방치 | `jetson-io.py` 동작에 영향을 줄 수 있다. 같은 overlay-name 이 중복되면 특히 그렇다 |
| BME680 을 I2C 로 우회해서 "일단 되게" 만들기 | SPI 문제를 덮는다. 또한 데이터시트 §6.1 에 따라 CSB 가 한 번 LOW 로 내려간 뒤에는 power-on-reset 전까지 I2C 가 비활성이다 |

**관측된 차이 (인과로 단정하지 않음)**: `config-by-function.py -o dt` 로 생성된 overlay 는
5핀 전부 `nvidia,tristate = <1>` 이었고, 공식 대화형 메뉴로 생성된 overlay 는 5핀 전부
`tristate = <0>` 이었다. 다만 두 실행의 function 집합이 달랐으므로
(`i2c2 i2c8 uarta spi1` vs `spi1`) 도구 차이라고 단정할 수는 없다.
어느 쪽이든 **실제로 동작한 설정은 메뉴 경로 쪽**이다.

---

## 12. 복구 방법

### 12-1. boot entry 로 되돌리기

`/boot/extlinux/extlinux.conf` 에 여러 entry 가 있으면, 부팅 중 `L4T boot options` 메뉴에서
원하는 entry 를 **수동 선택**할 수 있다. 이것은 자동 fallback 이 아니라 **수동 복구 경로를
보존하는 것**이다. `LABEL primary` 는 절대 삭제하거나 수정하지 않는다.

```bash
grep -nE '^DEFAULT|^LABEL|MENU LABEL|OVERLAYS' /boot/extlinux/extlinux.conf
```

`TIMEOUT 30` 은 syslinux 관례상 1/10초 단위(약 3초)이므로 모니터+키보드 또는 시리얼 콘솔이
붙어 있어야 선택할 수 있다. 화면 앞에서 선택할 수 없는 상황이면 `DEFAULT` 한 줄만 바꾼다
(entry 는 지우지 않는다).

### 12-2. 완전 초기 상태로 되돌리기

`jetson-io.py` 를 다시 실행해 40핀 설정에서 `spi1` 선택을 해제하고 저장하면 된다.
파일을 손으로 지우지 않는다.

### 12-3. 2026-09-02 진단 산출물 (historical diagnostic artifact)

아래는 수동 DTBO 경로에서 만들어진 실험 산출물이다. **canonical 설정이 아니다.**
사용자 승인 없이 삭제하지 않는다.

```
/boot/jetson-io-hdr40-spi1-master-fix.dtbo                   실험용 tristate fix overlay (canonical 아님)
/boot/extlinux/extlinux.conf 의 LABEL JetsonIO-SPI1FIX       위 overlay 를 쓰는 실험 entry
/boot/extlinux/extlinux.conf.pre_spi1_20260902_145420        SPI1 작업 이전 pristine 백업 (유지)
/boot/extlinux/extlinux.conf.pre_spi1fix_20260902_165804     중간 백업
/boot/extlinux/extlinux.conf.pre_primary_20260902_173306     중간 백업
/boot/extlinux/extlinux.conf.jetson-io-backup                jetson-io 자동 백업
```

현재 canonical 설정은 `LABEL JetsonIO` + `/boot/jetson-io-hdr40-user-custom.dtbo` 이며,
이것이 공식 메뉴 경로로 생성된 것이다.

---

## 참고

- BME680 데이터시트: `jetson_deploy/BME680.PDF`
  (Bosch BST-BME680-DS001-00 Rev 1.0) — §5.1 메모리 페이지, §5.3.1.5 chip id,
  §6.1 인터페이스 자동선택, §6.3 SPI 프로토콜/mode 자동판별, §6.4 타이밍
- 환경 전반 / 다른 센서 실측: [`JETSON_ENVIRONMENT.md`](JETSON_ENVIRONMENT.md)
- agent 규칙: [`../AGENTS.md`](../AGENTS.md) §4
