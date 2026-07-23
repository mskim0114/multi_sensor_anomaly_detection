# SPS30 Jetson Orin Nano I2C 연결 메모

이 문서는 Sensirion SPS30 미세먼지 센서를 Jetson Orin Nano 40-pin header에 I2C로 연결하기 위한 작업 메모다.

## 기준 자료

- 로컬 데이터시트: `/home/keti/Downloads/Sensirion_PM_Sensors_Datasheet_SPS30.pdf`
- Sensirion 기존 저장소: `https://github.com/Sensirion/embedded-sps`
- Sensirion 현재 I2C 전용 저장소: `https://github.com/Sensirion/embedded-i2c-sps30`
- Jetson 40-pin 핀맵: `docs/Jetson_Orin_Nano_40pin_pinmap.md`

`embedded-sps` 저장소는 archived 상태이며 README에서 I2C는 `embedded-i2c-sps30` 사용을 안내한다. 앞으로 SPS30 I2C 코드는 `embedded-i2c-sps30` 기준으로 본다.

## SPS30 핀맵

| SPS30 핀 | 이름 | 기능 | Jetson 연결 |
| --- | --- | --- | --- |
| 1 | VDD | 센서 전원 | 40-pin pin 2 또는 4, 5V |
| 2 | SDA | I2C data | 40-pin pin 27, SDA, `/dev/i2c-1` |
| 3 | SCL | I2C clock | 40-pin pin 28, SCL, `/dev/i2c-1` |
| 4 | SEL | 인터페이스 선택 | GND에 연결해서 I2C 선택 |
| 5 | GND | Ground | 40-pin pin 6/9/14 등 GND |

Sensirion 케이블 색상 기준은 보통 `red=VDD`, `green=SDA`, `yellow=SCL`, `blue=SEL`, `black=GND`다. 실제 케이블 색상은 제품/하네스에 따라 다를 수 있으므로 커넥터 핀 번호를 우선한다.

## 전기적 주의사항

- SPS30 VDD는 5V가 기준이다. 데이터시트 범위는 4.5V~5.5V다.
- 측정 중 소비전류는 보통 55mA, 팬 시작 구간은 최대 약 80mA로 잡는다.
- SPS30 I/O는 3.3V LVTTL을 지원하지만 Jetson GPIO/I2C 핀은 5V tolerant로 취급하면 안 된다.
- SDA/SCL 풀업 저항은 3.3V로 걸어야 한다. 5V로 풀업된 I2C 모듈/브레이크아웃이면 그대로 Jetson에 연결하지 않는다.
- I2C용으로는 SDA/SCL에 외부 풀업이 필요할 수 있다. 시작값은 10kΩ to 3.3V가 적당하다.
- ADS1115 모듈에도 풀업이 있을 수 있으므로 여러 모듈을 같은 버스에 묶을 때 풀업이 과도하게 병렬로 낮아지지 않는지 확인한다.

## I2C 구성

- SPS30 I2C 주소: `0x69`
- SPS30 I2C 최대 속도: standard mode, `100 kbit/s`
- 데이터시트 기준 clock stretching은 사용하지 않는다.
- `SEL`은 센서 power-up 시점부터 GND에 묶여 있어야 I2C 모드로 부팅된다. 연결 후 모드가 이상하면 전원을 완전히 껐다 켠다.
- 데이터 갱신 주기는 약 1초다.
- 20cm를 넘는 긴 케이블이나 노이즈가 많은 현장 배선은 UART 쪽이 더 견고하다. 짧은 프로토타입 배선은 I2C로 진행한다.

현재 Jetson Orin Nano Super DevKit에서 확인한 I2C bus 속도:

| Jetson 40-pin | Linux device | DT node | clock-frequency | SPS30 사용 |
| --- | --- | --- | --- | --- |
| pin 3/5 | `/dev/i2c-7` | `c250000.i2c` | 400 kHz | 비권장. SPS30 최대 100 kHz 초과 |
| pin 27/28 | `/dev/i2c-1` | `c240000.i2c` | 100 kHz | 권장 |

## 우리 프로젝트 버스 계획

같은 I2C 버스에 아래 장치들을 같이 둘 수 있다.

| 장치 | 주소 | 메모 |
| --- | --- | --- |
| ADS1115 #1 | `0x48` | NTC, CT 일부 |
| ADS1115 #2 | `0x49` 후보 | CT 나머지 |
| SPS30 | `0x69` | PM1.0, PM2.5, PM10 사용 |

SPS30은 프로젝트 센서 입력 `[NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4]` 중 PM 세 채널을 채운다. SPS30에서 PM4.0도 읽을 수 있지만 현재 모델 입력에는 넣지 않는다.

## 연결 순서

1. Jetson 전원을 끈다.
2. SPS30 `VDD`를 Jetson 5V에 연결한다.
3. SPS30 `GND`와 `SEL`을 Jetson GND에 연결한다.
4. SPS30 `SDA`를 Jetson pin 27 SDA에 연결한다.
5. SPS30 `SCL`을 Jetson pin 28 SCL에 연결한다.
6. SDA/SCL 풀업이 없으면 10kΩ을 각각 SDA->3.3V, SCL->3.3V로 추가한다.
7. Jetson 전원을 켠 뒤 I2C scan으로 확인한다.

## 연결 확인 명령

현재 Jetson Orin Nano Super DevKit에서는 SPS30을 100 kHz I2C 버스인 pin 27/28, `/dev/i2c-1`에 연결한다.

```bash
i2cdetect -l
sudo i2cdetect -r -y 1
```

정상 연결이면 `0x69` 위치에 장치가 보여야 한다.

보이지 않을 때 우선 확인할 것:

- `SEL`이 GND에 묶인 상태에서 센서 전원이 들어갔는지
- VDD가 5V인지
- Jetson GND와 SPS30 GND가 공통인지
- SDA/SCL이 서로 바뀌지 않았는지
- SDA/SCL 풀업이 3.3V 기준인지
- 실제 header bus가 `/dev/i2c-1`이 아닌 다른 bus인지

## 소프트웨어 드라이버

Sensirion의 `embedded-sps` 저장소는 C 기반 embedded driver이며 archived 상태다. 해당 README도 현재 I2C용 새 저장소로 `embedded-i2c-sps30`을 안내한다.

Jetson에서 빠른 Python 검증은 Sensirion 공식 Python 드라이버를 사용한다.

```bash
python3 -m pip install --user sensirion-i2c-sps30
cd /home/keti/projects/factory_safety/jetson_deploy
python3 scripts/07_read_sps30.py --i2c-port /dev/i2c-1
```

현재 추가한 테스트 스크립트:

```text
/home/keti/projects/factory_safety/jetson_deploy/scripts/07_read_sps30.py
```

## 2026-07-23 연결 테스트 상태

- `/dev/i2c-7` read scan에서 SPS30 주소 `0x69` 확인됨.
- `/dev/i2c-7`은 현재 device tree 기준 400 kHz라 SPS30 데이터시트 최대 속도 100 kHz를 초과함.
- Sensirion Python 패키지 설치 완료:
  - `sensirion-i2c-sps30==1.0.0`
  - `sensirion-i2c-driver==1.0.2`
  - `sensirion-driver-adapters==2.3.1`
- 실제 SPS30 명령 실행은 아직 실패:
  - Python driver: `[Errno 121] Remote I/O error`
  - firmware version read 중 `0xff` 응답/CRC 오류 확인
  - `i2ctransfer`로 `0xD100` 명령 write 후 read 시 `0xff 0xff 0xff`

해석:

- 드라이버 미설치 문제는 해결됐다.
- 주소 `0x69`가 보이므로 I2C 모드/주소 감지는 일부 성공했다.
- 실제 명령 응답이 깨진 직접 원인 후보는 bus 속도 초과다.

우선 조치:

1. Jetson 전원을 끄고 SPS30 SDA/SCL만 옮긴다: SDA pin 27, SCL pin 28.
2. `SEL`을 GND에 묶은 상태에서 SPS30 전원을 완전히 껐다 켠다.
3. SDA/SCL 각각에 10kΩ pull-up을 3.3V로 건다. 5V로 풀업하지 않는다.
4. VDD 5V, GND 공통, SDA pin 27, SCL pin 28을 다시 확인한다.
5. 테스트 중에는 배선을 짧게 둔다.
6. 다시 `sudo i2cdetect -r -y 1` 후 `python3 scripts/07_read_sps30.py --i2c-port /dev/i2c-1`를 실행한다.

## 2026-07-23 pin 27/28 재배선 후 성공

- SPS30 SDA/SCL을 100 kHz 버스인 Jetson pin 27/28, `/dev/i2c-1`로 옮긴 뒤 성공.
- `/dev/i2c-1` scan에서 `0x69` 확인.
- `scripts/07_read_sps30.py`로 실제 측정값 읽기 성공.
- 확인된 장치 정보:
  - serial number: `E95C50BEF297082A`
  - product type: `00080000`
  - firmware version: `(2, 3)`
  - device status: `0`
- 저장된 결과:
  - CSV: `/home/keti/projects/factory_safety/jetson_deploy/results/sps30/sps30_20260723_170737.csv`
  - JSON: `/home/keti/projects/factory_safety/jetson_deploy/results/sps30/sps30_20260723_170737.json`
- 샘플 측정 범위:
  - PM1.0: 약 `4.76 ~ 8.84 ug/m3`
  - PM2.5: 약 `9.10 ~ 9.89 ug/m3`
  - PM10: 약 `9.10 ~ 15.88 ug/m3`
