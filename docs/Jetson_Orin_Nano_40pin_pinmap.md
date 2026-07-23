# Jetson Orin Nano 40-pin Pin Map

이 문서는 현재 Jetson Orin Nano Super DevKit에서 센서 배선에 사용할 40-pin expansion header 기준 핀맵이다.

## 보드 확인

현재 보드는 device tree 기준으로 아래 모델로 잡힌다.

```text
nvidia,p3768-0000+p3767-0005-super
```

I2C adapter 매핑은 현재 시스템에서 아래처럼 확인됐다.

```text
/dev/i2c-1 -> c240000.i2c
/dev/i2c-7 -> c250000.i2c
```

Device tree symbol 기준:

```text
gen2_i2c   -> c240000.i2c
gen8_i2c   -> c250000.i2c
hdr40_i2c1 -> c250000.i2c
```

40-pin header의 pin 3/5 I2C는 `/dev/i2c-7`로 확인된다. 단, 현재 device tree 기준 이 버스는 400 kHz다.
SPS30처럼 standard mode 100 kbit/s까지만 지원하는 센서는 pin 27/28의 `/dev/i2c-1` 버스를 우선 사용한다.

## 번호 방향

- 40핀 헤더의 `pin 1`은 PCB의 작은 삼각형/1번 마킹 또는 사각 패드 쪽이다.
- `pin 1, 3, 5, ... 39`는 홀수 줄이고, `pin 2, 4, 6, ... 40`은 짝수 줄이다.
- 점퍼를 꽂기 전에 반드시 보드 실크의 `1` 표시를 먼저 확인한다.

## 전체 핀맵

| 홀수 핀 | 기능 | 짝수 핀 | 기능 |
| --- | --- | --- | --- |
| 1 | 3.3V | 2 | 5V |
| 3 | I2C SDA, `/dev/i2c-7`, 400 kHz | 4 | 5V |
| 5 | I2C SCL, `/dev/i2c-7`, 400 kHz | 6 | GND |
| 7 | GPIO09 | 8 | UART TX |
| 9 | GND | 10 | UART RX |
| 11 | UART RTS / GPIO | 12 | I2S SCLK / GPIO |
| 13 | SPI1 SCK / GPIO | 14 | GND |
| 15 | GPIO12 / PWM | 16 | SPI1 CS1 / GPIO |
| 17 | 3.3V | 18 | SPI1 CS0 / GPIO |
| 19 | SPI0 MOSI | 20 | GND |
| 21 | SPI0 MISO | 22 | SPI1 MISO / GPIO |
| 23 | SPI0 SCK | 24 | SPI0 CS0 |
| 25 | GND | 26 | SPI0 CS1 |
| 27 | I2C SDA, `/dev/i2c-1`, 100 kHz | 28 | I2C SCL, `/dev/i2c-1`, 100 kHz |
| 29 | GPIO01 | 30 | GND |
| 31 | GPIO11 | 32 | GPIO07 / PWM |
| 33 | GPIO13 / PWM | 34 | GND |
| 35 | I2S FS / GPIO | 36 | UART CTS / GPIO |
| 37 | SPI1 MOSI / GPIO | 38 | I2S SDIN / GPIO |
| 39 | GND | 40 | I2S SDOUT / GPIO |

## SPS30 연결에 필요한 핀

SPS30을 I2C로 붙일 때는 아래 5개만 쓰면 된다. SPS30 데이터시트의 I2C 최대 속도는 100 kbit/s이므로 pin 27/28을 권장한다.

| SPS30 | Jetson 40-pin |
| --- | --- |
| VDD | pin 2 또는 pin 4, 5V |
| SDA | pin 27, I2C SDA, `/dev/i2c-1` |
| SCL | pin 28, I2C SCL, `/dev/i2c-1` |
| SEL | pin 6/9/14/20/25/30/34/39 중 하나, GND |
| GND | pin 6/9/14/20/25/30/34/39 중 하나, GND |

확인 명령:

```bash
sudo i2cdetect -r -y 1
```

정상 연결이면 `0x69`가 보여야 한다.

## 전압 주의

- Jetson 40-pin 신호 핀은 3.3V 로직 기준으로 취급한다.
- 5V를 SDA/SCL/GPIO에 넣지 않는다.
- SPS30 전원은 5V가 맞지만, I2C 풀업은 3.3V 기준이어야 한다.
