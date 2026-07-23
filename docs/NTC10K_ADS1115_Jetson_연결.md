# NTC 10K 3950 + ADS1115 Jetson 연결 메모

이 문서는 에폭시타입 NTC 10K B3950 서미스터 프로브를 ADS1115 ADC를 통해 Jetson Orin Nano에서 읽기 위한 작업 메모다.

## 왜 ADS1115가 필요한가

Jetson Orin Nano 40-pin header에는 아날로그 입력이 없다. NTC 서미스터는 저항값이 온도에 따라 변하는 아날로그 센서이므로, 전압분배 회로로 전압을 만든 뒤 ADS1115 같은 ADC로 읽어야 한다.

## 기준 회로

첫 테스트는 아래 회로를 기준으로 한다.

```text
Jetson 3.3V
  |
  |
10kΩ fixed resistor
  |
  +---- ADS1115 A0
  |
NTC 10K B3950 probe
  |
  |
GND
```

이 회로에서는 온도가 올라가면 NTC 저항이 내려가고, ADS1115 A0 전압도 내려간다.

저항 계산식:

```text
R_ntc = R_fixed * Vout / (Vcc - Vout)
```

온도 계산식, Beta model:

```text
1 / T = 1 / T0 + ln(R_ntc / R0) / B
```

기본값:

```text
R_fixed = 10000Ω
R0 = 10000Ω at 25°C
B = 3950K
T0 = 25°C = 298.15K
Vcc = 3.3V
```

## ADS1115 배선

ADS1115 모듈은 Jetson I2C 신호 보호를 위해 3.3V로 구동한다.

| ADS1115 | Jetson 40-pin |
| --- | --- |
| VDD | pin 1 또는 pin 17, 3.3V |
| GND | pin 6/9/14/20/25/30/34/39 중 하나, GND |
| SDA | pin 27, I2C SDA, `/dev/i2c-1` |
| SCL | pin 28, I2C SCL, `/dev/i2c-1` |
| ADDR | GND, 주소 `0x48` |
| A0 | NTC 전압분배 중간 노드 |

주의:

- ADS1115 VDD를 5V에 연결하지 않는다. 많은 ADS1115 모듈은 SDA/SCL pull-up이 VDD로 걸려 있어서 5V 구동 시 Jetson I2C 핀에 5V가 들어갈 수 있다.
- 첫 번째 ADS1115는 `ADDR -> GND`로 `0x48`을 쓴다.
- 두 번째 ADS1115가 필요하면 `ADDR -> VDD`로 `0x49`를 사용한다.
- SPS30은 같은 `/dev/i2c-1` 버스의 `0x69`에 있으므로 ADS1115 `0x48`과 주소 충돌이 없다.

## 연결 순서

1. Jetson 전원을 끈다.
2. ADS1115 `VDD`를 Jetson 3.3V에 연결한다.
3. ADS1115 `GND`를 Jetson GND에 연결한다.
4. ADS1115 `SDA/SCL`을 Jetson pin 27/28에 연결한다.
5. ADS1115 `ADDR`를 GND에 연결한다.
6. 10kΩ 고정 저항을 Jetson 3.3V와 측정 노드 사이에 연결한다.
7. NTC 프로브를 측정 노드와 GND 사이에 연결한다.
8. 측정 노드를 ADS1115 `A0`에 연결한다.
9. Jetson 전원을 켠 뒤 I2C scan으로 `0x48`을 확인한다.

## 확인 명령

```bash
sudo i2cdetect -r -y 1
```

정상이라면 `0x48`이 보인다. SPS30도 연결되어 있으면 `0x69`도 같이 보인다.

온도 읽기:

```bash
cd /home/keti/projects/factory_safety/jetson_deploy
python3 scripts/08_read_ntc_ads1115.py --samples 10 --interval 1.0
```

결과는 아래에 저장된다.

```text
jetson_deploy/results/ntc/*.csv
jetson_deploy/results/ntc/*.json
```

## 정상값 감각

3.3V 기준, 10kΩ 고정 저항과 10kΩ NTC가 같은 값이면 A0는 약 1.65V다. 실온 25°C 근처에서는 대략 이 값에 가까워야 한다.

예상 범위:

| 상태 | A0 전압 | 해석 |
| --- | --- | --- |
| 약 1.65V | 약 25°C | 정상적인 실온 근처 |
| 0V 근처 | NTC short 또는 A0가 GND에 붙음 |
| 3.3V 근처 | NTC open 또는 GND 경로 끊김 |
| 음수/이상값 | A0가 떠 있거나 배선 오류 |

## 프로젝트 입력 채널

NTC 측정값은 모델 입력 순서에서 첫 번째 채널이다.

```text
[NTC, PM1.0, PM2.5, PM10, CT1, CT2, CT3, CT4]
```
