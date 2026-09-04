# Consortium Data Platform — Integration Questions

**목적**: 2027년 상반기 독일 robot test site 에 Jetson Orin Nano sensor module 을 설치하기
전에, **consortium partner 가 운영하는 DB / data platform 과의 integration 조건**을 확인하기
위한 질문 목록.

**이 문서는 답을 추측해서 채우지 않는다.** partner 의 product · API · DB · authentication ·
network · hosting 은 확인될 때까지 `TBD` 로 유지한다.

architecture 정본: [`DATA_PLATFORM_ARCHITECTURE.md`](DATA_PLATFORM_ARCHITECTURE.md)

## 운영 방법

답변을 받으면 해당 질문 블록의 네 필드를 갱신한다.

```
Answer            받은 내용
Status            OPEN -> CONFIRMED  (또는 PARTIAL / NOT_APPLICABLE)
Confirmed date    YYYY-MM-DD
Contact / source  누구에게서 어떤 경로로 확인했는지
```

초기 상태는 모든 질문이 `Answer = TBD`, `Status = OPEN` 이다.

---

## 먼저 확인해야 할 다섯 가지

아래 다섯 개는 **architecture 결정에 직접 영향을 준다.** 나머지 15개보다 먼저 답을 받아야 한다.

| ID | 항목 | 답에 따라 달라지는 것 |
|---|---|---|
| **P01** | outbound connectivity | 회선이 없으면 §7 local spool 이 "업로드 대기" 가 아니라 **오프라인 반출** 설계가 된다 |
| **P02** | upload interface | uploader 구현 방식 전체 |
| **P03** | authentication | credential 관리·보관 방식, 현장 설치 절차 |
| **P09** | Korea remote access | 귀국 후 연구 진행 가능 여부 |
| **P10** | programmatic export | **학습 automation 가능 여부.** 불가하면 학습 경로를 다시 설계해야 한다 |

---

## 우리 쪽 고정된 사실 (질문의 전제)

partner 에게 제시할 수 있는 우리 값이다. 이미 실측·검증된 것이다.

```
trial 단위        completed trial directory (파일 7종, 아래)
trial 1건 크기    약 8.8 MB (360초, thermal + CT raw 포함)
                  thermal NPZ 가 약 87 %
100 trials        약 0.9 GB
500 trials        약 4.4 GB
timestamp         UTC + 로컬 monotonic acquisition clock
checksum 후보     SHA-256
raw 원칙          immutable (수정하지 않는다)
```

파일 구성:

```
experiment.json      실험 메타데이터 (scenario, severity, phase plan, quality)
metadata.json        수집 설정, sensor manifest (센서 serial 포함), sensor profile
scalars.csv          tick 당 1행 (360행)
snapshots.jsonl      tick 당 1줄 전체 스냅샷 (360줄)
thermal_*.npz        열화상 30프레임 단위 (120x160 uint16)
timing_report.json   타이밍·품질 통계
ct_raw_*.npz         선택적, CT 원시 파형
```

---

## P01 — Network connectivity

| | |
|---|---|
| **Priority** | **HIGH** |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Will the Jetson Orin Nano installed at the German test site have outbound network
> connectivity to the partner-operated data platform?

**확인할 것**

- Internet / intranet / private network 중 무엇인가
- outbound HTTPS 가능 여부
- 필요한 port
- proxy 를 거쳐야 하는지 (있다면 인증 방식)

---

## P02 — Data upload interface

| | |
|---|---|
| **Priority** | **HIGH** |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> What interface should the Jetson use to upload completed trial data?

**확인할 것**

- REST API / SFTP / object storage / SDK / shared filesystem / custom platform connector
- 문서 또는 예제 제공 가능 여부

---

## P03 — Authentication

| | |
|---|---|
| **Priority** | **HIGH** |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> What authentication mechanism is required for Jetson data upload?

**확인할 것**

- API key / OAuth·OIDC / client certificate / VPN / username-password / 기타
- credential 발급·회전·폐기 절차
- 현장 장비에 credential 을 어떻게 안전하게 넣는지

---

## P04 — Upload data model

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Can the platform accept one completed trial as a group of files, or does the data need
> to be transformed into another schema?

**확인할 것**

- 파일 묶음(디렉터리) 그대로 받을 수 있는지
- 별도 schema 변환이 필요하면 그 스펙
- 우리 현재 trial 파일 구성은 위 "우리 쪽 고정된 사실" 참조

---

## P05 — File / object limits

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Are there limits on individual file size, total trial size, upload frequency, or
> storage quota?

**확인할 것**

- 파일 1건 최대 크기 (우리 최대 파일은 thermal NPZ, 약 0.6 MB)
- trial 1건 최대 크기 (우리 약 8.8 MB)
- 업로드 빈도 제한
- 총 저장 quota

---

## P06 — Integrity

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Does the platform support checksums or another mechanism to verify that uploaded data
> was received without corruption?

**확인할 것**

- checksum 지원 여부와 알고리즘 (우리 후보 **SHA-256**)
- 서버가 무결성 확인 결과를 응답으로 돌려주는지

우리 spool 정책은 **"보냈다" 가 아니라 "받았고 무결하다"** 를 확인한 뒤에만 로컬 raw 를
정리하도록 설계되어 있다. 이 질문의 답이 그 확인 수단을 결정한다.

---

## P07 — Idempotency / duplicate handling

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> How should duplicate uploads of the same `trial_id` be handled?

**확인할 것**

- reject / overwrite / version / idempotent success 중 어느 동작인지
- 재시도 시 중복이 생기지 않게 하는 권장 방법

---

## P08 — Interrupted upload

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Does the platform support retry or resume after a partial or interrupted upload?

**확인할 것**

- resume 지원 여부, 부분 업로드가 남았을 때의 정리 방법
- 실패한 업로드가 조회 결과에 노출되는지

---

## P09 — Korea remote access

| | |
|---|---|
| **Priority** | **HIGH** |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> How will the Korean research team remotely access the uploaded data after returning
> from the German site?

**확인할 것**

- web viewer / REST API / VPN / SFTP / SDK / remote desktop / 기타
- 한국에서의 접속 가능 시간대·대역폭 제약

---

## P10 — Programmatic export

| | |
|---|---|
| **Priority** | **HIGH** |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Can the Korean research workstation programmatically retrieve trial data for
> preprocessing, model training, evaluation, and batch re-inference?

**확인할 것**

- 프로그램으로 대량 조회·다운로드가 가능한지 (수동 다운로드만 가능한지)
- 인증된 자동화 접근이 허용되는지

**이 질문이 학습 automation 가능 여부를 직접 결정한다.** 불가하면
`DATA_PLATFORM_ARCHITECTURE.md` §10 의 학습 경로를 다시 설계해야 한다.

---

## P11 — Query / metadata interface

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Can trials be searched or queried by metadata such as date, `scenario_id`, `trial_id`,
> `status`, and sensor profile?

**확인할 것**

- 검색 가능한 필드 목록
- 우리 metadata 필드를 platform 에 어떻게 전달해야 검색 가능해지는지

---

## P12 — Raw data preservation

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Will the original uploaded files be preserved without modification?

**확인할 것**

- 업로드 원본을 변형(재인코딩·압축·schema 변환)하지 않고 보관하는지
- 변형이 불가피하면 원본을 함께 보관할 수 있는지

우리 원칙은 **raw immutable** 이다. annotation 은 원본을 수정하지 않고 별도 layer 에 둔다.

---

## P13 — Retention

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> What is the retention policy for uploaded raw trial data?

**확인할 것**

- 보관 기간, 자동 삭제 여부
- 과제 종료 후 데이터 처리 방침

---

## P14 — Backup responsibility

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Who is responsible for backup and disaster recovery of the uploaded data?

**확인할 것**

- partner 측 백업 범위와 주기
- 우리 쪽에서 별도 사본을 보유해야 하는지

---

## P15 — Timestamp / time synchronization

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> What timestamp and timezone convention should be used?

**확인할 것**

- platform 의 timestamp·timezone 규약
- 현장에서 **NTP 사용 가능 여부**

우리 현재 방식: **UTC timestamp + 로컬 monotonic acquisition clock.** tick 스케줄은 monotonic
deadline 이고 사람이 읽는 시각은 UTC 로 기록한다. NTP 가 없으면 벽시계 시각의 절대 정확도가
떨어지지만 tick 간격은 영향받지 않는다.

---

## P16 — Security / network controls

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Are TLS, VPN, firewall allow-listing, client certificates, or other network-security
> controls required?

**확인할 것**

- 필요한 보안 통제와 그 설정 주체
- Jetson 에 inbound 접속을 요구하는 요건이 있는지 (우리는 **outbound-only** 를 유지한다)

---

## P17 — User / institution permissions

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Can access permissions be separated by institution or user?

**확인할 것**

- 기관·사용자 단위 권한 분리 가능 여부
- 예: KETI read/export · partner admin · **Jetson upload-only**

우리 원칙은 **하나의 credential 에 모든 권한을 주지 않는 것**이다.

---

## P18 — API limits

| | |
|---|---|
| **Priority** | LOW |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Are there API rate limits, upload limits, or download quotas?

**확인할 것**

- rate limit 수치와 초과 시 동작
- 대량 다운로드(학습용 일괄 확보) 시 제약

---

## P19 — Data governance

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> Are there consortium, company, or regulatory requirements governing sensor data
> collected at the German test site?

**확인할 것**

- 컨소시엄·기업 내부 규정, 적용되는 법규
- 데이터 반출·공개 범위, 라이선스

**현재 우리 센서 데이터에 사람이 식별 가능한 정보가 포함된다고 단정하지 않는다.**
120×160 열화상의 식별 가능성, 현장에 사람이 등장하는지 여부는 확인이 필요한 별개 사안이다.
확인 전에 "포함된다" 또는 "포함되지 않는다" 로 결론짓지 않는다.

---

## P20 — Model delivery

| | |
|---|---|
| **Priority** | MEDIUM |
| **Answer** | TBD |
| **Status** | OPEN |
| **Confirmed date** | — |
| **Contact / source** | — |

**Question (EN)**

> What mechanism will be available to deliver a validated model artifact from the Korean
> research team back to the Jetson installed in Germany?

**확인할 것**

- 배포 경로 (platform 경유 / 별도 채널 / 현장 방문)
- 배포용 인증 수단 (업로드 credential 과 분리되어야 한다)

우리 현재 canonical portable artifact 후보는 **ONNX** 다. TensorRT serialized engine 은
runtime environment 에 민감해 서버에서 빌드한 engine 이 현장 Jetson 에서 동작한다고 가정하지
않는다. **automatic deployment 는 아직 결정하지 않았다.**

---

## 상태 요약

| ID | 항목 | Priority | Status |
|---|---|---|---|
| P01 | Network connectivity | **HIGH** | OPEN |
| P02 | Data upload interface | **HIGH** | OPEN |
| P03 | Authentication | **HIGH** | OPEN |
| P04 | Upload data model | MEDIUM | OPEN |
| P05 | File / object limits | MEDIUM | OPEN |
| P06 | Integrity | MEDIUM | OPEN |
| P07 | Idempotency / duplicate handling | MEDIUM | OPEN |
| P08 | Interrupted upload | MEDIUM | OPEN |
| P09 | Korea remote access | **HIGH** | OPEN |
| P10 | Programmatic export | **HIGH** | OPEN |
| P11 | Query / metadata interface | MEDIUM | OPEN |
| P12 | Raw data preservation | MEDIUM | OPEN |
| P13 | Retention | MEDIUM | OPEN |
| P14 | Backup responsibility | MEDIUM | OPEN |
| P15 | Timestamp / time synchronization | MEDIUM | OPEN |
| P16 | Security / network controls | MEDIUM | OPEN |
| P17 | User / institution permissions | MEDIUM | OPEN |
| P18 | API limits | LOW | OPEN |
| P19 | Data governance | MEDIUM | OPEN |
| P20 | Model delivery | MEDIUM | OPEN |

```
CONFIRMED  0 / 20
OPEN      20 / 20
```
