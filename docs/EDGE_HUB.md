# Edge hub — Jetson → 서버 데이터 전송 엔진과 원격 모니터링

작성 2026-10-06. 상태: **구현·검증 완료(개발 프로파일). 사내망 운영 시험 중. 외부 공개는 실제 서버 이전 후 회사 포트포워딩으로.**

## 0. 왜 만들었나 (방향 전환 기록)

2026-10-06 사용자 지시는 다음과 같다.
- 연구는 새 현장에 다시 설치해 데이터를 수집하고 학습하는 방식으로 진행한다.
- Jetson 의 역할은 **데이터를 수집해 우리 서버로 전달하는 것**이다.
- 학습과 (필요하면) 추론은 서버에서 할 수 있다. Jetson 추론은 필수가 아니다.
- 따라서 핵심 요구는 두 가지다.
  1. Jetson 설치 장소가 바뀌어도 서버로 데이터를 전달하는 엔진.
  2. 외부에서 접근해 서버를 모니터링하는 웹페이지.

이 결정은 `DATA_PLATFORM_ARCHITECTURE.md` 초안의 두 문장을 개발 프로파일에 한해 바꾼다.
- "partner 사양 전에는 uploader·API·viewer 를 만들지 않는다"
- "Jetson 실시간 추론"

나머지 원칙은 그대로 따른다:
- local spool 을 둔다.
- 수신 측이 "받았고 무결하다" 를 확인해야 완료다.
- 전송 단위는 완료된 trial 디렉터리 하나다.
- trial_id 와 파일별 SHA-256 으로 멱등성을 지킨다.
- raw 는 불변이다.
- Jetson 을 인터넷에 직접 노출하지 않는다(outbound only).
- 장치 자격증명은 업로드만 할 수 있고, 열람 자격증명과 분리한다.

2027 partner platform 이 정해지면 uploader 의 목적지 프로토콜만 바꾸면 된다. 수집기 코드와 raw 형식은 이번에도 바꾸지 않았다(태그 `jetson-dataset-v1-ready-2026` 의 수집기 그대로).

## 1. 구성

```
[현장 Jetson]                                        [KETI 학습 서버]
 collector (기존, 무변경)                             edge hub (src/edge_hub, stdlib only)
   └ dataset/<group>/<trial>/  ──┐                     ├ POST /device/units/begin    manifest 등록·재개 지점 응답
   └ results/sensor_collection/  │  HTTPS, outbound     ├ PUT  /device/units/file     offset 기반 이어받기, 완료 시 SHA-256 검증
   └ /run/user/…/preview (휘발)  │  ───────────────▶    ├ POST /device/units/commit   전체 검증 → raw/ 로 원자적 이동, read-only
 uplink agent (jetson_deploy/uplink)                   ├ POST /device/heartbeat      장치 상태 + 최신 센서값 + 열화상(주기)
   ├ ledger.sqlite (~/.cache, dataset 밖)              └ 웹페이지 (로그인)            장치 현황 / 실시간 / 수신 데이터 열람
   └ heartbeat 10 s
```

- **Jetson 쪽** (`jetson_deploy/uplink/`, 실행 `jetson_deploy/uplink_agent.py`)
  - 표준 라이브러리만 쓴다. `factory_runtime` venv 에 패키지를 추가하지 않는다.
  - 완료된 unit 만 보낸다.
    - trial: `experiment.json.status` 가 `completed / failed / aborted / preflight_failed` 인 것. 모두 보내며, 상태는 `status` 로 구분한다.
    - 자유 수집 run: `metadata.json.status` 가 `completed / stopped / failed` 인 것.
    - 위 조건에 더해, 마지막 수정 후 `settle_s`(기본 120 s) 동안 변경이 없어야 한다.
  - 다음 경우에는 보내지 않고 사유만 기록한다: symlink, 특수 파일, 허용되지 않은 파일명.
  - raw 는 읽기만 한다. 업로드 상태는 `~/.cache/factory_safety/uplink/ledger.sqlite` 에 둔다.
  - **로컬 파일은 절대 지우지 않는다.**
  - commit 후 로컬 unit 이 바뀌면 `local_changed_after_commit` 으로 보고만 한다. 서버 사본을 덮어쓰지 않는다.
- **서버 쪽** (`src/edge_hub/`, 실행 `python3 -E -s -m src.edge_hub …`)
  - 표준 라이브러리만 쓴다. 학습 venv 를 건드리지 않는다.
  - 받은 파일은 SHA-256 검증을 통과해야만 `raw/<device>/<unit>/` 에 나타난다. 이후 read-only(0444/0555)다.
  - `manifests/<device>/<unit>.sha256` 는 `sha256sum -c` 형식이다.
  - `raw/<device>/dataset/<group>` 은 `src/data/scripts/build_windows.py --scan-dir` 로 바로 읽을 수 있다.

## 2. 장소가 바뀌어도 동작하는 이유

1. **Jetson 이 먼저 접속한다.** outbound HTTPS 만 쓴다. 현장 방화벽에 inbound 포트, 공인 IP, DDNS 가 필요 없다.
2. **store-and-forward.** 네트워크가 끊겨도 수집은 계속되고, 다시 연결되면 순서대로 보낸다.
   - 수집 속도: 연속 수집 시 약 84 MB/h(2 GB/day). 360 s trial 은 약 8.8 MB.
   - Jetson 여유 130 GB 기준으로 연속 수집 약 2개월분을 버틸 수 있다.
3. **이어받기.** 서버가 실제로 받은 byte 수를 기준으로 재개한다. 실측은 §4.
4. **서버 주소 여러 개.** `server_urls` 를 순서대로 시도하고, 성공한 주소를 기억한다. 예: 사내 LAN 주소 + VPN 이름.
5. **TLS 인증서 고정.** 서버의 사설 CA(`hub-ca.pem`)를 고정한다. 주소가 바뀌어도 서버 인증서만 다시 발급하면 된다.
6. **현장을 옮길 때 사람이 할 일.**
   - 현장 네트워크 접속(Wi-Fi `nmcli` 또는 유선/LTE 동글).
   - 시간 동기(NTP)가 되는지 확인.
   - 나머지는 그대로 동작한다.

## 3. 외부 접속 방식 — 결정됨 (2026-10-06)

**결정**
- 지금 서버(10.252.219.59)는 외부 접속이 막혀 있다. 당분간 **사내망 안에서만** 운영하고 확인한다.
- 실제 운영 서버로 옮긴 뒤 **회사에 포트 하나를 열어 달라고 요청한다(포트포워딩).**
- Tailscale·Cloudflare 같은 외부 서비스는 쓰지 않는다.

**지금 (사내 운영)**
- hub 는 서버의 사내 주소 `https://10.252.219.59:8443` 에서 직접 TLS 로 서비스한다.
- Jetson 은 같은 사내망에서 이 주소로 올린다.
- 사내 PC 에서는 브라우저로 같은 주소를 연다. 사설 CA 를 신뢰 목록에 넣거나, 첫 접속 때 경고를 확인한다.

**나중 (실제 서버 + 포트포워딩) — 바꿀 것은 설정뿐이다. 코드는 그대로다.**
1. 회사 공인 주소의 포트 하나(가능하면 443)를 실제 서버의 hub 포트로 포워딩한다.
2. 서버 인증서를 다시 발급한다. `gen-cert --san IP:<공인 IP> --san DNS:<있으면 도메인> --san IP:<사내 IP>`
   - CA 는 그대로 두므로, Jetson 에 고정한 `hub-ca.pem` 은 바꿀 필요가 없다.
3. Jetson `uplink.json` 의 `server_urls` 를 `["https://<사내 IP>:8443", "https://<공인 IP>:<포트>"]` 로 둔다.
   - 사내에서는 첫 주소, 현장에서는 두 번째 주소로 자동 전환된다.
4. 공개 전 점검:
   - 열람 계정 비밀번호가 12자 이상인지.
   - `revoke-device` 로 쓰지 않는 장치 토큰을 정리했는지.
   - hub 로그에서 `login failed` 를 모니터링하는지.
   - 현장 네트워크가 그 포트로 나가는 것을 허용하는지(443 이 가장 무난하다).
5. 포트포워딩 장비가 원 접속 주소를 `X-Forwarded-For` 로 넘기는 reverse proxy 라면 `--trust-proxy` 를 켠다.
   - 단순 NAT 포트포워딩이면 끈 채로 둔다.

## 3a. 웹에서 수집 시작/정지 (원격 제어, 2026-10-12 추가)

**흐름** — 서버가 Jetson 에 직접 접속하지 않으므로, 현장이 바뀌어도 그대로 동작한다.
1. 운영자가 웹에서 [수집 시작] 또는 [수집 정지] 를 누른다. 누르면 확인 창이 뜬다.
2. hub 는 명령을 큐에 넣는다.
3. Jetson 이 다음 heartbeat(10 s 이내) 응답으로 명령을 받는다. 같은 명령은 한 번만 전달된다.
4. Jetson 이 명령을 실행하고 결과를 보고한다. 웹에는 "최근 명령" 으로 표시된다.

**안전장치**
- **장치 쪽 opt-in.**
  - Jetson `uplink.json` 의 `allow_remote_control` 이 `true` 일 때만 실행한다. 기본값은 `false` 다.
  - 서버가 탈취되어도 이 장치 설정은 바꿀 수 없다.
  - 꺼져 있으면 실행하지 않고, "장치에서 비활성" 으로 보고한다.
- **허용 동작은 셋뿐이다.**
  - `start` 는 `collect.py start` 다(자유 수집 run).
  - `stop` 은 `collect.py stop --timeout 90` 이다(정상 종료).
  - `reset_sps30` 은 `scripts/13_reset_sps30.py` 다(아래 참고).
  - 그 밖의 명령은 양쪽 모두에서 거부한다.
  - 실험 trial(`12_run_trial.py`)은 원격으로 시작하지 않는다. 현장에서 사람이 조건을 확인하는 절차이기 때문이다.
- **권한 분리.**
  - 열람 계정과 운영자 계정을 나눈다. 운영자 지정은 `set-role --role operator` 로 한다.
  - 장치 토큰으로는 명령을 만들 수 없다.
- **요청 검사.**
  - 같은 출처의 요청만 받는다. `X-Hub-Request` 헤더가 있어야 하고, Origin 이 Host 와 같아야 한다.
  - 쿠키는 SameSite=Strict 다.
- **명령 수명.**
  - 장치당 동시에 열린 명령은 하나뿐이다.
  - 120 s 안에 장치가 받지 못하면 **만료**된다. 만료된 명령은 나중에도 실행되지 않는다.
  - 장치가 받았는데 300 s 안에 결과가 오지 않으면 "결과 없음" 으로 표시한다.
- **중복 실행 방지.** 장치는 명령 id 를 ledger 에 기록해, 같은 명령을 두 번 실행하지 않는다.
- **우선순위 보존.**
  - 원격 시작은 `systemd-run --user` 의 별도 unit 으로 띄운다.
  - 그래서 수집기는 uplink 의 낮은 우선순위(Nice=10, idle I/O)를 물려받지 않고 정상 우선순위로 돈다(실측 nice 0).
- **기록.** 누가 언제 무엇을 요청했고 결과가 어땠는지 hub DB(`commands` 테이블)와 hub 로그에 남는다.

**SPS30 소프트웨어 리셋 (2026-10-12)**
- SPS30 이 I2C timeout(`Errno 110`) 등으로 멈추면, 지금까지는 뽑았다 꽂아야 했다.
- [SPS30 리셋] 버튼(또는 아래 명령)은 Sensirion `device_reset` 을 보낸 뒤 두 가지를 확인한다:
  - 제품 형식·시리얼·펌웨어·상태 레지스터를 읽는다.
  - 짧은 측정 1회를 한다.
- 수집기와 같은 하드웨어 lock 을 잡는다. 그래서 **수집이나 trial 이 돌고 있으면 버스에 아무것도 보내지 않고 거부한다.**
  - 장치 쪽 agent 도 수집기가 살아 있으면 먼저 거부한다.
- **자동으로 돌지 않는다.**
  - 수집기는 여전히 첫 SPS30 오류에서 멈추고 그 오류를 보존한다. 수집 중 재시도는 없다.
  - 운영자가 [SPS30 리셋] 을 누르고, 결과를 확인한 뒤 [수집 시작] 으로 새 run 을 만든다. 실패한 run 은 그대로 남아 업로드된다.
- **한계.**
  - 센서가 I2C 응답 자체를 못 하면 리셋 명령도 닿지 않는다. 결과에 "power cycle needed" 가 나온다.
  - 그때는 지금처럼 뽑았다 꽂거나, GPIO 로 끄고 켜는 전원 스위치를 달아야 한다.
- 기록은 `~/.cache/factory_safety/maintenance/sps30-reset-*.json` 에 남는다. raw 데이터와는 따로다.
- 명령: `./jetson_deploy/run_python.sh jetson_deploy/scripts/13_reset_sps30.py [--no-measure]`
  - 종료 코드: 0 정상, 1 리셋 후에도 비정상, 2 버스 열기 실패, 3 수집 중이라 거부.
- 실측(2026-10-12, 정상 센서): 10단계 모두 성공했다. `device_reset` 201 ms, 펌웨어 2.3, 상태 0, PM2.5 13.6 µg/m³.
  - **멈춘 센서에서의 효과는 아직 확인하지 못했다.** 다음 발생 때 기록한다.

**운영자 지정과 장치 허용**
```bash
# 서버: 계정에 운영자 권한 부여 (열람 계정은 그대로 둔다)
python3 -E -s -m src.edge_hub set-role --data-root $D --user <이름> --role operator
# Jetson: ~/.config/factory_safety/uplink.json 에 "allow_remote_control": true 추가 후
systemctl --user restart factory-uplink
```

**알아 둘 점**
- 사용자 linger 가 꺼져 있으면, 로그인 세션이 모두 끝날 때 원격으로 시작한 수집기도 함께 종료된다.
- 상시 운영하려면 `loginctl enable-linger` 를 켠다.

**보안 리뷰(2026-10-12) 반영과 남은 판단**
- **반영한 것**
  - 오프라인이거나 원격 제어를 허용하지 않은 장치에는 명령을 받지 않는다(409).
  - 시작이 20 s 를 넘겨 `starting` 상태여도 실패로 보고하지 않고 최대 40 s 더 기다린다.
  - 거부된 요청과 만료도 hub 로그에 남긴다.
  - `add-viewer` 를 다시 실행해도 기존 권한을 유지한다.
  - systemd unit 이름은 명령 id 로 만든다.
  - 결과 중복 전송을 막는다.
  - 기록은 hub 90일, 장치 최근 1000건만 보관한다.
- **설계상 감수하는 것**
  - **장치 토큰이 탈취되면** 그 장치에 온 명령을 가로채거나 상태와 결과를 위조할 수 있다.
    - 다른 장치를 조작하거나 운영자 권한을 얻을 수는 없다.
    - 실제 실행은 장치 opt-in 과 hub 가 보낸 명령에만 묶여 있다.
    - 의심되면 `revoke-device` 로 폐기한다.
  - **Origin 검사는 Host 헤더와 같은지만 본다.**
    - 브라우저는 다른 호스트로 hub 쿠키를 보내지 않으므로, 이것만으로 악용되지는 않는다.
    - 포트포워딩 후 reverse proxy 가 Host 를 바꾸면 운영자 요청이 403 이 될 수 있다. 그때는 proxy 설정에서 Host 를 보존한다.

 (2026-10-06)

**자동 테스트** `tests/test_edge_hub.py`: 26개(보안 리뷰 회귀 8개 포함), Jetson `run_python.sh` 에서 통과. 실제 TLS(사설 CA)로 localhost 서버를 띄워 확인한 항목:
- 업로드
  - byte 동일성과 read-only 권한, `sha256sum -c` 통과.
  - 두 번째 scan 은 재전송 0. ledger 를 잃어도 서버 영수증으로 복구된다.
  - 중단 지점(123,457 B)부터 재개.
  - offset 불일치는 서버 값으로 교정. 손상 bytes 는 422 로 버리고 commit 하지 않는다.
  - commit 후 로컬 변경은 덮어쓰지 않고 409 를 낸다.
- 전송 대상 판정
  - 미완료·settling·symlink unit 은 제외한다. aborted trial 과 자유 run 은 보낸다.
- 연결
  - 죽은 주소에서 다음 주소로 넘어간다. 모든 주소가 죽으면 pending 으로 둔다.
  - 고정 CA 가 아니면 TLS 실패.
- 인증과 보안
  - 토큰 없음·틀림 401. 다른 장치 id 사칭 403. 토큰은 hash 로만 저장한다.
  - 장치 간 격리.
  - 경로 탈출과 잘못된 이름 400. 선언 크기 초과 400.
  - 열람 API 는 로그인이 필요하다. 장치 토큰은 열람 권한이 아니다. 열람 세션은 장치 API 를 쓸 수 없다. 로그인 실패 제한, CSP 헤더.
- 모니터링
  - heartbeat, live, telemetry, unit 상세·scalars·열화상 frame 열람.
  - 수집기 control token 은 heartbeat 에 포함되지 않는다.

**실제 장비 end-to-end** (Jetson → 서버 scratch hub, SSH 터널, 서버는 127.0.0.1 bind 로 네트워크 노출 0):
- 실제 `_smoke` trial 5개(22.3 MB)가 commit 되었다.
  - 서버 manifest `sha256sum -c` 통과.
  - 저장소 독립 manifest `docs/manifests/development_baseline_20260904.sha256` 의 해당 36 줄과 대조해 통과.
  - `hub verify` 0 실패.
- 학습 쪽 reader(`src/data/window_builder.load_trial`)가 서버 사본과 Jetson 원본에서 **동일한 결과**를 냈다. 비교 항목은 snapshot 수, status, checksum.
  - aborted trial 은 양쪽 모두 `timing_report.json` 누락으로 같은 TrialError 를 낸다. 서버로 옮긴 뒤에도 원래 동작이 그대로 보존된다는 뜻이다.
- 업로드 중 터널을 실제로 끊었다.
  - 그 시점에 서버에는 7개 파일이 완료되어 있었고, `thermal_000003.npz` 는 524,288 B 까지 받은 상태였다.
  - 재연결 후 `offset=524288` 부터 이어 올렸고, 결과 SHA 가 일치했다.
- 수집기를 약 1분 실제로 돌렸다.
  - 웹페이지 실시간 화면에 NTC·CT1·SPS30·SCD30·BME680·FLIR 값과 120×160 열화상 프레임이 표시되었다(headless Chromium 캡처로 확인, JS 오류 0).
  - 수집기를 정지하면 "수집기 미실행" 으로 표시된다.
- 처리량은 Wi-Fi 에서 약 0.8 MB/s 였다. 연속 수집 요구량(약 0.023 MB/s)의 약 30배다.

**독립 보안 리뷰(2026-10-06)** 의 공격자 가정은 세 가지다: 자격증명 없음, 장치 토큰 탈취, 열람 계정. 지적과 반영은 아래와 같다.
- 확인된 결함 7건과 의심 항목은 모두 반영했다. 각각 회귀 테스트가 있다.
- 주요 반영:
  - **이어받기 영구 정지 수정.** 서버가 검증 직전에 죽어 크기가 다 찬 `.part` 가 남으면, begin 에서 검증해 마무리하거나 버린다.
  - **열화상 chunk 읽기 제한.** 압축 해제 크기에 상한을 두고, 요청한 frame 만 읽는다.
  - **장치 문서 처리.** 장치가 보낸 JSON 은 크기 상한을 두고 객체만 받는다. 요약이 실패해도 commit 과 인덱스는 유지된다.
  - **heartbeat 정제.** 깊이·키 수·문자열 길이를 제한하고 숫자만 추세에 넣는다. 크기 상한은 256 KB, telemetry 는 장치당 2 s 에 1점이다.
  - **인증 전 대용량 본문은 읽지 않는다.** 로그인 본문은 4 KB 로 제한하고, 동시 연결 상한은 64 다.
  - **로그인 제한.** 시도를 원자적으로 선기록하고, 없는 사용자도 PBKDF2 를 수행한다(이름 노출 방지). proxy 뒤에서는 오른쪽 끝 XFF 를 쓴다.
  - **staging 제한.** 장치당 미완료 64개, 여유 공간 20 GB 를 유지한다.
  - **기타.**
    - CSV 를 스트리밍으로 읽는다.
    - 장치 토큰을 폐기하는 CLI(`revoke-device`)를 추가했다.
    - 열람 계정을 삭제하면 그 세션을 즉시 끊는다.
    - `.part` 이름 금지를 경로 전체로 넓혔다.
    - symlink 인 unit·group 디렉터리는 보내지 않는다.

**commit 도중 서버가 죽은 경우 (수동 복구)**
- rename 과 receipt 기록 사이에 죽으면 해당 unit 이 `raw directory exists without a receipt` 409 를 낸다.
- 복구 절차:
  1. 서버를 멈춘다.
  2. `raw/<device>/<unit>` 의 각 파일을 장치의 manifest(Jetson ledger 또는 `pending` 출력)와 대조한다.
  3. 일치하면 receipt 를 손으로 쓰지 말고, 해당 raw 디렉터리를 `quarantine/` 로 옮긴다.
  4. Jetson 이 다시 올리게 한다.
- 자동화하지 않았다(raw 를 사람이 확인한 뒤에만 움직인다).

## 5. 설치 절차 (승인 후 사용자가 실행)

서버:
```bash
cd ~/projects/factory_safety            # 이 브랜치를 pull 한 뒤
D=/mnt/data-hdd/keti_data/edge_hub
python3 -E -s -m src.edge_hub init --data-root $D
python3 -E -s -m src.edge_hub gen-cert --data-root $D --san IP:10.252.219.59   # 포트포워딩 후 §3 대로 재발급
python3 -E -s -m src.edge_hub add-device --data-root $D --device-id jetson-keti-01 --token-out ~/jetson-keti-01.token
python3 -E -s -m src.edge_hub add-viewer --data-root $D --user <이름>    # 비밀번호는 프롬프트에 직접 입력
cp src/edge_hub/edge-hub.service ~/.config/systemd/user/ && systemctl --user enable --now edge-hub
```

Jetson:
```bash
install -m 600 <받은 토큰> ~/.config/factory_safety/uplink.token
cp <서버 $D/tls/ca.pem> ~/.config/factory_safety/hub-ca.pem
# ~/.config/factory_safety/uplink.json — 예시는 jetson_deploy/uplink/agent.py 맨 위
./jetson_deploy/run_python.sh jetson_deploy/uplink_agent.py check     # 연결·장치 확인
./jetson_deploy/run_python.sh jetson_deploy/uplink_agent.py pending   # 보낼 것 / 제외 사유 미리 보기
cp jetson_deploy/uplink/factory-uplink.service ~/.config/systemd/user/ && systemctl --user enable --now factory-uplink
```

운영:
- `python3 -E -s -m src.edge_hub verify --data-root $D`: 서버 사본 전체를 다시 hash 하는 무결성 감사.
- `uplink_agent.py status`: Jetson 쪽 전송 현황.

## 6. 하지 않은 것 / 다음 단계

- **Jetson 로컬 데이터 자동 삭제(보존 정책)**
  - 하지 않았다. 지금은 사람이 지운다.
  - 넣는다면 "서버 영수증 + N일 경과 + 서버 verify 통과" 를 조건으로 한다.
- **서버 측 추론·자동 처리**
  - 하지 않았다.
  - commit 직후 hook 으로 window 빌드나 서버 추론을 붙일 수 있다. 다만 라벨은 자동 생성하지 않는다.
- **실시간 스트리밍**
  - 하지 않았다. heartbeat 는 10 s 주기이고, 열화상은 30 s 주기 미리보기다. 원본은 commit 된 raw 다.
- **정식 배포 전 확인할 것**
  - `/mnt/data-hdd` 부팅 마운트(handoff 상 미검증).
  - 외부 접속 방식(§3).
  - 열람 계정 운영.
