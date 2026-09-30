# EXP004 raw cache 독립 검토 — Codex

2026-09-29 · 기준 HEAD `661bbbb9ba7bb9e35a663affb64e6b110a93b15c` + 미커밋 후보

**최종 판정: 수정본 fixture 검증 PASS, RC-01 닫힘.** 최초 후보의 window identity 검사 누락을
Claude가 수정했고 Codex가 새 SHA로 독립 재검증했다(§6). 실제 데이터 캐시 생성과 P1b 학습은 실행하지 않았다.
이 검토는 [Claude 갱신 (26)](REVIEW_20260929_EXP004_claude.md)에 대한 독립 검증이다.

## 1. 검증 대상과 환경

| 파일 | SHA-256 |
|---|---|
| `src/data/anomaly_raw_cache.py` | `5d5db02e15a1290922d6d3e4452106d9bda8bf779d0867944555396112eb34ac` |
| `tests/test_anomaly_raw_cache.py` | `6dcf7a2e5d7a3935787baf4376ad322bb3208c650077bf4f8653f4cddc0db281` |

후보와 의존 파일을 별도 패킷으로 동결하고 실행 전후 SHA를 확인했다. 기존 서버 운영 clone을
수정하지 않았고, 임시 인공 데이터만 읽었다. 실제 데이터 읽기·모델 optimizer step은 모두 0이다.

| 독립 실행 | 전체 테스트 | 통과 | 건너뜀 | 실패/오류 | 시간 |
|---|---:|---:|---:|---:|---:|
| Jetson `factory_runtime`, NumPy 1.26.4 | 50 | 48 | 2 | 0/0 | 21.61초 |
| 서버 `factory_training`, NumPy 2.2.6, torch 2.6.0+cu124 | 50 | 50 | 0 | 0/0 | 14.81초 |

Jetson의 두 건너뜀은 PyTorch가 필요한 pilot 선택 순서 비교와 `normalized_thermal` 비교다.
서버에서는 두 테스트도 실행했다. 이 표는 raw-cache 테스트만의 결과이며, 프로젝트 전체 테스트나
모델 성능 평가 결과가 아니다. Claude의 전체 suite 수치 정정(301개 중 213 통과·88 건너뜀,
새 파일 포함 351개 중 261 통과·90 건너뜀)은 별도 peer 보고로 보존한다. 여기서는 전체 suite를
다시 실행하지 않았다.

## 2. RC-01 — 최초 후보의 session/start identity 누락, §6에서 수정 확인

`window_identity()`와 `window_order_sha256`는 `session_id`, `start_index`를 포함한다.
하지만 `reader()`는 base ID·device·raw IDs·hard_valid·pure_Normal만 검사한다.
다음 두 호출은 Jetson과 서버에서 모두 예외 없이 원래 배열을 반환했다.

```python
read(dict(original, session_id=original["session_id"] + ":changed"))
read(dict(original, start_index=original["start_index"] + 1))
```

이는 원시 배열이 변했다는 증거는 아니다. 저장된 세션/위치 provenance와 호출자가 전달한
metadata가 다른 창을 허용한다는 문제다. 갱신 (26)의 “identity가 달라진 창 거부” 주장에도 미달한다.
저장된 role을 사용해 호출 창의 전체 `window_identity`를 JSON 정규화한 뒤 기록 identity와
비교하고, 두 필드 변경을 각각 거부하는 회귀 테스트를 요청했다. 코드 소유권은 Claude에 유지한다.

다른 fold의 role map으로 캐시를 재사용하는 문제는 위 수정과 구분한다. 실제 생성/소비 harness에서
아래 split·fold·roles digest를 확인해야 한다. 모듈의 `reader(window)`만으로 현재 외부 role map을
검증한다고 주장하지 않는다.

## 3. 실패 검출력 확인

원본 파일을 수정하지 않고 메모리의 후보 사본에 한 번에 한 가지 결함을 넣었다. 11개 모두
지정한 테스트 본문이 실패해 검출했다. fixture 준비 단계의 실패를 검출 성공으로 세지 않았다.

- dtype 자동 변환, calibration 창 허용
- flush 후 payload 비교 생략, 전체 npy 파일 비교 생략
- KeyboardInterrupt 실패 기록 누락, flush 실패 기록 누락
- 전체 창/원본 순서 대신 첫 창/원본만 해시
- reader의 hard_valid 검사 생략
- provenance JSON 왕복 정규화 생략
- attach 시 파일 변조 검사 생략

첫 mutation harness는 여섯 번째 변이 문자열이 함수 정의에도 매칭되어 중단됐다. 이는 검증 도구의
선택 오류다. 첫 5개 결과와 `failure.json`을 보존하고, 호출문으로 범위를 좁힌 별도 continuation에서
남은 6개를 검증했다. 원래 실행의 실패 상태를 성공으로 덮어쓰지 않았다. 이 11개 결과는 Claude가
보고한 46개/14개 mutation 실행과 별개이며, 모든 가능한 결함을 검출한다는 뜻은 아니다.

## 4. 실제 캐시 생성 harness 제안

출력은 raw root와 분리된 SSD의
`/mnt/data-ssd/keti_data/factory_safety/exp004_raw_cache/` 아래 새 고유 디렉터리를 제안한다.
fold·source digest·cache 코드 digest·UTC 시각을 이름에 넣고 기존 경로를 재사용하지 않는다.
현재 이 경로에 실제 캐시를 만들지 않았다.

원래 P0 manifest를 다시 해시하고 metadata만으로 계산한 바인딩은 다음과 같다.

| 항목 | 값 |
|---|---|
| fold / fit / dev | 0 / 2,088 / 525 |
| raw manifest payload SHA | `5d4b6cb7ab872470a2846586f0953e437332c0d0e2042b3757934507f7527b0b` |
| split manifest payload SHA | `7b092521f4eca8b3c9fff84ee748195a55e21adca5d8f38ab34cca228a972c61` |
| 전체 device→role map SHA | `1d522230f416fee026bde354aed18a154d003b8edc02a52a62f52def2c97f581` |
| fit→dev base ID 순서 SHA | `3a23c25e04ecb982d4d7d63a27cf388c7db464cb4581bfff5204cc147311c56b` |
| 전체 window identity 순서 SHA | `b6e7b26095dde546196886e9adc0e5572401cb4377032d30175e579b3123d6d7` |

thermal payload는 12,040,704,000 bytes, sensor는 5,016,960 bytes, labels는 627,120 bytes다.
합계 12,046,348,080 bytes에 npy header·manifest·여유 공간이 추가된다. sensor/thermal float64,
labels int64를 선언하되 canonical reader 출력이 다르면 변환 없이 실패시킨다. raw 8채널을 보존하므로
캐시 자체는 CT 선택이나 학습 seed에 의존하지 않는다. 특정 fold의 fit/dev 경계에는 의존한다.

생성 harness는 같은 root와 같은 raw manifest에서 `make_reader`를 직접 구성하고, 캐시 선택 창과
fold0 pilot 선택 창의 전체 identity/순서를 대조한다. provenance에는 저장소 SHA/dirty, 시각, 호스트,
Python/NumPy/환경 프로파일, 서버의 해당 없음 필드, 실제 실행 소스 SHA, 독립 검증 보고서 SHA,
fold·roles·split/raw manifest의 파일 및 payload SHA를 넣는다. 기존 manifest나 원시 파일을 바꾸지 않는다.

소비할 때도 현재 split/fold/roles 및 창 순서를 재대조하고 `attach`의 파일 검사와
`reader(verify_each_read=True)`를 사용한다. 실제 Normal·기존 주입의 최종 모델 입력 parity를 확인한
뒤에만 학습 연결 검증으로 넘어간다. 원시 파일 디코딩/재해시를 없애는 효과와 메모리 바이트 검사의
비용은 실제 측정 전 구분한다. BIN sampler/dev weighting은 이 검토에서 확정하지 않는다.

## 5. 재현 산출물

로컬 디렉터리: `/home/keti/agent-collaboration/factory-safety/`

- 코드/검증 harness: `exp004_raw_cache_independent_20260929T084723Z/`
- Jetson 결과: `exp004_raw_cache_independent_20260929T084723Z_jetson_result/`
- 서버 결과 사본: `exp004_raw_cache_independent_20260929T084723Z_server_result/`
- 독립 보고서: 코드 패킷의 `independent-review.json`, payload SHA
  `78ad9acc5325f967ba2fd0c43e1149948f71cd3b8e98ece81574527e84820b5b`

서버 코드: `/home/keti/review_runs/exp004_raw_cache_independent_20260929T084723Z_code/`.
서버 결과: `/home/keti/review_runs/exp004_raw_cache_independent_20260929T084723Z_server_result/`.
`summary.json`, `tests.log`, `identity-probes.json`, provenance 및 완료 기록을 보존했다.

## 6. 수정본 독립 재검증 — RC-01 닫힘

Claude가 기록된 전체 identity와 JSON 정규화한 호출 identity를 비교하도록 수정했다.
session/start 변경, session 누락, NumPy scalar metadata, device 변경은 거부하고 정상 창의
깊은 복사본은 허용하는 회귀 검증을 추가했다. 새 테스트 메서드를 늘리지 않고 기존 메서드의
subtest를 보강했으므로 총 메서드 수는 50이다.

| 파일 | 최종 SHA-256 |
|---|---|
| `src/data/anomaly_raw_cache.py` | `88a92e76d47edcc0ba3d3c034f0c1894072c5dfce7fa2796cc35b75cd264bede` |
| `tests/test_anomaly_raw_cache.py` | `d672a35a8f6f80106e893bec2e6d6ae36c0b5e4b5e7d7fe17f5e113f967f94eb` |

새 패킷 `exp004_raw_cache_fixed_20260929T085752Z`로 Jetson **48 통과·2 건너뜀, 25.30초**,
서버 **50 통과·건너뜀 0, 14.69초**를 확인했다. 최초 재현과 같은 독립 session/start 변경 probe도
양쪽 환경에서 거부됐다. hard_valid/session/start 검사를 각각 제거한 세 변이는 보강된 테스트
본문에서 모두 검출됐다. 기존 11개 변이는 최초 후보에서 수행한 기록이고, 새 수정본에는
변경 관련 3개만 수행했다. 실행 전후 동결 패킷 SHA와 현재 checkout의 SHA도 일치했다.

수정본의 코드 패킷과 `_jetson_result/`, `_server_result/`는 앞 절과 같은 로컬/서버 parent에 있다.
최종 `independent-review.json`의 payload SHA는
`acbcb8d26f731b79549328d2193059681f81f5f48f4b5922b4447379b87c3b5d`다.
판정 `PASS_FIXTURE_AND_RC01`은 모듈의 인공 fixture 검증에 한정한다. 실데이터 캐시/학습 결과,
full P0, 모델 성능의 PASS로 확대하지 않는다.

다음 단계는 §4의 실제 캐시 생성/검증 harness다. 최초에는 Claude 구현/Codex 검토로 배정했으나,
갱신 (27) D에서 Claude의 자체 확인 대기 사유가 명확해진 뒤 Codex 구현/Claude 검토로 조정했다.
원본 캐시 leaf 모듈과 해당 테스트는 계속 Claude 소유다. 실제 생성 실행에 앞서 frozen manifest·
fold/roles·canonical reader·소스/검증 보고서 바인딩, 실패 보존, 실제 CLI 성공/실패 경로를 검증한다.

## 7. 생성·검증 CLI 후보 — 코드/인공 fixture 및 실제 manifest 연결 확인

Codex가 `src/build_anomaly_raw_cache.py`, `tests/test_build_anomaly_raw_cache.py`를 작성했다.
이 작업은 기존 연구 범위의 가역적 구현이며, 실제 캐시 생성·학습·commit/push 실행을 포함하지 않는다.
Claude에는 두 파일을 수정하지 않는 peer review를 요청했다. **아래는 최초 28개 fixture 후보의
기록이다. 검토 (29)를 반영한 후속 후보와 검증은 §8에 기록한다.**

| 파일 | 후보 SHA-256 |
|---|---|
| `src/build_anomaly_raw_cache.py` | `a687f4d51576e416b6ee569411b555cccd9b5532715985bffe2cb2849e070761` |
| `tests/test_build_anomaly_raw_cache.py` | `bb8ba5a9e832338f0a02e51116eecfde6292b1dbca6d9d2cf5856c948291caba` |

실행기는 module review와 CLI fixture report의 **외부에서 지정한 파일 SHA** 및 자체 payload SHA,
전이 소스 파일 SHA를 확인한다. split/raw manifest도 파일·payload SHA를 모두 확인하고 fold0의
roles·전체 window 순서·normalizer fit provenance·dtype/용량을 동결 보고서와 대조한다.
caller가 reader를 전달하는 CLI 옵션은 없으며 동일 root/ledger로 `make_reader`를 직접 구성한다.

환경과 새 SSD 출력 경로 검사를 통과하면 새 run 디렉터리를 만들고, 이후 입력 gate 오류도 그
디렉터리의 `failure.json`에 보존한다. 캐시는 `run/cache/`에 있으며 성공은 부모의 `result.json`으로
판정한다. 부모에 실패 기록이 있거나 결과 기록이 없으면 자식 캐시의 존재만으로 성공 처리하지 않는다.
마지막 JSON 저장 실패·KeyboardInterrupt·실패 기록 자체의 쓰기 오류에서도 원래 예외를 유지한다.
기존 출력 경로를 덮어쓰거나 실패 출력을 재사용하지 않는다.

원시/Normal parity는 fit/dev 전 창에 대해 수행한다. Normal 모델 입력은 CT1/CT2 각각 기존
정규화와 최종 float32 변환 결과를 byte 단위로 비교한다. 합성 parity는 **role/device별 첫 창**에서
CT1/CT2 × 3 family × 3 strength, seed42를 비교하는 표본이다. 채택/거부를 모두 기록하고 원시 주입
배열·사건 metadata·최종 tensor를 대조한다. 전체 합성 bank 검증이나 성능 측정으로 표시하지 않는다.

| 검증 | 결과 |
|---|---|
| Jetson 정규 wrapper의 CLI suite | **28/28 PASS, skip 0**, 12.994초 |
| 서버 독립 패킷의 동일 suite | **28/28 PASS, skip 0**, 9.744초 |
| 서버의 실제 동결 manifest metadata preflight | **PASS**, 2,613창, 4.961초 |
| 실제 원시 파일 읽기 / 실제 캐시 생성 / optimizer step | **0 / 미실행 / 0** |

성공과 마지막 JSON 저장 실패 테스트 모두 실제 `main`→입력 검증→canonical 인공 파일 reader→
캐시 쓰기/flush/attach→raw/Normal/합성 parity→최종 저장 경로를 거친다. 테스트에서만 환경과 출력
parent를 임시 경로로 대체하고, 검증 보고서는 인공 fixture임을 명시한 모의 입력을 사용한다.
서버 metadata preflight는 실제 module review와 바로 생성한 서버 CLI fixture report, 원래 P0의
split/raw JSON을 읽는다. 이 preflight에서 `make_reader`가 호출되면 즉시 실패시키는 장치도 넣었다.

실제 데이터용 비용 계획은 build 2,613 + 독립 verification 2,613 = **canonical 창 읽기 5,226회**,
Normal tensor pair 5,226개, 합성 표본 20창·360 paired attempts다. 실제 wall 시간 추정은 `null`이며
fixture 시간으로 대체하지 않는다. 합성 표본의 대표성이나 미관측 이상 상태 성능을 주장하지 않는다.

서버 코드/결과는 `/home/keti/review_runs/exp004_raw_cache_cli_20260929T091748Z_code/`,
`exp004_raw_cache_cli_20260929T091748Z_server_result/`다. 로컬에는 같은 이름으로
`/home/keti/agent-collaboration/factory-safety/` 아래 패킷과 결과 사본을 보존했다.

- `cli-fixture-report.json` 파일 SHA:
  `2075468a43e7d18837c977eb07357418758262ff59497a7288c96e92f8224070`
- 위 보고서 payload SHA:
  `9e5b8f5c45200a234723830fc3e1f4cd57abf1a68bcfb2c75bb5fe386f3d8125`
- `frozen-metadata-preflight.json` payload SHA:
  `c698e7878bec6f8b3437cf1efa0a754a8d0155b132bb57bde94cf07dcf1cc3b9`

이 소스가 바뀌면 기존 fixture 보고서는 실제 생성 실행의 근거로 재사용할 수 없다.

## 8. 검토 (29) 반영 — 실패 보존 수정과 검증 공백 보완

Claude의 `a687f4d5/bb8ba5a9` 스냅샷 검토에서 확인한 코드 결함 3건을 수정했다.
출력은 SSD parent 바로 아래의 새 run 디렉터리만 허용해 성공/실패 run 내부의 숨은 재시도를
거부한다. 실패 파일 목록 수집에도 별도 예외 처리를 넣어 PermissionError나 두 번째 Ctrl-C가
원래 예외와 최소 실패 기록을 가리지 않게 했다. 입력 단계 실패에도 환경·소스·git metadata를
남긴다. 아직 검증된 evidence를 읽지 못한 경우 git SHA는 null이고 확인 전이라는 사유를 기록한다.

검사 개수만 확인하던 테스트를 보완했다. canonical 두 번째 읽기의 표본 외 창에만 thermal 1-ULP,
label dtype, sensor shape 변화를 주고 실제 main의 parity 단계 실패를 확인한다. attach 이후
캐시 파일 변경, raw/Normal/합성의 독립 양쪽 출처, 주입 직후 float32 변환에 가려지는 차이도
검증한다. CT·family·strength·role·seed 호출과 각 창의 두 번 읽기를 별도로 단언하고,
개별 환경 조건·재서명/재pin 음성 입력·Normal witness·실제 parity 반환값 기록을 확인한다.
canonical identity/role 밖의 읽기와 기존 JSON 임시 파일 덮어쓰기도 거부됨을 확인한다.

| 파일 | 수정 후보 SHA-256 |
|---|---|
| `src/build_anomaly_raw_cache.py` | `9c1fca3f49d238f6fe70af3b8a96ee17c84bad2a6bde62fb3a029411f95f312f` |
| `tests/test_build_anomaly_raw_cache.py` | `fdb04d04f78b07730bba911233d9f40056b83a00953a92221a8df20f9525ffd0` |

Jetson에서 51/51 PASS(skip 0, 43.926초) 후 split payload binding 음성 subcase를 한 개 더
추가했고 해당 메서드를 재실행해 통과했다. **최종 SHA의 서버 전체 suite는 51/51 PASS,
skip/failure/error 0, 26.944초**다. frozen metadata preflight도 2,613창 PASS(5.015초)다.
실행 전후 패킷 SHA와 현재 checkout의 소스 SHA가 일치한다. 서버 패킷은
`/home/keti/review_runs/exp004_raw_cache_cli_r29_final_20260929T102238Z_code/`,
결과는 같은 parent의 `exp004_raw_cache_cli_r29_final_20260929T102238Z_server_result/`에 있다.
로컬 사본은 `/home/keti/agent-collaboration/factory-safety/` 아래에 보존했다.

- 최종 `cli-fixture-report.json` 파일 SHA:
  `997275a5ed8a98b61df30a8a297abfb25530812ea12bd5b87538d68d88e7f1c3`
- 위 보고서 payload SHA:
  `9a9f0fb318360a75dd2fc19280ea5c044aa1672ca2b24a2d7531936c6a6eac29`
- frozen metadata preflight payload SHA:
  `1f7e7d5b4e77e4e5a18dc147de6e651352848bc7d74a7028e1da3b839f0e9cd9`

모든 변경은 인공 fixture와 metadata 검증 범위다. 실제 원시 데이터 읽기·실제 캐시 생성·optimizer
step은 0이며, full P0와 held-out 성능 평가는 여전히 미실행이다. Claude에 위 불변 스냅샷의
읽기 전용 재검토를 전달했다. Codex는 commit/push를 수행하지 않았다.

### 8.1 지적별 변이 재검증과 한계

Claude가 전달한 단일 변이 67개(M26은 여러 곳에 매칭돼 provenance 한 곳으로 한정), 조합 4개,
코드 결함 B1–B3을 되살린 3개를 **지적별 지정 테스트**로 확인했다. 74개 중 일반 실패로 66개,
테스트 본문에서 의도한 KeyboardInterrupt가 밖으로 나가는 것으로 2개를 검출했다. C1–C4와
B1–B3은 모두 검출했다. 이는 전체 suite를 각 변이에 반복한 mutation score가 아니다.

6개는 지정 테스트를 통과했다. M25는 읽기 실패 전에 내부 counter를 올려도 실패 시 결과 기록이
없어 성공 경로의 관측값이 달라지지 않는다. M60은 별도 source 비교문을 없애도 최종
`validate_inputs`가 같은 pinned evidence와 현재 source를 다시 비교해 source drift를 거부한다.
M39/M40(정규화 finite/shape), M41/M42(early raw_sources/check_windows)는 지정 테스트가 삭제를
검출하지 못했다. 검토 (29) E가 중복 검증으로 분류한 항목이며, 이번 재검증에서 전 suite의
동등성을 증명했다고 주장하지 않는다.

첫 sweep은 M58 뒤 턴 중단으로 종료했고 기록을 보존했다. M59와 B2는 unittest가 요약 footer를
출력하기 전에 KeyboardInterrupt로 종료해 **mutation harness의 결과 분류 assertion도 실패**했다.
이를 정상 unittest 실패로 바꾸어 기록하지 않았다. 실제 테스트 본문·원래 오류·누출된 interrupt
trace와 harness 오류를 함께 보존하고, 별도 continuation으로 나머지 항목을 확인했다.
이 과정에서 최종 repository 코드나 실제 데이터는 수정하지 않았다.

중간 51개 후보에서 보완이 더 필요했던 M01은 최종 split payload binding 음성 subcase로
검출했고, M34는 main을 거치지 않는 최초 selector를 바로잡아 검출했다. 두 후보 사이에서 바뀐
테스트 메서드는 M01 관련 하나뿐이며 AST 대조 결과를 남겼다. 각 결과에 사용된 정확한 사본과
로그 위치는 최종 패킷의 `targeted-mutation-review.json`에 있다. payload SHA:
`d03cb3d5bbc66ba7bb9d07707b9c157a4c4a8fe08f5fd26e76f35f0ddf4c915f`.

## 9. 검토 (30) 반영 — 운영 코드 유지, 테스트 공백 보완

Claude는 고정 후보를 재검토해 **추가 코드 결함 0건**, 검토 (29) B 세 건의 수정 유효성,
기존 변이 결과 및 M60 중복 방어를 독립 확인했다. 새 테스트 공백 중 medium 다섯 항목과
인접 low 항목을 Codex가 보강했다. 운영 CLI SHA는 §8의 `9c1fca3f…` 그대로다.

- 검증 전 실패를 review/CLI 파일 pin, payload digest, schema, source, git 형식별로 유발한다.
  git SHA가 null이고 확인 전 사유·요청 pin·실행 source가 남는지, 검증 후 실패에는 확인된
  SHA·사유·evidence pin이 남는지 확인한다. 39/41자리 git SHA도 거부한다.
- parity 집계는 실제 `inject_blind` 호출 결과를 따로 세어 accepted/rejected와 대조한다.
  두 수의 합, 54 paired attempts, 3 sample windows를 확인한다. 성공 provenance/source/binding도
  키 존재 여부에 더해 기대값과 비교한다.
- 기존 창별 읽기 시험에 **한 창만 세 번 읽는** 경우를 추가했다. dtype 비교는 shape와 bytes가
  같아도 signed/unsigned, integer/float, endian이 다르면 거부하는 사례를 추가했다.
- 실패 파일 목록의 목록형·파일 전용 구성, 합성 metadata와 최종 tensor의 witness 기여,
  출력 parent/out symlink 해석을 확인한다. 테스트 helper가 예상 밖 BaseException을 이름 있는
  assertion failure로 변환하므로 두 번째 Ctrl-C가 unittest 전체를 중단시키지 않는다.

최종 테스트 SHA는 `12cfc7855411ba5d8e092bf44581da47f9bb1394836396f1517c65ec09095af8`다.
Jetson **54/54 PASS, skip 0, 48.696초**, 서버 **54/54 PASS, skip/failure/error 0,
30.997초**를 확인했다. 서버 frozen metadata preflight도 2,613창 PASS(5.001초)다.
이번 지적과 직접 연결한 변이 **30/30개를 지정 테스트에서 검출**했다. 모든 변이는 unittest의
완전한 요약을 남겼으며, listing/writer의 두 번째 interrupt 변이는 `failures`로 검출됐다.
기존 67개나 새 광범위 변이 sweep은 반복하지 않았다.

새 패킷은 `exp004_raw_cache_cli_r30_20260930T024926Z`다. 로컬 parent는
`/home/keti/agent-collaboration/factory-safety/`, 서버 parent는 `/home/keti/review_runs/`이며,
서버 코드에는 `_code`, 결과에는 `_server_result` 접미사를 붙였다. 로컬 변이 결과는
같은 패킷 이름에 `_targeted_mutations/report.json`을 붙인 경로다.

| 증거 | SHA-256 |
|---|---|
| 서버 CLI fixture 보고서 파일 | `195a44d4e61d0f7e3a59d85aca3e2be1592518ecb21da25d527c24254d1fddce` |
| 위 보고서 payload | `795b4bb0bbd37ad5dae5007450f893a979c493cd6a39c987b48e3bb17ac22619` |
| frozen metadata preflight payload | `236bc8a5379ca952b2c029a321fc006ab918a723b28d5a015586da5c91842b68` |
| 지적별 변이 보고서 payload | `1371ce5b84672c065b949bc54870c3a938515d8793cd06f86f0764c3857ea2b1` |

실행 전후 source와 현재 checkout의 11개 파일 SHA, 보고서 자체 digest를 대조했다. 실제 원시
파일 읽기·실제 캐시 생성·optimizer step은 모두 0이다. Claude에는 이 고정 테스트 후보와 이번
보강 항목의 좁은 확인을 전달한다. 이전 51개 보고서는 이력으로 보존하며 새 코드 검증 근거는
위 54개 보고서다. Codex는 commit/push를 하지 않았다.
