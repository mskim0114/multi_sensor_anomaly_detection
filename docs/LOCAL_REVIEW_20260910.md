2026-09-10, Jetson `keti-kms`에서 기존 검토를 반증 중심으로 재검수하고 경로·캐시·샘플링·실행 안내의 국소 수정을 적용했다. 기준 HEAD는 `d84f661165ad1d387632d7b52c896985920ae0df`, 브랜치는 `feature/jetson-sensor-integration`이다. 이번 수정은 아직 미커밋 상태다.

Spark가 경로와 문서를 조사·수정했고 Sol이 기존 판단과 새 코드의 회귀를 독립 검토했다. 메인은 제안을 코드와 대조하고 수정·검증했다. 발견과 정정도 함께 남긴다.

- **재현된 결함:** 합성 fixture의 JSON 라벨을 0에서 3으로 바꾼 뒤에도 기존 캐시는 0을 반환했다. 캐시 입력 검증이 없다는 실제 동작을 확인한 뒤 수정했다.
- **이전 판단 정정:** 현재 보관된 인덱스가 0바이트 라벨에서 만들어졌다는 사실은 입증되지 않는다. 현재 builder는 JSON 파싱 실패 시 중단한다. 기존 캐시의 생성 당시 라벨 오염을 단정한 handoff 문장을 정정했다.
- **Spark 제안 정정:** `src.paths`를 import한 뒤 import 경로를 등록하는 순서는 직접 실행을 보장하지 못한다. 스크립트 자신의 위치에서 checkout 경로를 먼저 구하도록 수정했다.
- **수정 중 발견한 회귀:** 새로운 `src.paths` 의존성 때문에 파일 경로로 로드되는 offline builder가 정본 채널 목록을 읽지 못하고 fallback으로 진행했다. Sol이 이 문제를 찾아 메인이 builder의 import 경로를 수정했다. 실제 정본을 읽는지 확인하는 격리 subprocess 검증을 추가했다.
- **seed 전달 누락 보완:** V2+의 `--seed`가 DataConfig까지 전달되지 않아 새 전용 sampler가 기본 42에 남는 문제도 Sol 검수로 확인했다. CLI seed를 DataConfig에 전달하도록 보완하고 서버에서 실행할 연결 검증을 추가했다.

| 수정 | 동작 |
|---|---|
| [src/paths.py](../src/paths.py)와 학습·평가·export 진입점 | 존재하지 않는 옛 checkout 절대경로를 제거. 기본 자산·출력 경로를 현재 checkout에서 계산하고 직접 실행 시 import 경로를 먼저 등록 |
| [DataConfig](../src/data/config.py), [YAML](../configs/data_config.yaml) | 상대 데이터·캐시 경로는 checkout 기준. 외부 저장소의 절대경로는 유지 |
| [session_index.py](../src/data/session_index.py) | 캐시 버전, source/label 경로, split, gap, CSV basename 집합, JSON size/mtime/ctime을 확인한 경우만 재사용. 구형·손상·변경된 캐시는 다시 생성하고 임시 파일을 원자적으로 교체 |
| [sampler.py](../src/data/sampler.py), [datamodule.py](../src/data/datamodule.py) | WeightedRandomSampler에 전용 seed generator를 전달해 모델 초기화가 소비하는 전역 RNG와 분리. epoch 간 generator 상태는 계속 진행 |
| [window_builder.py](../src/data/window_builder.py) | 파일 경로로 독립 로드해도 현재 config의 채널 정의를 읽도록 import 경로 보완. 채널·품질·윈도 스키마 변경 없음 |
| [환경 점검 호환 진입점](../jetson_deploy/scripts/01_check_environment.py) | 과거 명령을 현재 run_python.sh/check_environment.py로 위임하고 실제 종료 상태를 전달 |
| README 및 codex_context 안내 | 현재 환경별 실행 경로로 정정. 과거 스냅샷은 역사 기록임을 명시하고 TensorRT EP 노출과 실제 추론 성공을 구분 |

캐시 signature는 일반적인 파일 변경을 탐지하기 위한 메타데이터다. CSV/BIN 내용은 세션 경계에 사용되지 않으므로 읽지 않고, JSON도 cache hit에서 내용 전체를 해시하지 않는다. **따라서 이 캐시는 데이터 무결성을 증명하지 않는다.** 원본 checksum과 파싱 검증은 기존 데이터 복구 절차에서 별도로 통과해야 한다. pickle 형식도 유지하므로 신뢰된 로컬 cache만 사용한다. 실제 보관된 캐시를 이번에 재생성하거나 읽지는 않았다.

cache hit에서도 입력 파일 수에 비례하는 메타데이터 조회 비용이 발생하므로 전체 데이터 규모의
성능은 서버에서 확인해야 한다. 입력은 복구·무결성 확인 후 학습 중에는 고정된 상태를 전제로 한다.
Sol의 최종 diff 재검수에서는 앞서 발견한 두 회귀가 해소됐고, 검토 범위의 추가 차단 결함은 발견되지 않았다.

실행한 검증은 다음과 같다.

| 검증 | 결과·범위 |
|---|---|
| [국소 회귀 검증](../tests/test_data_portability.py) | **28개 중 24개 PASS, 4개 SKIP**. cache hit/라벨 변경/경로·gap·split 변경/구형·손상 cache/입력 누락·동시 변경/원자적 교체 실패/경로 이식성/정본 채널/환경 점검 위임 확인 |
| 스크립트 경로 이식성 | 임시로 옮긴 checkout과 다른 cwd에서 실제 bootstrap 부분을 격리 실행. 학습 라이브러리와 모델 실행은 포함하지 않음 |
| 원본 보존 | baseline manifest의 **85개 파일 SHA-256, 실행 전후 모두 일치** |
| 기존 전처리 재현 | 5 trial·1800 snapshot → 60 window·유효 49·무효 11. 수집 시 품질 기록과 재계산 결과 일치 |
| 이전 산출물과 비교 | `windows.jsonl`, `baseline_stats.json` JSON 내용 동일. **60개 NPZ의 모든 배열 동일**. 이전 processed 파일은 덮어쓰지 않음 |
| 정본 채널 provenance | `legacy_channel_list_source = src/data/config.py:SENSOR_CHANNELS`, fallback 미사용 |

실행 환경은 기존 `factory_runtime`, `PYTHONNOUSERSITE=1`, bytecode 쓰기 비활성화다.
PyTorch가 설치되어 있지 않아 sampler의 실제 RNG 동작 3개 및 CLI seed 전달 1개 테스트는 SKIP했다.
이 4개와 전체 학습·export 실행은 실제 SERVER-TRAINING 환경에서 검증해야 한다.
정상 데이터 전처리 재현은 Jetson 국소 회귀검증이며 **서버 재현 G1-S 통과 또는 현장 모델 성능 검증이 아니다.**

국소 검증 실행:

```bash
./jetson_deploy/run_python.sh -m unittest discover -s tests -v
```

이번 작업 전 미커밋 파일 20개의 사본과 원래 diff를
`/home/keti/review_runs/20260910_portability/`에 보관했다. 같은 디렉터리에
`tests.txt`, `baseline-reproduction.txt`, `baseline-comparison.json` 및 별도 재현 산출물이 있다.
기존 모델 수정과 논문 수정은 보존했다. 데이터·센서·의존성 설치·모델 학습·export·네트워크 전송은 실행하지 않았다.

다음 단계는 실제 서버의 주소·사용자명·포트를 받아 환경을 먼저 확인하고, 기존 이관 절차로
원본을 전달·재현한 뒤 누락된 PyTorch 검증 및 데이터 복구를 수행하는 것이다.
현재 Jetson의 SSH 호스트 설정에는 서버 접속 대상이 없어 원격 단계를 진행하지 않았다.
