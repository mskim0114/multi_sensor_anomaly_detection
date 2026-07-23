# Jetson Codex CLI 사용 가이드

이 폴더(`codex_context/`)는 Jetson 위 Codex CLI에게 **이 프로젝트의 모든 맥락**을 전달하기 위한 패키지입니다.
PC 쪽 Claude가 지금까지 만든 모든 자산(코드·결과·논문·연구노트)이 한 곳에 정리되어 있습니다.

---

## 1. 설치 위치 권장

Jetson 홈 디렉토리 안에 영구 보관하는 걸 추천합니다.

```bash
# USB에서 Jetson 홈으로
mkdir -p ~/factory_safety
cp -r /media/<usb>/jetson_deploy  ~/factory_safety/
cd ~/factory_safety/jetson_deploy
ls
# codex_context/  model/  reference/  scripts/  results/  README.md
```

---

## 2. Codex CLI를 어디서 실행할지

**가장 중요한 규칙**: Codex CLI는 실행한 디렉토리(와 상위 디렉토리)에 있는 `AGENTS.md`를 **자동으로 system prompt에 포함**합니다.

따라서 작업할 때는 **항상 `codex_context/` 안에서 `codex` 명령을 실행**하세요.

```bash
cd ~/factory_safety/jetson_deploy/codex_context
codex
```

이렇게 하면 매번 새 세션을 시작해도 Codex가:
- 프로젝트 한 줄 설명
- V2+ 모델 구조와 선택 이유
- 데이터셋 구조
- 센서 하드웨어 계획
- 디렉토리 인덱스
- 작업 규칙

을 모두 알고 시작합니다.

---

## 3. 자주 쓰는 사용 패턴

### 패턴 A: 검증 결과를 함께 분석
```bash
cd ~/factory_safety/jetson_deploy/codex_context
codex
```
세션 안에서:
```
> 한 단계 위에 있는 scripts/02_benchmark_latency.py 를 돌렸을 때
  나온 results/jetson_latency.json 을 읽고, PC 측 기대치
  (CUDA EP < 5ms)에 비춰서 어떤지 평가해줘.
```

### 패턴 B: 코드 수정 / 추가 실험
```
> code/models/v2_plus.py 를 읽고, SE block 의 reduction ratio 를
  실험적으로 바꿔보는 게 의미 있을지 의견 줘. 근거는
  docs/연구노트/08_모델개선_리서치.md 와 09_V2Plus_결과.md 도 참고해서.
```

### 패턴 C: 센서가 도착한 뒤 결선 도움
```
> NTC 10kΩ 을 Jetson 의 ADS1115 ADC 에 붙이려고 해.
  docs/참고문서/실제 센서 및 엣지 보드 구매 사양서.md
  를 참고해서 결선도랑 캘리브레이션 절차를 알려줘.
```

### 패턴 D: 새 실험 결과 → STATE 업데이트
```
> 방금 TensorRT FP16 으로 변환했더니 mean latency 1.8ms 나왔고
  pred match rate 99.7% 였어. 이 결과를 STATE.md 와
  docs/연구노트/ 에 추가해줘.
```

---

## 4. 디렉토리 외부 작업

`codex_context/` 안에서 실행해도 Codex는 상위/하위 디렉토리의 파일에 자유롭게 접근할 수 있습니다.

상대 경로 예시:
- `../scripts/02_benchmark_latency.py` — 검증 스크립트
- `../model/model_v2plus.onnx` — ONNX 모델
- `../reference/val_reference_small.npz` — 참조 데이터
- `../results/jetson_summary.json` — 실행 결과

---

## 5. PC ↔ Jetson 간 동기화 규칙

이 패키지는 **PC가 만든 스냅샷**입니다. Jetson에서 작업해 새 결과가 나오면:

1. 항상 `STATE.md` 의 §A4 (또는 새 섹션)에 한 줄 추가
2. 에러를 만났으면 `docs/오류_및_해결_로그.md` 에 같은 형식으로 추가
3. 큰 변경 (모델 재학습, 새 코드)이 있으면 폴더 째 USB 로 PC 측에 다시 옮김

PC 쪽 Claude는 다음 USB 동기화 때 변경사항을 보고 자기 컨텍스트를 갱신합니다.

---

## 6. Codex 모델 선택

작업 성격에 따라:

| 작업 | 권장 모델 | 이유 |
|------|----------|------|
| 코드 수정 / 디버그 | GPT-5-codex (default) | 코드 능력 최강 |
| 긴 문서 분석 / 논문 리뷰 | GPT-5 medium~high | 사고력 |
| 빠른 질문 | GPT-5-codex low | 응답 속도 |

Codex CLI에서 `/model` 슬래시 명령으로 변경 가능.

---

## 7. 권한 모드

Jetson 에서 첫 작업은 **read-only**로 시작해서 익숙해진 뒤 권한을 올리는 걸 권장합니다.

```bash
codex                          # 기본 — 매 명령 승인 필요
codex --sandbox workspace-write  # 현재 폴더 수정 자동 허용
codex --full-auto              # 모든 액션 자동 (주의)
```

---

## 8. 트러블슈팅

| 증상 | 해결 |
|------|------|
| `AGENTS.md` 가 안 읽히는 듯 | 실행 디렉토리가 `codex_context/` 인지 확인 |
| 컨텍스트가 너무 길다는 에러 | `codex` 안에서 `/compact` 로 압축 |
| 한글 깨짐 | 터미널 `export LANG=ko_KR.UTF-8` |
| 모델 응답이 모호 | INDEX.md 의 해당 문서를 직접 지목해서 질문 |

---

## 9. 1분 시작 가이드

```bash
# 1. 환경 변수 (Codex 인증은 이미 되어 있다고 가정)
cd ~/factory_safety/jetson_deploy/codex_context

# 2. 진입
codex

# 3. 첫 prompt
> STATE.md 의 §B 첫 작업부터 시작하자. 어떻게 진행할까?
```

이렇게 시작하면 Codex가:
1. STATE.md 읽음
2. AGENTS.md 자동 로드된 컨텍스트 활용
3. § B의 ⓵ ONNX 검증 절차를 안내 + 실행 도움

---

마지막으로, **이 패키지를 만든 PC 측 Claude도 동일한 컨텍스트를 갖고 있으니**, Jetson 에서 한 작업을 다시 PC 로 가져오면 두 환경이 협업하듯 일관되게 굴러갈 겁니다.
