# RESEARCH STATUS

최종 갱신: 2026-09-18 · 기준 HEAD `d84f661` + **미커밋 63개** · 브랜치 `feature/jetson-sensor-integration`

> **이 문서는 현재 단계·최우선 작업·blocker만 담는다.** 배포/환경/데이터 획득의 canonical
> 현황은 [SERVER_WORKSTATION_HANDOFF.md](SERVER_WORKSTATION_HANDOFF.md)에 있고 여기서
> 복제하지 않는다. 상세 검증 근거는
> [연구노트 #16](연구노트/연구노트_16_외부검토_실측검증.md)이 단일 원본이다.

---

## 1. 현재 gate

업무 지침 [§4](research_agent_operating_brief_20260907.md) 기준.

9월 10일 경로·인덱스 캐시·샘플러 RNG 국소 수정([LOCAL_REVIEW_20260910.md](LOCAL_REVIEW_20260910.md)),
9월 17일 수집기 재작성·`collect.sh`([SENSOR_COLLECTION_REVIEW_20260917.md](SENSOR_COLLECTION_REVIEW_20260917.md)),
로컬 대시보드([JETSON_SENSOR_DASHBOARD.md](JETSON_SENSOR_DASHBOARD.md)), 서버 preflight·스토리지 이관·
센서 데이터 서버 보관([연구노트 #17](연구노트/연구노트_17_서버_스토리지_이관_및_실측.md))이 적용됐다.
**전부 미커밋이며 Jetson 로컬에만 있다.** 하드웨어 독립 테스트 69개 통과(skip 4, torch).

| gate | 상태 | 근거 |
|---|---|---|
| G0 현황 확인 | **완료** | [연구노트 #16](연구노트/연구노트_16_외부검토_실측검증.md) §1, §2, §6 |
| **G1-J** Jetson 원시 재현 | **PASS** (`8248e4a`) | handoff §4·§8, [manifest](manifests/development_baseline_20260904.sha256), `processed/development_baseline_v1/manifest.json` |
| G2a 논문 정정안 | **완료** | [claim_evidence.csv](논문/claim_evidence.csv), [PC0](논문/candidates/PC0.md) |
| G2b 결함 수정안 + 국소 검증 | **완료** | 연구노트 #16 §10 (LV-1 6/6, LV-2 통과) |
| **G2c** handoff 보완 + AI Hub manifest | **완료** | handoff §13, [manifests/aihub_zips.sha256](manifests/aihub_zips.sha256) |
| **G2h** 논문 확정 오류 정정 | **완료** | `paper_draft.md` 27+/22−, [claim_evidence.csv](논문/claim_evidence.csv) `applied_status` |
| **G2i** 국소 코드 결함 수정 | **완료** | F04·F03·F14a 적용 + 정적 검증. 연구노트 #16 §10 |
| **G1-S** 서버 원시 재현 | **pending** | preflight 완료(B-3 부분 해소). 남은 것: raw 85파일 전송 + 서버 clone 브랜치 갱신. handoff §7 |
| **G2d** AI Hub 데이터 검증 | **재정의** | **서버 사본은 온전**(0바이트 0). 재추출 불필요, L2~L4 검증만 남음. handoff §13-6 |
| G2e~G2g 수정 적용·재실행 | blocked | G1-S + G2d 선행 |
| G3 현장 문제 정의 | 미착수 | 로봇 소유기관 확인 필요 |
| G4~G7 | 미착수 | — |

**경계.** G1-J PASS가 서버 금지를 해제하지 않는다. G1-S 전에는 handoff §10의 training /
schema 변경 / export / CT imputation / threshold 생성 / raw mutation 금지가 유지된다.
G1-S PASS도 일괄 승인이 아니며 현장 모델 입력 결정은 handoff §11 별도 검토를 거친다.

---

## 2. 최우선 3개 작업

| # | 작업 | 통과 조건 | 막는 것 |
|---|---|---|---|
| 1 | **미커밋 63개 정리** — 커밋 단위 분리 후 feature 브랜치 push | 10일치 작업이 Jetson 디스크 하나에서 벗어남. 묶음 제안은 [연구노트 #17](연구노트/연구노트_17_서버_스토리지_이관_및_실측.md) 뒤 보고 참조 | 사용자 커밋 승인 (AGENTS §6), Codex TUI 종료 확인 |
| 2 | **G1-S** development baseline raw 85파일 서버 전송 → canonical 재현 | 85 파일 · 1800 snapshot · 60/49/11 · 구조오류 0 · 전후 raw 동일. 목적지는 `/mnt/data-hdd/keti_data/` 아래 | 서버 clone 을 작업 브랜치로 갱신 (현재 `5feabb5`) |
| 3 | **G2d** 서버 AI Hub 사본 L2→L4 검증 (재추출 없음) | handoff §13-7 STEP 10~12. **L2 전수 통과 없이 L4 숫자 일치만으로 성공 선언 금지** | 학습 환경 결정 (O-108) |

---

## 3. Blocker

| ID | 내용 | 영향 | 해소 조건 |
|---|---|---|---|
| **B-1** | **범위 축소:** Jetson 사본의 `extracted/`만 0바이트. **서버 사본은 온전**(Training/Validation 6종 모두 0바이트 0, 18 GB) | Jetson 에서의 재학습·재현만 불가. 서버는 L2~L4 검증 후 가능 | 서버 사본 L2~L4 검증 (handoff §13-7). Jetson 사본은 폐기/교체 대상 |
| **B-2** | 옛 절대경로는 9/10 checkout 기준으로 수정. 경로·캐시 국소 검증 통과 | 서버의 데이터 위치·권한·학습 진입점 통합 검증 대기 | 실제 서버 audit 및 실행 검증 (handoff §9) |
| **B-3** | **부분 해소:** read-only preflight 완료(`SERVER_ENVIRONMENT.md` = PARTIAL). 남은 것: 패키지 목록·`requirements-server.txt`·학습 venv 결정 | G1-S 는 착수 가능. 학습(G2e~)은 venv 결정 전 불가 | O-108 결정 + 패키지 audit |
| **B-4** | 이상 사건 0건 + 현장 모델(ModelAdapter) 미준비 | 사건 탐지율·선행시간·현장 오경보율 **계산 불가** | 이상 trial 확보(로봇 소유기관) + B0 이후 |
| **B-5** | AI Hub CT1~4 물리 정의가 공급자 배포 패키지에 없음 | 현장 CT 설계·legacy 재사용 판단 근거 부족 | 공급자 확인 |
| **B-6** | 공식 F1 averaging 방식·보고 split 확정 불가 (Jetson 의 `train_manager.py` 0바이트, docker 잘림). **서버에 docker 61 GB 완전본 존재** — 내부 미검증 | 공급자 보고치와의 직접 비교 불가 | 서버 docker 이미지에서 `/app/trainers/` 복구 시도, 또는 공급자 확인 |
| **B-7** | 센서 데이터 서버 import 를 수행한 **전송 스크립트가 저장소·서버 어디에도 없음.** manifest 스키마만 남음 | 다음 import 를 재현할 수 없음 | 도구를 저장소에 복원·작성 |
| **B-8** | **미커밋 63개**가 Jetson 로컬 디스크 한 곳에만 존재. 원격 두 저장소 모두 `d84f661`/`5feabb5` 에서 정지 | 10일치 작업 유실 위험 | 커밋 단위 분리 + push (사용자 승인) |
| **B-9** | 서버 clone 이 `5feabb5`(private main)에 있어 09-07~17 코드 변경 미포함. `public-upstream/feature/...` 는 fetch 만 됨 | G1-S 재현 시 서버 코드가 Jetson 과 다름 | 서버 clone 을 작업 브랜치로 checkout/merge |

---

## 4. 확인된 사실 (요약 — 상세는 연구노트 #16)

- **N1** 공급자 공식 모델은 Multimodal LSTM+CNN이며 Transformer가 아니다. 우리 `results/baseline`
  (2,833,412 params, F1 0.923537)이 같은 아키텍처다 → 연구노트 #16 §3.1
- **N2** AI Hub 학습 데이터·공식 소스 `.py`가 0바이트 → §6.1 (= B-1)
- **N3** legacy 정규화 구간과 현장 실측이 거의 겹치지 않는다 → §5.1
- **N4** ablation 20 epoch vs full model 30 epoch 예산 불일치 + LR schedule 동일성 미검증. 결합 이득을 요소 효과로도 예산으로도 분해할 수 없다 → §3.3
- **N5** AI Hub CT1~4 물리 정의 부재 → §3.6 (= B-5)
- **N6** 기존 연구 자산(결과 JSON 16개·checkpoint·ONNX/TRT·reference)은 이 장비에 전부 있다 → §6.2
- **F08 REFUTED** AI Hub 시간 정합은 검사 범위에서 확인됨(전 인접쌍 Δ=1 s, 30표본 span 29 s) → §1.3
- **AI Hub 압축 자산** 684 zip · 6.05 GB · 비압축 18.78 GB · member 478,695.
  존재 684/684 · 구조 무결성 684/684 · CRC 684/684, 실패 0 → handoff §13-2
- **Jetson 의 `extracted/` 는 파일 누락 0건이고 내용이 0바이트다** — Training csv/bin/json 각 99,476 전부
  0바이트, Validation json 만 8,147개 정상 → handoff §13-1
- **N7 서버 사본은 온전하다** — 6종 모두 0바이트 0, 18 GB, mtime 2025-03-10. docker 61 GB 완전본 → handoff §13-6, 연구노트 #17 §2
- **N8 서버 스토리지 이관 완료** — `data/aihub` 84 GB + `MONAI/physical-ai-research` 478 GB 를 HDD 로. 전수 체크섬 exit 0. 루트 97 % → 32 % → 연구노트 #17 §3
- **N9 센서 데이터 서버 보관 1회** — 19 run · 270 파일 · 106 MB, `copy_integrity_verified`, 자동 업로드 미설정 → 연구노트 #17 §4
- **N10 저장소가 둘** — public `multi_sensor_anomaly_detection`(Jetson 작업) + private `factory_safety`(서버, 09-10 신설, IEEE-ICA 논문·특허 자료) → 연구노트 #17 §1

**지표 표기.** `49/60 = 81.7 %`는 **v1 윈도 유효율**이다. 판단 가용률이나 전 센서 정상 관측
시간으로 환산하지 않는다. 판단 가용률은 현장 모델 부재로 계산 불가(B-4).

---

## 5. 관련 문서

| 문서 | 역할 |
|---|---|
| [research_agent_operating_brief_20260907.md](research_agent_operating_brief_20260907.md) | 업무 지침 (gate·가설·기록 체계) |
| [industrial_sensor_research_review_20260907.md](industrial_sensor_research_review_20260907.md) | 외부 검토 원문 (F01~F14) |
| [연구노트 #16](연구노트/연구노트_16_외부검토_실측검증.md) | 실측 검증 상세 근거 — **단일 원본** |
| [decisions.md](연구노트/decisions.md) | 결정 대장 |
| [experiment_index.csv](연구노트/experiment_index.csv) | 실행·계획 인덱스 |
| [claim_evidence.csv](논문/claim_evidence.csv) | 논문 문장 ↔ 근거 대응 |
| [candidates/](논문/candidates/) | 논문 후보 PC0~PC4 |
| [SERVER_WORKSTATION_HANDOFF.md](SERVER_WORKSTATION_HANDOFF.md) | 서버 이관 canonical |
| [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md) | 서버 실측 (PARTIAL) |
| [연구노트 #17](연구노트/연구노트_17_서버_스토리지_이관_및_실측.md) | 서버 실측·스토리지 이관·센서 데이터 보관 근거 |
| [LOCAL_REVIEW_20260910.md](LOCAL_REVIEW_20260910.md) | Codex 09-10 경로·캐시·RNG 수정 검증 |
| [SENSOR_COLLECTION_REVIEW_20260917.md](SENSOR_COLLECTION_REVIEW_20260917.md) | Codex 09-17 수집기 재작성·`collect.sh` 검증 |
| [JETSON_SENSOR_DASHBOARD.md](JETSON_SENSOR_DASHBOARD.md) | Codex 09-17 로컬 대시보드 |
| [SENSOR_DASHBOARD_REVIEW_20260917.md](SENSOR_DASHBOARD_REVIEW_20260917.md) | Codex 09-17 대시보드 구현·검증 기록 |
