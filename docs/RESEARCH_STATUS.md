# RESEARCH STATUS

최종 갱신: 2026-09-18 (야간) · 기준 HEAD `4fac5bc` + 이번 커밋 · 브랜치 `feature/jetson-sensor-integration`

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
09-18 커밋 6개(`33b56d1`~`572cff8`)로 push 됐고, 서버에도 같은 revision 이 반영됐다(연구노트 #17 §6).
하드웨어 독립 테스트 69개 통과(skip 4, torch).

| gate | 상태 | 근거 |
|---|---|---|
| G0 현황 확인 | **완료** | [연구노트 #16](연구노트/연구노트_16_외부검토_실측검증.md) §1, §2, §6 |
| **G1-J** Jetson 원시 재현 | **PASS** (`8248e4a`) | handoff §4·§8, [manifest](manifests/development_baseline_20260904.sha256), `processed/development_baseline_v1/manifest.json` |
| G2a 논문 정정안 | **완료** | [claim_evidence.csv](논문/claim_evidence.csv), [PC0](논문/candidates/PC0.md) |
| G2b 결함 수정안 + 국소 검증 | **완료** | 연구노트 #16 §10 (LV-1 6/6, LV-2 통과) |
| **G2c** handoff 보완 + AI Hub manifest | **완료** | handoff §13, [manifests/aihub_zips.sha256](manifests/aihub_zips.sha256) |
| **G2h** 논문 확정 오류 정정 | **완료** | `paper_draft.md` 27+/22−, [claim_evidence.csv](논문/claim_evidence.csv) `applied_status` |
| **G2i** 국소 코드 결함 수정 | **완료** | F04·F03·F14a 적용 + 정적 검증. 연구노트 #16 §10 |
| **G1-S** 서버 원시 재현 | **PASS** (2026-09-18, 서버 clone `572cff8`) | manifest 85 OK · 60/49/11 · tick-quality 일치 · raw 전후 불변 · **NPZ 2,280 배열 Jetson 과 동일, `baseline_stats.json` 바이트 동일**. 연구노트 #17 §6 |
| **G2d** AI Hub 데이터 검증 | **PASS** (2026-09-18, SSD 작업 사본) | L2 전수 파싱(CSV·BIN·JSON 각 111,870) 실패 0 · L3 대응·규약·timestamp 111,870 일치 · L4 session 303/39, window 9,313/1,157, 장비 32/4 교집합 0, Δ=1 s. `session_index.py` 강제 재생성으로 확인. 연구노트 #17 §7 |
| **G2e** V2+ 3-seed 재실행 (EXP-20260907-002) | **완료** (2026-09-18, 첫 서버 GPU 학습) | F1 0.954825/0.952084/0.956400 (mean 0.954436, std 0.001783) vs legacy mean 0.955659. provenance·초기 가중치 hash 기록. [연구노트 #18](연구노트/연구노트_18_G2e_V2plus_재실행.md) |
| **G2f** 2³ ablation (EXP-20260907-003) | **완료** (2026-09-18, 24 run, `4fac5bc` clean) | multiscale +0.0116 ± 0.0008, SupCon +0.0042 ± 0.0006, SE +0.0001 ± 0.0011. interaction 은 전체 이득의 23 %. [연구노트 #19](연구노트/연구노트_19_EXP003_2x3_ablation_및_재현성.md) |
| G2g 장비 K-fold (EXP-20260907-006) | 미착수 | — |
| bench 데이터 (현장 아님) | **시작** (2026-09-18) | soak #2 진행 중(05:08 KST 종료 예정), xcal 3회 관측(보정 불가 조건). [프로토콜 §15](JETSON_DATASET_PROTOCOL.md), 연구노트 #19 §7 |
| G3 현장 문제 정의 | 미착수 | 로봇 소유기관 확인 필요 |
| G4~G7 | 미착수 | — |

**경계.** G1-J PASS가 서버 금지를 해제하지 않는다. G1-S 전에는 handoff §10의 training /
schema 변경 / export / CT imputation / threshold 생성 / raw mutation 금지가 유지된다.
G1-S PASS도 일괄 승인이 아니며 현장 모델 입력 결정은 handoff §11 별도 검토를 거친다.

---

## 2. 최우선 3개 작업

| # | 작업 | 통과 조건 | 막는 것 |
|---|---|---|---|
| 1 | **O-111 논문 Table 8·§7.3·H1/H2 교체 여부 결정** — 교체 문안은 연구노트 #19 §6 | 사용자 판단 후 적용 | 사용자 결정 |
| 2 | **cuDNN 결정론 모드 seed 42 재실행 1회** (~20분) + **SE 추가 기여 확인 seed 3개 추가**(6 run, ~1시간) | 전자: epoch 별 val F1 30/30 일치 여부. 후자: ms1_se1_sc1 − ms1_se0_sc1 이 6 seed 에서 부호 유지 | 사용자 학습 승인 |
| 3 | **B-7** 센서 데이터 서버 import 도구 복원 · bench_thermal_xcal 유효 조건(NTC 위치 표시 + 온도 변화 구간) 재실행 | import 재현 가능 / xcal 기울기 추정 가능 | 없음 / 사용자 물리 조작 |

완료(09-18 야간): 동일 seed 재실행 → 초기 가중치 sha256 동일, F1 Δ −0.0008, 학습 경로는 epoch 1 부터 분기(비결정성). EXP-003 24 run 완료.

---

## 3. Blocker

| ID | 내용 | 영향 | 해소 조건 |
|---|---|---|---|
| **B-1** | **범위 축소:** Jetson 사본의 `extracted/`만 0바이트. **서버 사본은 온전**(Training/Validation 6종 모두 0바이트 0, 18 GB) | Jetson 에서의 재학습·재현만 불가. 서버는 L2~L4 검증 후 가능 | 서버 사본 L2~L4 검증 (handoff §13-7). Jetson 사본은 폐기/교체 대상 |
| **B-2** | 옛 절대경로는 9/10 checkout 기준으로 수정. 경로·캐시 국소 검증 통과 | 서버의 데이터 위치·권한·학습 진입점 통합 검증 대기 | 실제 서버 audit 및 실행 검증 (handoff §9) |
| ~~B-3~~ | **해소(09-18):** 전용 venv `factory_training` 생성·검증(pin 11/11, CUDA 2장), `requirements-server.txt` 확정, `SERVER_ENVIRONMENT.md` READY | — | — |
| **B-4** | 이상 사건 0건 + 현장 모델(ModelAdapter) 미준비 | 사건 탐지율·선행시간·현장 오경보율 **계산 불가** | 이상 trial 확보(로봇 소유기관) + B0 이후 |
| **B-5** | AI Hub CT1~4 물리 정의가 공급자 배포 패키지에 없음 | 현장 CT 설계·legacy 재사용 판단 근거 부족 | 공급자 확인 |
| **B-6** | 공식 F1 averaging 방식·보고 split 확정 불가 (Jetson 의 `train_manager.py` 0바이트, docker 잘림). **서버에 docker 61 GB 완전본 존재** — 내부 미검증 | 공급자 보고치와의 직접 비교 불가 | 서버 docker 이미지에서 `/app/trainers/` 복구 시도, 또는 공급자 확인 |
| **B-7** | 센서 데이터 서버 import 를 수행한 **전송 스크립트가 저장소·서버 어디에도 없음.** manifest 스키마만 남음 | 다음 import 를 재현할 수 없음 | 도구를 저장소에 복원·작성 |
| ~~B-8~~ | **해소(09-18):** 6개 커밋으로 `572cff8` push. Jetson·GitHub·서버 동일 revision | — | — |
| ~~B-9~~ | **해소(09-18):** 서버에 `/home/keti/projects/factory_safety` 새 clone(`572cff8`). 기존 `/home/keti/factory_safety`(private main, 특허 dirty 51)는 무변경 | — | 두 clone 의 장기 정책은 O-109 |

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
- **N11 HDD 는 학습 읽기용으로 부적합** — 무작위 소파일(bin 153 KB + csv 82 B) 읽기 HDD 51 샘플/s vs SSD 13,287 샘플/s. 1 epoch 이 558,780 파일 읽기. `extracted/` 18 GB 작업 사본을 SSD 에 두고 HDD 는 불변 아카이브 → 연구노트 #17 §6
- **N12 서버 G1-S PASS** — 회계 60/49/11 일치, NPZ 2,280 배열 동일, `baseline_stats.json` 바이트 동일. NPZ 파일 해시는 zip 타임스탬프로 달라지므로 **파일 해시가 아닌 배열 단위**가 재현 기준 → 연구노트 #17 §6
- **N13 서버 기존 clone 의 `results/v2plus` 는 논문 run 이 아니다** — F1 0.9551277 best@21 (08-06 Qwen 이후 재실행). 논문 run(0.9550 best@27)은 Jetson 사본이며 새 clone `results/` 로 전송 → 연구노트 #17 §6
- **N15 열화상 BIN 값 범위가 논문·정규화 범위를 벗어난다** — 전수 검사에서 Training min −110.29 °C / max 172.06 °C,
  Validation 14.04 / 138.96 °C. 논문 `:133` "약 31~146 °C, clipping 불필요" 및 `ThermalStats(30.98, 146.10)` 과 불일치.
  범위 밖은 Training 프레임 212개(픽셀 0.029 %), Validation 31개(픽셀 0.15 %). 음수 프레임 11개는 물리적으로 불가능한 값.
  무결성 문제는 아니며 원인·학습 영향 미확인 → 연구노트 #17 §7.1, claim_evidence CE-047
- **N16 G2e 재실행 결과** — clean revision `be16342`, seed 42/123/456, 각 ~20분. F1 mean 0.954436 (std 0.001783) vs
  legacy 0.955659 (std 0.000467), Δ −0.0012. best epoch 24/29/26 (legacy 는 2개가 best@30). plateau LR 감소 시점이
  seed 마다 다름(8 또는 13). **논문 수치를 대체하지 않는다**(D-020) → 연구노트 #18
- **N17 seed 는 초기화만 재현한다** — seed 42 동일 조건 재실행: 초기 가중치 sha256 동일(`58f43109…`), 최종 F1 0.954040 vs
  0.954825, epoch 별 val F1 30 중 0 일치, LR 감소 경로 분기. cuDNN 비결정성 추정(미확인). 재현성 주장은 "초기 가중치 + 결과 분포" 로
  한정 → 연구노트 #19 §5
- **N18 Table 8 의 "단독 무효·결합 시너지" 는 통제 조건에서 성립하지 않는다** — 2³ × 3 seed(EXP-003, `4fac5bc` clean):
  multiscale 주효과 +0.0116 ± 0.0008 (V2→V2+ 이득의 61 % 단독), SupCon +0.0042 ± 0.0006, SE +0.0001 ± 0.0011.
  interaction 합 +0.0039 (23 %), 3-way +0.0011. SupCon 단독 NM −9 %, 결합 −39 % → H2 의 SupCon 귀속 NOT SUPPORTED.
  legacy 와 직접 비교는 classifier dropout 통일 때문에 불가 → 연구노트 #19 §2~4, D-021
- **N19 bench 데이터는 현장 데이터가 아니다** — 사무실 책상 수집은 학습·평가에 쓰지 않고 파이프라인 내구·센서 교차 관측·전이
  리허설로 한정. xcal 1차 3회는 온도 변화 1.2 °C 뿐이라 기울기 추정 불가, 최고온 ROI − NTC ≈ +9.5 °C 일정 → 프로토콜 §15,
  연구노트 #19 §7
- **N14 `monai_env` 는 수술 영상 프로젝트용 공유 env 다** — 2025-12 생성, 336 패키지(monai·pydicom·SimpleITK·transformers). 기존 논문 실험이 이 위에서 돌았으나 정책상 전용 venv 로 교체(D-019). pin 은 그대로 옮겨 비교 가능성 유지 → 연구노트 #17 §6.3

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
| [연구노트 #18](연구노트/연구노트_18_G2e_V2plus_재실행.md) | G2e V2+ 3-seed 재실행 결과 |
| [연구노트 #19](연구노트/연구노트_19_EXP003_2x3_ablation_및_재현성.md) | EXP-003 2³ ablation · seed 재현성 · bench 데이터 시작 |
| [LOCAL_REVIEW_20260910.md](LOCAL_REVIEW_20260910.md) | Codex 09-10 경로·캐시·RNG 수정 검증 |
| [SENSOR_COLLECTION_REVIEW_20260917.md](SENSOR_COLLECTION_REVIEW_20260917.md) | Codex 09-17 수집기 재작성·`collect.sh` 검증 |
| [JETSON_SENSOR_DASHBOARD.md](JETSON_SENSOR_DASHBOARD.md) | Codex 09-17 로컬 대시보드 |
| [SENSOR_DASHBOARD_REVIEW_20260917.md](SENSOR_DASHBOARD_REVIEW_20260917.md) | Codex 09-17 대시보드 구현·검증 기록 |
