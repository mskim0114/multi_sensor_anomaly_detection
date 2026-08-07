# 연구노트 #13: 3-Phase 로드맵 확정 · 논문 RQ/H 도입

**작성일:** 2026-07-28
**프로젝트:** 제조공장 멀티모달 센서 기반 이상상태 예측 AI

---

## 1. 배경

논문 draft (paper_draft.md, 570 lines) 완성 후 다음 두 질문에 답해야 했다.

1. **이 논문의 목표 venue 는 어디이며 언제 제출하는가?**
2. **논문의 서사가 falsifiable 한 명시적 hypothesis 구조인가?**

첫 번째는 프로젝트 로드맵 문제, 두 번째는 논문 구조 문제. 둘 다 이 시점에 결정.

---

## 2. 로드맵 확정 (사용자 지시 반영)

논문 초안이 MDPI Sensors 급인지 IJCAI 급인지 냉정하게 판정 후 3-phase 로드맵으로 정리.

### Venue 별 판정

| Venue | 채택 가능성 | 근거 |
|:---|:---:|:---|
| MDPI Sensors | 높음 (60-75%) | 응용 지향, 우리 성격 맞음 |
| IEEE Access | 높음 | 광범위 |
| IEEE Trans Industrial Informatics | 중 | Depth 조금 부족 |
| **IJCAI 2027 Main Track** | 매우 낮음 | Novelty 얕음 (combination paper), 단일 데이터셋, 개선폭 marginal (+1.20%), 이론 없음 |
| **IJCAI 2027 Applied Track** | 낮음 (지금은) | 8-12개월 확장 후 도전 가능 |

**결정**: 현재 draft 는 **MDPI Sensors** 로 제출 (Phase 1). IJCAI 는 확장판 (Phase 3).

### 3-Phase 로드맵 (2026-07-28 확정)

| Phase | 기간 | 산출물 |
|:-----:|:-----|:------|
| **1a** | 2026-08 ~ 2026-12 | ① 포스터 논문 1편 (경량, 2026 하반기) ② MDPI/IEEE Access 저널 원고 완성 |
| **1b** | 2026-12 ~ 2027-05 | 저널 제출 (MDPI Sensors 또는 IEEE Access) |
| **2** | 2027-01 ~ 2027-05 | 실센서 데이터 + 벤치마크 2개 (CMAPSS, FEMTO-ST) + 이론 요소 (embedding geometry, MI) |
| **3** | 2027-06 | IJCAI 2027 Applied Track 확장판 제출 |

**주의**: IJCAI 실제 CFP 데드라인 관례 1-2월 → 실 제출은 2028-01 가능성 (IJCAI 2028 로 자연 이동).

---

## 3. 논문 RQ/H 명시적 추가

원 draft (2026-04-09 완성) 는 Section 1 에서 "3 key observations + 3 contributions" 만 있고 명시적 RQ/H 가 없었다. Reviewer 가 "이 논문의 hypothesis 가 뭐냐" 물으면 우회적으로만 답할 수 있는 구조. 이를 정면 대응 형태로 재구성.

### 도입 위치

- **Section 1 (Introduction) 끝** — Contributions 뒤에 RQ1-3 · H1-3 블록 삽입
- **Section 6.9 (Hypothesis Validation) 신설** — 각 H 를 empirical evidence 와 매핑하는 Table 15

### RQ1 / H1 (rate-of-change)

- **RQ1**: 다중 스케일 시간차분 피처가 절대값 대비 열화 상태 구분에 더 유효한가?
- **H1**: LSTM baseline 에 multi-scale diff [1,5,10] 을 붙이면 single-lag 및 absolute-only 대비 macro F1 이 통계적으로 유의미하게 상승. 파라미터 비례 증가 없이.

### RQ2 / H2 (SupCon 로 잠재 공간 분리)

- **RQ2**: SupCon 을 CE 와 결합하면 Normal-Mild boundary 를 개선할 수 있나?
- **H2**: λ=0.1 SupCon 추가 시 Normal↔Mild 오분류 ≥ 40% 감소, F1 유지 or 상승.

### RQ3 (domain-informed lightweight vs general-purpose deep)

- **RQ3**: Domain-informed 경량 LSTM 이 TimesNet · PatchTST · CATFT (최대 10.79 M) 를 능가하며 edge 배포 지연 제약 (<10 ms) 을 만족하는가?
- **H3**: V2+ (~2.85 M) 가 macro F1 에서 세 모델 모두 능가 + latency budget 통과.

### Section 6.9 Table 15 (Hypothesis Validation)

각 H 를 다음 3-column 으로 매핑:

| H | Result | Verdict |
|:-:|:-------|:-------:|
| H1 | V2+ (multi-scale) 0.9550 vs V2 (single) 0.9430 (+1.20%), V1 (absolute) 0.9235 (+3.15%). Lag [1,5,10] 이 [1], [1,3,7], [1,10,20] 대비 최적. | **Supported** |
| H2 | NM 오류 45 → 24 (46.7% 감소, 사전 등록 40% 임계치 초과). F1 0.9430 → 0.9557 ± 0.0006 동시 상승. | **Supported** |
| H3 | V2+ F1 0.9557 vs TimesNet 0.9189 / PatchTST 0.9311 / CATFT V5 0.9252. Latency ONNX GPU 2.61 ms · CPU 7.27 ms 모두 < 10 ms. | **Supported** |

H1 의 nuance (**combined-effect form**) 은 §7.3 Discussion 에서 후속 서술: V2a (multi-scale diff only) 는 F1 미미 상승 → 세 컴포넌트 결합 시너지가 진짜 원인.

---

## 4. 논문 line 변화

| 변경 전 | 변경 후 | Δ |
|:---:|:---:|:---:|
| 570 lines | **603 lines** | **+33 lines** |

---

## 5. 후속 파장 (다음 세션들)

- 2026-07-31 아침 (연구노트 #14): AI Hub 데이터 스키마 재분석 시작 → sensor 4-way heterogeneous 가설 폐기, 단일 sensor set 확정
- 2026-07-31 저녁 (연구노트 #14): AI Hub 활용 가이드라인 문서 발견 → MMTransformer baseline (F1 0.9109), 라벨 정의, sampling, CT 채널 의미 등 확인
- 2026-08-06 (연구노트 #15): 위 발견들이 논문 pending #2 로 정리되어 최종 반영, 그동안 Qwen AI 개선 커밋도 함께 도착 → 검증 · 수정 · push

---

## 6. 파일 위치

| 파일 | 경로 |
|------|------|
| 논문 draft | `docs/논문/paper_draft.md` (Section 1 line 45-56, Section 6.9 line 502-521) |
| 로드맵 메모리 | `~/.claude/projects/-home-keti/memory/project_roadmap.md` |
| 논문 pending 메모리 | `~/.claude/projects/-home-keti/memory/project_paper_pending_updates.md` |

---

## 7. 결론

- 논문 target venue 를 MDPI Sensors 로 확정. IJCAI 는 Phase 3 확장판 목표.
- 논문 Section 1 · Section 6.9 에 명시적 RQ1-3 · H1-3 구조 도입 완료 (+33 lines).
- 3-phase 로드맵을 영구 메모리에 저장 → 세션 간 유지.
- 다음 큰 목표: 포스터 논문 1편 (하반기) + MDPI 원고 완성 (연말 or 2027 상반기).
