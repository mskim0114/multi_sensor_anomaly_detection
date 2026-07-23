# INDEX — 빠른 파일 탐색

## "지금 뭘 해야 해?"
→ [`STATE.md`](STATE.md) § B

## "프로젝트 한눈에 보기"
→ [`AGENTS.md`](AGENTS.md) (Codex가 자동 로드, 사람도 먼저 읽으면 좋음)

## 데이터셋
| 질문 | 파일 |
|------|------|
| AI Hub 원본 구조 | `docs/연구노트/02_원본데이터셋_분석.md` |
| 4-class 라벨 정의 근거 | `docs/참고문서/AI Hub 데이터셋 심층 분석 및 3종 이상상태 정의 보고서.md` |
| 파이프라인 (split, 정규화) | `docs/연구노트/03_데이터_파이프라인_구축.md`, `code/data/` |

## 모델
| 질문 | 파일 |
|------|------|
| 최종 모델 V2+ 코드 | `code/models/v2_plus.py` |
| V2+ 결과 | `docs/연구노트/09_V2Plus_결과.md` |
| 베이스라인 결과 | `docs/연구노트/04_베이스라인_모델_결과.md` |
| CATFT (실패 사례) | `docs/연구노트/05_CATFT_모델_결과.md` |
| Ablation study | `docs/연구노트/06_Ablation_Study_결과.md`, `11_논문_추가실험.md` |
| 외부 SOTA 비교 | `code/models/external_baselines.py` |
| 왜 LSTM? 왜 SE? 왜 SupCon? | `docs/연구노트/08_모델개선_리서치.md` |

## 학습
| 질문 | 파일 |
|------|------|
| V2+ 학습 진입점 | `code/train_v2plus.py` |
| Demo 추론 | `code/demo_inference.py` |

## 배포 / Jetson
| 질문 | 파일 |
|------|------|
| ONNX 변환 코드 | `code/deploy/export_v2plus_onnx.py` |
| 실시간 파이프라인 (구버전) | `code/deploy/realtime_pipeline.py` |
| Jetson 검증 스크립트 | `../scripts/01~05_*.py` |
| 검증 패키지 README | `../README.md` |
| 경량화 전략 | `docs/연구노트/07_모델_경량화_배포.md` |
| 실시간 시스템 설계 | `docs/연구노트/10_실시간_추론_파이프라인.md`, `docs/참고문서/실시간 추론 시스템 구축 및 연구 전략 로드맵.md` |

## 센서 / 하드웨어
| 질문 | 파일 |
|------|------|
| 어떤 센서? Burden resistor? | `docs/참고문서/실제 센서 및 엣지 보드(Jetson) 구매 사양서 및 매칭 가이드.md` |

## 논문
| 질문 | 파일 |
|------|------|
| 논문 본문 (full draft) | `docs/논문/paper_draft.md` |
| Target journal | MDPI Sensors |
| 보고서(국문) | `docs/실험결과_보고서.md` |

## 에러 / 트러블슈팅
→ [`docs/오류_및_해결_로그.md`](docs/오류_및_해결_로그.md)
