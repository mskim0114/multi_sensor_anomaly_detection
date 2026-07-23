# STATE.md — 프로젝트 현재 상태

**스냅샷 시점**: 2026-05-22 (USB로 Jetson에 이동된 시점)
**작성자**: PC 측 Claude (claude-opus-4-7) 세션
**Jetson 측 Codex가 이 파일을 매번 시작할 때 갱신·확인할 것.**

---

## A. 지금까지 완료된 것 (PC 측)

### A1. 데이터 / 모델
- AI Hub #71802 원본 데이터 분석 + 4-class 라벨 정의 (`docs/연구노트/02`)
- 누수 없는 데이터 파이프라인 구축 — 세션 단위 분리, 정규화, weighted sampler (`code/data/`)
- 5종 모델 학습 완료 (Baseline / CATFT / V2 / V2+ / TimesNet / PatchTST)
- **V2+** 최종 선정: F1 = 0.9550 ± 0.0006 (3 seeds), Severe recall = 100%

### A2. 논문
- `docs/논문/paper_draft.md` (570 lines, 14 tables, 6 figures placeholder)
- Target: **MDPI Sensors** journal
- 자체 리뷰 3회 + Major issue 4건 모두 대응 완료
  - V2+ ablation 분리 (V2a/V2b/V2c) ✔
  - External SOTA 비교 (TimesNet, PatchTST) ✔
  - 3-seed 통계 유의성 ✔
  - Validation set 사용 명시 ✔
- Minor 이슈도 모두 fix (table 순서, °C 단위, per-class F1 등)

### A3. 배포 (PC 측 검증)
- V2+ → ONNX export (11 MB)
- PC ONNX CPU latency: 4.3 ms
- PyTorch ↔ ONNX 예측 100% 일치 (1157/1157)
- Jetson 검증용 ref dataset 생성 (`../reference/val_reference[_small].npz`)
- Jetson 실행 스크립트 5개 작성 (`../scripts/01~05`)

### A4. 하드웨어
- Jetson Orin Nano 8 GB 보드 도착 + 환경 구축 완료 (JetPack + PyTorch)
- 센서들 구매 완료, **미도착**
- NVLink: 없음 (PC 측 2× RTX 6000 은 SYS/PCIe로만 연결)

---

## B. 지금 Jetson에서 할 일 (우선 순위 순)

### ⓵ ONNX 추론 검증 — **이번 USB 작업의 핵심**
```bash
cd ../   # codex_context 의 상위 = jetson_deploy/
python3 scripts/01_check_environment.py
python3 scripts/02_benchmark_latency.py --runs 200
python3 scripts/03_verify_accuracy.py --small   # 통과하면 --small 빼고 풀
python3 scripts/04_realtime_pipeline.py --n 300
python3 scripts/05_summary.py
```
기대치
- CUDA EP mean latency < 5 ms
- Pred match rate ≥ 99% (정상이면 100%)
- F1 ≈ 0.95~0.96

결과(`../results/jetson_summary.json`)를 PC로 다시 가져와 비교.

### ⓶ TensorRT FP16 변환 (⓵ 통과 후)
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=../model/model_v2plus.onnx \
  --saveEngine=../model/v2plus_fp16.trt \
  --fp16 --workspace=2048 --verbose
```
- 목표: < 2 ms
- 정확도 손실 확인 필요 (FP16은 가끔 1~2% 떨어짐)

### ⓷ 전력/온도 모니터링
```bash
sudo nvpmodel -q          # 현재 모드 확인
sudo nvpmodel -m 0        # MAXN 모드 (최고 성능)
sudo jetson_clocks        # 클럭 고정
sudo tegrastats           # 측정 (별도 터미널)
```
추론 중 GPU/CPU 사용률·온도·전력 기록 → 보고서에 활용.

### ⓸ 결과 패키징
`../results/` 의 모든 JSON과 `tegrastats` 로그를 USB로 PC에 다시 옮김.

---

## C. 센서 도착 후 (예상 ~6월)

1. 회로 조립 (Burden resistor 33 Ω + DC bias for CT, ADS1115 ADC 결선)
2. 캘리브레이션 (전류 영점, NTC 보정 곡선)
3. 실데이터 수집 시스템 구축 (sampling = 1 Hz)
4. 도메인 적응 fine-tuning (AI Hub vs 실제 분포 차이 보정)
5. 현장 PoC 보고서

---

## D. 논문 제출 (R&D 마무리 시점)

- [ ] MDPI Sensors 템플릿 변환 (LaTeX)
- [ ] 참고문헌 MDPI 포맷
- [ ] Figures **사용자가 직접 그리기** (현재 placeholder)
- [ ] cover letter
- [ ] supplementary material (실험 raw)

---

## E. Open issues / known limits

| ID | 항목 | 메모 |
|----|------|------|
| L1 | CATFT 학습 불안정 | Transformer 9.3k window에 과적합 → 논문에서 limitation으로 인정 |
| L2 | NVLink 없음 | 멀티 GPU 학습은 PCIe 통신 → 현재 모델 크기에선 영향 없음 |
| L3 | Thermal 기여도 작음 | F1 +0.37%, NM 오류 -8 — sensor-only 만으로도 0.37 M 파라미터 0.9513 가능 |
| L4 | Severe sample 적음 | 5.7% 만 — class weight 로 보완. 실데이터에서 보강 필요 |
| L5 | Lepton 3.5 raw temp 단위 | C 단위 raw값을 그대로 사용. 새 장비에서 raw 단위 다를 수 있음 |

---

## F. 이 STATE 파일 업데이트 규칙

Jetson Codex가 새 작업을 끝낼 때마다:
1. **A4 또는 새 섹션** 에 한 줄 추가 (날짜 포함)
2. B 의 우선순위 작업이 끝나면 ✔ 표기
3. 새 이슈는 E 에 추가
4. 이 파일과 `docs/오류_및_해결_로그.md` 만 잘 유지하면 PC ↔ Jetson 간 동기화 가능
