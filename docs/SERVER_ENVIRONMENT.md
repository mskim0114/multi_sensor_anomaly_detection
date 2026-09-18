# 서버 학습 환경 (SERVER-TRAINING)

> **First read: [SERVER_WORKSTATION_HANDOFF.md](SERVER_WORKSTATION_HANDOFF.md)** — 서버에서
> 처음 작업할 때의 canonical handoff (현재 상태, 준비된 데이터, first-run 절차).
> 이 문서의 실제 hardware/software 값은 workstation audit 전까지 아래 상태를 유지한다.

## 상태

```
PARTIAL: read-only preflight 완료 (2026-09-17), 패키지·requirements audit 미완료
```

아래 "실측" 절은 2026-09-17 서버에 SSH 로 접속해 읽기 전용으로 확인한 값이다. 추측이 아니다.
아직 확정하지 않은 것: 학습 환경으로 쓸 venv 결정, `src/` 의존성 목록, `requirements-server.txt`.

## 실측 (2026-09-17, keti@10.252.219.59)

| 항목 | 값 |
|---|---|
| hostname / OS | `keti-Precision-7920-Tower` / Ubuntu 22.04.5 LTS, x86_64 |
| CPU / RAM | 2× Intel Xeon Silver 4214R (24C/48T) / 187 GiB |
| GPU | 2× Quadro RTX 6000 24,576 MiB · driver **580.178.04** · `nvidia-smi` 정상 (재부팅 전 mismatch 있었음) |
| CUDA | `nvcc` 없음. torch 가 보고하는 CUDA 12.4 |
| 루트 디스크 | 908 GB · **이관 후 269 GB 사용 / 593 GB 여유 / 32 %** (이관 전 97 %) |
| 데이터 디스크 | SSD `/mnt/data-ssd` 1 TB(비어 있음) · HDD `/mnt/data-hdd` 7.3 TB(2.9 TB 여유). UUID fstab, `nofail`. **boot-time 마운트 미검증** |
| Python | 시스템 3.10 (torch 없음) · `/home/keti/monai_env` 3.12.13 |
| torch 환경 | **`$HOME/venvs/factory_training` 없음.** `/home/keti/monai_env` (python venv, miniconda 3.12 base) 에 torch **2.6.0+cu124**, `torch.cuda.is_available()=True` |
| conda envs | etc / stock / time-llm / uni2ts — torch 없음 |
| Docker | 29.6.0 · compose 5.1.4 · keti 접근 가능 · 이미지 23 GB |
| 저장소 clone | `/home/keti/factory_safety` = private `mskim0114/factory_safety` main `5feabb5` (+ `public-upstream` 원격) |
| 프로젝트 데이터 | `factory_safety/data/aihub` → **심볼릭 링크** → `/mnt/data-hdd/keti_data/factory_safety/aihub` (84 GB, extracted 0바이트 없음) |

근거와 이관 과정: [연구노트 #17](연구노트/연구노트_17_서버_스토리지_이관_및_실측.md),
`~/review_runs/20260917_monitoring/server_preflight.json` (Jetson 보관).

**미결.** `monai_env` 를 SERVER-TRAINING 환경으로 채택할지, 정책대로 `factory_training` 을 새로 만들지는
결정되지 않았다(decisions O-108). 결정 전에는 `monai_env` 에 패키지를 설치하지 않는다.

## 왜 처음에 비워 두었는가

(초판 작성 시점의 설명. 실측 전에는 아래 이유로 비워 두었다.)

이 문서를 처음 작성한 시점의 작업 호스트는 Jetson Orin Nano (`keti-kms`, aarch64, L4T R36.5.0) 였다.
Jetson에서 학습 서버의 의존성 버전을 추측하면 다음이 어긋난다.

- CPU 아키텍처 (aarch64 vs x86_64) — 휠 태그가 다르다
- CUDA 버전과 그에 묶인 PyTorch 빌드
- PyTorch 자체 (Jetson은 NVIDIA 전용 휠, 서버는 PyPI/conda 채널)
- cuDNN, NCCL, GPU 개수 (연구노트에 NVLink 부재가 기록되어 있음)

따라서 **추측한 버전을 여기에 적지 않는다.**

## 확정된 정책

| 항목 | 값 |
|---|---|
| 프로파일 이름 | `SERVER-TRAINING` |
| 목적 | PC/서버 학습, 모델 개발, 논문 실험, ONNX export |
| 코드 영역 | `src/` |
| venv 경로 | `$HOME/venvs/factory_training` |
| requirements | `requirements-server.txt` (저장소 루트) |
| Jetson 센서 패키지 | **설치하지 않는다** |

전체 정책은 [ENVIRONMENT_POLICY.md](ENVIRONMENT_POLICY.md) 참조.

## audit 절차 (학습 서버에서 수행할 것)

실제 학습 서버에 접속한 뒤 다음을 수집해서 이 문서를 채운다.

```bash
# 1. 호스트 기본 정보
uname -a
cat /etc/os-release
python3 --version
nvidia-smi                          # GPU 모델, 개수, 드라이버, CUDA 버전
nvcc --version

# 2. 현재 학습 환경의 실제 패키지 (기존 환경이 있다면)
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "import torchvision; print(torchvision.__version__)"
python3 -m pip list

# 3. provenance 확인 - 어느 site-packages 에서 오는지
python3 -c "import torch, numpy; print(torch.__file__); print(numpy.__file__)"
```

수집 후 이 문서에 기록할 항목:

- 호스트명, OS, 커널, 아키텍처
- GPU 모델 / 개수 / VRAM / 드라이버 버전
- CUDA / cuDNN 버전
- Python 버전
- PyTorch / torchvision 버전과 설치 출처(PyPI, conda, NVIDIA index 중 무엇인지)
- `src/` 가 실제로 import하는 패키지 목록
- `requirements-server.txt` 에 넣을 직접 의존성

## 알려진 단서 (검증 필요, 그대로 신뢰하지 말 것)

저장소 문서에 남아 있는 흔적이며 **확인 전에는 사실로 취급하지 않는다.**

- `README.md` 가 `conda activate monai_env` 를 안내한다 → **실측: `monai_env` 는 conda env 가 아니라
  miniconda 의 python 으로 만든 venv 다** (`pyvenv.cfg`: `home = /home/keti/miniconda3/bin`)
- 학습 스크립트 전부 `--gpu` 기본값이 `1` 이다 → **실측: RTX 6000 2장 확인**
- 연구노트에 NVLink 부재가 한계로 기록되어 있다 (미재확인)
- `src/` 코드가 요구하는 것: `torch`, `torchvision`(EfficientNet-B0 pretrained), `numpy`,
  `scikit-learn`(metrics), `matplotlib`, `seaborn`(confusion matrix), `pyyaml`, `onnx`, `onnxruntime`
  (export/검증), `tqdm` 여부는 확인 필요

## 주의: 경로 하드코딩 문제

Jetson 작업 브랜치에서는 2026-09-10 `src/paths.py` 도입으로 checkout 기준 경로가 되었다
([LOCAL_REVIEW_20260910.md](LOCAL_REVIEW_20260910.md)). **서버 clone 은 `5feabb5` 로 이 수정을 포함하지
않아** 여전히 `/home/keti/factory_safety/...` 를 하드코딩하지만, 서버에서는 그 경로가 실제 위치이고
`data/aihub` 가 HDD 로의 심볼릭 링크라 그대로 동작한다. 서버가 작업 브랜치를 받으면 이 절은 소멸한다.
