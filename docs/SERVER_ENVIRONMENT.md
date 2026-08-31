# 서버 학습 환경 (SERVER-TRAINING)

## 상태

```
PENDING SERVER ENVIRONMENT AUDIT
```

이 문서는 아직 실제 버전을 담고 있지 않다. **의도적으로 비어 있다.**

## 왜 비어 있는가

이 문서를 작성한 시점의 작업 호스트는 Jetson Orin Nano (`keti-kms`, aarch64, L4T R36.5.0) 였다.
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

- `README.md` 가 `conda activate monai_env` 를 안내한다 → 학습 환경이 conda 기반이었을 가능성
- 학습 스크립트 전부 `--gpu` 기본값이 `1` 이다 → GPU가 최소 2장인 서버였을 가능성
- 연구노트에 NVLink 부재가 한계로 기록되어 있다
- `src/` 코드가 요구하는 것: `torch`, `torchvision`(EfficientNet-B0 pretrained), `numpy`,
  `scikit-learn`(metrics), `matplotlib`, `seaborn`(confusion matrix), `pyyaml`, `onnx`, `onnxruntime`
  (export/검증), `tqdm` 여부는 확인 필요

## 주의: 경로 하드코딩 문제

`src/` 전반과 `configs/data_config.yaml` 이 `/home/keti/factory_safety/...` 를 하드코딩하고 있으나
저장소의 실제 위치는 `/home/keti/projects/factory_safety` 다. 학습 서버에서도 동일한 문제가
발생할 수 있으므로 audit 시 함께 확인한다. 아직 수정하지 않았다.
