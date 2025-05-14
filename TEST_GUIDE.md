# RoboVLMs 테스트 가이드

이 문서는 RoboVLMs 프레임워크를 다양한 환경에서 설치하고 테스트하는 방법을 설명합니다.

## 목차
1. [개요](#개요)
2. [설치 방법](#설치-방법)
   - [MacOS](#macos-환경)
   - [Ubuntu](#ubuntu-환경)
   - [Windows](#windows-환경)
3. [테스트 방법](#테스트-방법)
4. [자주 발생하는 문제](#자주-발생하는-문제)
5. [메모리 요구사항](#메모리-요구사항)

## 개요

RoboVLMs는 Vision-Language-Action(VLA) 모델을 구축하고 테스트하기 위한 프레임워크입니다. 이 프레임워크는 다양한 비전-언어 모델(VLM)을 로봇 작업에 적용할 수 있도록 설계되었습니다.

본 테스트 가이드에서는 다음 모델들을 테스트할 수 있습니다:
- PaliGemma 3B (Google의 비전-언어 모델)
- Flamingo 3B (MPT-3B 기반 비전-언어 모델)
- Flamingo 7B (MPT-7B 기반 비전-언어 모델)

## 설치 방법

### 공통 요구사항
- Python 3.8 이상 (CALVIN 시뮬레이션의 경우 Python 3.8.10 권장)
- PyTorch 2.0 이상

### MacOS 환경

Apple Silicon(M1/M2/M3) 또는 Intel 프로세서 환경에서 RoboVLMs를 설치하는 방법입니다.

#### 1. 환경 설정
```bash
# 가상 환경 생성 (CALVIN 시뮬레이션용)
conda create -n robovlms python=3.8.10 -y

# 또는 SimplerEnv 시뮬레이션용
# conda create -n robovlms python=3.10 -y

conda activate robovlms

# PyTorch 설치 (MPS 지원 버전)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

#### 2. RoboVLMs 설치
```bash
# 저장소 클론
git clone https://github.com/Robot-VLAs/RoboVLMs.git
cd RoboVLMs

# 필요한 패키지 설치
pip install -e .

# 추가 필수 패키지 설치
pip install transformers==4.42.0 accelerate einops pandas
```

#### 3. MacOS 특별 조치
MacOS 환경에서 발생하는 '`from turtle import pd`' 오류 수정:
```bash
# 파일 수정
sed -i '' 's/from turtle import pd/import pandas as pd/g' robovlms/data/vid_llava_dataset.py
```

### Ubuntu 환경

#### 1. 환경 설정
```bash
# CUDA 설치 확인 (CUDA 11.7 이상 권장)
nvidia-smi

# 가상 환경 생성
conda create -n robovlms python=3.8.10 -y
conda activate robovlms

# CUDA 툴킷 설치
conda install cudatoolkit cudatoolkit-dev -y

# PyTorch 설치 (CUDA 11.8 기준)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. RoboVLMs 설치
```bash
# 저장소 클론
git clone https://github.com/Robot-VLAs/RoboVLMs.git
cd RoboVLMs

# 필요한 패키지 설치
pip install -e .

# 추가 패키지 설치
pip install transformers==4.42.0 accelerate einops pandas
```

### Windows 환경

#### 1. 환경 설정
```bash
# 가상 환경 생성
conda create -n robovlms python=3.8.10 -y
conda activate robovlms

# PyTorch 설치 (CUDA 11.8 기준, Windows에서 지원하는 버전 사용)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. RoboVLMs 설치
```bash
# 저장소 클론 (Git Bash 또는 명령 프롬프트 사용)
git clone https://github.com/Robot-VLAs/RoboVLMs.git
cd RoboVLMs

# 필요한 패키지 설치
pip install -e .

# 추가 패키지 설치
pip install transformers==4.42.0 accelerate einops pandas
```

#### 3. Windows 특별 조치
Windows 환경에서 발생하는 '`from turtle import pd`' 오류 수정:
```bash
# 텍스트 편집기로 파일 열기 후 해당 라인을 수정하거나,
# PowerShell을 사용하여 다음 명령 실행:
(Get-Content robovlms\data\vid_llava_dataset.py) -replace 'from turtle import pd', 'import pandas as pd' | Set-Content robovlms\data\vid_llava_dataset.py
```

## 테스트 방법

RoboVLMs 테스트를 위해 제공된 스크립트를 사용할 수 있습니다.

### 1. 직접 테스트 (PaliGemma)
```bash
# MacOS (MPS 사용)
bash run_robovlms_test.sh --model direct-paligemma --device mps

# Ubuntu (CUDA 사용)
bash run_robovlms_test.sh --model direct-paligemma --device cuda

# Windows (CUDA 사용)
bash run_robovlms_test.sh --model direct-paligemma --device cuda

# CPU 사용
bash run_robovlms_test.sh --model direct-paligemma --device cpu
```

### 2. RoboVLMs 테스트
```bash
# PaliGemma 모델 테스트
bash run_robovlms_test.sh --model paligemma

# Flamingo 3B 모델 테스트
bash run_robovlms_test.sh --model flamingo-3b

# Flamingo 7B 모델 테스트
bash run_robovlms_test.sh --model flamingo
```

### 3. 커스텀 이미지 및 지시문 테스트
```bash
bash run_robovlms_test.sh --model paligemma --image path/to/your/image.jpg --instruction "이 로봇 작업을 어떻게 수행할까요?"
```

## 자주 발생하는 문제

### 1. tkinter 오류
**문제**: `from turtle import pd` 불러오는 과정에서 tkinter 관련 오류 발생
**해결**: `robovlms/data/vid_llava_dataset.py` 파일에서 `from turtle import pd`를 `import pandas as pd`로 수정

### 2. 메모리 부족 오류
**문제**: CUDA/MPS 메모리 부족 오류
**해결**: 
- CPU 모드로 전환: `--device cpu` 옵션 사용
- `direct-paligemma` 모델 사용: `--model direct-paligemma`
- 배치 사이즈 및 이미지 크기 축소

### 3. transformers 버전 충돌
**문제**: 모델에 따라 필요한 transformers 버전이 다름
**해결**:
- PaliGemma: `pip install transformers>=4.42.0`
- Flamingo: `pip install transformers==4.33.2`

### 4. 파일 경로 오류
**문제**: 상대 경로 참조 오류
**해결**: 항상 RoboVLMs 루트 디렉토리에서 명령을 실행

## 메모리 요구사항

각 모델별 대략적인 메모리 요구사항은 다음과 같습니다:

| 모델 | GPU VRAM 요구량 | CPU RAM 요구량 |
|-----|---------------|-------------|
| PaliGemma 3B | 약 8GB | 약 12GB |
| Flamingo 3B | 약 8GB | 약 12GB |
| Flamingo 7B | 약 16GB | 약 24GB |

NVIDIA Jetson 16GB VRAM 환경에서는:
- PaliGemma 3B: 작동 가능
- Flamingo 3B: 작동 가능
- Flamingo 7B: 메모리 최적화 필요

### Jetson 환경 최적화 팁

NVIDIA Jetson 환경에서는 다음과 같은 최적화를 적용할 수 있습니다:

1. 메모리 효율적인 설정 사용:
```python
# Jetson 환경을 위한 메모리 최적화 옵션
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
```

2. 그래디언트 체크포인팅 활성화:
```
train_setup:
  gradient_checkpointing: true
```

3. 더 낮은 해상도 이미지 사용:
```
image_size: 160  # 기본값 224보다 작게 설정
``` 