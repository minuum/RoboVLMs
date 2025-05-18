# RoboVLMs 테스트 가이드 (Ubuntu 중점)

이 문서는 RoboVLMs 프레임워크를 다양한 환경, 특히 Ubuntu에서 설치하고 테스트하는 방법을 설명합니다.

## 목차
1. [개요](#개요)
2. [Ubuntu 환경 설치 방법](#ubuntu-환경-설치-방법)
   - [2.1. 사전 요구사항](#21-사전-요구사항)
   - [2.2. Conda 환경 설정](#22-conda-환경-설정)
   - [2.3. RoboVLMs 설치 (Poetry 사용)](#23-robovlms-설치-poetry-사용)
   - [2.4. `vid_llava_dataset.py` 수정](#24-vid_llava_datasetpy-수정)
3. [테스트 방법](#테스트-방법)
   - [3.1. 테스트 스크립트 실행 (`run_robovlms_test.sh`)](#31-테스트-스크립트-실행-run_robovlms_testsh)
4. [자주 발생하는 문제 (Ubuntu)](#자주-발생하는-문제-ubuntu)
5. [메모리 요구사항](#메모리-요구사항)
6. [기타 환경 설치 정보](#기타-환경-설치-정보)
   - [MacOS 환경](#macos-환경)
   - [Windows 환경](#windows-환경)


## 1. 개요

RoboVLMs는 Vision-Language-Action(VLA) 모델을 구축하고 테스트하기 위한 프레임워크입니다. 이 프레임워크는 다양한 비전-언어 모델(VLM)을 로봇 작업에 적용할 수 있도록 설계되었습니다.

본 테스트 가이드에서는 다음 모델들을 테스트할 수 있습니다 (설정 파일 및 스크립트에 따라 다름):
- PaliGemma 3B (Google의 비전-언어 모델)
- Flamingo 3B (MPT-3B 기반 비전-언어 모델)
- Flamingo 7B (MPT-7B 기반 비전-언어 모델)
- 기타 `pyproject.toml` 및 프로젝트 내에서 지원하는 모델

## 2. Ubuntu 환경 설치 방법

### 2.1. 사전 요구사항
- Ubuntu (예: 20.04 LTS, 22.04 LTS)
- NVIDIA GPU 및 최신 NVIDIA 드라이버 설치 완료
- Conda (Miniconda 또는 Anaconda) 설치 완료

### 2.2. Conda 환경 설정

`pyproject.toml` 파일에 명시된 Python 버전은 `>=3.9,<3.11` 입니다. 여기서는 Python 3.10을 기준으로 설명합니다.

1.  **Conda 가상 환경 생성 및 활성화:**
    ```bash
    conda create -n robovlms_py310 python=3.10 -y
    conda activate robovlms_py310
    ```

2.  **PyTorch 및 관련 라이브러리 설치:**
    `pyproject.toml`에 명시된 버전은 `torch="2.1.0"`, `torchvision="0.16.0"`, `torchaudio="2.1.0"` 입니다.
    시스템에 설치된 NVIDIA 드라이버 및 사용하고자 하는 CUDA 버전에 맞춰 PyTorch를 설치합니다.

    *   **CUDA 11.8 사용 시:**
        ```bash
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **CUDA 12.1 사용 시:**
        ```bash
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        ```
    *다른 CUDA 버전을 사용 중이라면, [PyTorch 공식 웹사이트](https://pytorch.org/get-started/previous-versions/)에서 해당 버전에 맞는 설치 명령을 확인하세요.*

### 2.3. RoboVLMs 설치 (Poetry 사용)

1.  **RoboVLMs 저장소 클론:**
    ```bash
    git clone https://github.com/Robot-VLAs/RoboVLMs.git
    cd RoboVLMs
    ```

2.  **Poetry 설치 (아직 설치하지 않았다면):**
    ```bash
    pip install poetry
    ```

3.  **의존성 설치:**
    Poetry를 사용하여 `pyproject.toml`에 정의된 모든 프로젝트 의존성을 설치합니다.
    ```bash
    poetry install
    ```
    이 명령은 `transformers`, `einops`, `opencv-python`, `pandas` 등 필요한 모든 패키지를 설치합니다.

### 2.4. `vid_llava_dataset.py` 수정

`turtle` 모듈 import로 인해 발생할 수 있는 `tkinter` 관련 오류를 방지하기 위해 해당 파일을 수정합니다.
```bash
# RoboVLMs 프로젝트 루트 디렉토리에서 실행
sed -i 's/from turtle import pd/import pandas as pd/g' robovlms/data/vid_llava_dataset.py
```

## 3. 테스트 방법

### 3.1. 테스트 스크립트 실행 (`run_robovlms_test.sh`)

**참고:** `run_robovlms_test.sh` 스크립트는 사용자가 작성하거나 프로젝트에서 제공하는 테스트 실행 스크립트를 의미합니다. 이 스크립트는 모델, 장치, 이미지 경로, 지시문 등을 인자로 받을 수 있도록 구성되어야 합니다. 아래는 예시 명령어입니다.

1.  **PaliGemma 모델 테스트 (GPU 사용):**
    ```bash
    bash run_robovlms_test.sh --model paligemma --device cuda
    ```

2.  **PaliGemma 모델 테스트 (CPU 사용):**
    ```bash
    bash run_robovlms_test.sh --model paligemma --device cpu
    ```

3.  **Flamingo 3B 모델 테스트 (GPU 사용):**
    ```bash
    bash run_robovlms_test.sh --model flamingo-3b --device cuda
    ```
    *(참고: `flamingo-3b`는 예시 모델명이며, 실제 설정 파일이나 스크립트에서 사용하는 이름을 확인해야 합니다.)*

4.  **Flamingo 7B 모델 테스트 (GPU 사용):**
    ```bash
    bash run_robovlms_test.sh --model flamingo --device cuda
    ```
    *(참고: `flamingo`는 Flamingo 7B 모델을 지칭하는 예시입니다.)*

5.  **커스텀 이미지 및 지시문 테스트 (예: PaliGemma, GPU 사용):**
    ```bash
    bash run_robovlms_test.sh --model paligemma --image path/to/your/image.jpg --instruction "이 물체를 어떻게 옮길 수 있을까?" --device cuda
    ```

**실행 전 확인사항:**
- `run_robovlms_test.sh` 스크립트가 실행 가능하도록 권한이 설정되어 있는지 확인 (`chmod +x run_robovlms_test.sh`).
- 스크립트 내에서 모델 경로, 설정 파일 경로 등이 올바르게 지정되어 있는지 확인합니다.

## 4. 자주 발생하는 문제 (Ubuntu)

### 1. CUDA 및 PyTorch 버전 불일치
-   **문제**: `RuntimeError: CUDA error: ...` , `CUBLAS_STATUS_NOT_INITIALIZED` 등.
-   **해결**: 시스템 NVIDIA 드라이버, CUDA 툴킷 버전, PyTorch 빌드 시 사용된 CUDA 버전이 호환되는지 확인합니다. `nvidia-smi`로 드라이버 버전을 확인하고, PyTorch 설치 시 적절한 CUDA 버전을 명시합니다. (예: `cu118`, `cu121`)

### 2. `ModuleNotFoundError`
-   **문제**: 특정 Python 패키지를 찾을 수 없음 (예: `einops`, `timm`, `opencv-python`).
-   **해결**: `poetry install` 명령이 성공적으로 완료되었는지 확인합니다. 가상 환경이 올바르게 활성화되었는지 확인합니다. 문제가 지속되면 `poetry show`로 설치된 패키지를 확인하고, 필요한 경우 `poetry add <package_name>`으로 추가합니다.

### 3. `tkinter` 관련 오류 (예: `_tkinter.TclError: no display name and no $DISPLAY environment variable`)
-   **문제**: GUI 라이브러리 `tkinter`가 필요하지만, headless 서버 환경이거나 X11 포워딩이 설정되지 않은 경우 발생. `robovlms/data/vid_llava_dataset.py`의 `from turtle import pd`가 원인일 수 있습니다.
-   **해결**: [2.4. `vid_llava_dataset.py` 수정](#24-vid_llava_datasetpy-수정) 섹션의 `sed` 명령을 실행하여 수정합니다. `opencv-python`의 `cv2.imshow` 등 GUI 함수를 직접 호출하는 코드가 테스트 스크립트에 있다면, headless 환경에서는 문제를 일으킬 수 있으므로 주의합니다.

### 4. 메모리 부족 (OOM: Out Of Memory)
-   **문제**: GPU VRAM 또는 시스템 RAM 부족.
-   **해결**:
    -   더 작은 모델 사용 (예: 7B 대신 3B).
    -   테스트 시 배치 크기를 줄입니다 (스크립트 또는 설정에서).
    -   `--device cpu` 옵션으로 CPU에서 실행 (속도 저하 감수).
    -   모델 로드 시 양자화 옵션 고려 (예: `load_in_4bit=True`, `load_in_8bit=True` - 모델 로딩 코드에서 지원해야 함).

### 5. `transformers` 버전 충돌
-   **문제**: 특정 VLM 백본은 특정 버전의 `transformers` 라이브러리를 요구할 수 있습니다.
-   **해결**: `pyproject.toml`에 명시된 `transformers` 버전 (`^4.42.0`)은 PaliGemma 등 최신 모델과 호환성이 좋습니다. 만약 특정 모델(예: 구버전 Flamingo)이 다른 버전을 요구한다면, 해당 모델 테스트를 위해 별도의 환경을 구성하거나 `poetry update transformers==<required_version>` 등으로 버전을 조정해야 할 수 있습니다. (단, 이 경우 다른 모델과의 호환성 문제가 발생할 수 있습니다.)

## 5. 메모리 요구사항

각 모델별 대략적인 메모리 요구사항은 다음과 같습니다 (실제 사용량은 입력 데이터, 코드 최적화 수준에 따라 다를 수 있음):

| 모델 | GPU VRAM 요구량 (FP16) | CPU RAM 요구량 (로드 시) |
|-----|-----------------------|-------------------------|
| PaliGemma 3B | 약 7-8GB | 약 10-12GB |
| Flamingo 3B  | 약 7-8GB | 약 10-12GB |
| Flamingo 7B  | 약 15-16GB | 약 20-24GB |

NVIDIA Jetson (예: 16GB VRAM 모델) 환경에서는:
-   PaliGemma 3B, Flamingo 3B: FP16 또는 INT8/INT4 양자화 적용 시 작동 가능성 높음.
-   Flamingo 7B: 양자화 및 최적화 필수.

### Jetson 등 제한된 환경 최적화 팁

1.  **모델 로드 시 정밀도 및 양자화 조절:**
    ```python
    # 예시: Hugging Face Transformers 사용 시
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    # FP16 사용
    model = AutoModelForVision2Seq.from_pretrained(
        "google/paligemma-3b-pt-224", # 모델에 따라 다름
        torch_dtype=torch.float16,   # FP16
        device_map="auto"            # 가능하면 GPU에 할당
    )
    # 4비트 양자화 사용 (BitsAndBytes 필요)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     "google/paligemma-3b-pt-224",
    #     load_in_4bit=True,
    #     device_map="auto"
    # )
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    ```

2.  **그래디언트 체크포인팅 (학습 시):**
    학습 설정에서 그래디언트 체크포인팅을 활성화하여 메모리를 절약할 수 있습니다. (테스트 시에는 직접적인 영향은 적음)

3.  **입력 이미지 해상도 줄이기:**
    모델 설정이나 전처리 과정에서 이미지 크기를 줄이면 메모리 사용량을 낮출 수 있습니다. (예: `image_size: 160`)

## 6. 기타 환경 설치 정보

### MacOS 환경

Apple Silicon(M1/M2/M3) 또는 Intel CPU 환경에서의 설치 방법입니다.

1.  **Conda 환경 설정 (Python 3.10 기준):**
    ```bash
    conda create -n robovlms_macos python=3.10 -y
    conda activate robovlms_macos
    ```
2.  **PyTorch 설치 (MPS 지원):**
    ```bash
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    ```
3.  **RoboVLMs 설치 (Poetry 사용):**
    ```bash
    git clone https://github.com/Robot-VLAs/RoboVLMs.git
    cd RoboVLMs
    pip install poetry
    poetry install
    ```
4.  **`vid_llava_dataset.py` 수정:**
    ```bash
    sed -i '' 's/from turtle import pd/import pandas as pd/g' robovlms/data/vid_llava_dataset.py
    ```
    *MacOS에서는 `sed -i ''` 와 같이 빈 문자열을 제공해야 할 수 있습니다.*

### Windows 환경

1.  **Conda 환경 설정 (Python 3.10 기준):**
    ```bash
    conda create -n robovlms_win python=3.10 -y
    conda activate robovlms_win
    ```
2.  **PyTorch 설치 (CUDA 또는 CPU):**
    *   CUDA 11.8 사용 시:
        ```bash
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        ```
    *   CPU 버전:
        ```bash
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        ```
3.  **RoboVLMs 설치 (Poetry 사용):**
    ```bash
    git clone https://github.com/Robot-VLAs/RoboVLMs.git
    cd RoboVLMs
    pip install poetry
    poetry install
    ```
4.  **`vid_llava_dataset.py` 수정 (PowerShell 예시):**
    ```powershell
    (Get-Content robovlms\data\vid_llava_dataset.py) -replace 'from turtle import pd', 'import pandas as pd' | Set-Content robovlms\data\vid_llava_dataset.py
    ```
---
이 가이드가 RoboVLMs 프로젝트를 Ubuntu 환경에서 설정하고 테스트하는 데 도움이 되기를 바랍니다. 