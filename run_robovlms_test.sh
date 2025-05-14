#!/bin/bash

# 로깅 설정
log_file="robovlms_test.log"
echo "--- RoboVLMs 테스트 시작 $(date) ---" > $log_file

# 사용법 표시
usage() {
  echo "사용법: $0 [옵션]"
  echo "옵션:"
  echo "  --model MODEL_TYPE    테스트할 모델 (flamingo, flamingo-3b, paligemma, direct-paligemma)"
  echo "  --image IMAGE_PATH    테스트할 이미지 경로 또는 URL"
  echo "  --instruction TEXT    모델에 전달할 지시문"
  echo "  --device DEVICE       사용할 장치 (mps, cuda, cpu)"
  echo "  --help                이 도움말을 표시합니다"
}

# 기본값 설정
MODEL="direct-paligemma"
IMAGE="SCR-20250513-omus.png"
INSTRUCTION="Provide step-by-step instructions for the robot arm to pick up the red pepper in front of it."
DEVICE=""

# 인수 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --instruction)
      INSTRUCTION="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# 모델 유효성 검사
if [[ ! "$MODEL" =~ ^(flamingo|flamingo-3b|paligemma|direct-paligemma)$ ]]; then
  echo "오류: 유효하지 않은 모델 타입입니다: $MODEL" | tee -a $log_file
  echo "가능한 모델: flamingo, flamingo-3b, paligemma, direct-paligemma" | tee -a $log_file
  exit 1
fi

# 이미지 파일 확인
if [[ ! -f "$IMAGE" && ! "$IMAGE" =~ ^https?:// ]]; then
  echo "경고: 이미지 파일을 찾을 수 없습니다: $IMAGE" | tee -a $log_file
  echo "현재 디렉토리에서 이미지 파일 확인 중..." | tee -a $log_file
  
  # 현재 디렉토리의 이미지 파일 목록
  IMAGE_FILES=$(find . -maxdepth 1 -type f -name "*.png" -o -name "*.jpg" -o -name "*.jpeg")
  if [ -z "$IMAGE_FILES" ]; then
    echo "사용 가능한 이미지 파일을 찾을 수 없습니다." | tee -a $log_file
    echo "기본 URL 이미지를 사용합니다." | tee -a $log_file
    IMAGE="https://raw.githubusercontent.com/Robot-VLAs/RoboVLMs/main/imgs/robovlms.png"
  else
    echo "사용 가능한 이미지 파일: $IMAGE_FILES" | tee -a $log_file
  fi
fi

# 환경 설정 확인
echo "=== 환경 확인 ===" | tee -a $log_file
echo "Python 버전: $(python --version 2>&1)" | tee -a $log_file
echo "PyTorch 버전: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch 없음")" | tee -a $log_file
echo "CUDA 사용 가능: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "PyTorch 없음")" | tee -a $log_file
echo "MPS 사용 가능: $(python -c "import torch; print(torch.backends.mps.is_available())" 2>&1)" | tee -a $log_file

# 필요한 패키지 확인
echo "=== 필수 패키지 확인 ===" | tee -a $log_file
python -c "from importlib import import_module; [import_module(m) for m in ['transformers', 'PIL', 'requests', 'huggingface_hub']]" > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "필요한 패키지가 설치되지 않았습니다. 설치를 진행합니다..." | tee -a $log_file
  pip install transformers pillow requests huggingface_hub accelerate
fi

# direct-paligemma 모드일 때는 RoboVLMs 패키지 없이도 작동 가능
if [ "$MODEL" != "direct-paligemma" ]; then
  echo "=== RoboVLMs 패키지 확인 ===" | tee -a $log_file
  python -c "import robovlms" > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "RoboVLMs 패키지를 찾을 수 없습니다. 확인해주세요." | tee -a $log_file
    echo "현재 디렉토리에서 pip install -e . 명령어를 실행했는지 확인하세요." | tee -a $log_file
    
    # RoboVLMs이 없을 경우 direct-paligemma 모드로 변경
    echo "RoboVLMs 패키지가 없어 direct-paligemma 모드로 변경합니다." | tee -a $log_file
    MODEL="direct-paligemma"
  fi
fi

# 장치 파라미터 구성
DEVICE_PARAM=""
if [ ! -z "$DEVICE" ]; then
  DEVICE_PARAM="--device $DEVICE"
else
  # 기본 장치 자동 감지
  if python -c "import torch; print(torch.backends.mps.is_available())" 2>&1 | grep -q "True"; then
    DEVICE="mps"
    DEVICE_PARAM="--device mps"
  elif python -c "import torch; print(torch.cuda.is_available())" 2>&1 | grep -q "True"; then
    DEVICE="cuda"
    DEVICE_PARAM="--device cuda"
  else
    DEVICE="cpu"
    DEVICE_PARAM="--device cpu"
  fi
  echo "자동으로 감지된 장치: $DEVICE" | tee -a $log_file
fi

# 스크립트 실행
echo "=== 테스트 실행 ===" | tee -a $log_file
echo "모델: $MODEL" | tee -a $log_file
echo "이미지: $IMAGE" | tee -a $log_file
echo "장치: $DEVICE" | tee -a $log_file
echo "지시문: $INSTRUCTION" | tee -a $log_file

# MPS 백엔드 폴백 활성화
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 모델 다운로드 디렉토리 생성
if [ ! -d ".vlms/paligemma-3b-pt-224" ]; then
  echo "모델 캐시 디렉토리 생성 중..." | tee -a $log_file
  mkdir -p .vlms/paligemma-3b-pt-224
fi

# 테스트 실행
echo "테스트 실행 중..." | tee -a $log_file
python robovlms_test.py --model $MODEL $DEVICE_PARAM --image "$IMAGE" --instruction "$INSTRUCTION" | tee -a $log_file

echo "=== 테스트 완료 ===" | tee -a $log_file
echo "로그 파일: $log_file" 