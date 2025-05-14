#!/bin/bash

# RoboVLMs 테스트 스크립트 (M3 PRO 환경)

# 환경 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Poetry 환경 활성화 (이미 활성화된 경우 주석 처리)
# source $(poetry env info --path)/bin/activate

# 사용할 모델 선택 - PaliGemma 3B 모델 사용
MODEL="paligemma"

# 테스트 이미지
IMAGE_PATH="https://raw.githubusercontent.com/Robot-VLAs/RoboVLMs/main/imgs/robovlms.png"

# 명령어 (로봇 지시)
INSTRUCTION="로봇이 테이블 위의 물체를 집어서 상자에 넣으려면 어떻게 해야 할까요?"

# 테스트 실행
python test.py \
  --model "$MODEL" \
  --image "$IMAGE_PATH" \
  --instruction "$INSTRUCTION"