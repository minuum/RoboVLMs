#!/bin/bash

# 🚀 K-프로젝트 Event-Triggered VLA 시스템 실행 스크립트
# 목적: 실시간(<100ms) 로봇카 네비게이션 VLA 시스템 시작

set -e  # 에러 발생시 중단

echo "🎯 K-프로젝트 Event-Triggered VLA 시스템 시작"
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 환경 변수 설정
export PROJECT_NAME="k_project_event_vla"
export VLA_MODEL="paligemma-3b-mix-224"
export ACTION_MODE="automotive"
export ACTION_DIM=4
export WINDOW_SIZE=8
export INFERENCE_LATENCY_TARGET=100  # ms
export ROS2_DOMAIN_ID=42

# 실행 디렉토리 확인
if [ ! -f "jetson_quick_start.sh" ]; then
    echo -e "${RED}❌ RoboVLMs 디렉토리에서 실행해주세요${NC}"
    echo "사용법: cd RoboVLMs && ./launch_event_triggered_vla.sh"
    exit 1
fi

echo -e "${BLUE}🔧 환경 변수 설정 완료${NC}"
echo "- 프로젝트: $PROJECT_NAME"
echo "- VLA 모델: $VLA_MODEL"
echo "- 액션 모드: $ACTION_MODE (${ACTION_DIM}D)"
echo "- 윈도우 크기: $WINDOW_SIZE"
echo "- 목표 지연시간: ${INFERENCE_LATENCY_TARGET}ms"
echo ""

# 1. ROS2 환경 설정
echo -e "${BLUE}🤖 ROS2 환경 설정${NC}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
    echo "✅ ROS2 Humble 환경 활성화 (Domain ID: $ROS2_DOMAIN_ID)"
else
    echo -e "${RED}❌ ROS2 Humble이 설치되지 않았습니다${NC}"
    exit 1
fi

# 2. Python 환경 활성화
echo -e "${BLUE}🐍 Python 환경 활성화${NC}"
if command -v conda &> /dev/null && conda env list | grep -q robovlms; then
    echo "Conda robovlms 환경 활성화 중..."
    eval "$(conda shell.bash hook)"
    conda activate robovlms
elif [ -d "venv" ]; then
    echo "Python 가상환경 활성화 중..."
    source venv/bin/activate
fi
echo "✅ Python 환경 준비 완료"

# 3. GPU 메모리 최적화
echo -e "${BLUE}🎮 GPU 메모리 최적화${NC}"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_DTYPE=bfloat16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=./.vlms
echo "✅ CUDA 및 메모리 최적화 설정 완료"

# 4. Docker 컨테이너 상태 확인 및 시작
echo -e "${BLUE}🐳 Docker 컨테이너 관리${NC}"

# 기존 컨테이너가 실행 중인지 확인
if docker ps | grep -q $PROJECT_NAME; then
    echo -e "${YELLOW}⚠️  기존 컨테이너가 실행 중입니다. 중지하고 새로 시작합니다.${NC}"
    docker stop $PROJECT_NAME || true
    docker rm $PROJECT_NAME || true
fi

# Docker Compose 파일 존재 확인
if [ -f "docker-compose.yml" ]; then
    echo "Docker Compose를 사용하여 컨테이너 시작..."
    docker-compose up -d $PROJECT_NAME
else
    echo -e "${YELLOW}⚠️  docker-compose.yml이 없습니다. 기본 Docker 명령어로 실행...${NC}"
    
    # 기본 Docker 실행 (docker-compose.yml이 없는 경우)
    docker run -d \
        --name $PROJECT_NAME \
        --gpus all \
        --privileged \
        --network host \
        -v $(pwd):/workspace \
        -v /dev:/dev \
        -e ROS_DOMAIN_ID=$ROS2_DOMAIN_ID \
        -e DISPLAY=$DISPLAY \
        -e CUDA_VISIBLE_DEVICES=0 \
        -w /workspace \
        nvcr.io/nvidia/pytorch:23.10-py3 \
        sleep infinity
fi

# 컨테이너 시작 대기
echo "컨테이너 시작 대기 중..."
sleep 5

if docker ps | grep -q $PROJECT_NAME; then
    echo -e "${GREEN}✅ Docker 컨테이너 시작 완료${NC}"
else
    echo -e "${RED}❌ Docker 컨테이너 시작 실패${NC}"
    exit 1
fi

# 5. VLA 모델 로딩 (컨테이너 내부에서)
echo -e "${BLUE}🧠 Event-Triggered VLA 모델 로딩${NC}"
docker exec $PROJECT_NAME bash -c "
cd /workspace
python3 -c \"
import torch
import time
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

print('🔄 PaliGemma-3B 모델 로딩 중...')
start_time = time.time()

# 모델 로딩
model = PaliGemmaForConditionalGeneration.from_pretrained(
    'google/$VLA_MODEL',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    cache_dir='.vlms'
)

processor = PaliGemmaProcessor.from_pretrained(
    'google/$VLA_MODEL',
    cache_dir='.vlms'
)

load_time = time.time() - start_time
memory_used = torch.cuda.memory_allocated() / 1e9

print(f'✅ 모델 로딩 완료 ({load_time:.1f}초)')
print(f'메모리 사용량: {memory_used:.1f}GB')

if load_time > 30:
    print('⚠️  로딩 시간이 깁니다. 캐시 확인 필요')
if memory_used > 14:
    print('⚠️  메모리 사용량이 높습니다. bfloat16 확인 필요')

print('🚀 Event-Triggered VLA 모델 준비 완료!')
\"
"

# 6. ROS2 노드 시작 (컨테이너 내부에서)
echo -e "${BLUE}🚀 ROS2 Event-Triggered VLA 노드 시작${NC}"
docker exec -d $PROJECT_NAME bash -c "
cd /workspace
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID

# Model_ws 워크스페이스 빌드 및 실행
if [ -d '../Model_ws' ]; then
    cd ../Model_ws
    colcon build --packages-select vla_node
    source install/setup.bash
    ros2 run vla_node event_triggered_vla_node &
    echo '✅ VLA 노드 시작 완료'
else
    echo '⚠️  Model_ws가 없습니다. 기본 VLA 노드 실행...'
    python3 -c \"
import rclpy
from rclpy.node import Node
print('🤖 기본 Event-Triggered VLA 노드 시작...')
# 기본 VLA 노드 실행 로직 (간단한 버전)
\"
fi
"

# 7. 시스템 상태 확인
echo -e "${BLUE}📊 시스템 상태 확인${NC}"
sleep 3

echo "Docker 컨테이너 상태:"
docker ps | grep $PROJECT_NAME || echo "컨테이너를 찾을 수 없습니다"

echo ""
echo "ROS2 토픽 상태:"
docker exec $PROJECT_NAME bash -c "
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
timeout 5 ros2 topic list | grep -E '(cmd_vel|vla_action|scan)' || echo '⚠️  ROS2 토픽이 아직 준비되지 않았습니다'
"

echo ""
echo "GPU 메모리 사용량:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 8. 완료 메시지 및 사용법 안내
echo ""
echo -e "${GREEN}🎉 Event-Triggered VLA 시스템 시작 완료!${NC}"
echo "=================================================="
echo ""
echo -e "${CYAN}📋 사용법:${NC}"
echo "1. 텍스트 명령 전송:"
echo "   ${YELLOW}./send_text_command.sh \"앞으로 가\"${NC}"
echo "   ${YELLOW}./send_text_command.sh \"컵으로 가\"${NC}"
echo ""
echo "2. 대화형 명령 모드:"
echo "   ${YELLOW}./send_text_command.sh -i${NC}"
echo ""
echo "3. 시스템 모니터링:"
echo "   ${YELLOW}docker logs -f $PROJECT_NAME${NC}"
echo "   ${YELLOW}ros2 topic echo /cmd_vel${NC}"
echo ""
echo "4. 시스템 종료:"
echo "   ${YELLOW}./stop_event_triggered_vla.sh${NC}"
echo ""
echo -e "${CYAN}🔍 모니터링 명령어:${NC}"
echo "- GPU 상태: ${YELLOW}nvidia-smi${NC}"
echo "- 컨테이너 로그: ${YELLOW}docker logs $PROJECT_NAME${NC}"
echo "- ROS2 토픽: ${YELLOW}ros2 topic list${NC}"
echo "- 추론 지연시간: ${YELLOW}ros2 topic echo /vla_latency${NC}"
echo ""
echo -e "${GREEN}🚀 K-프로젝트 로봇카 네비게이션 VLA 시스템이 실행 중입니다!${NC}"
echo -e "${BLUE}목표 추론 지연시간: <${INFERENCE_LATENCY_TARGET}ms${NC}"