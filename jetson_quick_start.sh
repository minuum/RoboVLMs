#!/bin/bash

# 🚀 K-프로젝트 Jetson 환경 빠른 시작 스크립트
# 목적: NVIDIA Jetson에서 RoboVLMs 로봇카 네비게이션 실험을 즉시 시작

set -e  # 에러 발생시 중단

echo "🎯 K-프로젝트 RoboVLMs Jetson 빠른 시작"
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. 시스템 정보 확인
echo -e "${BLUE}📊 시스템 정보 확인${NC}"
echo "CUDA 버전: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "GPU 메모리: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "사용 가능한 메모리: $(free -m | awk 'NR==2{printf "%.0f MB (%.1f%%)", $7, $7*100/$2}')"
echo ""

# 2. 환경 변수 설정
echo -e "${BLUE}🔧 환경 변수 설정${NC}"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_DTYPE=bfloat16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo "✅ CUDA 환경 설정 완료"

# 3. ROS2 환경 활성화
echo -e "${BLUE}🤖 ROS2 환경 활성화${NC}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✅ ROS2 Humble 환경 활성화"
else
    echo -e "${RED}❌ ROS2 Humble이 설치되지 않았습니다${NC}"
    exit 1
fi

# 4. Python 환경 확인
echo -e "${BLUE}🐍 Python 환경 확인${NC}"
if command -v conda &> /dev/null; then
    echo "Conda 사용 가능"
    # robovlms 환경이 있으면 활성화
    if conda env list | grep -q robovlms; then
        echo "robovlms 환경 활성화 중..."
        conda activate robovlms
    fi
elif [ -d "venv" ]; then
    echo "Python 가상환경 활성화 중..."
    source venv/bin/activate
fi

python --version
echo "✅ Python 환경 준비 완료"

# 5. 실행 권한 확인
echo -e "${BLUE}📋 실행 권한 확인${NC}"
chmod +x *.sh 2>/dev/null || echo "⚠️ 일부 스크립트 파일이 없습니다"
echo "✅ 스크립트 실행 권한 설정 완료"

# 6. GPU 및 모델 테스트
echo -e "${BLUE}🧪 GPU 및 모델 로딩 테스트${NC}"
python3 -c "
import torch
print(f'CUDA 사용가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 장치: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ CUDA를 사용할 수 없습니다')
    exit(1)
"

echo ""
echo -e "${YELLOW}⚠️  PaliGemma-3B 모델 로딩 테스트 (시간이 걸릴 수 있습니다)${NC}"
python3 -c "
try:
    from transformers import PaliGemmaForConditionalGeneration
    import torch
    
    print('PaliGemma 모델 로딩 중...')
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        'google/paligemma-3b-mix-224',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    
    memory_used = torch.cuda.memory_allocated() / 1e9
    print(f'✅ PaliGemma-3B 로딩 성공')
    print(f'메모리 사용량: {memory_used:.1f}GB')
    
    if memory_used > 14:
        print('⚠️  메모리 사용량이 높습니다. bfloat16 모드 확인 필요')
    
    del model
    torch.cuda.empty_cache()
    print('✅ 모델 테스트 완료 및 메모리 정리')
    
except Exception as e:
    print(f'❌ 모델 로딩 실패: {e}')
    print('Hugging Face 캐시를 정리하고 다시 시도하세요')
    print('rm -rf ~/.cache/huggingface/')
    exit(1)
"

# 7. 누락된 파일 확인
echo -e "${BLUE}📁 필수 파일 확인${NC}"
missing_files=0

# 핵심 스크립트 확인
if [ ! -f "launch_event_triggered_vla.sh" ]; then
    echo -e "${RED}❌ launch_event_triggered_vla.sh 누락${NC}"
    missing_files=$((missing_files + 1))
fi

if [ ! -f "send_text_command.sh" ]; then
    echo -e "${RED}❌ send_text_command.sh 누락${NC}"  
    missing_files=$((missing_files + 1))
fi

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml 누락${NC}"
    missing_files=$((missing_files + 1))
fi

# 8. 완료 및 다음 단계 안내
echo ""
if [ $missing_files -eq 0 ]; then
    echo -e "${GREEN}🎉 Jetson 환경 준비 완료!${NC}"
else
    echo -e "${YELLOW}⚠️  Jetson 환경 부분 준비 완료 ($missing_files개 파일 누락)${NC}"
fi
echo "=================================================="
echo ""
echo -e "${BLUE}📋 다음 단계:${NC}"

if [ $missing_files -eq 0 ]; then
    echo "1. Event-Triggered VLA 시스템 시작:"
    echo "   ${YELLOW}./launch_event_triggered_vla.sh${NC}"
    echo ""
    echo "2. 기본 명령어 테스트:"
    echo "   ${YELLOW}./send_text_command.sh \"앞으로 가\"${NC}"
    echo "   ${YELLOW}./send_text_command.sh \"멈춰\"${NC}"
    echo ""
    echo "3. 대화형 명령 모드:"
    echo "   ${YELLOW}./send_text_command.sh -i${NC}"
else
    echo "1. 누락된 스크립트 파일들을 먼저 복구하세요:"
    echo "   - launch_event_triggered_vla.sh"
    echo "   - send_text_command.sh"  
    echo "   - docker-compose.yml"
    echo ""
    echo "2. 복구 후 시스템을 다시 시작하세요"
fi

echo ""
echo "4. 시스템 모니터링:"
echo "   ${YELLOW}docker ps${NC}"
echo "   ${YELLOW}ros2 topic list${NC}"
echo ""
echo -e "${BLUE}📖 자세한 가이드:${NC}"
echo "- 전체 맥락: ../Robo+/K-프로젝트/RoboVLMs_실험설계_대화요약_20250725.md"
echo "- 다음 단계: ../Robo+/K-프로젝트/다음단계_액션아이템.md"
echo ""

if [ $missing_files -eq 0 ]; then  
    echo -e "${GREEN}🚀 K-프로젝트 로봇카 네비게이션 실험을 시작하세요!${NC}"
else
    echo -e "${YELLOW}🔧 누락된 파일들을 복구한 후 실험을 시작하세요!${NC}"
fi