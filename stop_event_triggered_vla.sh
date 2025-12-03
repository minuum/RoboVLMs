#!/bin/bash

# 🛑 K-프로젝트 Event-Triggered VLA 시스템 안전 종료 스크립트
# 목적: VLA 시스템을 안전하게 종료하고 리소스 정리

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_NAME="k_project_event_vla"
ROS2_DOMAIN_ID=42

echo -e "${BLUE}🛑 K-프로젝트 Event-Triggered VLA 시스템 종료${NC}"
echo "=================================================="

# 1. ROS2 노드 종료
echo -e "${BLUE}🤖 ROS2 노드 종료 중...${NC}"
if docker ps | grep -q $PROJECT_NAME; then
    docker exec $PROJECT_NAME bash -c "
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
        
        # VLA 노드 프로세스 찾아서 종료
        pkill -f 'event_triggered_vla_node' || true
        pkill -f 'vla_node' || true
        pkill -f 'ros2 run vla_node' || true
        
        echo '✅ ROS2 노드 종료 완료'
    "
else
    echo -e "${YELLOW}⚠️  컨테이너가 실행되지 않았습니다${NC}"
fi

# 2. GPU 메모리 정리
echo -e "${BLUE}🎮 GPU 메모리 정리 중...${NC}"
if docker ps | grep -q $PROJECT_NAME; then
    docker exec $PROJECT_NAME bash -c "
        python3 -c \"
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU 메모리 캐시 정리 완료')
gc.collect()
print('✅ Python 메모리 정리 완료')
\" 2>/dev/null || echo '⚠️  GPU 메모리 정리 실패'
    "
fi

# 3. Docker 컨테이너 종료
echo -e "${BLUE}🐳 Docker 컨테이너 종료 중...${NC}"

# 컨테이너 상태 확인
if docker ps | grep -q $PROJECT_NAME; then
    echo "컨테이너 종료 중..."
    
    # Graceful shutdown (10초 대기)
    timeout 10 docker stop $PROJECT_NAME || {
        echo -e "${YELLOW}⚠️  Graceful shutdown 실패. 강제 종료합니다.${NC}"
        docker kill $PROJECT_NAME
    }
    
    # 컨테이너 제거
    docker rm $PROJECT_NAME || echo -e "${YELLOW}⚠️  컨테이너 제거 실패${NC}"
    
    echo -e "${GREEN}✅ Docker 컨테이너 종료 완료${NC}"
else
    echo -e "${YELLOW}⚠️  실행 중인 컨테이너가 없습니다${NC}"
fi

# 4. Docker Compose 종료 (있는 경우)
if [ -f "docker-compose.yml" ]; then
    echo -e "${BLUE}🐳 Docker Compose 서비스 종료 중...${NC}"
    docker-compose down || echo -e "${YELLOW}⚠️  Docker Compose 종료 실패${NC}"
fi

# 5. 네트워크 정리
echo -e "${BLUE}🌐 네트워크 리소스 정리 중...${NC}"
# 사용하지 않는 Docker 네트워크 정리
docker network prune -f > /dev/null 2>&1 || true
echo -e "${GREEN}✅ 네트워크 정리 완료${NC}"

# 6. 최종 상태 확인
echo -e "${BLUE}📊 시스템 상태 확인${NC}"

echo "Docker 컨테이너 상태:"
if docker ps | grep -q $PROJECT_NAME; then
    echo -e "${RED}❌ 컨테이너가 여전히 실행 중입니다${NC}"
    docker ps | grep $PROJECT_NAME
else
    echo -e "${GREEN}✅ 컨테이너가 완전히 종료되었습니다${NC}"
fi

echo ""
echo "GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | while read line; do
    echo "  $line"
done

echo ""
echo "ROS2 프로세스 상태:"
if pgrep -f "ros2\|vla_node" > /dev/null; then
    echo -e "${YELLOW}⚠️  일부 ROS2 프로세스가 여전히 실행 중일 수 있습니다${NC}"
    pgrep -f "ros2\|vla_node" | head -5
else
    echo -e "${GREEN}✅ ROS2 프로세스가 모두 종료되었습니다${NC}"
fi

# 7. 종료 완료 메시지
echo ""
echo -e "${GREEN}🎉 Event-Triggered VLA 시스템 종료 완료!${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}📋 다음 단계:${NC}"
echo "1. 시스템 재시작:"
echo "   ${YELLOW}./launch_event_triggered_vla.sh${NC}"
echo ""
echo "2. 로그 확인 (문제가 있었던 경우):"
echo "   ${YELLOW}docker logs $PROJECT_NAME${NC} (종료 전에만 가능)"
echo ""
echo "3. 완전한 정리 (필요한 경우):"
echo "   ${YELLOW}docker system prune -f${NC}"
echo "   ${YELLOW}rm -rf .vlms/models--google--*${NC} (모델 캐시 삭제)"
echo ""

# 8. 안전 종료 확인
sleep 2
final_check=$(docker ps | grep $PROJECT_NAME | wc -l)
if [ "$final_check" -eq "0" ]; then
    echo -e "${GREEN}✅ 안전 종료 검증 완료${NC}"
    exit 0
else
    echo -e "${RED}❌ 완전한 종료에 실패했습니다${NC}"
    echo "수동으로 다음 명령어를 실행하세요:"
    echo "  ${YELLOW}docker kill $PROJECT_NAME && docker rm $PROJECT_NAME${NC}"
    exit 1
fi