#!/bin/bash

# 📝 K-프로젝트 텍스트 명령 전송 스크립트
# 목적: Event-Triggered VLA 시스템에 네비게이션 명령 전송

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 환경 변수
PROJECT_NAME="k_project_event_vla"
ROS2_DOMAIN_ID=42

# 도움말 함수
show_help() {
    echo -e "${CYAN}📝 K-프로젝트 텍스트 명령 전송기${NC}"
    echo "=================================================="
    echo ""
    echo -e "${BLUE}사용법:${NC}"
    echo "  ./send_text_command.sh \"명령어\"          # 단일 명령 전송"
    echo "  ./send_text_command.sh -i               # 대화형 모드"
    echo "  ./send_text_command.sh -h               # 도움말"
    echo ""
    echo -e "${BLUE}예시 명령어:${NC}"
    echo "  ${YELLOW}./send_text_command.sh \"앞으로 2미터 가\"${NC}"
    echo "  ${YELLOW}./send_text_command.sh \"오른쪽으로 회전해\"${NC}"
    echo "  ${YELLOW}./send_text_command.sh \"빨간 원뿔을 찾아가\"${NC}"
    echo "  ${YELLOW}./send_text_command.sh \"멈춰\"${NC}"
    echo ""
    echo -e "${BLUE}Calvin 스타일 Sequential Task:${NC}"
    echo "  1. \"앞으로 2미터 이동해\""
    echo "  2. \"오른쪽으로 90도 회전해\""
    echo "  3. \"빨간 원뿔을 찾아가\""
    echo "  4. \"장애물을 피해서 벽까지 가\""
    echo "  5. \"출발점으로 돌아와\""
}

# ROS2 환경 확인 함수
check_ros2_environment() {
    if ! command -v ros2 &> /dev/null; then
        echo -e "${RED}❌ ROS2가 설치되지 않았습니다${NC}"
        return 1
    fi
    
    # ROS2 환경 설정
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
    fi
    
    return 0
}

# VLA 시스템 상태 확인 함수
check_vla_system() {
    echo -e "${BLUE}🔍 VLA 시스템 상태 확인 중...${NC}"
    
    # Docker 컨테이너 상태 확인
    if ! docker ps | grep -q $PROJECT_NAME; then
        echo -e "${RED}❌ Event-Triggered VLA 시스템이 실행되지 않았습니다${NC}"
        echo "먼저 VLA 시스템을 시작하세요:"
        echo "  ${YELLOW}./launch_event_triggered_vla.sh${NC}"
        return 1
    fi
    
    # ROS2 토픽 확인 (컨테이너 내부에서)
    local topic_check=$(docker exec $PROJECT_NAME bash -c "
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
        timeout 3 ros2 topic list 2>/dev/null | grep -E '(vla_command|cmd_vel)' | wc -l
    " 2>/dev/null || echo "0")
    
    if [ "$topic_check" -eq "0" ]; then
        echo -e "${YELLOW}⚠️  ROS2 토픽이 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ VLA 시스템이 정상적으로 실행 중입니다${NC}"
    return 0
}

# 명령 전송 함수
send_command() {
    local command="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo -e "${BLUE}📤 명령 전송 중...${NC}"
    echo "명령어: ${MAGENTA}\"$command\"${NC}"
    echo "시각: $timestamp"
    echo ""
    
    # ROS2 토픽으로 명령 전송 (컨테이너 내부에서)
    local result=$(docker exec $PROJECT_NAME bash -c "
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
        
        # VLA 명령 토픽으로 전송
        timeout 10 ros2 topic pub --once /vla_text_command std_msgs/msg/String \"data: '$command'\" 2>/dev/null
        echo \$?
    " 2>/dev/null || echo "1")
    
    if [ "$result" = "0" ]; then
        echo -e "${GREEN}✅ 명령이 성공적으로 전송되었습니다${NC}"
        
        # 액션 결과 대기 (선택적)
        echo -e "${BLUE}🤖 VLA 처리 중... (최대 5초 대기)${NC}"
        
        # 액션 결과 확인
        docker exec $PROJECT_NAME bash -c "
            source /opt/ros/humble/setup.bash
            export ROS_DOMAIN_ID=$ROS2_DOMAIN_ID
            
            echo '예상 액션:'
            timeout 5 ros2 topic echo /cmd_vel --once 2>/dev/null | head -10 || echo '⚠️  액션 결과를 받지 못했습니다'
        " 2>/dev/null
        
        echo ""
        echo -e "${GREEN}🎯 명령 처리 완료!${NC}"
        
    else
        echo -e "${RED}❌ 명령 전송 실패${NC}"
        echo "문제 해결 방법:"
        echo "1. VLA 시스템이 실행 중인지 확인: ${YELLOW}docker ps | grep $PROJECT_NAME${NC}"
        echo "2. ROS2 토픽 상태 확인: ${YELLOW}ros2 topic list${NC}"
        echo "3. 시스템 재시작: ${YELLOW}./launch_event_triggered_vla.sh${NC}"
        return 1
    fi
}

# 대화형 모드 함수
interactive_mode() {
    echo -e "${CYAN}🎮 대화형 명령 모드 시작${NC}"
    echo "=================================================="
    echo ""
    echo "사용법:"
    echo "- 네비게이션 명령을 입력하세요"
    echo "- 'quit' 또는 'exit'로 종료하세요"
    echo "- 'help'로 예시 명령어를 확인하세요"
    echo ""
    
    while true; do
        echo -ne "${BLUE}VLA> ${NC}"
        read -r user_input
        
        # 입력 처리
        case "$user_input" in
            "quit"|"exit"|"종료")
                echo -e "${GREEN}대화형 모드를 종료합니다.${NC}"
                break
                ;;
            "help"|"도움말")
                echo ""
                echo -e "${YELLOW}💡 예시 명령어:${NC}"
                echo "  - 앞으로 가"
                echo "  - 뒤로 가"  
                echo "  - 왼쪽으로 회전해"
                echo "  - 오른쪽으로 회전해"
                echo "  - 멈춰"
                echo "  - 빨간 원뿔을 찾아가"
                echo "  - 장애물을 피해서 가"
                echo "  - 출발점으로 돌아와"
                echo ""
                ;;
            "")
                # 빈 입력 무시
                continue
                ;;
            *)
                echo ""
                send_command "$user_input"
                echo ""
                ;;
        esac
    done
}

# 메인 실행 로직
main() {
    # 인자 처리
    case "$1" in
        "-h"|"--help"|"help")
            show_help
            exit 0
            ;;
        "-i"|"--interactive")
            # 대화형 모드
            if ! check_ros2_environment; then
                exit 1
            fi
            
            if ! check_vla_system; then
                exit 1
            fi
            
            interactive_mode
            exit 0
            ;;
        "")
            echo -e "${RED}❌ 명령어를 입력해주세요${NC}"
            echo ""
            show_help
            exit 1
            ;;
        *)
            # 단일 명령 모드
            local command="$1"
            
            if ! check_ros2_environment; then
                exit 1
            fi
            
            if ! check_vla_system; then
                exit 1
            fi
            
            send_command "$command"
            exit 0
            ;;
    esac
}

# 스크립트 시작
echo -e "${CYAN}📝 K-프로젝트 텍스트 명령 전송기${NC}"
echo "=================================================="

# 메인 함수 실행
main "$@"