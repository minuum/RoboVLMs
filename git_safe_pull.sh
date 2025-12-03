#!/bin/bash

# 🛡️ K-프로젝트 안전한 Git Pull 스크립트
# 목적: 로봇 환경에서 git pull 시 중요 파일 손실 방지

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
CRITICAL_FILES=(
    "jetson_quick_start.sh"
    "launch_event_triggered_vla.sh" 
    "send_text_command.sh"
    "stop_event_triggered_vla.sh"
    "docker-compose.yml"
    "configs/k_project/"
    "../Robo+/K-프로젝트/"
    "../Model_ws/src/vla_node/"
    "models_cache/"
    ".vlms/"
)

echo -e "${CYAN}🛡️ K-프로젝트 안전한 Git Pull${NC}"
echo "=================================================="

# 도움말 함수
show_help() {
    echo -e "${BLUE}사용법:${NC}"
    echo "  ./git_safe_pull.sh                 # 안전한 pull (권장)"
    echo "  ./git_safe_pull.sh --force         # 강제 pull (주의)"
    echo "  ./git_safe_pull.sh --backup-only   # 백업만 수행"
    echo "  ./git_safe_pull.sh --restore       # 백업에서 복구"
    echo ""
    echo -e "${YELLOW}⚠️ 주의사항:${NC}"
    echo "- 항상 VLA 시스템을 먼저 종료하세요: ./stop_event_triggered_vla.sh"
    echo "- 중요한 작업 중이라면 백업을 먼저 수행하세요"
    echo "- 문제 발생 시 즉시 복구할 수 있도록 백업을 보관하세요"
}

# VLA 시스템 상태 확인
check_vla_system() {
    if docker ps | grep -q "k_project_event_vla"; then
        echo -e "${RED}❌ VLA 시스템이 실행 중입니다!${NC}"
        echo "먼저 시스템을 안전하게 종료하세요:"
        echo "  ${YELLOW}./stop_event_triggered_vla.sh${NC}"
        echo ""
        read -p "시스템을 자동으로 종료하시겠습니까? (y/N): " auto_stop
        if [[ $auto_stop =~ ^[Yy]$ ]]; then
            echo "VLA 시스템 종료 중..."
            ./stop_event_triggered_vla.sh
        else
            echo "수동으로 시스템을 종료한 후 다시 실행하세요."
            exit 1
        fi
    fi
}

# 중요 파일 백업
backup_critical_files() {
    echo -e "${BLUE}📦 중요 파일 백업 중...${NC}"
    
    mkdir -p "$BACKUP_DIR"
    
    for file in "${CRITICAL_FILES[@]}"; do
        if [ -e "$file" ]; then
            echo "백업 중: $file"
            
            # 디렉토리 구조 유지하면서 복사
            parent_dir=$(dirname "$file")
            mkdir -p "$BACKUP_DIR/$parent_dir"
            
            if [ -d "$file" ]; then
                cp -r "$file" "$BACKUP_DIR/$file"
            else
                cp "$file" "$BACKUP_DIR/$file"
            fi
        else
            echo -e "${YELLOW}⚠️ 파일이 없습니다: $file${NC}"
        fi
    done
    
    # 백업 정보 파일 생성
    cat > "$BACKUP_DIR/backup_info.txt" << EOF
K-프로젝트 백업 정보
==================
백업 시간: $(date)
Git 브랜치: $(git branch --show-current)
Git 커밋: $(git rev-parse HEAD)
Git 상태:
$(git status --porcelain)

백업된 파일들:
$(find "$BACKUP_DIR" -type f | sort)
EOF
    
    echo -e "${GREEN}✅ 백업 완료: $BACKUP_DIR${NC}"
}

# Git 상태 확인
check_git_status() {
    echo -e "${BLUE}📊 Git 상태 확인${NC}"
    
    # 스테이징되지 않은 변경사항 확인
    if ! git diff --quiet; then
        echo -e "${YELLOW}⚠️ 스테이징되지 않은 변경사항이 있습니다:${NC}"
        git diff --name-only
        echo ""
        
        read -p "변경사항을 stash하시겠습니까? (y/N): " stash_changes
        if [[ $stash_changes =~ ^[Yy]$ ]]; then
            git stash push -m "안전한 pull 전 백업 - $(date)"
            echo -e "${GREEN}✅ 변경사항을 stash했습니다${NC}"
        fi
    fi
    
    # 스테이징된 변경사항 확인
    if ! git diff --cached --quiet; then
        echo -e "${YELLOW}⚠️ 스테이징된 변경사항이 있습니다:${NC}"
        git diff --cached --name-only
        echo ""
        
        read -p "변경사항을 커밋하시겠습니까? (y/N): " commit_changes
        if [[ $commit_changes =~ ^[Yy]$ ]]; then
            echo "커밋 메시지를 입력하세요:"
            read -p "메시지: " commit_msg
            git commit -m "$commit_msg"
            echo -e "${GREEN}✅ 변경사항을 커밋했습니다${NC}"
        fi
    fi
}

# 안전한 pull 수행
safe_pull() {
    echo -e "${BLUE}⬇️ 안전한 Git Pull 수행${NC}"
    
    # 원격 저장소 fetch
    echo "원격 저장소 정보 가져오는 중..."
    git fetch origin
    
    # 로컬과 원격 비교
    local current_branch=$(git branch --show-current)
    local behind_count=$(git rev-list --count HEAD..origin/$current_branch)
    local ahead_count=$(git rev-list --count origin/$current_branch..HEAD)
    
    echo "현재 브랜치: $current_branch"
    echo "원격보다 뒤처짐: $behind_count 커밋"
    echo "원격보다 앞섬: $ahead_count 커밋"
    
    if [ "$behind_count" -eq "0" ]; then
        echo -e "${GREEN}✅ 이미 최신 상태입니다${NC}"
        return 0
    fi
    
    # Pull 충돌 시뮬레이션 (dry-run)
    echo "Pull 충돌 가능성 검사 중..."
    if git merge-tree $(git merge-base HEAD origin/$current_branch) HEAD origin/$current_branch | grep -q "<<<<<<< "; then
        echo -e "${RED}⚠️ 병합 충돌이 예상됩니다!${NC}"
        echo "충돌 파일들:"
        git merge-tree $(git merge-base HEAD origin/$current_branch) HEAD origin/$current_branch | grep -E "^(\+\+\+|---)" | sort | uniq
        echo ""
        
        read -p "계속 진행하시겠습니까? (y/N): " continue_pull
        if [[ ! $continue_pull =~ ^[Yy]$ ]]; then
            echo "Pull을 취소했습니다."
            return 1
        fi
    fi
    
    # 실제 pull 수행
    echo "Git pull 수행 중..."
    if git pull origin $current_branch; then
        echo -e "${GREEN}✅ Git pull 성공${NC}"
    else
        echo -e "${RED}❌ Git pull 실패${NC}"
        echo ""
        echo "복구 방법:"
        echo "1. 백업에서 복구: ${YELLOW}./git_safe_pull.sh --restore${NC}"
        echo "2. 충돌 해결 후 계속: ${YELLOW}git status${NC} 확인 후 충돌 해결"
        echo "3. pull 취소: ${YELLOW}git merge --abort${NC}"
        return 1
    fi
}

# 백업에서 복구
restore_from_backup() {
    echo -e "${BLUE}🔄 백업에서 복구${NC}"
    
    # 가장 최근 백업 찾기
    local latest_backup=$(ls -dt backup_* 2>/dev/null | head -1)
    
    if [ -z "$latest_backup" ]; then
        echo -e "${RED}❌ 백업을 찾을 수 없습니다${NC}"
        return 1
    fi
    
    echo "복구할 백업: $latest_backup"
    
    if [ -f "$latest_backup/backup_info.txt" ]; then
        echo "백업 정보:"
        cat "$latest_backup/backup_info.txt"
        echo ""
    fi
    
    read -p "이 백업에서 복구하시겠습니까? (y/N): " confirm_restore
    if [[ ! $confirm_restore =~ ^[Yy]$ ]]; then
        echo "복구를 취소했습니다."
        return 1
    fi
    
    # 파일 복구
    for file in "${CRITICAL_FILES[@]}"; do
        if [ -e "$latest_backup/$file" ]; then
            echo "복구 중: $file"
            
            # 기존 파일 백업 (복구 실패에 대비)
            if [ -e "$file" ]; then
                mv "$file" "$file.before_restore"
            fi
            
            # 복구 수행
            parent_dir=$(dirname "$file")
            mkdir -p "$parent_dir"
            
            if [ -d "$latest_backup/$file" ]; then
                cp -r "$latest_backup/$file" "$file"
            else
                cp "$latest_backup/$file" "$file"
            fi
        fi
    done
    
    echo -e "${GREEN}✅ 복구 완료${NC}"
    echo ""
    echo "복구 후 확인 사항:"
    echo "1. 파일 권한 확인: ${YELLOW}chmod +x *.sh${NC}"
    echo "2. VLA 시스템 테스트: ${YELLOW}./jetson_quick_start.sh${NC}"
}

# 메인 실행 로직
main() {
    case "$1" in
        "--help"|"-h")
            show_help
            exit 0
            ;;
        "--backup-only")
            backup_critical_files
            echo -e "${GREEN}✅ 백업만 완료했습니다${NC}"
            exit 0
            ;;
        "--restore")
            restore_from_backup
            exit 0
            ;;
        "--force")
            echo -e "${RED}⚠️ 강제 모드 - 백업 없이 진행합니다${NC}"
            check_vla_system
            check_git_status
            safe_pull
            ;;
        "")
            # 기본 안전 모드
            echo -e "${GREEN}🛡️ 안전 모드로 진행합니다${NC}"
            check_vla_system
            backup_critical_files
            check_git_status
            if safe_pull; then
                echo ""
                echo -e "${GREEN}🎉 안전한 Git Pull 완료!${NC}"
                echo ""
                echo "다음 단계:"
                echo "1. 파일 권한 확인: ${YELLOW}chmod +x RoboVLMs/*.sh${NC}"
                echo "2. 시스템 테스트: ${YELLOW}cd RoboVLMs && ./jetson_quick_start.sh${NC}"
                echo "3. VLA 시스템 시작: ${YELLOW}./launch_event_triggered_vla.sh${NC}"
            else
                echo ""
                echo -e "${RED}Pull 실패 - 복구 옵션:${NC}"
                echo "1. 백업 복구: ${YELLOW}./git_safe_pull.sh --restore${NC}"
                echo "2. 수동 해결 후 재시도"
            fi
            ;;
        *)
            echo -e "${RED}❌ 알 수 없는 옵션: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 시작 메시지
echo -e "${MAGENTA}📍 현재 위치: $(pwd)${NC}"
echo -e "${MAGENTA}📍 Git 브랜치: $(git branch --show-current)${NC}"
echo -e "${MAGENTA}📍 Git 상태: $(git status --porcelain | wc -l) 변경사항${NC}"
echo ""

# 메인 함수 실행
main "$@"