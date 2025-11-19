#!/bin/bash
# minuum 계정으로 RoboVLMs 저장소 푸시 설정 스크립트

cd "$(dirname "$0")"

echo "=========================================="
echo "RoboVLMs 저장소 minuum 계정 푸시 설정"
echo "=========================================="
echo ""

# 방법 1: 토큰을 직접 입력받아 URL에 포함
if [ -z "$1" ]; then
    echo "사용법 1: 토큰을 인자로 전달"
    echo "  ./setup_minuum_push.sh <YOUR_TOKEN>"
    echo ""
    echo "사용법 2: 토큰을 직접 입력"
    echo "  ./setup_minuum_push.sh"
    echo ""
    read -sp "minuum 계정의 Personal Access Token을 입력하세요: " TOKEN
    echo ""
else
    TOKEN="$1"
fi

if [ -z "$TOKEN" ]; then
    echo "오류: 토큰이 입력되지 않았습니다."
    exit 1
fi

# 원격 URL에 토큰 포함
git remote set-url origin "https://${TOKEN}@github.com/minuum/RoboVLMs.git"

echo "✅ 원격 URL이 업데이트되었습니다."
echo ""
echo "이제 다음 명령어로 푸시할 수 있습니다:"
echo "  git push origin main"
echo ""
echo "⚠️  주의: 토큰이 URL에 포함되어 있으므로 .git/config 파일을 공유하지 마세요!"

