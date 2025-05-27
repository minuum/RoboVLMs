FROM robovla:latest

# 루트 사용자로 실행
USER root
ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# PyTorch + torchaudio 설치
# 예시: CUDA 11.8을 기준으로 설치
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.17.2 \
    torchaudio==2.3.0

# 그 외 필수 패키지
RUN pip install --no-cache-dir \
    transformers>=4.33 \
    accelerate \
    scipy

# TTS 모델 사전 다운로드
RUN python3 -c "from transformers import VitsModel, AutoTokenizer; \
               VitsModel.from_pretrained('Matthijs/mms-tts-kor'); \
               AutoTokenizer.from_pretrained('Matthijs/mms-tts-kor')"

# 실행 스크립트 복사
COPY run_tts.py .

# 기본 명령어 설정
CMD ["bash"]
