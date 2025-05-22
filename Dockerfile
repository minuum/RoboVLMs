# 기존 robovla:latest 이미지를 기반으로 시작
FROM robovla:latest

# USER를 root로 설정하여 권한 문제 없이 패키지 설치
USER root

# 환경 변수 설정 (비대화형 설치를 위해)
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 업데이트 및 필수 유틸리티 설치 (curl, gnupg, lsb-release 등)
# X11 및 Qt 관련 라이브러리도 여기에 포함
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    libxcb-randr0-dev \
    libxcb-xtest0-dev \
    libxcb-xinerama0-dev \
    libxcb-shape0-dev \
    libxcb-xkb-dev \
    libqt5x11extras5 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    # Python3 pip (Poetry 및 기타 Python 패키지 설치에 필요할 수 있음)
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble 설치를 위한 단일 RUN 레이어
RUN apt-get update && \
    # ROS 2 GPG 키 추가
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    # ROS 2 저장소 추가
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    # 패키지 목록 업데이트 (저장소 추가 후 필수)
    apt-get update && \
    # ROS 2 패키지 설치
    apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-cv-bridge \
        python3-opencv \
        ros-humble-audio-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Poetry 실행 경로 환경 변수 설정 (robovla:latest에 이미 유사한 설정이 있다면 중복 확인 필요)
ENV PATH="/root/.local/bin:$PATH"

# PyTorch/CUDA 테스트 스크립트 복사 및 실행 권한 부여
COPY pytorch_cuda_test.py /usr/local/bin/pytorch_cuda_test.py
RUN chmod +x /usr/local/bin/pytorch_cuda_test.py

# .bashrc에 ROS 2 환경 설정, Poetry 프로젝트 경로 이동 및 활성화 안내, 테스트 alias 추가
RUN echo "" >> /root/.bashrc && \
    echo "# ROS 2 Humble environment setup" >> /root/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=20" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# RoboVLMs Poetry Project Helper" >> /root/.bashrc && \
    echo "alias activate_robovlms='cd /workspace/RoboVLMs && source \$(poetry env info --path)/bin/activate'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'activate_robovlms' to navigate to /workspace/RoboVLMs and activate Poetry environment.\"" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# PyTorch & CUDA Test Alias" >> /root/.bashrc && \
    echo "alias torch_cuda_test='python3 /usr/local/bin/pytorch_cuda_test.py'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'torch_cuda_test' to check PyTorch and CUDA setup (after activating Poetry env if PyTorch is managed by Poetry).\"" >> /root/.bashrc

# 컨테이너 시작 시 실행될 기본 명령어
CMD ["bash"]