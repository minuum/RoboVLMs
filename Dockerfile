# 1. 기본 이미지 설정
FROM robovla:latest

# 2. USER를 root로 설정하여 권한 문제 없이 패키지 설치
USER root

# 3. 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONUNBUFFERED=1
# Poetry 실행 경로 환경 변수 설정 (기존 Dockerfile 내용 유지)
ENV PATH="/root/.local/bin:$PATH"
# Transformers 캐시 경로 변경 (선택 사항, HF_HOME 사용 권장)
ENV HF_HOME="/root/.cache/huggingface" 
# ENV TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers" # 이전 방식


# 4. ROS 2 저장소 설정 및 기본 유틸리티, 핵심 ROS 패키지 설치 (하나의 RUN 레이어로 통합)
#    - curl, gnupg, lsb-release: ROS 저장소 추가에 필요
#    - python3-pip: Python 패키지 설치 도구
#    - ros-humble-desktop: ROS 2 데스크톱 전체 설치 (기본 이미지에 이미 있다면 일부 중복 가능)
#    - ros-dev-tools: colcon 등 ROS 2 개발 도구
#    - python3-colcon-common-extensions, python3-rosdep: colcon 및 rosdep 관련
#    - ros-humble-cv-bridge, python3-opencv: OpenCV 및 ROS 이미지 변환 관련
#    - ros-humble-rmw-cyclonedds-cpp: CycloneDDS RMW 구현
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
        libxcb-randr0-dev \
        libxcb-xtest0-dev \
        libxcb-xinerama0-dev \
        libxcb-shape0-dev \
        libxcb-xkb-dev \
        libqt5x11extras5 \
        libxkbcommon-x11-0 \
        libgl1-mesa-glx \
        python3-pip \
        ros-humble-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-cv-bridge \
        python3-opencv \
        ros-humble-rmw-cyclonedds-cpp \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Python (pip) 패키지 설치
#    pip 업그레이드 후, transformers, tokenizers, triton 및 기타 필요할 수 있는 패키지 설치
#    --no-cache-dir 옵션으로 캐시 문제 방지
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        "transformers==4.52.3" \
        "tokenizers==0.21.1" \
        # 위 버전들은 이전 테스트에서 PaliGemmaForConditionalGeneration 임포트는 성공했으나 triton이 필요했던 조합입니다.
        # 만약 openvla, optimum 등 다른 패키지와의 호환성이 중요하다면,
        # 해당 패키지들이 요구하는 transformers 버전을 명시해야 합니다. (예: "transformers==4.40.1", "tokenizers==0.19.1")
        # 그 경우 PaliGemmaForConditionalGeneration 임포트 문제를 다시 겪을 수 있습니다.
        triton \
        Pillow \
        # torch는 robovla:latest 이미지에 이미 포함되어 있을 가능성이 높습니다.
        # 만약 특정 버전의 torch가 필요하다면 여기에 명시 (예: "torch>=2.0")
        # "accelerate" # 모델 로딩 및 분산 학습 지원 라이브러리 (필요시 추가)
    && rm -rf /root/.cache/pip

# 6. PyTorch/CUDA 테스트 스크립트 복사 및 실행 권한 부여 (기존 Dockerfile 내용 유지)
#    COPY 명령어는 Dockerfile과 같은 디렉터리에 해당 파일이 있다고 가정합니다.
#    만약 pytorch_cuda_test.py 파일이 없다면 이 부분을 주석 처리하거나 파일을 준비해야 합니다.
COPY pytorch_cuda_test.py /usr/local/bin/pytorch_cuda_test.py
RUN chmod +x /usr/local/bin/pytorch_cuda_test.py

# 7. .bashrc에 ROS 2 환경 설정 등 추가 (기존 Dockerfile 내용 유지)
#    컨테이너 시작 시 자동으로 /opt/ros/humble/setup.bash가 소싱되도록 합니다.
#    ROS_DOMAIN_ID도 여기서 설정합니다.
RUN echo "" >> /root/.bashrc && \
    echo "# ROS 2 Humble environment setup" >> /root/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=20" >> /root/.bashrc && \
    # 작업 공간 setup.bash는 컨테이너 실행 후 수동으로 소싱하거나, 
    # 또는 CMD에서 실행할 스크립트 내에서 소싱하는 것이 일반적입니다.
    # 여기에 추가하면 항상 /workspace/ROS_action이 존재하고 빌드되어 있어야 합니다.
    # echo "if [ -f /workspace/ROS_action/install/setup.bash ]; then source /workspace/ROS_action/install/setup.bash; fi" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# RoboVLMs Poetry Project Helper (기존 내용 유지)" >> /root/.bashrc && \
    echo "alias activate_robovlms='cd /workspace/RoboVLMs && source \$(poetry env info --path)/bin/activate'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'activate_robovlms' to navigate to /workspace/RoboVLMs and activate Poetry environment.\"" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# PyTorch & CUDA Test Alias (기존 내용 유지)" >> /root/.bashrc && \
    echo "alias torch_cuda_test='python3 /usr/local/bin/pytorch_cuda_test.py'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'torch_cuda_test' to check PyTorch and CUDA setup (after activating Poetry env if PyTorch is managed by Poetry).\"" >> /root/.bashrc

# 8. 작업 공간 설정 (컨테이너 실행 시 기본 경로)
WORKDIR /workspace/ROS_action

# 9. 컨테이너 시작 시 실행될 기본 명령어 (bash 셸 실행)
CMD ["bash"]
