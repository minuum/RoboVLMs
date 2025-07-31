FROM robovla-ros-poetry:humble

# 2. USER를 root로 설정
USER root

# 3. 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH" 
ENV HF_HOME="/root/.cache/huggingface" 

# 4. ROS 2 저장소 설정 및 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
        # 기본 유틸리티 및 X11/Qt 라이브러리
        libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev libqt5x11extras5 libxkbcommon-x11-0 libgl1-mesa-glx \
        python3-pip \
        # ROS 2 핵심 및 필요 패키지
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
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        "transformers==4.52.3" \
        "tokenizers==0.21.1" \
        triton \
        Pillow \
    && rm -rf /root/.cache/pip

# 6. (선택 사항) 테스트 스크립트 복사 (Dockerfile과 같은 위치에 파일이 있어야 함)
# COPY pytorch_cuda_test.py /usr/local/bin/pytorch_cuda_test.py
# RUN chmod +x /usr/local/bin/pytorch_cuda_test.py

# 7. .bashrc에 ROS 2 환경 자동 소싱 및 사용자 편의 alias 추가
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

# 8. 기본 작업 디렉터리 설정
WORKDIR /workspace/ROS_action

# 9. 컨테이너 시작 시 기본 명령어
CMD ["bash"]