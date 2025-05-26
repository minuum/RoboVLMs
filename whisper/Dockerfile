# Dockerfile

################################################################################
# 1) 베이스 이미지: robovla:latest (CUDA + PyTorch 환경 포함)
################################################################################
FROM robovla:latest

################################################################################
# 2) 사용자 및 환경 설정
################################################################################
USER root
ENV ROS_DISTRO=humble \
    APP_DIR=/app \
    ROS_WS=/ros2_ws \
    ROS_DOMAIN_ID=20

################################################################################
# 3) ROS 2 APT 저장소 등록 및 ROS 설치
################################################################################
RUN apt-get update && \
    # ROS 2 키 추가
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    # ROS 2 저장소 등록
    echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    # ROS 2 패키지 및 오디오 관련 라이브러리 설치
    apt-get install -y --no-install-recommends \
      ros-humble-desktop \
      ros-dev-tools \
      python3-colcon-common-extensions \
      python3-rosdep \
      ros-humble-rmw-cyclonedds-cpp \
      ffmpeg libsndfile1 libportaudio2 portaudio19-dev \
  && rm -rf /var/lib/apt/lists/*

################################################################################
# 4) ROS2 워크스페이스 설정
################################################################################
RUN mkdir -p ${ROS_WS}/src && \
    cd ${ROS_WS} && \
    # rosdep 초기화 및 업데이트
    rosdep init || true && \
    rosdep update && \
    # 워크스페이스 빌드 (빈 워크스페이스)
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

################################################################################
# 5) Whisper 관련 Python 라이브러리 설치
################################################################################
# robovla:latest에 이미 PyTorch가 설치되어 있으므로, Whisper 관련 라이브러리만 추가 설치
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    pip3 install --no-cache-dir \
      transformers==4.41.2 \
      accelerate \
      sounddevice scipy librosa \
      soundfile

################################################################################
# 6) 앱 파일 복사 및 bashrc 설정
################################################################################
WORKDIR ${APP_DIR}
COPY whisper_test.py ./whisper_test.py

# ROS 환경 자동 source 및 ROS_DOMAIN_ID 설정
RUN echo "" >> /root/.bashrc && \
    echo "# ROS 2 Humble environment setup" >> /root/.bashrc && \
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source ${ROS_WS}/install/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# Whisper STT Node Helper" >> /root/.bashrc && \
    echo "alias stt_test='cd ${APP_DIR} && python3 whisper_test.py'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'stt_test' to start Whisper STT node.\"" >> /root/.bashrc

################################################################################
# 7) 기본 실행 명령어 (whisper_test.py 실행)
################################################################################
CMD ["bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && source ${ROS_WS}/install/setup.bash && python3 whisper_test.py"]
