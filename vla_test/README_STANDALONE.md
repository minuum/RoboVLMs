# 독립형 VLA 테스트 시스템

ROS2 없이 VLA (Vision-Language-Action) 모델을 테스트할 수 있는 독립적인 시스템입니다.

## 📁 파일 구조

```
ROS_action/
├── standalone_vla_test.py    # 메인 VLA 추론 시스템
├── action_parser.py          # 액션 파싱 모듈
├── test_runner.py            # 통합 테스트 실행기
├── requirements.txt          # Python 의존성
└── README_STANDALONE.md      # 이 파일
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# 가상환경 생성 (권장)
python -m venv vla_env
source vla_env/bin/activate  # macOS/Linux
# vla_env\Scripts\activate   # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 액션 파서만 테스트 (모델 없이)

```bash
# 액션 파서 단독 테스트
python action_parser.py

# 또는 통합 러너로 파서만 테스트
python test_runner.py --no-vla --mode batch
```

### 3. VLA 모델 포함 전체 테스트

```bash
# 대화형 모드 (기본)
python test_runner.py

# 배치 테스트 모드
python test_runner.py --mode batch

# 단일 명령어 테스트
python test_runner.py --mode single --command "move forward"
```

## 🔧 사용법 상세

### standalone_vla_test.py

핵심 VLA 추론 시스템입니다.

```python
from standalone_vla_test import StandaloneVLAInference, CameraHandler

# VLA 시스템 초기화
vla = StandaloneVLAInference(
    model_id="google/paligemma-3b-mix-224",
    device_preference="cuda"  # 또는 "cpu"
)

# 카메라 핸들러
camera = CameraHandler()

# 테스트 이미지 로드
image = camera.load_test_image("test.jpg")

# 추론 실행
linear_x, linear_y, angular_z = vla.simple_command_inference(image, "move forward")
print(f"제어 명령: linear_x={linear_x}, angular_z={angular_z}")
```

### action_parser.py

VLA 결과를 로봇 액션으로 파싱합니다.

```python
from action_parser import VLAActionParser, ActionValidator

parser = VLAActionParser()
validator = ActionValidator()

# 텍스트 출력 파싱
action = parser.parse_text_output("move forward slowly")

# 세그멘테이션 출력 파싱  
action = parser.parse_segmentation_output("<loc0500><loc0300><loc0700><loc0600>cup", 640, 480)

# 액션 유효성 검사
action = validator.validate_action(action)

print(f"액션: {action.action_type.value}")
print(f"속도: linear_x={action.linear_x}, angular_z={action.angular_z}")
```

### test_runner.py

통합 테스트 실행기입니다.

```bash
# 기본 대화형 모드
python test_runner.py

# CPU 사용
python test_runner.py --device cpu

# 특정 이미지로 테스트
python test_runner.py --image path/to/image.jpg

# 카메라 사용
python test_runner.py --camera

# VLA 모델 없이 파서만 테스트
python test_runner.py --no-vla

# 단일 명령어 테스트
python test_runner.py --mode single --command "turn left"

# 배치 테스트
python test_runner.py --mode batch
```

## 🎮 대화형 모드 명령어

대화형 모드에서 사용 가능한 특수 명령어들:

- `quit` 또는 `exit`: 테스트 종료
- `capture`: 카메라에서 새 프레임 캡처 (카메라 모드)
- `load <파일경로>`: 새 테스트 이미지 로드
- `toggle_vla`: VLA 모델 사용 ON/OFF 토글

## 📊 지원하는 액션 타입

- **MOVE**: 이동 (전진, 후진)
- **TURN**: 회전 (좌회전, 우회전) 
- **STOP**: 정지
- **GRAB**: 객체 잡기
- **RELEASE**: 객체 놓기
- **POINT**: 가리키기
- **LOOK**: 관찰하기
- **NAVIGATE**: 내비게이션
- **AVOID**: 회피
- **UNKNOWN**: 미지의 액션

## 🧪 테스트 명령어 예시

### 기본 이동 명령어

```
move forward
move backward  
turn left
turn right
stop
```

### 한국어 명령어

```
전진하세요
우회전 하세요
정지
```

### 복합 명령어

```
move forward slowly
turn left quickly
navigate to door
avoid obstacle
grab the cup
```

### 세그멘테이션 토큰 명령어

```
<loc0500><loc0300><loc0700><loc0600>cup segment
```

## 🔧 설정 옵션

### 모델 설정

다른 VLA 모델 사용:

```bash
python test_runner.py --model "google/paligemma-3b-pt-224"
```

### 디바이스 설정

```bash
# CUDA 사용 (기본)
python test_runner.py --device cuda

# CPU 사용  
python test_runner.py --device cpu
```

### 안전성 설정

`ActionValidator`에서 속도 제한 설정:

```python
validator = ActionValidator(
    max_linear_speed=0.5,    # 최대 직선 속도
    max_angular_speed=1.0    # 최대 회전 속도
)
```

## 🐛 문제 해결

### CUDA 메모리 부족

```bash
# CPU 사용으로 전환
python test_runner.py --device cpu
```

### 모델 다운로드 실패

```bash
# 캐시 디렉토리 확인/생성
mkdir -p .vla_models_cache
```

### 카메라 접근 오류

```bash
# 카메라 권한 확인 또는 다른 카메라 ID 시도
python test_runner.py --camera --camera-id 1
```

### 이미지 로드 실패

```bash
# 지원 형식: jpg, png, bmp
# OpenCV로 읽을 수 있는 형식이어야 함
```

## 📈 성능 최적화

### 모델 양자화 (선택사항)

```bash
# requirements.txt에서 주석 해제
pip install bitsandbytes>=0.41.0
```

### Flash Attention (선택사항)

```bash
# requirements.txt에서 주석 해제  
pip install flash-attn>=2.0.0
```

## 🔍 디버깅

### 로그 레벨 조정

코드에서 디버그 출력 추가:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 모델 상태 확인

```python
# standalone_vla_test.py에서
print(f"Model loaded: {vla.model is not None}")
print(f"Device: {vla.device}")
```

## 🤝 확장

### 새로운 액션 타입 추가

`action_parser.py`의 `ActionType` enum에 추가:

```python
class ActionType(Enum):
    # 기존 타입들...
    CUSTOM_ACTION = "custom_action"
```

### 새로운 명령어 패턴 추가

`VLAActionParser`의 키워드 사전에 추가:

```python
self.action_keywords = {
    ActionType.CUSTOM_ACTION: ["custom", "special", "맞춤"]
}
```

## 📝 라이센스

이 코드는 원본 VLA 시스템과 동일한 라이센스를 따릅니다. 