#!/usr/bin/env python3
"""
ROS2 없이 VLA 모델을 테스트하기 위한 독립적인 파일
"""

import torch
from PIL import Image as PilImage
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import cv2
import numpy as np
import os
import sys
import re
from pathlib import Path
import time
from typing import Tuple, List, Dict, Optional

class StandaloneVLAInference:
    def __init__(self, 
                 model_id: str = "google/paligemma-3b-mix-224",
                 device_preference: str = "cuda",
                 model_cache_dir: str = ".vla_models_cache",
                 max_new_tokens: int = 128):
        
        print("🤖 독립형 VLA 추론 시스템 초기화 중...")
        
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model_cache_dir = model_cache_dir
        
        # 디바이스 설정
        if device_preference == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🎯 CUDA 사용: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("🎯 CPU 사용")
        
        # 모델 로드
        self.model = None
        self.processor = None
        self.load_model()
        
        print("✅ VLA 추론 시스템 초기화 완료")

    def load_model(self):
        """VLA 모델 로드"""
        try:
            print(f"📥 모델 로딩 중: {self.model_id}")
            
            model_save_path = Path(self.model_cache_dir) / self.model_id.split('/')[-1]
            model_save_path.mkdir(parents=True, exist_ok=True)

            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                cache_dir=model_save_path
            )

            # 모델 로드
            model_kwargs = {
                "cache_dir": model_save_path,
                "low_cpu_mem_usage": True
            }
            
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id, 
                **model_kwargs
            )
            
            if self.device.type != "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            print("✅ 모델 로딩 완료")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise

    def infer_from_image_and_text(self, image: np.ndarray, text_prompt: str) -> str:
        """이미지와 텍스트로부터 VLA 추론 수행"""
        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        try:
            print(f"🧠 추론 실행: '{text_prompt}'")
            
            # 이미지 전처리 (BGR -> RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            pil_image = PilImage.fromarray(rgb_image)
            
            # 모델 입력 준비
            inputs = self.processor(
                images=pil_image, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            # 추론 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            print(f"🤖 VLA 결과: {result}")
            return result
            
        except Exception as e:
            print(f"❌ 추론 오류: {e}")
            return ""

    def simple_command_inference(self, image: np.ndarray, command: str) -> Tuple[float, float, float]:
        """간단한 명령어 추론"""
        command_lower = command.lower()
        
        # 직접 처리 가능한 명령어들
        if "stop" in command_lower or "halt" in command_lower:
            print("🛑 정지 명령")
            return 0.0, 0.0, 0.0
        elif "move forward" in command_lower or "go forward" in command_lower:
            print("➡️ 전진 명령")
            return 0.3, 0.0, 0.0
        elif "move backward" in command_lower or "go backward" in command_lower:
            print("⬅️ 후진 명령")
            return -0.3, 0.0, 0.0
        elif "turn left" in command_lower:
            print("↪️ 좌회전 명령")
            return 0.0, 0.0, 0.5
        elif "turn right" in command_lower:
            print("↩️ 우회전 명령")
            return 0.0, 0.0, -0.5
        
        # VLA 모델 사용
        return self.vla_model_inference(image, command)

    def vla_model_inference(self, image: np.ndarray, command: str) -> Tuple[float, float, float]:
        """VLA 모델을 이용한 복잡한 추론"""
        try:
            # 다양한 추론 타입
            if "navigate to" in command.lower() or "go to" in command.lower():
                return self.navigation_inference(image, command)
            elif "avoid" in command.lower() or "obstacle" in command.lower():
                return self.obstacle_avoidance_inference(image, command)
            else:
                return self.general_inference(image, command)
                
        except Exception as e:
            print(f"❌ VLA 추론 오류: {e}")
            return 0.0, 0.0, 0.0

    def navigation_inference(self, image: np.ndarray, command: str) -> Tuple[float, float, float]:
        """내비게이션 추론"""
        target = command.lower().replace("navigate to", "").replace("go to", "").strip()
        prompt = f"find {target} in the image and determine robot movement direction"
        
        result = self.infer_from_image_and_text(image, prompt)
        return self.parse_action_to_twist(result)

    def obstacle_avoidance_inference(self, image: np.ndarray, command: str) -> Tuple[float, float, float]:
        """장애물 회피 추론"""
        prompt = "detect obstacles and suggest safe movement direction"
        result = self.infer_from_image_and_text(image, prompt)
        
        # 장애물 감지 시 안전한 행동
        if "obstacle" in result.lower() or "blocked" in result.lower():
            print("🛑 장애물 감지 - 정지")
            return 0.0, 0.0, 0.0
        else:
            print("✅ 경로 안전 - 천천히 전진")
            return 0.1, 0.0, 0.0

    def general_inference(self, image: np.ndarray, command: str) -> Tuple[float, float, float]:
        """일반적인 추론"""
        prompt = f"Robot action for command: {command}"
        result = self.infer_from_image_and_text(image, prompt)
        return self.parse_action_to_twist(result)

    def parse_action_to_twist(self, action_text: str) -> Tuple[float, float, float]:
        """VLA 결과를 로봇 제어 명령으로 변환"""
        linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
        
        action_lower = action_text.lower()
        
        if "forward" in action_lower or "ahead" in action_lower:
            linear_x = 0.2
        elif "backward" in action_lower or "back" in action_lower:
            linear_x = -0.2
        elif "left" in action_lower:
            angular_z = 0.5
        elif "right" in action_lower:
            angular_z = -0.5
        elif "stop" in action_lower or "halt" in action_lower:
            linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
            
        return linear_x, linear_y, angular_z

class CameraHandler:
    """카메라 입력 처리"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
    def init_camera(self) -> bool:
        """카메라 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ 카메라 {self.camera_id} 열기 실패")
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ 카메라 {self.camera_id} 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 초기화 오류: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """프레임 캡처"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame

    def load_test_image(self, image_path: str) -> Optional[np.ndarray]:
        """테스트 이미지 로드"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                return None
            
            print(f"✅ 테스트 이미지 로드: {image_path}")
            return image
            
        except Exception as e:
            print(f"❌ 이미지 로드 오류: {e}")
            return None

    def release(self):
        """카메라 해제"""
        if self.cap:
            self.cap.release()
            print("🔒 카메라 해제 완료")

if __name__ == "__main__":
    # 테스트 실행 예시
    print("🚀 독립형 VLA 테스트 시작")
    
    # VLA 시스템 초기화
    vla = StandaloneVLAInference()
    
    # 카메라 핸들러 초기화 
    camera = CameraHandler()
    
    # 테스트 명령어들
    test_commands = [
        "move forward",
        "turn left", 
        "stop",
        "navigate to door",
        "avoid obstacle"
    ]
    
    print("\n🧪 테스트 명령어들:")
    for i, cmd in enumerate(test_commands):
        print(f"{i+1}. {cmd}")
    
    # 간단한 테스트 (테스트 이미지 필요)
    test_image_path = "../RoboVLMs/cat.jpg"  # 예시 이미지 경로
    if os.path.exists(test_image_path):
        test_image = camera.load_test_image(test_image_path)
        if test_image is not None:
            for cmd in test_commands:
                print(f"\n🧠 테스트: '{cmd}'")
                linear_x, linear_y, angular_z = vla.simple_command_inference(test_image, cmd)
                print(f"🚀 결과: linear_x={linear_x:.2f}, linear_y={linear_y:.2f}, angular_z={angular_z:.2f}")
                time.sleep(1)
    
    print("\n✅ 테스트 완료") 