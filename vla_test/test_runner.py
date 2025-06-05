#!/usr/bin/env python3
"""
독립형 VLA 시스템 통합 테스트 실행기
"""

import cv2
import time
import os
import sys
import argparse
from typing import Optional
import numpy as np

from standalone_vla_test import StandaloneVLAInference, CameraHandler
from action_parser import VLAActionParser, ActionValidator, RobotAction

class VLATestRunner:
    """VLA 시스템 통합 테스트 클래스"""
    
    def __init__(self, 
                 model_id: str = "google/paligemma-3b-mix-224",
                 device: str = "cuda",
                 use_camera: bool = False,
                 camera_id: int = 0):
        
        print("🚀 VLA 테스트 러너 초기화 중...")
        
        # VLA 추론 시스템 초기화
        self.vla_system = StandaloneVLAInference(
            model_id=model_id,
            device_preference=device
        )
        
        # 액션 파서 초기화
        self.action_parser = VLAActionParser()
        self.action_validator = ActionValidator()
        
        # 카메라 핸들러 초기화
        self.camera_handler = CameraHandler(camera_id)
        self.use_camera = use_camera
        
        if self.use_camera:
            if not self.camera_handler.init_camera():
                print("⚠️ 카메라 초기화 실패 - 테스트 이미지 모드로 전환")
                self.use_camera = False
        
        # 테스트 상태
        self.test_results = []
        self.current_image = None
        
        print("✅ VLA 테스트 러너 초기화 완료")

    def load_test_image(self, image_path: str) -> bool:
        """테스트 이미지 로드"""
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일이 존재하지 않음: {image_path}")
            return False
        
        self.current_image = self.camera_handler.load_test_image(image_path)
        return self.current_image is not None

    def capture_current_frame(self) -> bool:
        """현재 카메라 프레임 캡처"""
        if not self.use_camera:
            print("❌ 카메라가 활성화되지 않음")
            return False
        
        self.current_image = self.camera_handler.capture_frame()
        return self.current_image is not None

    def run_single_test(self, command: str, use_vla_model: bool = True) -> RobotAction:
        """단일 명령어 테스트 실행"""
        if self.current_image is None:
            print("❌ 테스트용 이미지가 없습니다")
            return RobotAction(action_type="UNKNOWN")
        
        print(f"\n🧠 테스트 실행: '{command}'")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            if use_vla_model:
                # VLA 모델 사용한 전체 파이프라인
                linear_x, linear_y, angular_z = self.vla_system.simple_command_inference(
                    self.current_image, command
                )
                
                # 간단한 RobotAction 생성 (실제로는 VLA 결과를 파싱해야 함)
                action = RobotAction(
                    action_type="MOVE",
                    linear_x=linear_x,
                    linear_y=linear_y,
                    angular_z=angular_z,
                    description=f"VLA result for: {command}",
                    confidence=0.7
                )
            else:
                # 액션 파서만 사용한 테스트
                action = self.action_parser.parse_text_output(command)
            
            # 액션 유효성 검사
            action = self.action_validator.validate_action(action)
            
            processing_time = time.time() - start_time
            
            # 결과 출력
            print(f"🎯 액션 타입: {action.action_type.value}")
            print(f"🚀 제어 명령:")
            print(f"   - linear_x: {action.linear_x:.3f}")
            print(f"   - linear_y: {action.linear_y:.3f}")
            print(f"   - angular_z: {action.angular_z:.3f}")
            print(f"🎯 목표 객체: {action.target_object or 'N/A'}")
            print(f"📊 신뢰도: {action.confidence:.2f}")
            print(f"⏱️ 처리 시간: {processing_time:.2f}초")
            print(f"✅ 안전성: {'통과' if self.action_validator.is_safe_action(action) else '실패'}")
            
            # 테스트 결과 저장
            test_result = {
                "command": command,
                "action": action.to_dict(),
                "processing_time": processing_time,
                "safe": self.action_validator.is_safe_action(action),
                "use_vla_model": use_vla_model
            }
            self.test_results.append(test_result)
            
            return action
            
        except Exception as e:
            print(f"❌ 테스트 실행 오류: {e}")
            return RobotAction(action_type="UNKNOWN")

    def run_batch_tests(self, commands: list, use_vla_model: bool = True):
        """배치 테스트 실행"""
        print(f"\n📋 배치 테스트 시작 (VLA 모델 사용: {use_vla_model})")
        print("=" * 60)
        
        for i, command in enumerate(commands, 1):
            print(f"\n📍 테스트 {i}/{len(commands)}")
            self.run_single_test(command, use_vla_model)
            
            if i < len(commands):
                print("\n⏳ 잠시 대기...")
                time.sleep(1)
        
        print(f"\n✅ 배치 테스트 완료 ({len(commands)}개 명령)")

    def run_interactive_test(self):
        """대화형 테스트 실행"""
        print("\n🎮 대화형 테스트 모드")
        print("=" * 40)
        print("명령어를 입력하세요 (종료: 'quit' 또는 'exit')")
        print("카메라 캡처: 'capture' (카메라 모드에서만)")
        print("이미지 로드: 'load <파일경로>'")
        print("VLA 모델 토글: 'toggle_vla'")
        print("-" * 40)
        
        use_vla_model = True
        
        while True:
            try:
                user_input = input("\n💬 명령어: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 테스트 종료")
                    break
                elif user_input.lower() == 'capture':
                    if self.capture_current_frame():
                        print("📸 프레임 캡처 완료")
                    else:
                        print("❌ 프레임 캡처 실패")
                elif user_input.lower().startswith('load '):
                    image_path = user_input[5:].strip()
                    if self.load_test_image(image_path):
                        print(f"✅ 이미지 로드 완료: {image_path}")
                    else:
                        print(f"❌ 이미지 로드 실패: {image_path}")
                elif user_input.lower() == 'toggle_vla':
                    use_vla_model = not use_vla_model
                    print(f"🔄 VLA 모델 사용: {'ON' if use_vla_model else 'OFF'}")
                elif user_input.strip():
                    self.run_single_test(user_input, use_vla_model)
                
            except KeyboardInterrupt:
                print("\n👋 테스트 종료")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

    def show_results_summary(self):
        """테스트 결과 요약 출력"""
        if not self.test_results:
            print("📊 테스트 결과가 없습니다")
            return
        
        print("\n📊 테스트 결과 요약")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        safe_tests = sum(1 for result in self.test_results if result["safe"])
        avg_time = np.mean([result["processing_time"] for result in self.test_results])
        
        print(f"총 테스트: {total_tests}개")
        print(f"안전한 액션: {safe_tests}개 ({safe_tests/total_tests*100:.1f}%)")
        print(f"평균 처리 시간: {avg_time:.2f}초")
        
        # 액션 타입별 분석
        action_types = {}
        for result in self.test_results:
            action_type = result["action"]["action_type"]
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        print(f"\n액션 타입 분포:")
        for action_type, count in action_types.items():
            print(f"  {action_type}: {count}개")

    def cleanup(self):
        """리소스 정리"""
        if self.use_camera:
            self.camera_handler.release()
        print("🧹 리소스 정리 완료")

def main():
    parser = argparse.ArgumentParser(description="VLA 시스템 테스트 러너")
    parser.add_argument("--model", default="google/paligemma-3b-mix-224", 
                       help="사용할 VLA 모델 ID")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="추론에 사용할 디바이스")
    parser.add_argument("--camera", action="store_true",
                       help="카메라 사용")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="카메라 ID")
    parser.add_argument("--image", type=str,
                       help="테스트용 이미지 파일 경로")
    parser.add_argument("--mode", choices=["batch", "interactive", "single"], 
                       default="interactive",
                       help="테스트 모드")
    parser.add_argument("--command", type=str,
                       help="단일 테스트 명령어")
    parser.add_argument("--no-vla", action="store_true",
                       help="VLA 모델 없이 파서만 테스트")
    
    args = parser.parse_args()
    
    # 테스트 러너 초기화
    test_runner = VLATestRunner(
        model_id=args.model,
        device=args.device,
        use_camera=args.camera,
        camera_id=args.camera_id
    )
    
    try:
        # 테스트 이미지 로드
        if args.image:
            if not test_runner.load_test_image(args.image):
                print("❌ 테스트 이미지 로드 실패")
                return
        elif not args.camera:
            # 기본 테스트 이미지 찾기
            default_images = [
                "../RoboVLMs/cat.jpg",
                "test_image.jpg", 
                "sample.jpg"
            ]
            
            image_loaded = False
            for img_path in default_images:
                if os.path.exists(img_path):
                    if test_runner.load_test_image(img_path):
                        image_loaded = True
                        break
            
            if not image_loaded:
                print("⚠️ 기본 테스트 이미지를 찾을 수 없습니다")
                print("카메라를 사용하거나 --image 옵션으로 이미지를 지정하세요")
        
        use_vla_model = not args.no_vla
        
        # 테스트 모드에 따른 실행
        if args.mode == "single":
            if not args.command:
                print("❌ 단일 테스트 모드에서는 --command 옵션이 필요합니다")
                return
            test_runner.run_single_test(args.command, use_vla_model)
            
        elif args.mode == "batch":
            default_commands = [
                "move forward",
                "turn left",
                "turn right", 
                "stop",
                "navigate to door",
                "avoid obstacle",
                "grab the cup",
                "look around"
            ]
            test_runner.run_batch_tests(default_commands, use_vla_model)
            
        elif args.mode == "interactive":
            test_runner.run_interactive_test()
        
        # 결과 요약 출력
        test_runner.show_results_summary()
        
    finally:
        test_runner.cleanup()

if __name__ == "__main__":
    main() 