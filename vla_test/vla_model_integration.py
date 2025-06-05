#!/usr/bin/env python3
"""
VLA 모델과 RoboVLM 액션 파서 통합 시스템
"""

import torch
import numpy as np
from PIL import Image
import json
from typing import Dict, Any, Optional, List
import sys
import os

# RoboVLM 파서 임포트
from robovlm_action_parser import (
    RoboVLMActionParser, 
    ActionValidator, 
    RoboAction, 
    ActionSpace, 
    RobotControl
)

class VLAModelWrapper:
    """VLA 모델 래퍼 클래스"""
    
    def __init__(self, 
                 model_name: str = "openvla/openvla-7b",
                 device: str = "auto"):
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # 모델과 프로세서 로드
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # 액션 파서 초기화
        self.action_parser = RoboVLMActionParser(
            action_space=ActionSpace.CONTINUOUS,
            action_dim=7,  # 6DOF + 그리퍼
            prediction_horizon=1
        )
        self.action_validator = ActionValidator(
            max_translation_speed=0.5,
            max_rotation_speed=1.0
        )
        
        print(f"🤖 VLA 모델 래퍼 초기화 완료")
        print(f"   모델: {model_name}")
        print(f"   디바이스: {self.device}")

    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_model(self) -> bool:
        """VLA 모델 로드"""
        try:
            print("🔄 VLA 모델 로딩 중...")
            
            # OpenVLA 모델 로드 시도
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device if self.device != "mps" else None,
                    trust_remote_code=True
                )
                
                if self.device == "mps":
                    self.model = self.model.to(self.device)
                
                self.is_loaded = True
                print("✅ OpenVLA 모델 로드 완료")
                return True
                
            except Exception as e:
                print(f"⚠️ OpenVLA 로드 실패: {e}")
                
                # PaliGemma 대안 시도
                print("🔄 PaliGemma 대안 모델 시도...")
                try:
                    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
                    
                    self.processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
                    self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                        "google/paligemma-3b-pt-224",
                        torch_dtype=torch.float16,
                        device_map=self.device if self.device != "mps" else None
                    )
                    
                    if self.device == "mps":
                        self.model = self.model.to(self.device)
                    
                    self.is_loaded = True
                    print("✅ PaliGemma 모델 로드 완료")
                    return True
                    
                except Exception as e2:
                    print(f"❌ PaliGemma 로드도 실패: {e2}")
                    return False
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def predict_action(self, 
                      image: Image.Image,
                      text_instruction: str,
                      return_raw: bool = False) -> Dict[str, Any]:
        """이미지와 텍스트로부터 액션 예측"""
        
        if not self.is_loaded:
            print("⚠️ 모델이 로드되지 않음. 텍스트 기반 fallback 사용")
            return self._text_fallback_action(text_instruction)
        
        try:
            # PaliGemma는 특별한 프롬프트 형식 필요
            prompt = f"action: {text_instruction}"
            
            # 입력 전처리
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # MPS 디바이스로 이동
            if self.device == "mps":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # 결정적 출력
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 입력 길이만큼 제거하여 새로 생성된 토큰만 추출
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            # 출력 디코딩
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 액션 파싱 시도
            action = self._parse_model_output(generated_text, text_instruction)
            
            # 안전성 검증
            validated_action = self.action_validator.validate_action(action)
            
            result = {
                "success": True,
                "action": validated_action,
                "raw_output": generated_text if return_raw else None,
                "confidence": validated_action.confidence,
                "is_safe": self.action_validator.is_safe_action(validated_action)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 모델 추론 실패: {e}")
            return self._text_fallback_action(text_instruction)

    def _parse_model_output(self, 
                           generated_text: str,
                           instruction: str) -> RoboAction:
        """모델 출력 파싱"""
        
        # 생성된 텍스트에서 액션 추출 시도
        try:
            # 숫자 패턴 찾기
            import re
            numbers = re.findall(r'-?\d+\.?\d*', generated_text)
            
            if len(numbers) >= 6:
                # 충분한 숫자가 있으면 6DOF 액션으로 파싱
                action_array = np.array([float(x) for x in numbers[:7]])
                # 정규화 (-1 ~ 1 범위로)
                action_array = np.clip(action_array / 100.0, -1.0, 1.0)
                
                # RoboVLM 파서로 처리
                action_tensor = torch.tensor(action_array).unsqueeze(0)
                return self.action_parser.parse_continuous_action(
                    action_tensor, instruction
                )
            else:
                # 숫자가 부족하면 텍스트 기반으로 fallback
                return self.action_parser._text_only_action(instruction)
                
        except Exception as e:
            print(f"⚠️ 출력 파싱 실패: {e}, fallback to text-only")
            return self.action_parser._text_only_action(instruction)

    def _text_fallback_action(self, instruction: str) -> Dict[str, Any]:
        """텍스트 기반 fallback 액션"""
        action = self.action_parser._text_only_action(instruction)
        validated_action = self.action_validator.validate_action(action)
        
        return {
            "success": False,
            "action": validated_action,
            "raw_output": None,
            "confidence": validated_action.confidence,
            "is_safe": self.action_validator.is_safe_action(validated_action),
            "fallback": True
        }

    def batch_predict(self, 
                     image_instruction_pairs: List[tuple],
                     return_raw: bool = False) -> List[Dict[str, Any]]:
        """배치 예측"""
        results = []
        
        for i, (image, instruction) in enumerate(image_instruction_pairs):
            print(f"🔄 배치 예측 {i+1}/{len(image_instruction_pairs)}: {instruction}")
            result = self.predict_action(image, instruction, return_raw)
            results.append(result)
        
        return results

class VLATestSuite:
    """VLA 모델 테스트 스위트"""
    
    def __init__(self, model_wrapper: VLAModelWrapper):
        self.model = model_wrapper
        
    def create_test_image(self, scenario: str = "kitchen") -> Image.Image:
        """테스트용 이미지 생성"""
        # 간단한 더미 이미지 생성 (실제로는 카메라 이미지 사용)
        image = Image.new('RGB', (224, 224), color='lightblue')
        
        # 시나리오별 간단한 요소 추가 (실제로는 복잡한 장면)
        if scenario == "kitchen":
            # 주방 시나리오 시뮬레이션
            pass
        elif scenario == "table":
            # 테이블 시나리오 시뮬레이션
            pass
        
        return image

    def run_basic_tests(self) -> Dict[str, Any]:
        """기본 테스트 실행"""
        print("🧪 VLA 모델 기본 테스트 시작")
        print("=" * 50)
        
        test_cases = [
            ("Move forward slowly", "navigation"),
            ("Turn left and stop", "navigation"),
            ("Pick up the red cup", "manipulation"),
            ("Put the object on the table", "manipulation"),
            ("전진하세요", "navigation_kr"),
            ("물건을 잡아주세요", "manipulation_kr")
        ]
        
        results = []
        test_image = self.create_test_image("kitchen")
        
        for instruction, category in test_cases:
            print(f"\n📝 테스트: '{instruction}' ({category})")
            
            result = self.model.predict_action(test_image, instruction, return_raw=True)
            
            if result["success"]:
                action = result["action"]
                linear_x, linear_y, angular_z = action.to_twist_like()
                safety_icon = "✅" if result["is_safe"] else "❌"
                
                print(f"   액션 타입: {action.action_type}")
                print(f"   제어 모드: {action.control_mode.value}")
                print(f"   Linear: ({linear_x:.2f}, {linear_y:.2f})")
                print(f"   Angular: {angular_z:.2f}")
                print(f"   그리퍼: {action.gripper:.2f}")
                print(f"   신뢰도: {action.confidence:.2f}")
                print(f"   안전성: {safety_icon}")
                
                if result.get("raw_output"):
                    print(f"   원본 출력: {result['raw_output'][:100]}...")
            else:
                print(f"   ❌ 실패 (fallback 사용)")
                if result.get("fallback"):
                    action = result["action"]
                    linear_x, linear_y, angular_z = action.to_twist_like()
                    print(f"   Fallback 액션: ({linear_x:.2f}, {linear_y:.2f}, {angular_z:.2f})")
            
            results.append({
                "instruction": instruction,
                "category": category,
                "result": result
            })
        
        return {
            "total_tests": len(test_cases),
            "successful_predictions": sum(1 for r in results if r["result"]["success"]),
            "safe_actions": sum(1 for r in results if r["result"]["is_safe"]),
            "results": results
        }

    def run_sequence_tests(self) -> Dict[str, Any]:
        """시퀀스 테스트 실행"""
        print("\n🔄 VLA 모델 시퀀스 테스트 시작")
        print("=" * 50)
        
        sequence_instructions = [
            "Move to the table",
            "Pick up the cup", 
            "Bring it to the counter",
            "Put it down gently"
        ]
        
        test_image = self.create_test_image("kitchen")
        sequence_results = []
        
        for i, instruction in enumerate(sequence_instructions):
            print(f"\n🔗 시퀀스 {i+1}: '{instruction}'")
            
            result = self.model.predict_action(test_image, instruction)
            action = result["action"]
            
            linear_x, linear_y, angular_z = action.to_twist_like()
            print(f"   액션: ({linear_x:.2f}, {linear_y:.2f}, {angular_z:.2f})")
            print(f"   그리퍼: {action.gripper:.2f}")
            print(f"   신뢰도: {action.confidence:.2f}")
            
            sequence_results.append(result)
        
        return {
            "sequence_length": len(sequence_instructions),
            "results": sequence_results
        }

def main():
    """메인 실행 함수"""
    print("🚀 VLA 모델 통합 테스트 시작")
    print("=" * 60)
    
    # VLA 모델 래퍼 초기화
    vla_model = VLAModelWrapper(
        model_name="openvla/openvla-7b",
        device="auto"
    )
    
    # 모델 로드 시도
    if not vla_model.load_model():
        print("⚠️ 모델 로드 실패. 텍스트 기반 모드로 계속...")
    
    # 테스트 스위트 실행
    test_suite = VLATestSuite(vla_model)
    
    # 기본 테스트
    basic_results = test_suite.run_basic_tests()
    
    print(f"\n📊 기본 테스트 결과:")
    print(f"   전체 테스트: {basic_results['total_tests']}")
    print(f"   성공한 예측: {basic_results['successful_predictions']}")
    print(f"   안전한 액션: {basic_results['safe_actions']}")
    
    # 시퀀스 테스트
    sequence_results = test_suite.run_sequence_tests()
    
    print(f"\n📊 시퀀스 테스트 결과:")
    print(f"   시퀀스 길이: {sequence_results['sequence_length']}")
    
    # 결과 저장
    final_results = {
        "basic_tests": basic_results,
        "sequence_tests": sequence_results,
        "model_info": {
            "name": vla_model.model_name,
            "device": vla_model.device,
            "loaded": vla_model.is_loaded
        }
    }
    
    with open("vla_test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 테스트 완료! 결과가 'vla_test_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    main() 