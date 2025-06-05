#!/usr/bin/env python3
"""
RoboVLM 액션 파서 고급 데모
실제 RoboVLMs 모델의 출력을 시뮬레이션하여 파서 성능 확인
"""

import torch
import numpy as np
from typing import Dict, List, Any
import json

from robovlm_action_parser import (
    RoboVLMActionParser, 
    ActionValidator, 
    RoboAction, 
    ActionSpace, 
    RobotControl
)

class RoboVLMSimulator:
    """RoboVLM 모델 출력 시뮬레이터"""
    
    def __init__(self):
        self.action_parser = RoboVLMActionParser(
            action_space=ActionSpace.CONTINUOUS,
            action_dim=7,
            prediction_horizon=1
        )
        self.validator = ActionValidator()
        
        # 시뮬레이션용 액션 템플릿
        self.action_templates = {
            "move_forward": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "move_backward": [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "turn_left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            "turn_right": [0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0],
            "grab_object": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "release_object": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "stop": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "complex_navigate": [0.2, 0.1, 0.0, 0.0, 0.0, 0.3, 0.0]
        }
    
    def simulate_vlm_output(self, instruction: str) -> torch.Tensor:
        """명령어에 따른 VLM 출력 시뮬레이션"""
        instruction_lower = instruction.lower()
        
        # 키워드 매칭으로 적절한 액션 선택
        if "forward" in instruction_lower or "전진" in instruction_lower:
            base_action = self.action_templates["move_forward"]
        elif "backward" in instruction_lower or "후진" in instruction_lower:
            base_action = self.action_templates["move_backward"]
        elif "left" in instruction_lower or "좌" in instruction_lower:
            base_action = self.action_templates["turn_left"]
        elif "right" in instruction_lower or "우" in instruction_lower:
            base_action = self.action_templates["turn_right"]
        elif "grab" in instruction_lower or "pick" in instruction_lower or "잡" in instruction_lower:
            base_action = self.action_templates["grab_object"]
        elif "release" in instruction_lower or "drop" in instruction_lower or "놓" in instruction_lower:
            base_action = self.action_templates["release_object"]
        elif "stop" in instruction_lower or "정지" in instruction_lower:
            base_action = self.action_templates["stop"]
        else:
            base_action = self.action_templates["complex_navigate"]
        
        # 노이즈 추가로 현실적인 모델 출력 시뮬레이션
        noise = np.random.normal(0, 0.05, 7)
        noisy_action = np.array(base_action) + noise
        
        # 배치 차원 추가 (bs=1, seq_len=1, action_dim=7)
        return torch.tensor(noisy_action).unsqueeze(0).unsqueeze(0).float()
    
    def simulate_trajectory_output(self, instruction: str, steps: int = 3) -> torch.Tensor:
        """궤적 시퀀스 출력 시뮬레이션"""
        base_action = self.simulate_vlm_output(instruction).squeeze()
        
        trajectory = []
        for i in range(steps):
            # 시간에 따른 액션 변화 시뮬레이션
            decay_factor = 0.8 ** i
            step_action = base_action * decay_factor
            
            # 마지막 스텝에서는 정지
            if i == steps - 1:
                step_action[:6] = 0  # 움직임 정지, 그리퍼 상태 유지
            
            trajectory.append(step_action)
        
        return torch.stack(trajectory).unsqueeze(0)  # (1, seq_len, action_dim)

def demo_continuous_parsing():
    """연속 액션 파싱 데모"""
    print("🔄 연속 액션 파싱 데모")
    print("=" * 40)
    
    simulator = RoboVLMSimulator()
    
    test_instructions = [
        "Move forward to the table",
        "Turn left slowly",
        "Grab the red cup carefully", 
        "Navigate around the obstacle",
        "테이블로 전진하세요",
        "천천히 우회전하세요",
        "빨간 컵을 조심스럽게 잡으세요"
    ]
    
    results = []
    
    for instruction in test_instructions:
        print(f"\n📝 명령어: '{instruction}'")
        
        # VLM 출력 시뮬레이션
        simulated_output = simulator.simulate_vlm_output(instruction)
        print(f"   시뮬레이션 출력 형태: {simulated_output.shape}")
        
        # 액션 파싱
        action = simulator.action_parser.parse_continuous_action(
            simulated_output, instruction
        )
        
        # 검증
        validated_action = simulator.validator.validate_action(action)
        
        # 결과 출력
        linear_x, linear_y, angular_z = validated_action.to_twist_like()
        safety_icon = "✅" if simulator.validator.is_safe_action(validated_action) else "❌"
        
        print(f"   액션 타입: {validated_action.action_type}")
        print(f"   제어 모드: {validated_action.control_mode.value}")
        print(f"   Linear: ({linear_x:.3f}, {linear_y:.3f})")
        print(f"   Angular: {angular_z:.3f}")
        print(f"   그리퍼: {validated_action.gripper:.3f}")
        print(f"   신뢰도: {validated_action.confidence:.3f}")
        print(f"   안전성: {safety_icon}")
        
        results.append({
            "instruction": instruction,
            "action_type": validated_action.action_type,
            "linear_x": linear_x,
            "linear_y": linear_y,
            "angular_z": angular_z,
            "gripper": validated_action.gripper,
            "confidence": validated_action.confidence,
            "is_safe": simulator.validator.is_safe_action(validated_action)
        })
    
    return results

def demo_trajectory_parsing():
    """궤적 시퀀스 파싱 데모"""
    print("\n🔄 궤적 시퀀스 파싱 데모") 
    print("=" * 40)
    
    simulator = RoboVLMSimulator()
    
    trajectory_tasks = [
        ("Approach and pick up the object", 4),
        ("Navigate to destination and stop", 3),
        ("Avoid obstacle and continue", 5)
    ]
    
    for task, steps in trajectory_tasks:
        print(f"\n📝 궤적 태스크: '{task}' ({steps}스텝)")
        
        # 궤적 시뮬레이션
        trajectory_output = simulator.simulate_trajectory_output(task, steps)
        print(f"   궤적 출력 형태: {trajectory_output.shape}")
        
        # 궤적 파싱
        actions = simulator.action_parser.parse_trajectory_sequence(
            trajectory_output, task
        )
        
        print(f"   파싱된 액션 시퀀스:")
        for i, action in enumerate(actions):
            validated = simulator.validator.validate_action(action)
            linear_x, linear_y, angular_z = validated.to_twist_like()
            
            print(f"     Step {i+1}/{steps}: linear_x={linear_x:.3f}, "
                  f"angular_z={angular_z:.3f}, gripper={validated.gripper:.3f}, "
                  f"remaining={validated.prediction_horizon}")

def demo_vlm_output_parsing():
    """VLM 전체 출력 파싱 데모"""
    print("\n🔄 VLM 전체 출력 파싱 데모")
    print("=" * 40)
    
    simulator = RoboVLMSimulator()
    
    # RoboVLM 스타일 출력 시뮬레이션
    mock_vlm_outputs = [
        {
            "action_pred": torch.tensor([[[0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]]]),
            "confidence": 0.85
        },
        {
            "action_pred": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]),
            "confidence": 0.92
        },
        {
            # 이산 액션 대신 연속 액션으로 변경
            "action_pred": torch.tensor([[[0.1, 0.1, 0.0, 0.0, 0.0, 0.2, 0.5]]]),
            "confidence": 0.78
        }
    ]
    
    instructions = [
        "Move forward and turn slightly",
        "Close the gripper to grab",
        "Follow the instruction sequence"
    ]
    
    for i, (vlm_output, instruction) in enumerate(zip(mock_vlm_outputs, instructions)):
        print(f"\n📝 VLM 출력 {i+1}: '{instruction}'")
        print(f"   출력 키: {list(vlm_output.keys())}")
        
        # VLM 출력 파싱
        action = simulator.action_parser.parse_vision_language_action(
            vlm_output, instruction
        )
        
        validated = simulator.validator.validate_action(action)
        linear_x, linear_y, angular_z = validated.to_twist_like()
        
        print(f"   결과: {validated.action_type} - "
              f"({linear_x:.3f}, {linear_y:.3f}, {angular_z:.3f})")

def demo_comparison_with_simple_parser():
    """간단한 파서와 RoboVLM 파서 비교"""
    print("\n🔄 파서 성능 비교 데모")
    print("=" * 40)
    
    # 간단한 기존 파서 (기존 프로젝트 스타일)
    def simple_parse(text: str) -> Dict[str, float]:
        linear_x = angular_z = 0.0
        
        if "forward" in text.lower() or "전진" in text:
            linear_x = 0.3
        elif "backward" in text.lower() or "후진" in text:
            linear_x = -0.3
        elif "left" in text.lower() or "좌" in text:
            angular_z = 0.5
        elif "right" in text.lower() or "우" in text:
            angular_z = -0.5
        
        return {"linear_x": linear_x, "angular_z": angular_z}
    
    simulator = RoboVLMSimulator()
    
    test_cases = [
        "Move forward to the kitchen table",
        "Turn left and avoid the obstacle", 
        "Grab the cup and bring it here",
        "Navigate carefully around furniture",
        "부엌 테이블로 전진하세요"
    ]
    
    print("\n비교 결과:")
    print("명령어 | 간단한 파서 | RoboVLM 파서")
    print("-" * 60)
    
    for instruction in test_cases:
        # 간단한 파서
        simple_result = simple_parse(instruction)
        
        # RoboVLM 파서
        simulated_output = simulator.simulate_vlm_output(instruction)
        robovlm_action = simulator.action_parser.parse_continuous_action(
            simulated_output, instruction
        )
        robovlm_result = robovlm_action.to_twist_like()
        
        print(f"{instruction[:25]:<25} | "
              f"({simple_result['linear_x']:.2f}, {simple_result['angular_z']:.2f}) | "
              f"({robovlm_result[0]:.2f}, {robovlm_result[2]:.2f}) [{robovlm_action.action_type}]")

def main():
    """메인 데모 실행"""
    print("🤖 RoboVLM 액션 파서 고급 데모")
    print("=" * 60)
    
    # 1. 연속 액션 파싱 데모
    continuous_results = demo_continuous_parsing()
    
    # 2. 궤적 파싱 데모
    demo_trajectory_parsing()
    
    # 3. VLM 출력 파싱 데모
    demo_vlm_output_parsing()
    
    # 4. 성능 비교 데모
    demo_comparison_with_simple_parser()
    
    # 결과 통계
    print("\n📊 데모 결과 통계")
    print("=" * 40)
    
    total_tests = len(continuous_results)
    safe_actions = sum(1 for r in continuous_results if r["is_safe"])
    avg_confidence = sum(r["confidence"] for r in continuous_results) / total_tests
    
    action_types = {}
    for r in continuous_results:
        action_types[r["action_type"]] = action_types.get(r["action_type"], 0) + 1
    
    print(f"전체 테스트: {total_tests}")
    print(f"안전한 액션: {safe_actions} ({safe_actions/total_tests*100:.1f}%)")
    print(f"평균 신뢰도: {avg_confidence:.3f}")
    print(f"액션 타입 분포: {action_types}")
    
    # 결과 저장
    demo_results = {
        "continuous_parsing": continuous_results,
        "statistics": {
            "total_tests": total_tests,
            "safe_actions": safe_actions,
            "avg_confidence": avg_confidence,
            "action_type_distribution": action_types
        }
    }
    
    with open("robovlm_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 데모 완료! 결과가 'robovlm_demo_results.json'에 저장되었습니다.")

if __name__ == "__main__":
    main() 