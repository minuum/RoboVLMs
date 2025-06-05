#!/usr/bin/env python3
"""
RoboVLM 기반 고급 액션 파서
실제 RoboVLMs 모델의 출력 형태를 처리하는 전문 파서
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import copy

class ActionSpace(Enum):
    """액션 공간 타입"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

class RobotControl(Enum):
    """로봇 제어 타입"""
    POSITION = "position"      # 위치 제어
    VELOCITY = "velocity"      # 속도 제어  
    FORCE = "force"           # 힘 제어
    TRAJECTORY = "trajectory"  # 궤적 제어

@dataclass
class RoboAction:
    """RoboVLMs 스타일 로봇 액션"""
    # 6DOF 액션 (x, y, z, roll, pitch, yaw)
    translation: np.ndarray = None  # (3,) [x, y, z]
    rotation: np.ndarray = None     # (3,) [roll, pitch, yaw] 
    gripper: float = 0.0           # 그리퍼 상태 (0: 열림, 1: 닫힘)
    
    # 메타데이터
    action_type: str = "unknown"
    confidence: float = 0.0
    control_mode: RobotControl = RobotControl.VELOCITY
    
    # 시퀀스 정보
    sequence_length: int = 1
    prediction_horizon: int = 1
    
    def to_6dof_array(self) -> np.ndarray:
        """6DOF 배열로 변환"""
        if self.translation is None:
            self.translation = np.zeros(3)
        if self.rotation is None:
            self.rotation = np.zeros(3)
        
        return np.concatenate([self.translation, self.rotation])
    
    def to_twist_like(self) -> Tuple[float, float, float]:
        """ROS Twist 메시지 스타일로 변환"""
        if self.translation is None:
            linear_x, linear_y = 0.0, 0.0
        else:
            linear_x, linear_y = float(self.translation[0]), float(self.translation[1])
            
        if self.rotation is None:
            angular_z = 0.0
        else:
            angular_z = float(self.rotation[2])  # yaw만 사용
            
        return linear_x, linear_y, angular_z
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "translation": self.translation.tolist() if self.translation is not None else None,
            "rotation": self.rotation.tolist() if self.rotation is not None else None,
            "gripper": self.gripper,
            "action_type": self.action_type,
            "confidence": self.confidence,
            "control_mode": self.control_mode.value,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon
        }

class RoboVLMActionParser:
    """RoboVLMs 모델 출력을 파싱하는 고급 파서"""
    
    def __init__(self, 
                 action_space: ActionSpace = ActionSpace.CONTINUOUS,
                 action_dim: int = 6,
                 bins: int = 256,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 prediction_horizon: int = 1):
        
        self.action_space = action_space
        self.action_dim = action_dim
        self.bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.prediction_horizon = prediction_horizon
        
        # 액션 정규화를 위한 빈 생성 (ActionTokenizer 스타일)
        if action_space == ActionSpace.DISCRETE:
            self.action_bins = np.linspace(min_action, max_action, bins)
            self.bin_centers = (self.action_bins[:-1] + self.action_bins[1:]) / 2.0
        
        # 액션 타입별 기본 설정
        self.action_configs = {
            "move": {
                "control_mode": RobotControl.VELOCITY,
                "default_speed": 0.3,
                "gripper_action": False
            },
            "turn": {
                "control_mode": RobotControl.VELOCITY, 
                "default_angular_speed": 0.5,
                "gripper_action": False
            },
            "stop": {
                "control_mode": RobotControl.VELOCITY,
                "gripper_action": False
            },
            "grab": {
                "control_mode": RobotControl.POSITION,
                "approach_speed": 0.1,
                "gripper_action": True,
                "gripper_state": 1.0
            },
            "release": {
                "control_mode": RobotControl.POSITION,
                "gripper_action": True,
                "gripper_state": 0.0
            },
            "navigate": {
                "control_mode": RobotControl.TRAJECTORY,
                "approach_speed": 0.2,
                "gripper_action": False
            }
        }

    def parse_continuous_action(self, 
                              action_tensor: torch.Tensor,
                              text_instruction: str = "",
                              vision_features: Optional[torch.Tensor] = None) -> RoboAction:
        """연속 액션 텐서 파싱 (BaseRoboVLM.forward_continuous 출력)"""
        
        if isinstance(action_tensor, torch.Tensor):
            action_array = action_tensor.detach().cpu().numpy()
        else:
            action_array = np.array(action_tensor)
            
        # 형태 확인 및 정규화
        if action_array.ndim == 3:  # (batch, seq_len, action_dim)
            action_array = action_array[0, -1]  # 마지막 시퀀스 사용
        elif action_array.ndim == 2:  # (seq_len, action_dim)
            action_array = action_array[-1]     # 마지막 시퀀스 사용
        
        # 액션 정규화 ([-1, 1] -> 실제 제어 값)
        action_array = np.clip(action_array, self.min_action, self.max_action)
        
        # 6DOF 액션 분해
        if len(action_array) >= 6:
            translation = action_array[:3]
            rotation = action_array[3:6]
            gripper = action_array[6] if len(action_array) > 6 else 0.0
        else:
            # 부족한 차원은 0으로 패딩
            padded_action = np.zeros(6)
            padded_action[:len(action_array)] = action_array
            translation = padded_action[:3]
            rotation = padded_action[3:6]
            gripper = 0.0
        
        # 텍스트 기반 액션 타입 추론
        action_type = self._infer_action_type(text_instruction)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(action_array, text_instruction)
        
        return RoboAction(
            translation=translation,
            rotation=rotation,
            gripper=gripper,
            action_type=action_type,
            confidence=confidence,
            control_mode=self.action_configs.get(action_type, {}).get(
                "control_mode", RobotControl.VELOCITY
            ),
            prediction_horizon=self.prediction_horizon
        )

    def parse_discrete_action(self, 
                            action_token_ids: Union[torch.Tensor, List[int]],
                            text_instruction: str = "") -> RoboAction:
        """이산 액션 토큰 파싱 (ActionTokenizer.decode_token_ids_to_actions 스타일)"""
        
        if isinstance(action_token_ids, torch.Tensor):
            token_ids = action_token_ids.detach().cpu().numpy()
        else:
            token_ids = np.array(action_token_ids)
        
        # 토큰 ID를 연속 액션으로 변환
        if self.action_space == ActionSpace.DISCRETE:
            # ActionTokenizer 스타일 디코딩
            discretized_actions = self.bins - token_ids
            discretized_actions = np.clip(
                discretized_actions - 1, 
                a_min=0, 
                a_max=self.bin_centers.shape[0] - 1
            )
            action_array = self.bin_centers[discretized_actions]
        else:
            raise ValueError("Discrete parsing requires ActionSpace.DISCRETE")
        
        # 연속 액션 파싱과 동일하게 처리
        return self.parse_continuous_action(action_array, text_instruction)

    def parse_trajectory_sequence(self,
                                action_sequence: torch.Tensor,
                                text_instruction: str = "") -> List[RoboAction]:
        """궤적 시퀀스 파싱 (trajectory_gpt2 출력)"""
        
        if isinstance(action_sequence, torch.Tensor):
            seq_array = action_sequence.detach().cpu().numpy()
        else:
            seq_array = np.array(action_sequence)
        
        # (batch, seq_len, action_dim) or (seq_len, action_dim)
        if seq_array.ndim == 3:
            seq_array = seq_array[0]  # 첫 번째 배치 사용
        
        actions = []
        for i, action_step in enumerate(seq_array):
            action = self.parse_continuous_action(action_step, text_instruction)
            action.sequence_length = len(seq_array)
            action.prediction_horizon = len(seq_array) - i  # 남은 스텝
            actions.append(action)
        
        return actions

    def parse_vision_language_action(self,
                                   vlm_output: Dict[str, Any],
                                   text_instruction: str = "",
                                   image_features: Optional[torch.Tensor] = None) -> RoboAction:
        """VLM 전체 출력 파싱 (BaseRoboVLM.forward 출력)"""
        
        # VLM 출력에서 액션 관련 부분 추출
        if "action_pred" in vlm_output:
            action_pred = vlm_output["action_pred"]
            
            if self.action_space == ActionSpace.CONTINUOUS:
                return self.parse_continuous_action(action_pred, text_instruction, image_features)
            else:
                return self.parse_discrete_action(action_pred, text_instruction)
        
        elif "instr_and_action_pred" in vlm_output:
            # 명령어와 액션이 함께 예측된 경우
            pred_tokens = vlm_output["instr_and_action_pred"]
            return self.parse_discrete_action(pred_tokens, text_instruction)
        
        else:
            # fallback: 텍스트만으로 액션 생성
            return self._text_only_action(text_instruction)

    def _infer_action_type(self, text: str) -> str:
        """텍스트에서 액션 타입 추론"""
        text_lower = text.lower()
        
        action_keywords = {
            "move": ["move", "go", "forward", "backward", "전진", "후진", "이동"],
            "turn": ["turn", "rotate", "left", "right", "회전", "돌다"],
            "stop": ["stop", "halt", "brake", "정지", "멈춤"],
            "grab": ["grab", "grasp", "pick", "take", "잡다", "들다"],
            "release": ["release", "drop", "put", "place", "놓다", "내려놓다"],
            "navigate": ["navigate", "find", "reach", "go to", "찾아가다"]
        }
        
        for action_type, keywords in action_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return action_type
        
        return "unknown"

    def _calculate_confidence(self, action_array: np.ndarray, text: str) -> float:
        """액션 신뢰도 계산"""
        base_confidence = 0.7
        
        # 액션 크기 기반 신뢰도
        action_magnitude = np.linalg.norm(action_array)
        magnitude_confidence = min(action_magnitude * 0.2, 0.2)
        
        # 텍스트 매칭 기반 신뢰도  
        text_confidence = 0.1 if text and len(text) > 3 else 0.0
        
        return min(base_confidence + magnitude_confidence + text_confidence, 1.0)

    def _text_only_action(self, text: str) -> RoboAction:
        """텍스트만으로 기본 액션 생성 (fallback)"""
        action_type = self._infer_action_type(text)
        config = self.action_configs.get(action_type, {})
        
        # 기본 액션 생성
        translation = np.zeros(3)
        rotation = np.zeros(3)
        gripper = 0.0
        
        if action_type == "move":
            if "forward" in text.lower() or "전진" in text:
                translation[0] = config.get("default_speed", 0.3)
            elif "backward" in text.lower() or "후진" in text:
                translation[0] = -config.get("default_speed", 0.3)
        
        elif action_type == "turn":
            if "left" in text.lower() or "좌" in text:
                rotation[2] = config.get("default_angular_speed", 0.5)
            elif "right" in text.lower() or "우" in text:
                rotation[2] = -config.get("default_angular_speed", 0.5)
        
        elif action_type == "grab":
            gripper = config.get("gripper_state", 1.0)
            translation[0] = config.get("approach_speed", 0.1)
        
        return RoboAction(
            translation=translation,
            rotation=rotation,
            gripper=gripper,
            action_type=action_type,
            confidence=0.6,  # 텍스트만으로는 낮은 신뢰도
            control_mode=config.get("control_mode", RobotControl.VELOCITY)
        )

class ActionValidator:
    """RoboVLM 액션 검증기"""
    
    def __init__(self,
                 max_translation_speed: float = 0.5,
                 max_rotation_speed: float = 1.0,
                 safety_bounds: Dict[str, Tuple[float, float]] = None):
        
        self.max_translation_speed = max_translation_speed
        self.max_rotation_speed = max_rotation_speed
        
        # 안전 경계값
        self.safety_bounds = safety_bounds or {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0), 
            "z": (-0.5, 0.5),
            "roll": (-np.pi, np.pi),
            "pitch": (-np.pi/2, np.pi/2),
            "yaw": (-np.pi, np.pi)
        }

    def validate_action(self, action: RoboAction) -> RoboAction:
        """액션 유효성 검사 및 클리핑"""
        if action.translation is not None:
            # 속도 제한
            speed = np.linalg.norm(action.translation)
            if speed > self.max_translation_speed:
                action.translation = action.translation * (self.max_translation_speed / speed)
            
            # 경계값 클리핑
            action.translation[0] = np.clip(action.translation[0], 
                                          self.safety_bounds["x"][0], 
                                          self.safety_bounds["x"][1])
            action.translation[1] = np.clip(action.translation[1],
                                          self.safety_bounds["y"][0],
                                          self.safety_bounds["y"][1])
            action.translation[2] = np.clip(action.translation[2],
                                          self.safety_bounds["z"][0], 
                                          self.safety_bounds["z"][1])
        
        if action.rotation is not None:
            # 회전 속도 제한
            angular_speed = np.linalg.norm(action.rotation)
            if angular_speed > self.max_rotation_speed:
                action.rotation = action.rotation * (self.max_rotation_speed / angular_speed)
            
            # 각도 제한
            action.rotation[0] = np.clip(action.rotation[0],
                                       self.safety_bounds["roll"][0],
                                       self.safety_bounds["roll"][1])
            action.rotation[1] = np.clip(action.rotation[1],
                                       self.safety_bounds["pitch"][0], 
                                       self.safety_bounds["pitch"][1])
            action.rotation[2] = np.clip(action.rotation[2],
                                       self.safety_bounds["yaw"][0],
                                       self.safety_bounds["yaw"][1])
        
        # 그리퍼 상태 클리핑
        action.gripper = np.clip(action.gripper, 0.0, 1.0)
        
        return action

    def is_safe_action(self, action: RoboAction) -> bool:
        """액션 안전성 검사"""
        if action.translation is not None:
            if np.any(np.abs(action.translation) > self.max_translation_speed):
                return False
        
        if action.rotation is not None:
            if np.any(np.abs(action.rotation) > self.max_rotation_speed):
                return False
        
        if action.confidence < 0.3:
            return False
        
        return True

if __name__ == "__main__":
    # 테스트 코드
    print("🤖 RoboVLM 액션 파서 테스트")
    print("=" * 50)
    
    # 파서 초기화
    parser = RoboVLMActionParser(
        action_space=ActionSpace.CONTINUOUS,
        action_dim=7,  # 6DOF + 그리퍼
        prediction_horizon=1
    )
    validator = ActionValidator()
    
    # 테스트 1: 연속 액션 파싱
    print("\n1. 연속 액션 파싱 테스트")
    action_tensor = torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.8]])  # 전진 + 좌회전 + 그리퍼 닫기
    action = parser.parse_continuous_action(action_tensor, "전진하면서 물건을 잡아")
    print(f"   액션 타입: {action.action_type}")
    print(f"   위치: {action.translation}")
    print(f"   회전: {action.rotation}")
    print(f"   그리퍼: {action.gripper}")
    print(f"   신뢰도: {action.confidence:.2f}")
    
    # Twist 변환 테스트
    linear_x, linear_y, angular_z = action.to_twist_like()
    print(f"   ROS Twist: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
    
    # 테스트 2: 텍스트만으로 액션 생성
    print("\n2. 텍스트 전용 액션 생성")
    text_commands = [
        "move forward slowly",
        "turn right quickly", 
        "stop immediately",
        "grab the red cup",
        "우회전하세요"
    ]
    
    for cmd in text_commands:
        action = parser._text_only_action(cmd)
        action = validator.validate_action(action)
        linear_x, linear_y, angular_z = action.to_twist_like()
        safety = "✅" if validator.is_safe_action(action) else "❌"
        print(f"   '{cmd}' → {action.action_type}: ({linear_x:.2f}, {linear_y:.2f}, {angular_z:.2f}) {safety}")
    
    # 테스트 3: 궤적 시퀀스 파싱
    print("\n3. 궤적 시퀀스 파싱")
    trajectory = torch.tensor([
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 천천히 전진
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 더 빠르게
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 정지하고 그리퍼 닫기
    ])
    
    actions = parser.parse_trajectory_sequence(trajectory, "컵에 접근해서 잡기")
    for i, action in enumerate(actions):
        linear_x, _, angular_z = action.to_twist_like()
        print(f"   Step {i+1}: linear_x={linear_x:.2f}, gripper={action.gripper:.1f}, remaining={action.prediction_horizon}")
    
    print("\n✅ RoboVLM 액션 파서 테스트 완료") 