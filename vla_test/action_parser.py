#!/usr/bin/env python3
"""
VLA 모델의 추론 결과를 로봇 액션으로 파싱하는 모듈
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """액션 타입 정의"""
    MOVE = "move"
    TURN = "turn"
    STOP = "stop"
    GRAB = "grab"
    RELEASE = "release"
    POINT = "point"
    LOOK = "look"
    NAVIGATE = "navigate"
    AVOID = "avoid"
    UNKNOWN = "unknown"

@dataclass
class RobotAction:
    """로봇 액션 데이터 클래스"""
    action_type: ActionType
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    target_object: Optional[str] = None
    confidence: float = 0.0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "action_type": self.action_type.value,
            "linear_x": self.linear_x,
            "linear_y": self.linear_y,
            "angular_z": self.angular_z,
            "target_object": self.target_object,
            "confidence": self.confidence,
            "description": self.description
        }

class VLAActionParser:
    """VLA 결과를 액션으로 파싱하는 클래스"""
    
    def __init__(self):
        # 키워드 기반 액션 매핑
        self.action_keywords = {
            ActionType.MOVE: [
                "move", "go", "forward", "backward", "advance", "proceed",
                "drive", "walk", "travel", "전진", "후진", "이동"
            ],
            ActionType.TURN: [
                "turn", "rotate", "spin", "pivot", "twist", "left", "right",
                "회전", "돌다", "좌회전", "우회전"
            ],
            ActionType.STOP: [
                "stop", "halt", "pause", "brake", "freeze", "wait",
                "정지", "멈춤", "대기"
            ],
            ActionType.GRAB: [
                "grab", "grasp", "pick", "take", "hold", "catch", "seize",
                "잡다", "들다", "집다"
            ],
            ActionType.RELEASE: [
                "release", "drop", "let", "place", "put", "set",
                "놓다", "내려놓다", "둔다"
            ],
            ActionType.POINT: [
                "point", "indicate", "show", "direct", "aim",
                "가리키다", "지시하다"
            ],
            ActionType.LOOK: [
                "look", "see", "watch", "observe", "scan", "search",
                "보다", "관찰하다", "찾다"
            ],
            ActionType.NAVIGATE: [
                "navigate", "find", "reach", "approach", "head",
                "네비게이트", "찾아가다", "도달하다"
            ],
            ActionType.AVOID: [
                "avoid", "dodge", "evade", "bypass", "circumvent",
                "피하다", "회피하다"
            ]
        }
        
        # 방향성 키워드
        self.direction_keywords = {
            "forward": (0.3, 0.0, 0.0),
            "backward": (-0.3, 0.0, 0.0),
            "left": (0.0, 0.0, 0.5),
            "right": (0.0, 0.0, -0.5),
            "up": (0.0, 0.0, 0.0),
            "down": (0.0, 0.0, 0.0),
            "전진": (0.3, 0.0, 0.0),
            "후진": (-0.3, 0.0, 0.0),
            "좌측": (0.0, 0.0, 0.5),
            "우측": (0.0, 0.0, -0.5)
        }
        
        # 속도 수식어
        self.speed_modifiers = {
            "slowly": 0.5,
            "fast": 1.5,
            "quickly": 1.5,
            "carefully": 0.3,
            "천천히": 0.5,
            "빠르게": 1.5,
            "조심스럽게": 0.3
        }

    def parse_text_output(self, vla_output: str, original_prompt: str = "") -> RobotAction:
        """VLA 텍스트 출력을 로봇 액션으로 파싱"""
        text = vla_output.lower().strip()
        
        # 프롬프트 제거
        if original_prompt:
            cleaned_prompt = original_prompt.lower().strip()
            if text.startswith(cleaned_prompt):
                text = text[len(cleaned_prompt):].strip()
        
        # 액션 타입 결정
        action_type = self._determine_action_type(text)
        
        # 기본 액션 생성
        action = RobotAction(
            action_type=action_type,
            description=vla_output,
            confidence=self._calculate_confidence(text, action_type)
        )
        
        # 세부 파싱
        if action_type == ActionType.MOVE:
            action = self._parse_movement(text, action)
        elif action_type == ActionType.TURN:
            action = self._parse_turn(text, action)
        elif action_type == ActionType.STOP:
            action = self._parse_stop(text, action)
        elif action_type == ActionType.GRAB:
            action = self._parse_grab(text, action)
        elif action_type == ActionType.NAVIGATE:
            action = self._parse_navigation(text, action)
        elif action_type == ActionType.AVOID:
            action = self._parse_avoidance(text, action)
        else:
            action = self._parse_general(text, action)
        
        return action

    def parse_segmentation_output(self, vla_output: str, image_width: int, image_height: int) -> RobotAction:
        """세그멘테이션 토큰이 포함된 출력 파싱"""
        # 위치 토큰 추출
        loc_tokens = re.findall(r"<loc(\d{4})>", vla_output)
        seg_tokens = re.findall(r"<seg(\d{3})>", vla_output)
        
        # 객체 레이블 추출
        label_text = vla_output
        for token in re.findall(r"<loc\d{4}>", label_text):
            label_text = label_text.replace(token, "")
        for token in re.findall(r"<seg\d{3}>", label_text):
            label_text = label_text.replace(token, "")
        
        target_object = label_text.strip()
        
        action = RobotAction(
            action_type=ActionType.NAVIGATE,
            target_object=target_object,
            description=vla_output
        )
        
        # 위치 토큰이 있으면 이동 계산
        if len(loc_tokens) == 4:
            y_min, x_min, y_max, x_max = [int(t) / 1023.0 for t in loc_tokens]
            bbox = [x_min * image_width, y_min * image_height,
                   x_max * image_width, y_max * image_height]
            
            linear_x, linear_y, angular_z = self._calculate_movement_from_bbox(
                bbox, image_width, image_height
            )
            
            action.linear_x = linear_x
            action.linear_y = linear_y
            action.angular_z = angular_z
            action.confidence = 0.8
        
        return action

    def _determine_action_type(self, text: str) -> ActionType:
        """텍스트에서 액션 타입 결정"""
        scores = {}
        
        for action_type, keywords in self.action_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            scores[action_type] = score
        
        # 가장 높은 점수의 액션 타입 반환
        best_action = max(scores, key=scores.get)
        
        if scores[best_action] == 0:
            return ActionType.UNKNOWN
        
        return best_action

    def _calculate_confidence(self, text: str, action_type: ActionType) -> float:
        """신뢰도 계산"""
        base_confidence = 0.5
        
        # 키워드 매칭 점수
        keywords = self.action_keywords.get(action_type, [])
        keyword_matches = sum(1 for keyword in keywords if keyword in text)
        keyword_score = min(keyword_matches * 0.2, 0.4)
        
        # 텍스트 길이 기반 점수
        length_score = min(len(text.split()) * 0.05, 0.1)
        
        return min(base_confidence + keyword_score + length_score, 1.0)

    def _parse_movement(self, text: str, action: RobotAction) -> RobotAction:
        """이동 액션 파싱"""
        linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
        
        # 방향 감지
        for direction, (lx, ly, az) in self.direction_keywords.items():
            if direction in text:
                linear_x, linear_y, angular_z = lx, ly, az
                break
        
        # 속도 수정자 적용
        speed_multiplier = 1.0
        for modifier, multiplier in self.speed_modifiers.items():
            if modifier in text:
                speed_multiplier = multiplier
                break
        
        action.linear_x = linear_x * speed_multiplier
        action.linear_y = linear_y * speed_multiplier
        action.angular_z = angular_z * speed_multiplier
        
        return action

    def _parse_turn(self, text: str, action: RobotAction) -> RobotAction:
        """회전 액션 파싱"""
        if "left" in text or "좌" in text:
            action.angular_z = 0.5
        elif "right" in text or "우" in text:
            action.angular_z = -0.5
        else:
            action.angular_z = 0.3  # 기본 회전
        
        # 속도 수정자 적용
        for modifier, multiplier in self.speed_modifiers.items():
            if modifier in text:
                action.angular_z *= multiplier
                break
        
        return action

    def _parse_stop(self, text: str, action: RobotAction) -> RobotAction:
        """정지 액션 파싱"""
        action.linear_x = 0.0
        action.linear_y = 0.0
        action.angular_z = 0.0
        action.confidence = 0.9
        return action

    def _parse_grab(self, text: str, action: RobotAction) -> RobotAction:
        """잡기 액션 파싱"""
        # 목표 객체 추출
        target_patterns = [
            r"grab (\w+)",
            r"pick up (\w+)",
            r"take (\w+)",
            r"hold (\w+)"
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                action.target_object = match.group(1)
                break
        
        return action

    def _parse_navigation(self, text: str, action: RobotAction) -> RobotAction:
        """네비게이션 액션 파싱"""
        # 목표 위치 추출
        target_patterns = [
            r"navigate to (\w+)",
            r"go to (\w+)",
            r"find (\w+)",
            r"reach (\w+)"
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                action.target_object = match.group(1)
                break
        
        # 기본 전진 명령
        action.linear_x = 0.2
        
        return action

    def _parse_avoidance(self, text: str, action: RobotAction) -> RobotAction:
        """회피 액션 파싱"""
        # 회피 방향 결정
        if "left" in text:
            action.angular_z = 0.3
        elif "right" in text:
            action.angular_z = -0.3
        else:
            action.linear_x = -0.1  # 후진
        
        return action

    def _parse_general(self, text: str, action: RobotAction) -> RobotAction:
        """일반적인 텍스트 파싱"""
        # 방향성 키워드 탐지
        for direction, (lx, ly, az) in self.direction_keywords.items():
            if direction in text:
                action.linear_x = lx * 0.5  # 보수적인 속도
                action.linear_y = ly * 0.5
                action.angular_z = az * 0.5
                break
        
        return action

    def _calculate_movement_from_bbox(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float]:
        """바운딩 박스로부터 이동 명령 계산"""
        x_min, y_min, x_max, y_max = bbox
        
        # 중심점 계산
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 이미지 중심과의 오차 계산
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        error_x = center_x - img_center_x
        error_y = center_y - img_center_y
        
        # 제어 게인
        kp_angular = 0.005
        kp_linear = 0.0005
        
        # 정렬 허용 오차
        tolerance_x = img_width * 0.05
        tolerance_y = img_height * 0.05
        
        linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
        
        # 객체 크기 기반 거리 판단
        bbox_area = (x_max - x_min) * (y_max - y_min)
        img_area = img_width * img_height
        area_ratio = bbox_area / img_area
        
        # 너무 가까우면 정지
        if area_ratio > 0.3:
            return 0.0, 0.0, 0.0
        
        # 수평 정렬
        if abs(error_x) > tolerance_x:
            angular_z = -kp_angular * error_x
            angular_z = max(-0.5, min(0.5, angular_z))
        else:
            # 정렬됨 - 전진
            linear_x = 0.2
        
        return linear_x, linear_y, angular_z

class ActionValidator:
    """액션 유효성 검사 클래스"""
    
    def __init__(self, max_linear_speed=0.5, max_angular_speed=1.0):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    def validate_action(self, action: RobotAction) -> RobotAction:
        """액션 유효성 검사 및 수정"""
        # 속도 제한
        action.linear_x = max(-self.max_linear_speed, 
                             min(self.max_linear_speed, action.linear_x))
        action.linear_y = max(-self.max_linear_speed, 
                             min(self.max_linear_speed, action.linear_y))
        action.angular_z = max(-self.max_angular_speed, 
                              min(self.max_angular_speed, action.angular_z))
        
        # 신뢰도 제한
        action.confidence = max(0.0, min(1.0, action.confidence))
        
        return action

    def is_safe_action(self, action: RobotAction) -> bool:
        """액션 안전성 검사"""
        # 속도가 너무 높은지 확인
        if abs(action.linear_x) > self.max_linear_speed:
            return False
        if abs(action.angular_z) > self.max_angular_speed:
            return False
        
        # 신뢰도가 너무 낮은지 확인
        if action.confidence < 0.3:
            return False
        
        return True

if __name__ == "__main__":
    # 테스트 코드
    parser = VLAActionParser()
    validator = ActionValidator()
    
    test_outputs = [
        "move forward slowly",
        "turn left quickly",
        "stop immediately",
        "grab the red ball",
        "navigate to the door",
        "avoid the obstacle on the right",
        "<loc0500><loc0300><loc0700><loc0600>cup segment",
        "전진하세요",
        "우회전 하세요"
    ]
    
    print("🧪 액션 파서 테스트")
    print("=" * 50)
    
    for i, output in enumerate(test_outputs, 1):
        print(f"\n{i}. 입력: '{output}'")
        
        if "<loc" in output:
            action = parser.parse_segmentation_output(output, 640, 480)
        else:
            action = parser.parse_text_output(output)
        
        action = validator.validate_action(action)
        
        print(f"   액션 타입: {action.action_type.value}")
        print(f"   속도: linear_x={action.linear_x:.2f}, angular_z={action.angular_z:.2f}")
        print(f"   목표: {action.target_object or 'N/A'}")
        print(f"   신뢰도: {action.confidence:.2f}")
        print(f"   안전성: {'✅' if validator.is_safe_action(action) else '❌'}")
    
    print("\n✅ 테스트 완료") 