"""
Hybrid Action Head: 방향 Classification + 크기 Regression

VLA 학습에서 방향(left/right)은 discrete한 선택이고,
크기(속도)는 continuous한 값입니다.

이를 분리하여 학습하면:
1. 방향: Binary Classification (CrossEntropy Loss)
2. 크기: Regression (MSE Loss)

작성일: 2025-12-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class HybridActionHead(nn.Module):
    """
    Hybrid Action Head: 방향과 크기를 분리하여 학습
    
    출력:
    - direction: (B, seq_len, 2) - left/right logits
    - magnitude: (B, seq_len, 1) - 속도 크기 (0~1)
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        lstm_hidden: int = 512,
        direction_classes: int = 2,  # left, right
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden
        
        # 공유 LSTM
        self.shared_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 방향 분류 헤드
        self.direction_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden // 2, direction_classes)  # left/right
        )
        
        # 크기 회귀 헤드 (Tanh로 0~1 범위)
        self.magnitude_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden // 2, 2),  # linear_x, linear_y magnitude
            nn.Sigmoid()  # 0~1 범위
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        direction_labels: Optional[torch.Tensor] = None,
        magnitude_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (B, seq_len, hidden_size)
            direction_labels: (B, seq_len) - 0: left, 1: right
            magnitude_labels: (B, seq_len, 2) - [|linear_x|, |linear_y|]
            
        Returns:
            dict with:
            - direction_logits: (B, seq_len, 2)
            - magnitude: (B, seq_len, 2)
            - direction_loss: scalar (if labels provided)
            - magnitude_loss: scalar (if labels provided)
        """
        B, seq_len, _ = hidden_states.shape
        
        # LSTM forward
        lstm_out, _ = self.shared_lstm(hidden_states)  # (B, seq_len, lstm_hidden)
        
        # 방향 예측
        direction_logits = self.direction_head(lstm_out)  # (B, seq_len, 2)
        
        # 크기 예측
        magnitude = self.magnitude_head(lstm_out)  # (B, seq_len, 2)
        
        result = {
            'direction_logits': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude,
        }
        
        # 손실 계산
        if direction_labels is not None:
            direction_loss = F.cross_entropy(
                direction_logits.view(-1, 2),
                direction_labels.view(-1).long()
            )
            result['direction_loss'] = direction_loss
        
        if magnitude_labels is not None:
            magnitude_loss = F.mse_loss(magnitude, magnitude_labels)
            result['magnitude_loss'] = magnitude_loss
        
        # 전체 손실
        if 'direction_loss' in result and 'magnitude_loss' in result:
            result['total_loss'] = result['direction_loss'] + result['magnitude_loss']
        
        return result
    
    def predict_action(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        최종 action 예측 (방향 + 크기 결합)
        
        Returns:
            actions: (B, seq_len, 2) - [linear_x, linear_y]
        """
        result = self.forward(hidden_states)
        
        direction_probs = result['direction_probs']  # (B, seq_len, 2)
        magnitude = result['magnitude']  # (B, seq_len, 2)
        
        # 방향 결정: left(0) → +1, right(1) → -1
        direction = torch.argmax(direction_probs, dim=-1)  # (B, seq_len)
        direction_sign = torch.where(direction == 0, 1.0, -1.0)  # left=+1, right=-1
        
        # linear_y에 방향 적용
        actions = magnitude.clone()
        actions[:, :, 1] = actions[:, :, 1] * direction_sign
        
        # linear_x는 항상 양수 (전진)
        
        return actions


# 테스트 코드
if __name__ == "__main__":
    print("Testing HybridActionHead...")
    
    # 모델 생성
    model = HybridActionHead(hidden_size=2048, lstm_hidden=512)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 더미 입력
    B, seq_len, hidden = 2, 8, 2048
    hidden_states = torch.randn(B, seq_len, hidden)
    
    # Forward (예측만)
    result = model(hidden_states)
    print(f"Direction logits shape: {result['direction_logits'].shape}")
    print(f"Magnitude shape: {result['magnitude'].shape}")
    
    # Forward with labels
    direction_labels = torch.randint(0, 2, (B, seq_len))
    magnitude_labels = torch.rand(B, seq_len, 2)
    
    result = model(hidden_states, direction_labels, magnitude_labels)
    print(f"Direction loss: {result['direction_loss'].item():.4f}")
    print(f"Magnitude loss: {result['magnitude_loss'].item():.4f}")
    print(f"Total loss: {result['total_loss'].item():.4f}")
    
    # Action 예측
    actions = model.predict_action(hidden_states)
    print(f"Predicted actions shape: {actions.shape}")
    print(f"Predicted actions sample: {actions[0, 0]}")
    
    print("✅ Test passed!")
