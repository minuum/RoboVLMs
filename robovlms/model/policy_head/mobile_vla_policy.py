"""
Mobile VLA 전용 Policy Head
2D 속도 (linear_x, linear_y) 처리에 특화
LSTMDecoder를 기반으로 하되, gripper 없이 2D 속도만 출력
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from robovlms.model.policy_head.base_policy import BasePolicyHead, lstm_decoder, MLPTanhHead, initialize_param


class MobileVLALSTMDecoder(BasePolicyHead):
    """
    Mobile VLA 전용 LSTMDecoder
    
    특징:
    - 2D 속도 (linear_x, linear_y)만 출력 - 0.4초 동안의 이동 방향 속도 조정
    - Gripper 없음
    - BasePolicyHead.loss를 오버라이드하여 2D 속도 Loss 계산
    """
    
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        super(MobileVLALSTMDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        
        # LSTM Decoder
        self.rnn = lstm_decoder(
            in_features * latent, hidden_size * latent, num_layers, policy_rnn_dropout_p
        )
        
        # 2D 속도 출력 (gripper 없음) - 0.4초 동안의 이동 방향 속도 조정
        # action_dim=2 (linear_x, linear_y)이므로 fwd_pred_next_n * 2 차원 출력
        self.velocities = MLPTanhHead(
            self.hidden_size * latent, fwd_pred_next_n * action_dim
        )
        
        self.hidden_state = None
        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError
        initialize_param(self)

    def reset(self):
        self.hidden_state = None
        self.history_memory = []

    def forward(self, tok_seq, h_0=None, **kwargs):
        """
        Forward pass (LSTMDecoder와 동일한 구조)
        
        Args:
            tok_seq: (B, seq_len, latent_num, feature_dim) 또는 (B, seq_len, in_features * latent)
            h_0: 초기 hidden state (optional)
        
        Returns:
            velocities: (B, seq_len, fwd_pred_next_n, action_dim) - 2D 속도 (linear_x, linear_y)
            None: gripper 없음 (BasePolicyHead.loss 호환성을 위해)
        """
        print(f"DEBUG: MobileVLALSTMDecoder forward input tok_seq shape: {tok_seq.shape}")
        # Down sample 처리 (LSTMDecoder와 동일)
        # tok_seq shape 확인 및 처리
        if len(tok_seq.shape) == 4:
            # (B, seq_len, latent_num, feature_dim)
            if self.down_sample == "pooling":
                bs, seq_len = tok_seq.shape[:2]
                tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
                tok_seq = self.global_1d_pool(
                    tok_seq.permute(0, 2, 1)
                )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
                tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
            elif self.down_sample == "resampler":
                raise NotImplementedError
            elif self.down_sample == "none":
                tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
            else:
                raise NotImplementedError
        elif len(tok_seq.shape) == 3:
            # (B, seq_len, feature_dim) - 이미 flatten된 경우
            # latent=1이므로 그대로 사용
            pass
        else:
            raise ValueError(f"Unexpected tok_seq shape: {tok_seq.shape}")

        # History memory 처리 (LSTMDecoder와 동일)
        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)
            if len(self.history_memory) <= self.history_len:
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
        else:
            self.hidden_state = h_0
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n

        # 2D 속도 출력 - 0.4초 동안의 이동 방향 속도 조정
        velocities = self.velocities(x)
        # (B, seq_len, fwd_pred_next_n * action_dim) -> (B, seq_len, fwd_pred_next_n, action_dim)
        velocities = rearrange(velocities, "b l (n d) -> b l n d", n=self.fwd_pred_next_n, d=self.action_dim)

        # gripper 없음 (None 반환)
        return velocities, None

    def loss(self, pred_action, labels, attention_mask=None):
        """
        Mobile VLA용 Loss 계산
        2D 속도 (linear_x, linear_y)만 처리 - 0.4초 동안의 이동 방향 속도 조정
        
        Args:
            pred_action: (velocities, gripper) - gripper는 None
            labels: (velocity_chunck, gripper_action_chunck) - gripper_action_chunck는 None
            attention_mask: (B, seq_len, chunk_size)
        
        Returns:
            dict: {"loss_velocity": ..., "loss_gripper": None, "acc_gripper": None}
        """
        if labels is None or labels[0] is None:
            return {"loss_velocity": None, "loss_gripper": None, "acc_gripper": None}

        # pred_action는 (velocities, None) 형태
        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            velocities = pred_action[0]  # (B, seq_len, chunk_size, 2) - [linear_x, linear_y]
        else:
            velocities = pred_action

        # labels는 (velocity_chunck, None) 형태
        velocity_labels = labels[0]  # (B, seq_len, chunk_size, 2) - [linear_x, linear_y]

        # 2D 속도 Loss 계산 (Huber Loss) - 0.4초 동안의 이동 방향 속도 조정
        if attention_mask is None:
            loss_velocity = torch.nn.functional.huber_loss(velocities, velocity_labels)
        else:
            loss_velocity = torch.nn.functional.huber_loss(
                velocities, velocity_labels, reduction="none"
            )
            attention_mask = attention_mask.bool()
            loss_velocity = loss_velocity[attention_mask].mean()

        return {
            "loss_velocity": loss_velocity,  # loss_arm -> loss_velocity
            "loss_gripper": None,  # Mobile VLA는 gripper 없음
            "acc_gripper": None,  # Mobile VLA는 gripper 없음
        }

