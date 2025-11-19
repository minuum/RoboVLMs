"""
Mobile VLA 전용 Trainer
2D 속도 (linear_x, linear_y) 처리에 특화된 Trainer
BaseTrainer를 상속받아 _process_batch 메서드를 오버라이드
"""

import torch
from robovlms.train.base_trainer import BaseTrainer


class MobileVLATrainer(BaseTrainer):
    """
    Mobile VLA 전용 Trainer
    
    특징:
    - 2D 속도 처리 (linear_x, linear_y) - 0.4초 동안의 이동 방향 속도 조정
    - Gripper 액션 없음
    - BaseTrainer의 _process_batch를 오버라이드하여 2D 속도 처리
    """
    
    def _process_batch(self, batch):
        """
        Mobile VLA용 배치 처리
        2D 속도 (linear_x, linear_y) 처리에 특화
        
        BaseTrainer의 _process_batch와의 차이점:
        - 7D 액션 (6D arm + 1D gripper) 대신 2D 속도 사용
        - Gripper 관련 로직 제거
        - velocity_chunck를 2D 속도로 직접 사용
        """
        # BaseTrainer의 기본 처리 (rgb, language 등)
        if isinstance(batch, list):
            batch = batch[0]
        if isinstance(batch["rgb"], list):
            rgb = [_.cuda() for _ in batch["rgb"]]
        else:
            rgb = batch["rgb"].cuda()
            if len(rgb.shape) == 4:
                rgb = rgb.unsqueeze(1)
            assert len(rgb.shape) == 5

        if isinstance(batch["text"], list) and isinstance(batch["text"][0], str):
            raise ValueError("The raw text data is not supported")
        else:
            seq_len = self.configs["window_size"]
            language = batch["text"].cuda()
            text_mask = batch["text_mask"].cuda()

        # 2D 속도 처리 (Mobile VLA) - 0.4초 동안의 이동 방향 속도 조정
        if batch.get("action", None) is not None:
            action = batch["action"].cuda()  # (B, seq_len, 2) - [linear_x, linear_y]
            # 2D 속도를 velocity로 직접 사용 (gripper 없음)
            velocity = action  # (B, seq_len, 2) - 속도 명령
            gripper_action = None  # Mobile VLA는 gripper 없음
        else:
            velocity = None
            gripper_action = None

        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = batch["attention_mask"].cuda()

        if self.use_hand_rgb and batch.get("hand_rgb", None) is not None:
            hand_rgb = batch["hand_rgb"].cuda()
        else:
            hand_rgb = None

        # Forward prediction chunks
        fwd_rgb_chunck = batch.get("fwd_rgb_chunck", None)
        fwd_hand_rgb_chunck = batch.get("fwd_hand_rgb_chunck", None)
        if fwd_rgb_chunck is not None:
            fwd_rgb_chunck = fwd_rgb_chunck.cuda()
        if fwd_hand_rgb_chunck is not None:
            fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

        # 2D 속도 chunk 처리 (Mobile VLA) - 0.4초 동안의 이동 방향 속도 조정
        velocity_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get("action_chunck", None)
        if action_chunck is not None:
            action_chunck = action_chunck.cuda()  # (B, seq_len, chunk_size, 2) - [linear_x, linear_y]
            # 2D 속도를 velocity_chunck으로 직접 사용
            velocity_chunck = action_chunck  # (B, seq_len, chunk_size, 2) - 속도 명령 시퀀스
            gripper_action_chunck = None  # Mobile VLA는 gripper 없음

        if isinstance(rgb, torch.Tensor):
            rgb = rgb[:, :seq_len]
            if hand_rgb is not None:
                hand_rgb = hand_rgb[:, :seq_len]

        chunck_mask = batch.get("chunck_mask", None)
        if chunck_mask is not None:
            chunck_mask = chunck_mask.cuda()

        fwd_mask = batch.get("fwd_mask", None)
        if fwd_mask is not None:
            fwd_mask = fwd_mask.bool().cuda()

        # data preparation for discrete action inputs and labels
        instr_and_action_ids = batch.get("instr_and_action_ids", None)
        if instr_and_action_ids is not None:
            instr_and_action_ids = instr_and_action_ids.cuda()

        instr_and_action_labels = batch.get("instr_and_action_labels", None)
        if instr_and_action_labels is not None:
            instr_and_action_labels = instr_and_action_labels.cuda()

        instr_and_action_mask = batch.get("instr_and_action_mask", None)
        if instr_and_action_mask is not None:
            instr_and_action_mask = instr_and_action_mask.cuda()

        rel_state = batch.get("rel_state", None)
        raw_text = batch.get("raw_text", None)
        data_source = batch.get("data_source", "mobile_vla_action")
        
        return (
            rgb,
            hand_rgb,
            attention_mask,
            language,
            text_mask,
            fwd_rgb_chunck,
            fwd_hand_rgb_chunck,
            velocity,  # arm_action -> velocity (2D 속도)
            gripper_action,
            velocity_chunck,  # arm_action_chunck -> velocity_chunck (2D 속도 시퀀스)
            gripper_action_chunck,
            chunck_mask,
            fwd_mask,
            instr_and_action_ids,
            instr_and_action_labels,
            instr_and_action_mask,
            raw_text,
            rel_state,
            data_source,
        )

