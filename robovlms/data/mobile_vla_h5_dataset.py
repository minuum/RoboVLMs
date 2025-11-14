# GitHub Citation: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py
# Mobile VLA용 HDF5 데이터셋 로더 (RoboVLMs CALVIN 데이터셋 구조 참고)

import glob
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from robovlms.utils.model_utils import build_tokenizer


class MobileVLAH5Dataset(Dataset):
    """
    Mobile VLA HDF5 데이터셋 로더
    
    Args:
        data_dir: HDF5 파일들이 있는 디렉토리
        episode_pattern: 에피소드 파일 패턴 (예: "episode_20251106_*.h5")
        window_size: 히스토리 윈도우 크기
        action_chunk_size: 액션 청크 크기 (fwd_pred_next_n)
        model_name: 모델 이름 (토크나이저용)
        image_size: 이미지 크기
        rgb_pad: RGB 증강 패딩
        train_split: 학습/검증 분할 비율
        is_validation: 검증 데이터셋 여부
    """
    
    def __init__(
        self,
        data_dir,
        episode_pattern="episode_*.h5",
        window_size=8,
        action_chunk_size=10,
        model_name="kosmos",
        image_size=224,
        rgb_pad=10,
        train_split=0.8,
        is_validation=False,
        shift_first=False,
        **kwargs
    ):
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.window_size = window_size
        self.action_chunk_size = action_chunk_size
        self.model_name = model_name
        self.image_size = image_size
        self.rgb_pad = rgb_pad
        self.train_split = train_split
        self.is_validation = is_validation
        self.shift_first = shift_first
        
        # 에피소드 파일 로드
        episode_files = sorted(glob.glob(f"{data_dir}/{episode_pattern}"))
        if len(episode_files) == 0:
            raise ValueError(f"No episodes found in {data_dir} with pattern {episode_pattern}")
        
        # Train/Val 분할
        split_idx = int(len(episode_files) * train_split)
        if is_validation:
            self.episode_files = episode_files[split_idx:]
        else:
            self.episode_files = episode_files[:split_idx]
        
        # 각 에피소드의 프레임 수 계산
        self.episode_lengths = []
        self.cumulative_lengths = [0]
        for ep_file in self.episode_files:
            with h5py.File(ep_file, 'r') as f:
                length = len(f['images'])  # 'observations/images' -> 'images'
                self.episode_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
        self.total_frames = self.cumulative_lengths[-1]
        
        print(f"MobileVLAH5Dataset initialized:")
        print(f"  data_dir: {data_dir}")
        print(f"  episode_pattern: {episode_pattern}")
        print(f"  num_episodes: {len(self.episode_files)}")
        print(f"  total_frames: {self.total_frames}")
        print(f"  window_size: {window_size}")
        print(f"  action_chunk_size: {action_chunk_size}")
        print(f"  shift_first: {shift_first}")
        print(f"  model_name: {model_name}")
        print(f"  rgb_pad: {rgb_pad}")
        print(f"  train_split: {train_split}")
        print(f"  is_training: {not is_validation}")
        
        # 토크나이저 및 text_fn 초기화
        self.tokenizer = kwargs.get("tokenizer", None)
        self.tokenizer_config = kwargs.get("tokenizer_config", None)
        self.model_name = model_name
        self.window_size = window_size
        self.action_chunk_size = action_chunk_size  # 사용하지 않음 (호환성 유지)
        self.fwd_pred_next_n = kwargs.get("fwd_pred_next_n", action_chunk_size)  # kwargs에서 가져오기
        
        # text_fn 초기화 (tokenizer_config가 있으면)
        if self.tokenizer_config is not None and self.tokenizer is not None:
            from robovlms.data.data_utils import get_text_function
            tokenizer_type = self.tokenizer_config.get("tokenizer_type", "kosmos")
            max_text_len = self.tokenizer_config.get("max_text_len", 256)
            self.text_fn = get_text_function(self.tokenizer, tokenizer_type, max_text_len)
        else:
            self.text_fn = None
    
    def __len__(self):
        # 각 에피소드에서 window_size + fwd_pred_next_n 만큼의 프레임이 필요
        # RoboVLMs 구조: 8 + 10 = 18 프레임
        total_frames_needed = self.window_size + self.fwd_pred_next_n
        valid_frames = 0
        for length in self.episode_lengths:
            if length >= total_frames_needed:
                valid_frames += length - total_frames_needed + 1
        return max(1, valid_frames)
    
    def _find_episode_and_frame(self, idx):
        """전체 인덱스를 에피소드와 프레임 인덱스로 변환"""
        for ep_idx, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if idx < end - start:
                frame_idx = idx
                return ep_idx, frame_idx
            idx -= (end - start)
        return len(self.episode_files) - 1, 0
    
    def __getitem__(self, idx):
        # 에피소드와 프레임 인덱스 찾기
        # RoboVLMs 구조: window_size + fwd_pred_next_n = 8 + 10 = 18 프레임
        total_frames_needed = self.window_size + self.fwd_pred_next_n
        ep_idx = 0
        frame_idx = idx
        for i, length in enumerate(self.episode_lengths):
            if length >= total_frames_needed:
                valid_frames = length - total_frames_needed + 1
                if frame_idx < valid_frames:
                    ep_idx = i
                    break
                frame_idx -= valid_frames
        
        # HDF5 파일 로드
        with h5py.File(self.episode_files[ep_idx], 'r') as f:
            # 이미지 로드 (window_size + fwd_pred_next_n 프레임)
            # DiskCalvinDataset과 동일한 구조: 전체 시퀀스 로드
            images = []
            for t in range(frame_idx, frame_idx + total_frames_needed):
                img_array = f['images'][t]  # 'observations/images' -> 'images'
                # (H, W, C) -> PIL Image
                img = Image.fromarray(img_array.astype(np.uint8))
                # Resize to image_size
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                # PIL -> numpy -> tensor
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                # (C, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
                images.append(img_tensor)
            
            # 액션 로드 (window_size + fwd_pred_next_n 프레임)
            # DiskCalvinDataset과 동일: 시퀀스 형태로 반환
            # Shape: (window_size + fwd_pred_next_n, 2)
            actions = []
            for t in range(frame_idx, frame_idx + total_frames_needed):
                if t < len(f['actions']):
                    action_2d = f['actions'][t][:2]  # linear_x, linear_y만 사용
                    # 2D 액션 그대로 사용 (패딩 없음)
                    action = action_2d.copy()
                else:
                    action = np.zeros(2)  # 패딩 (2D)
                actions.append(action)
            
            # 언어 명령 로드 (기본 명령 사용)
            language = "Navigate to the target location"  # 기본 명령
        
        # 텐서 변환
        images_tensor = torch.stack(images)  # (window_size + fwd_pred_next_n, C, H, W)
        actions_tensor = torch.from_numpy(np.array(actions)).float()  # (window_size + fwd_pred_next_n, 2)
        
        # 액션 정규화 [-1, 1] (2D 액션 기준)
        actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
        
        # 언어 토크나이징 (간단한 더미 토큰 사용)
        # 실제 토크나이징은 collate_fn에서 처리됨
        input_ids = torch.zeros(256, dtype=torch.long)  # 더미
        attention_mask = torch.ones(256, dtype=torch.long)  # 더미
        
        # RoboVLMs 형식에 맞춰 반환 (배치 전 개별 샘플)
        # DiskCalvinDataset과 동일한 구조: 시퀀스 형태로 반환
        # CRITICAL: data_source must contain 'action' for forward_action to be called!
        return {
            'rgb': images_tensor,  # (window_size + fwd_pred_next_n, C, H, W)
            'hand_rgb': torch.zeros_like(images_tensor),  # 더미 gripper 이미지
            'actions': actions_tensor,  # (window_size + fwd_pred_next_n, 2) - DiskCalvinDataset과 동일
            'action_mask': torch.ones(total_frames_needed),  # (window_size + fwd_pred_next_n,)
            'image_mask': torch.ones(total_frames_needed),  # (window_size + fwd_pred_next_n,)
            'text': input_ids,  # (seq_len,)
            'text_mask': attention_mask,  # (seq_len,)
            'lang': language,  # DiskCalvinDataset과 동일한 키 이름
            'raw_text': language,
            'data_source': 'mobile_vla_action',  # Must contain 'action'!
            'attention_mask': torch.ones(total_frames_needed),  # 이미지 마스크 (모두 유효)
        }
    
    def collater(self, data):
        """
        배치 데이터를 처리하는 collater 메서드
        DiskCalvinDataset의 collater를 참고하여 구현
        """
        # 액션 텐서 스택 (DiskCalvinDataset과 동일한 구조)
        # s["actions"]는 (window_size + fwd_pred_next_n, 2) 형태
        action_tensors = torch.from_numpy(
            np.array([s["actions"].numpy() for s in data])
        )[:, :-1]  # 마지막 프레임 제거 (DiskCalvinDataset과 동일)
        # Shape: (B, window_size + fwd_pred_next_n - 1, 2)
        
        # 액션 마스크 스택
        action_mask = torch.from_numpy(
            np.array([s["action_mask"].numpy() for s in data])
        )[:, :-1]  # 마지막 프레임 제거
        
        # 이미지 텐서 스택
        image_tensors = torch.stack([s["rgb"] for s in data], dim=0)  # (B, window_size + fwd_pred_next_n, C, H, W)
        
        # 이미지 마스크 스택
        image_mask = torch.from_numpy(
            np.array([s["image_mask"].numpy() for s in data])
        )
        
        # Gripper 이미지 스택 (더미)
        gripper_tensors = torch.stack([s["hand_rgb"] for s in data], dim=0)  # (B, window_size + fwd_pred_next_n, C, H, W)
        
        # 언어 토크나이징 (text_fn 사용)
        stacked_language = [s["lang"] for s in data]
        if self.text_fn is not None:
            text_tensors, attention_mask = self.text_fn(stacked_language)
        else:
            # text_fn이 없으면 더미 사용 (초기화 시점)
            text_tensors = torch.stack([s["text"] for s in data], dim=0)
            attention_mask = torch.stack([s["text_mask"] for s in data], dim=0)
        
        # DiskCalvinDataset과 동일한 방식으로 chunk 생성 (unfold 사용)
        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]  # 첫 번째 제거
        image_tensors = image_tensors[:, : self.window_size]  # window_size만 사용
        
        gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]
        gripper_tensors = gripper_tensors[:, : self.window_size]
        
        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]
        
        # 액션 chunk 생성 (unfold 사용)
        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 3, 2
        )
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)
        
        res = {
            "rgb": image_tensors,  # (B, window_size, C, H, W)
            "hand_rgb": gripper_tensors,  # (B, window_size, C, H, W)
            "action": action_tensors,  # (B, window_size + fwd_pred_next_n - 1, 2)
            "text": text_tensors,  # (B, seq_len)
            "text_mask": attention_mask,  # (B, seq_len)
            "fwd_rgb_chunck": image_chunk,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, C, H, W)
            "fwd_hand_rgb_chunck": gripper_chunk,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, C, H, W)
            "fwd_mask": fwd_mask,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n)
            "action_chunck": action_chunck,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, 2)
            "chunck_mask": action_mask,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n)
            "raw_text": stacked_language,  # List[str]
            "data_source": "mobile_vla_action",
        }
        return res

