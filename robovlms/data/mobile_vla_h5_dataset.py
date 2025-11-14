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
        
        # 토크나이저는 나중에 외부에서 설정됨
        self.tokenizer = None
        self.model_name = model_name
    
    def __len__(self):
        # 각 에피소드에서 window_size + action_chunk_size 만큼의 프레임이 필요
        valid_frames = 0
        for length in self.episode_lengths:
            if length >= self.window_size + self.action_chunk_size:
                valid_frames += length - self.window_size - self.action_chunk_size + 1
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
        ep_idx = 0
        frame_idx = idx
        cumsum = 0
        for i, length in enumerate(self.episode_lengths):
            if length >= self.window_size + self.action_chunk_size:
                valid_frames = length - self.window_size - self.action_chunk_size + 1
                if frame_idx < valid_frames:
                    ep_idx = i
                    break
                frame_idx -= valid_frames
        
        # HDF5 파일 로드
        with h5py.File(self.episode_files[ep_idx], 'r') as f:
            # 이미지 로드 (window_size 프레임)
            images = []
            for t in range(frame_idx, frame_idx + self.window_size):
                img_array = f['images'][t]  # 'observations/images' -> 'images'
                # (H, W, C) -> PIL Image
                img = Image.fromarray(img_array.astype(np.uint8))
                # Resize to image_size
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                # PIL -> numpy -> tensor
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                # (H, W, C) -> (C, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
                images.append(img_tensor)
            
            # 액션 로드 (window_size x action_chunk_size 프레임)
            # RoboVLMs는 각 window frame마다 future action chunk를 기대
            # Shape: (window_size, action_chunk_size, 7)
            actions = []
            for w in range(self.window_size):
                window_actions = []
                for t in range(frame_idx + w, frame_idx + w + self.action_chunk_size):
                    if t < len(f['actions']):
                        action_2d = f['actions'][t][:2]  # linear_x, linear_y만 사용
                        # 7D로 패딩: [linear_x, linear_y, 0, 0, 0, 0, gripper]
                        action = np.zeros(7)
                        action[:2] = action_2d
                        action[6] = 0.0  # gripper는 항상 0 (열림)
                    else:
                        action = np.zeros(7)  # 패딩
                    window_actions.append(action)
                actions.append(window_actions)
            
            # 언어 명령 로드 (기본 명령 사용)
            language = "Navigate to the target location"  # 기본 명령
        
        # 텐서 변환
        images_tensor = torch.stack(images)  # (window_size, C, H, W)
        actions_tensor = torch.from_numpy(np.array(actions)).float()  # (window_size, action_chunk_size, 7)
        
        # 액션 정규화 [-1, 1]
        actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
        
        # 언어 토크나이징 (간단한 더미 토큰 사용)
        # 실제 토크나이징은 collate_fn에서 처리됨
        input_ids = torch.zeros(256, dtype=torch.long)  # 더미
        attention_mask = torch.ones(256, dtype=torch.long)  # 더미
        
        # RoboVLMs 형식에 맞춰 반환 (배치 전 개별 샘플)
        # CRITICAL: data_source must contain 'action' for forward_action to be called!
        return {
            'rgb': images_tensor,  # (window_size, C, H, W)
            'hand_rgb': torch.zeros_like(images_tensor),  # 더미 gripper 이미지
            'action': actions_tensor,  # (window_size, action_chunk_size, 7)
            'text': input_ids,  # (seq_len,)
            'text_mask': attention_mask,  # (seq_len,)
            'action_chunck': actions_tensor,  # (window_size, action_chunk_size, 7)
            'chunck_mask': torch.ones(self.window_size, self.action_chunk_size),  # (window_size, action_chunk_size)
            'fwd_rgb_chunck': None,
            'fwd_hand_rgb_chunck': None,
            'fwd_mask': None,
            'raw_text': language,
            'data_source': 'mobile_vla_action',  # Must contain 'action'!
        }

