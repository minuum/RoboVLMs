# GitHub Citation: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
# 이 함수는 LLaVA 프로젝트에서 가져온 것으로, LoRA를 적용할 Linear 레이어를 찾습니다.

import torch.nn as nn


def find_all_linear_names(model):
    """
    LoRA를 적용할 모든 Linear 레이어의 이름을 찾습니다.
    
    Args:
        model: PyTorch 모델
        
    Returns:
        list: Linear 레이어 이름 리스트
    """
    cls = nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


