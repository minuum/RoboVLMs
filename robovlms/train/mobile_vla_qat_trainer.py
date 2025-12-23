"""
Mobile VLA QAT (Quantization-Aware Training) Trainer
INT8 Vision Encoder + INT4 LLM으로 학습

목표:
- Vision Encoder: INT8 (QAT)
- LLM: INT4 (BitsAndBytes)
- Action Head: FP16 (학습 대상)
- 전체 메모리: ~9GB (Jetson 16GB 목표)
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from torch.quantization import get_default_qat_qconfig
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from transformers import BitsAndBytesConfig
import warnings


class QuantizedVisionWrapper(nn.Module):
    """Vision Encoder QAT Wrapper"""
    
    def __init__(self, vision_model):
        super().__init__()
        self.quant = QuantStub()
        self.vision_model = vision_model
        self.dequant = DeQuantStub()
        
    def forward(self, *args, **kwargs):
        """
        Forward with flexible arguments to support different vision models.
        
        Kosmos2 calls: forward(pixel_values=x, output_attentions=..., ...)
        Other models might call: forward(x)
        
        CRITICAL: Converts FP16 to FP32 for fake quantization compatibility
        Mixed precision (AMP) uses FP16, but fake_quantize requires FP32
        """
        # Extract the main input (could be positional or keyword)
        if len(args) > 0:
            # Positional argument case: forward(x)
            x = args[0]
            # Convert to FP32 for fake quantization
            original_dtype = x.dtype
            x = x.float()  # FP16 → FP32
            x = self.quant(x)
            output = self.vision_model(x, *args[1:], **kwargs)
        elif 'pixel_values' in kwargs:
            # Keyword argument case: forward(pixel_values=x, ...)
            x = kwargs['pixel_values']
            # Convert to FP32 for fake quantization
            original_dtype = x.dtype
            x = x.float()  # FP16 → FP32
            x = self.quant(x)
            kwargs['pixel_values'] = x
            output = self.vision_model(**kwargs)
        else:
            # Fallback: just pass everything through
            output = self.vision_model(*args, **kwargs)
        
        # Dequantize the output
        # Handle different output types (tensor, tuple, BaseModelOutput)
        if isinstance(output, tuple):
            # If output is tuple, dequantize first element
            first_elem = output[0]
            # Ensure it's FP32 before dequant
            if first_elem.dtype != torch.float32:
                first_elem = first_elem.float()
            output = (self.dequant(first_elem),) + output[1:]
        elif hasattr(output, 'last_hidden_state'):
            # If output is BaseModelOutput, dequantize last_hidden_state
            hs = output.last_hidden_state
            # Ensure it's FP32 before dequant
            if hs.dtype != torch.float32:
                hs = hs.float()
            output.last_hidden_state = self.dequant(hs)
        else:
            # If output is tensor, dequantize directly
            if hasattr(output, 'dtype') and output.dtype != torch.float32:
                output = output.float()
            output = self.dequant(output)
        
        return output


class MobileVLAQATTrainer(MobileVLATrainer):
    """
    QAT (Quantization-Aware Training) Trainer for Mobile VLA
    
    특징:
    - Vision Encoder: INT8 QAT (fake quantization during training)
    - LLM: INT4 BitsAndBytes (frozen)
    - Action Head: FP16 (학습 가능)
    
    메모리 목표:
    - Model: ~2GB
    - Activation: ~1.5GB
    - Total on Jetson: ~9GB
    """
    
    def __init__(self, configs):
        # QAT config 확인
        self.qat_config = configs.get('quantization', {})
        self.qat_enabled = self.qat_config.get('enable', False)
        
        # Parent init
        super().__init__(configs)
        
        # QAT setup
        if self.qat_enabled:
            self._setup_quantization()
    
    def _setup_quantization(self):
        """QAT 설정"""
        print("\n" + "="*60)
        print("🔧 Setting up Quantization-Aware Training (QAT)")
        print("="*60)
        
        # 1. Vision Encoder INT8 QAT
        if self.qat_config.get('vision_encoder', {}).get('enable', True):
            self._setup_vision_qat()
        
        # 2. LLM INT4 BitsAndBytes (이미 로딩 시 적용되어야 함)
        if self.qat_config.get('llm', {}).get('enable', True):
            self._verify_llm_int4()
        
        print("="*60)
        print("✅ QAT Setup Complete")
        print("="*60 + "\n")
    
    def _setup_vision_qat(self):
        """Vision Encoder INT8 QAT 설정"""
        print("\n📊 Setting up Vision Encoder INT8 QAT...")
        
        try:
            # Kosmos2 모델 구조: self.model.model.vision_model
            # RoboKosMos.vision_tower -> self.model.vision_model
            vision_model = None
            
            # Try different paths
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                # Kosmos2ForConditionalGeneration structure
                vision_model = self.model.model.vision_model
                vision_model_path = "self.model.model.vision_model"
            elif hasattr(self, 'model') and hasattr(self.model, 'vision_tower'):
                # RoboVLM wrapper structure
                vision_model = self.model.vision_tower
                vision_model_path = "self.model.vision_tower"
            else:
                raise AttributeError(f"Cannot find vision encoder. Model type: {type(self.model)}")
            
            print(f"   📍 Found vision encoder at: {vision_model_path}")
            
            # Freeze vision encoder (QAT에서도 frozen)
            for param in vision_model.parameters():
                param.requires_grad = False
            
            # QAT config 설정
            qconfig_spec = self.qat_config['vision_encoder'].get('qconfig', 'fbgemm')
            qat_qconfig = get_default_qat_qconfig(qconfig_spec)
            
            # Vision model을 wrapper로 감싸기
            wrapped_vision = QuantizedVisionWrapper(vision_model)
            wrapped_vision.qconfig = qat_qconfig
            
            # QAT 준비
            wrapped_vision = prepare_qat(wrapped_vision, inplace=False)
            
            # 모델에 다시 할당
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                self.model.model.vision_model = wrapped_vision
            elif hasattr(self.model, 'vision_tower'):
                self.model.vision_tower = wrapped_vision
            
            print("   ✅ Vision Encoder prepared for INT8 QAT")
            print(f"   - QConfig: {qconfig_spec}")
            print(f"   - Frozen: True")
            print(f"   - Fake Quantization: Enabled")
            
        except Exception as e:
            warnings.warn(f"Vision QAT setup failed: {e}")
            print(f"   ⚠️ Warning: {e}")
            import traceback
            traceback.print_exc()
    
    def _verify_llm_int4(self):
        """LLM INT4 설정 확인"""
        print("\n📊 Verifying LLM INT4 configuration...")
        
        llm_config = self.qat_config.get('llm', {})
        
        print("   ℹ️ LLM should be loaded with BitsAndBytes INT4")
        print("   - Expected: load_in_4bit=True")
        print("   - Expected: bnb_4bit_quant_type='nf4'")
        print("   ⚠️ Note: LLM INT4 must be configured in model loading stage")
        
        # Config에서 확인
        if llm_config.get('load_in_4bit', False):
            print("   ✅ INT4 configuration detected")
        else:
            print("   ⚠️ INT4 configuration not found in config")
            print("   Please ensure model is loaded with BitsAndBytesConfig")
    
    def on_train_end(self):
        """학습 종료 시 QAT 모델을 실제 INT8로 변환"""
        if self.qat_enabled and self.qat_config.get('convert_after_training', True):
            print("\n" + "="*60)
            print("🔄 Converting QAT model to INT8 quantized model")
            print("="*60)
            
            try:
                # Vision encoder 찾기
                vision_wrapper = None
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                    vision_wrapper = self.model.model.vision_model
                    vision_path = "self.model.model.vision_model"
                elif hasattr(self.model, 'vision_tower'):
                    vision_wrapper = self.model.vision_tower
                    vision_path = "self.model.vision_tower"
                else:
                    raise AttributeError(f"Cannot find vision encoder. Model type: {type(self.model)}")
                
                print(f"   📍 Converting vision encoder at: {vision_path}")
                
                # QAT → INT8 변환
                vision_wrapper.eval()
                quantized_vision = convert(vision_wrapper, inplace=False)
                
                # 모델에 다시 할당
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                    self.model.model.vision_model = quantized_vision
                elif hasattr(self.model, 'vision_tower'):
                    self.model.vision_tower = quantized_vision
                
                print("   ✅ Vision Encoder converted to INT8")
                print("="*60 + "\n")
                
            except Exception as e:
                warnings.warn(f"Conversion failed: {e}")
                print(f"   ⚠️ Conversion failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Parent cleanup
        super().on_train_end()
    
    def on_save_checkpoint(self, checkpoint):
        """체크포인트 저장 시 QAT 정보 추가"""
        super().on_save_checkpoint(checkpoint)
        
        if self.qat_enabled:
            checkpoint['qat_config'] = self.qat_config
            checkpoint['qat_enabled'] = True
            print("   ℹ️ QAT configuration saved to checkpoint")
    
    def on_load_checkpoint(self, checkpoint):
        """체크포인트 로딩 시 QAT 정보 복원"""
        super().on_load_checkpoint(checkpoint)
        
        if checkpoint.get('qat_enabled', False):
            self.qat_config = checkpoint.get('qat_config', self.qat_config)
            self.qat_enabled = True
            print("   ℹ️ QAT configuration loaded from checkpoint")
