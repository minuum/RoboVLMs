import os
import torch
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import functools
import json
import requests
from io import BytesIO
# hf_hub_download는 ensure_model_downloaded 함수가 제거되므로 더 이상 필요하지 않을 수 있습니다.
# from huggingface_hub import hf_hub_download 
import logging
import sys
import time
from torchvision import transforms

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 모델 관련 상수
MODEL_ID = "google/paligemma-3b-mix-224"
# LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vlms/paligemma-3b-mix-224") # 제거

# 메모리 최적화 설정
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 사용 가능한 장치 확인 및 설정
def get_device(device_arg=None):
    if device_arg:
        return device_arg
        
    if torch.backends.mps.is_available():
        logger.info("MPS 사용 가능: Apple Silicon 가속을 사용합니다.")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA 사용 가능: NVIDIA GPU 가속을 사용합니다.")
        return "cuda"
    else:
        logger.info("가속 하드웨어가 감지되지 않았습니다. CPU를 사용합니다.")
        return "cpu"

# 모델 다운로드 및 캐싱 함수 - 제거
# def ensure_model_downloaded():
#     """
#     모델이 로컬에 있는지 확인하고, 없으면 다운로드합니다
#     """
#     model_dir = Path(LOCAL_MODEL_DIR)
    
#     if model_dir.exists() and any(model_dir.iterdir()):
#         logger.info(f"로컬에 캐시된 모델 발견: {LOCAL_MODEL_DIR}")
#         return LOCAL_MODEL_DIR
    
#     model_dir.mkdir(parents=True, exist_ok=True)
    
#     logger.info(f"로컬에 모델이 없습니다. {MODEL_ID}에서 다운로드를 시작합니다...")
#     try:
#         hf_hub_download(repo_id=MODEL_ID, filename="config.json", local_dir=LOCAL_MODEL_DIR)
#         logger.info(f"모델 구성 파일 다운로드 완료 (실제 실행시에는 전체 모델 파일이 필요할 수 있습니다)")
#         return LOCAL_MODEL_DIR
#     except Exception as e:
#         logger.error(f"모델 다운로드 실패: {e}")
#         return MODEL_ID

# 이미지 불러오기
def load_image(image_path_or_url):
    logger.info(f"이미지 불러오기 시작: {image_path_or_url}")
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
            logger.info("URL에서 이미지 로드 성공")
        else:
            image = Image.open(image_path_or_url)
            logger.info("로컬 파일에서 이미지 로드 성공")
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("이미지를 RGB 형식으로 변환했습니다.")
        return image
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise e

def preprocess_image(image): 
    """
    이미지를 모델 입력 형식에 맞게 전처리합니다.
    (주의: test_direct_paligemma에서는 AutoProcessor가 전처리하므로 직접 호출되지 않음)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    logger.info(f"수동 이미지 전처리 완료. 형태: {img_tensor.shape}")
    return img_tensor

def test_direct_paligemma(image_path, instruction, device):
    """
    PaliGemma 모델을 직접 사용하여 추론하는 함수
    """
    try:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        import torch
        
        model_kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        model_load_path = MODEL_ID

        logger.info(f"PaliGemma 모델 및 프로세서 로드 중... ({model_load_path})")
        processor = AutoProcessor.from_pretrained(model_load_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_load_path, **model_kwargs)
        
        if device != "cuda":
            model.to(device)
        
        logger.info("모델 로드 완료")
        
        raw_image = load_image(image_path)
        logger.info(f"원본 이미지 로드 완료: mode={raw_image.mode}, size={raw_image.size}")

        if device == "mps":
            logger.info("MPS 장치에서 실행: 이미지를 CPU에서 처리 후 MPS로 이동합니다.")
            with torch.no_grad():
                with torch.device("cpu"): 
                    inputs = processor(
                        text=instruction, 
                        images=raw_image, 
                        return_tensors="pt",
                    )
                inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            logger.info(f"{device} 장치에서 이미지 처리 중...")
            inputs = processor(
                text=instruction,
                images=raw_image,
                return_tensors="pt"
            ).to(device)
        
        logger.info(f"Hugging Face Processor를 통한 이미지 및 텍스트 처리 완료. 입력 형태: { {k: v.shape for k, v in inputs.items()} }")
        logger.info(f"Pixel values tensor: shape={inputs['pixel_values'].shape}, dtype={inputs['pixel_values'].dtype}, min={inputs['pixel_values'].min()}, max={inputs['pixel_values'].max()}")
        
        logger.info("추론 실행 중...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5
            )
        
        logger.info("추론 완료")
        logger.info(f"생성된 ID (전체): {generated_ids}")
        
        generated_text_with_prompt = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        generated_text_answer_only = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"모델 출력 (특수 토큰 포함, 프롬프트 포함): {generated_text_with_prompt}")
        logger.info(f"모델 출력 (특수 토큰 제외, 프롬프트 포함): {generated_text_answer_only}")
        
        input_len = inputs["input_ids"].shape[1]
        logger.info(f"입력 프롬프트 길이 (input_len): {input_len}")
        
        answer_ids = generated_ids[0][input_len:]
        logger.info(f"추출된 답변 ID (answer_ids): {answer_ids}")
        
        result = processor.decode(answer_ids, skip_special_tokens=True)
        logger.info(f"최종 추출된 답변 (디코딩 후): {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"PaliGemma 모델 로드 및 추론 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"오류: {str(e)}"

def test_robovlms_paligemma(image_path, instruction, device):
    """
    RoboVLMs의 RoboPaliGemma 모델을 사용하여 추론하는 함수
    """
    try:
        from robovlms.model.backbone import RoboPaligemma
        
        logger.info("RoboVLMs의 PaliGemma 모델 초기화...")
        
        # model_load_path = ensure_model_downloaded() # 제거하고 MODEL_ID 직접 사용
        model_load_path = MODEL_ID 
        logger.info(f"모델 경로: {model_load_path}")
        
        config = {
            "robovlm_name": "RoboPaligemma",
            "model": "paligemma",
            "model_url": model_load_path, 
            "image_size": 224, 
            "window_size": 8,
            "train_setup": {
                "train_vision": True,
                "freeze_backbone": False,
                "bits": 16 if device != "cpu" else 32
            },
            "vlm": { 
                "type": "PaliGemmaForConditionalGeneration",
                "pretrained_model_name_or_path": model_load_path,
                "name": "paligemma"
            },
            "tokenizer": { 
                "type": "AutoProcessor", 
                "pretrained_model_name_or_path": model_load_path,
                "tokenizer_type": "paligemma", 
                "max_text_len": 256
            }
        }
        
        model = RoboPaligemma(config, config["train_setup"])
        model.to(device)
        logger.info("RoboVLMs 모델 초기화 및 장치로 이동 완료")
        
        raw_image = load_image(image_path) 
        logger.info(f"원본 이미지 로드 완료: mode={raw_image.mode}, size={raw_image.size}")
        
        logger.info("추론 실행 중 (RoboVLMs)...")
        with torch.no_grad():
            output = model.generate(raw_image, instruction, max_new_tokens=128) 
        return output
    
    except ImportError as e:
        logger.error(f"RoboVLMs 모듈 임포트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"오류: RoboVLMs 모듈을 로드할 수 없습니다. {str(e)}"
    
    except Exception as e:
        logger.error(f"RoboVLMs 모델 처리 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"오류: {str(e)}"

def test_flamingo(image_path, instruction, device, model_size="3b"):
    """
    RoboVLMs의 RoboFlamingo 모델을 사용하여 추론하는 함수
    """
    try:
        from robovlms.model.backbone import RoboFlamingo
        
        model_name = "flamingo-3b" if model_size == "3b" else "flamingo"
        logger.info(f"RoboVLMs의 {model_name} 모델 초기화...")
        
        config = {
            "robovlm_name": "RoboFlamingo", 
            "model": model_name, 
            "image_size": 224, 
            "window_size": 8, 
            "train_setup": {
                "train_vision": True,
                "freeze_backbone": False,
            }
        }
        
        model = RoboFlamingo(config, config["train_setup"])
        model.to(device)
        logger.info("RoboVLMs Flamingo 모델 초기화 및 장치로 이동 완료")
        
        raw_image = load_image(image_path) 
        logger.info(f"원본 이미지 로드 완료: mode={raw_image.mode}, size={raw_image.size}")

        logger.info("추론 실행 중 (RoboVLMs Flamingo)...")
        with torch.no_grad():
            output = model.generate(raw_image, instruction, max_new_tokens=128) 
        return output
    
    except ImportError as e:
        logger.error(f"RoboVLMs 모듈 임포트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"오류: RoboVLMs 모듈을 로드할 수 없습니다. {str(e)}"
    
    except Exception as e:
        logger.error(f"RoboVLMs 모델 처리 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"오류: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="RoboVLMs 테스트 스크립트")
    parser.add_argument("--model", type=str, default="direct-paligemma", 
                        choices=["flamingo", "flamingo-3b", "paligemma", "direct-paligemma"],
                        help="테스트할 모델 타입 (flamingo, flamingo-3b, paligemma, direct-paligemma)")
    parser.add_argument("--image", type=str, 
                        default="SCR-20250513-omus.png", 
                        help="이미지 경로 또는 URL")
    parser.add_argument("--image_path", type=str, help="[사용 안 함] 이미지 경로 또는 URL (--image 사용)")
    parser.add_argument("--instruction", type=str, 
                        default="로봇이 이 물체를 집어서 옮기려면 어떻게 해야 할까요?",
                        help="모델에게 전달할 지시문")
    parser.add_argument("--device", type=str, default=None,
                        help="사용할 장치 (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    if args.image_path and not args.image: 
        args.image = args.image_path
    elif args.image_path and args.image:
        logger.warning("--image와 --image_path가 모두 제공되었습니다. --image 값을 사용합니다.")

    final_device = get_device(args.device)
    logger.info(f"사용 장치: {final_device}")
    
    start_time = time.time()
    if args.model == "direct-paligemma":
        logger.info("직접 PaliGemma 모델 사용하여 추론 시작")
        output = test_direct_paligemma(args.image, args.instruction, final_device)
    elif args.model == "paligemma":
        logger.info("RoboVLMs의 PaliGemma 모델 사용하여 추론 시작")
        output = test_robovlms_paligemma(args.image, args.instruction, final_device)
    elif args.model == "flamingo-3b":
        logger.info("RoboVLMs의 Flamingo 3B 모델 사용하여 추론 시작")
        output = test_flamingo(args.image, args.instruction, final_device, model_size="3b")
    elif args.model == "flamingo": 
        logger.info("RoboVLMs의 Flamingo 모델 사용하여 추론 시작")
        output = test_flamingo(args.image, args.instruction, final_device, model_size="default") 
    else:
        logger.error(f"알 수 없는 모델 타입: {args.model}")
        output = f"오류: 알 수 없는 모델 타입 ({args.model})"

    end_time = time.time()
    logger.info(f"추론 소요 시간: {end_time - start_time:.2f}초")
    
    print("\n=== 결과 ===")
    print(f"모델: {args.model}")
    print(f"입력 이미지: {args.image}")
    print(f"입력 지시문: {args.instruction}")
    print(f"출력: {output}")

if __name__ == "__main__":
    main() 