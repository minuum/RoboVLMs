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
from huggingface_hub import hf_hub_download
import logging
import sys
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 올바른 경로로 RoboVLMs 관련 임포트 수정
logger.info("RoboVLMs 모듈 불러오기 시작...")
try:
    from robovlms.model.backbone import RoboPaligemma, RoboFlamingo
    from robovlms.utils.model_utils import build_tokenizer
    from robovlms.data.data_utils import preprocess_image
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoTokenizer
    logger.info("RoboVLMs 모듈 불러오기 성공")
except ImportError as e:
    logger.error(f"모듈 불러오기 실패: {e}")
    if 'tkinter' in str(e):
        logger.error("tkinter가 설치되지 않았습니다. 'vid_llava_dataset.py' 파일에서 'from turtle import pd'를 'import pandas as pd'로 수정하세요.")
    sys.exit(1)

# 모델 매핑 및 설정 파일 경로
MODEL_PATHS = {
    "flamingo": "flamingo_mpt_7b",
    "flamingo-3b": "flamingo_mpt_3b",
    "paligemma": "paligemma-3b-pt-224",
}

CONFIG_FILES = {
    "flamingo": "flamingo_config.json",
    "flamingo-3b": "flamingo_3b_config.json",
    "paligemma": "paligemma_config.json",
}

MODEL_CLASSES = {
    "flamingo": RoboFlamingo,
    "flamingo-3b": RoboFlamingo,
    "paligemma": RoboPaligemma,
}

# 모델 설정 파일 및 체크포인트 확보
def ensure_model_files(model_type):
    logger.info(f"모델 설정 파일 및 체크포인트 확인 중: {model_type}")
    
    # 모델 유형 확인
    if model_type not in MODEL_PATHS:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        logger.info(f"지원되는 모델 유형: {list(MODEL_PATHS.keys())}")
        sys.exit(1)
    
    model_path = MODEL_PATHS[model_type]
    config_file = CONFIG_FILES[model_type]
    config_path = os.path.join(os.getcwd(), config_file)
    
    # 설정 파일 확인 및 다운로드
    if not os.path.exists(config_path):
        logger.info("설정 파일이 존재하지 않습니다. 허깅페이스에서 다운로드 중...")
        try:
            repo_id = f"google/{model_path}" if model_type == "paligemma" else f"Robot-VLAs/RoboVLMs"
            filename = "config.json" if model_type == "paligemma" else f"configs/{model_type}_config.json"
            
            config_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.getcwd(),
                local_dir_use_symlinks=False
            )
            
            # PaliGemma 모델은 파일 이름 변경
            if model_type == "paligemma":
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"모델 설정 로드 완료: {config_path}")
            else:
                logger.info(f"모델 설정 로드 완료: {config_file}")
                config_path = config_file
        except Exception as e:
            logger.error(f"설정 파일 다운로드 실패: {e}")
            sys.exit(1)
    else:
        logger.info(f"모델 설정 로드 완료: {config_path}")
    
    # PaliGemma 모델의 경우 파일을 다운로드하는 로직 추가
    if model_type == "paligemma":
        model_dir = os.path.join(os.getcwd(), ".vlms", model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 가중치 파일 확인
        model_file = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_file):
            logger.info("모델 가중치가 존재하지 않습니다. 자동 다운로드 설정...")
            # 모델 디렉토리에 huggingface 접근을 위한 설정 만들기
            with open(os.path.join(model_dir, ".gitattributes"), "w") as f:
                f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text")
                
            # 모델 설치 로직 변경
            try:
                logger.info(f"transformers와 accelerate를 사용하여 모델 다운로드 준비...")
                # 필요한 파일만 추출
                logger.info(f"PaliGemma 모델을 로컬에 저장하지 않고 직접 허깅페이스에서 불러오는 방식으로 변경합니다.")
            except Exception as e:
                logger.error(f"모델 다운로드 설정 실패: {e}")
                sys.exit(1)
    
    return config_path

# 이미지 불러오기 및 전처리
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
        return image
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        sys.exit(1)

# 기본 인퍼런스 함수
def inference(model, image, instruction, device="cpu"):
    logger.info("인퍼런스 시작...")
    start_time = time.time()
    
    try:
        # 이미지 전처리
        logger.info("이미지 전처리 중...")
        if isinstance(image, str):
            image = load_image(image)
        
        preprocessed_image = preprocess_image(image, model.configs["image_size"])
        preprocessed_image = preprocessed_image.to(device)
        
        # 토크나이저로 지시문 인코딩
        logger.info("텍스트 인코딩 중...")
        encoded_text = model.encode_text(instruction)
        
        # 모델 인퍼런스 실행
        logger.info("모델 인퍼런스 실행 중...")
        with torch.no_grad():
            output = model.generate(
                preprocessed_image, 
                encoded_text, 
                max_new_tokens=128, 
                temperature=0.7
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"인퍼런스 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return output
    except Exception as e:
        logger.error(f"인퍼런스 오류: {e}")
        return f"오류 발생: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="RoboVLMs 테스트 스크립트")
    parser.add_argument("--model", type=str, default="paligemma", 
                        choices=["flamingo", "flamingo-3b", "paligemma"],
                        help="테스트할 모델 타입 (flamingo, flamingo-3b, paligemma)")
    parser.add_argument("--image", type=str, 
                        default="https://raw.githubusercontent.com/Robot-VLAs/RoboVLMs/main/imgs/robovlms.png", 
                        help="이미지 경로 또는 URL")
    parser.add_argument("--instruction", type=str, 
                        default="로봇이 이 물체를 집어서 옮기려면 어떻게 해야 할까요?",
                        help="모델에게 전달할 지시문")
    parser.add_argument("--device", type=str, default=None,
                        help="사용할 장치 (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # MPS(Apple Silicon) 가속 확인
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("MPS 사용 가능: Apple Silicon 가속을 사용합니다.")
        elif torch.cuda.is_available():
            device = "cuda"
            print("CUDA 사용 가능: NVIDIA GPU 가속을 사용합니다.")
        else:
            device = "cpu"
            print("가속 하드웨어가 감지되지 않았습니다. CPU를 사용합니다.")
    else:
        device = args.device
    
    print(f"사용 장치: {device}")
    
    # 모델 설정 파일 로드
    config_path = ensure_model_files(args.model)
    
    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 모델 타입에 따라 적절한 클래스 선택
    model_class = MODEL_CLASSES[args.model]
    print(f"모델 타입: {model_class.__name__}")
    
    # PaliGemma 모델일 경우 직접 허깅페이스에서 로드
    if args.model == "paligemma":
        try:
            print("PaliGemma 모델 및 프로세서 로드 중...")
            
            # 모델 설정에 train_setup 정보가 있는지 확인
            if 'train_setup' not in config:
                config['train_setup'] = {
                    "train_batch_size": 1,
                    "weight_decay": 0.0,
                    "learning_rate": 1e-5,
                    "warmup_steps": 100
                }
            
            # 직접 Hugging Face에서 PaliGemma 모델 로드 (RoboVLMs 래퍼 사용하지 않음)
            processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
            
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
            print(f"모델을 {torch_dtype} 정밀도로 로드합니다...")
            
            model_hf = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-pt-224",
                device_map=device,
                torch_dtype=torch_dtype
            )
            
            # 이미지 로드 및 전처리
            image = load_image(args.image)
            print("이미지 로드 완료")
            
            # 직접 추론
            print("입력 처리 중...")
            inputs = processor(text=args.instruction, images=image, return_tensors="pt").to(device)
            
            # 모델 실행
            print("모델 추론 실행 중...")
            start_time = time.time()
            
            generated_ids = model_hf.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7
            )
            
            elapsed_time = time.time() - start_time
            print(f"추론 완료 (소요 시간: {elapsed_time:.2f}초)")
            
            # 결과 디코딩
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            input_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
            
            # 입력 부분 제거하여 실제 생성된 텍스트만 추출
            if output.startswith(input_text):
                output = output[len(input_text):].strip()
            
            print("\n=== 결과 ===")
            print(f"입력: {args.instruction}")
            print(f"출력: {output}")
            
        except Exception as e:
            logger.error(f"PaliGemma 모델 로드 및 추론 실패: {e}")
            print(f"모델 로드 및 추론 실패: {e}")
            print("추가적인 옵션: 더 작은 모델 사용 또는 CPU에서 실행")
    else:
        # Flamingo 모델은 기존 방식으로 로드
        print(f"모델 초기화 중: {args.model}...")
        model = model_class(config, config['train_setup'])
        
        # 이미지 로드 및 전처리
        image = load_image(args.image)
        
        # 추론 실행
        output = inference(model, image, args.instruction, device)
        
        print("\n=== 결과 ===")
        print(f"입력: {args.instruction}")
        print(f"출력: {output}")

if __name__ == "__main__":
    main()