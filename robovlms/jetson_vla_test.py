# jetson_vla_test.py

import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration # PaliGemma 직접 사용
import cv2
import numpy as np
from enum import Enum, auto
import time # 파일 이름용
import logging # 로깅 모듈 추가

# --- VLA Node Constants: VLA 파이프라인의 각 주요 단계를 나타내는 상수 --- 
VLA_NODE_SETUP = "SETUP"              # 스크립트 초기 설정 및 파라미터 처리 단계
VLA_NODE_VISION = "VISION"            # 이미지 데이터 획득 및 전처리 단계 (웹캠 또는 파일)
VLA_NODE_LANGUAGE = "LANGUAGE"        # VLM에 입력될 프롬프트 생성 및 처리 단계
VLA_NODE_VLM = "VLM_INFERENCE"      # Vision-Language Model 추론 실행 단계
VLA_NODE_ACTION = "ACTION"            # VLM의 출력을 기반으로 실제 액션 수행 단계
VLA_NODE_GENERAL = "GENERAL"          # 특정 노드에 국한되지 않는 일반적인 정보 또는 상태 로깅

# --- Logger Setup: 스크립트 전체에서 사용할 로거 설정 --- 
# 로그 포맷은 시간, 로그 레벨, VLA 노드, 파일명, 줄번호, 함수명, 메시지 순서로 구성됩니다.
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(vla_node)s] [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler = logging.StreamHandler() # 로그를 콘솔(터미널)에 출력하는 핸들러
handler.setFormatter(formatter)   # 핸들러에 위에서 정의한 포맷 적용

logger = logging.getLogger(__name__) # 현재 모듈(__name__)을 위한 로거 객체 생성
logger.addHandler(handler)           # 로거에 핸들러 추가
logger.setLevel(logging.INFO)        # 기본 로그 레벨을 INFO로 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.propagate = False             # 이 로거의 메시지가 상위(루트) 로거로 전파되지 않도록 설정 (중복 출력 방지)

class Goal(Enum): # CameraAction -> Goal 로 변경하고, 값들도 일반적인 목표로 수정
    """
    VLA 시스템이 수행할 수 있는 목표(goal)의 종류를 정의하는 열거형 클래스입니다.
    """
    DESCRIBE_SCENE = "describe_scene"
    IDENTIFY_MAIN_OBJECT = "identify_main_object"
    SAVE_IMAGE = "save_image"
    FIND_OBJECT = "find_object" # 새로운 목표 추가
    UNKNOWN_COMMAND = "unknown"

def get_primary_vlm_prompt(target_object: str) -> str:
    """
    1차 VLM 질의용 프롬프트를 생성합니다. (주로 객체 유무 확인용)
    Args:
        target_object (str): 찾고자 하는 객체 이름.
    Returns:
        str: 1차 VLM 질의용 프롬프트 문자열.
    """
    _vla_node = VLA_NODE_LANGUAGE
    prompt = f"<image> Is there a {target_object} in this image? Answer with only 'yes' or 'no'."
    logger.debug(f"Generated primary VLM prompt: '{prompt}' for target: '{target_object}'", extra={'vla_node': _vla_node})
    return prompt

def parse_yes_no_answer(answer: str) -> bool | None:
    """
    VLM의 답변에서 'yes' 또는 'no'를 파싱합니다.
    Args:
        answer (str): VLM의 텍스트 답변.
    Returns:
        bool | None: 'yes'면 True, 'no'면 False, 불분명하면 None.
    """
    _vla_node = VLA_NODE_LANGUAGE
    normalized_answer = answer.lower().strip().replace(".","")
    
    # 응답 문자열의 끝부분이 "yes" 또는 "no"로 끝나는지 확인
    # 모델이 프롬프트를 반복하고 줄바꿈 후 답변하는 경우 등을 고려
    last_part = normalized_answer.split('\\n')[-1].strip() # 마지막 줄을 가져와서 공백 제거

    if last_part == "yes":
        logger.debug(f"Parsed 'yes' from VLM answer (last part): '{answer}' -> '{last_part}'", extra={'vla_node': _vla_node})
        return True
    elif last_part == "no":
        logger.debug(f"Parsed 'no' from VLM answer (last part): '{answer}' -> '{last_part}'", extra={'vla_node': _vla_node})
        return False
    else:
        # 만약 마지막 줄에서 못찾으면, 전체 문자열의 끝에서 한번 더 시도 (예: "프롬프트 yes" 같은 경우)
        if normalized_answer.endswith("yes"):
            logger.debug(f"Parsed 'yes' from VLM answer (endswith): '{answer}'", extra={'vla_node': _vla_node})
            return True
        elif normalized_answer.endswith("no"):
            logger.debug(f"Parsed 'no' from VLM answer (endswith): '{answer}'", extra={'vla_node': _vla_node})
            return False
            
        logger.warning(f"Could not parse 'yes' or 'no' from VLM answer: '{answer}'. Normalized last part: '{last_part}', Normalized full: '{normalized_answer}'", extra={'vla_node': _vla_node})
        return None

def get_vlm_prompt(goal: Goal, custom_prompt: str = None, target_object: str = None, primary_query_result: bool | None = None) -> str:
    """
    주어진 목표 유형 및 1차 질의 결과에 따라 VLM에 전달할 적절한 프롬프트를 생성합니다.
    모든 프롬프트는 "<image>" 토큰으로 시작합니다.

    Args:
        goal (Goal): 수행할 목표의 유형.
        custom_prompt (str, optional): 사용자가 직접 제공하는 커스텀 프롬프트 문자열.
        target_object (str, optional): FIND_OBJECT 목표 시 찾고자 하는 객체 이름.
        primary_query_result (bool | None, optional): 1차 질의 결과 (예: 객체 존재 유무).
                                                    FIND_OBJECT goal 시 이 값을 참조합니다.

    Returns:
        str: 생성된 프롬프트 문자열.
    """
    _vla_node = VLA_NODE_LANGUAGE
    logger.debug(f"Generating VLM prompt for goal: {goal}, custom_prompt: '{custom_prompt if custom_prompt else ''}', target_object: '{target_object if target_object else ''}', primary_result: {primary_query_result}", extra={'vla_node': _vla_node})
    
    base_prompt = "<image> "

    if goal == Goal.DESCRIBE_SCENE:
        prompt = base_prompt + (custom_prompt if custom_prompt else "Describe this image in detail.")
    elif goal == Goal.IDENTIFY_MAIN_OBJECT:
        prompt = base_prompt + (custom_prompt if custom_prompt else "What is the main object in this image? Answer with the object's name or a very short phrase.")
    elif goal == Goal.FIND_OBJECT:
        if custom_prompt:
            prompt = base_prompt + custom_prompt
        elif target_object:
            if primary_query_result is True: # 1차 질의 결과 객체가 있다고 판단된 경우
                prompt = base_prompt + f"Describe the {target_object} in this image in detail."
            elif primary_query_result is False: # 1차 질의 결과 객체가 없다고 판단된 경우
                prompt = base_prompt + f"Okay, I understand there is no {target_object}. Then, describe the overall scene."
            else: # 1차 질의를 사용하지 않거나 결과를 알 수 없는 경우 (기존 방식 또는 fallback)
                prompt = base_prompt + f"Is there a {target_object} in this image? If yes, describe it. Otherwise, describe the scene."
        else:
            prompt = base_prompt + "What objects are in this image? Please list them."
    else: # SAVE_IMAGE 또는 UNKNOWN_COMMAND
        prompt = base_prompt + "Describe this image briefly."
    
    logger.info(f"Generated prompt: '{prompt}'", extra={'vla_node': _vla_node})
    return prompt

def execute_action(goal: Goal, vlm_output: str, frame: np.ndarray = None, output_dir: str = "saved_images", target_object: str = None):
    """
    VLM의 출력을 바탕으로 정의된 목표(goal)에 따른 액션을 수행합니다.

    Args:
        goal (Goal): 수행할 목표의 유형.
        vlm_output (str): VLM으로부터 생성된 텍스트 출력.
        frame (np.ndarray, optional): 현재 처리 중인 이미지 프레임.
        output_dir (str, optional): 이미지를 저장할 디렉토리 경로.
        target_object (str, optional): FIND_OBJECT 목표 시 참조할 수 있는 객체 이름.
    """
    _vla_node = VLA_NODE_ACTION
    logger.info(f"Executing action for goal: {goal} with VLM output: '{vlm_output[:50]}...'", extra={'vla_node': _vla_node})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    action_result = ""

    if goal == Goal.DESCRIBE_SCENE:
        action_result = f"[Scene Description] {vlm_output}"
    elif goal == Goal.IDENTIFY_MAIN_OBJECT:
        action_result = f"[Main Object] {vlm_output}"
    elif goal == Goal.FIND_OBJECT:
        # FIND_OBJECT의 경우, vlm_output은 객체 유무 및 설명일 수 있음.
        # 여기서는 단순히 VLM 출력을 그대로 보여주는 것으로 단순화.
        # 추후 1차/2차 VLM 호출 로직이 도입되면 이 부분은 더 정교해져야 함.
        action_result = f"[Object Finding Result for '{target_object if target_object else 'any objects'}'] {vlm_output}"
    elif goal == Goal.SAVE_IMAGE:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/image_{timestamp}.png"
        try:
            if frame is not None: # 웹캠 프레임 데이터가 있는 경우에만 저장 시도
                cv2.imwrite(filename, frame) # OpenCV를 사용하여 이미지 파일 저장
                action_result = f"[Image Saved] as {filename}"
                logger.info(action_result, extra={'vla_node': _vla_node})
            else:
                action_result = "[Image Save Skipped] No frame provided for saving."
                logger.warning(action_result, extra={'vla_node': _vla_node})
        except Exception as e:
            action_result = f"[Image Save Failed] Could not save image to {filename}. Error: {e}"
            logger.error(action_result, exc_info=True, extra={'vla_node': _vla_node})
            
        if vlm_output: # VLM이 이미지 저장에 대한 추가적인 코멘트를 생성한 경우 (현재 로직에서는 비어있을 수 있음)
             action_result += f"\n[VLM Comment] {vlm_output}"
    elif goal == Goal.UNKNOWN_COMMAND:
        action_result = "[Error] Unknown command received. Available tasks: describe, identify, save."
        logger.warning(action_result, extra={'vla_node': _vla_node})
    else:
        action_result = "[Error] An undefined action type was encountered."
        logger.error(action_result, extra={'vla_node': _vla_node})
    
    if action_result: # 액션 결과가 있는 경우에만 로그 출력
        logger.info(f"Action result: {action_result}", extra={'vla_node': _vla_node})

def main():
    """
    VLA 추론 스크립트의 메인 실행 함수입니다.
    명령줄 인자를 파싱하고, 모델을 로드하며, 선택된 작업(이미지 파일 처리 또는 웹캠 실시간 처리)을 수행합니다.
    """
    _vla_node_main_setup = VLA_NODE_SETUP # main 함수 시작 시점의 노드는 SETUP으로 간주
    logger.info("Starting VLA inference script.", extra={'vla_node': _vla_node_main_setup})
    
    # --- 명령줄 인자 파싱 설정 --- 
    parser = argparse.ArgumentParser(
        description="Enhanced PaliGemma VLA Inference Script with Action Set. "
                    "Processes either a single image or a webcam feed based on defined actions."
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the JSON configuration file for model and tokenizer (e.g., ./configs/inference_paligemma_mps.json)."
    )
    parser.add_argument( # --task에서 --goal로 변경
        "--goal", type=str, default=Goal.DESCRIBE_SCENE.value,
        choices=[g.value for g in Goal if g != Goal.UNKNOWN_COMMAND],
        help=f"Goal to achieve. Available: {[g.value for g in Goal if g != Goal.UNKNOWN_COMMAND]}. Default: {Goal.DESCRIBE_SCENE.value}."
    )
    parser.add_argument(
        "--custom_prompt", type=str, default=None, 
        help="Optional custom prompt to override default prompts for certain goals."
    )
    parser.add_argument( # 새로운 인자 추가
        "--target_object", type=str, default=None,
        help="Specify the target object for goals like 'find_object'."
    )
    parser.add_argument( # 새로운 인자 추가: 1차 질의 사용 여부
        "--use_primary_query", action='store_true',
        help="Enable two-step VLM query (primary query for context, then main query). Default is False (single query)."
    )
    parser.add_argument(
        "--image_path", type=str, default=None, 
        help="Path to a single image file for processing. If not provided, the script will attempt to use the webcam."
    )
    parser.add_argument(
        "--model_cache_dir", type=str, default=".vlms_cache", 
        help="Directory to cache downloaded VLM models and tokenizers. Default: .vlms_cache/"
    )
    parser.add_argument( # --max_length에서 --max_new_tokens로 변경
        "--max_new_tokens", type=int, default=256, # 기본값 변경
        help="Maximum number of new tokens to be generated by VLM. Default: 256."
    )
    parser.add_argument(
        "--output_image_dir", type=str, default="runs/saved_images", 
        help="Directory where captured images (via 'save_image' goal) will be stored. Default: runs/saved_images/"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging output level. Default: INFO."
    )

    args = parser.parse_args()
    
    # --- 로거 레벨 설정 --- 
    # 명령줄 인자에서 받은 log_level 값으로 로거의 레벨을 동적으로 설정합니다.
    try:
        logger.setLevel(getattr(logging, args.log_level.upper()))
        logger.info(f"Logger level set to {args.log_level.upper()} 이후.", extra={'vla_node': _vla_node_main_setup})
    except AttributeError:
        logger.error(f"Invalid log level specified: '{args.log_level}'. Defaulting to INFO.", extra={'vla_node': _vla_node_main_setup})
        logger.setLevel(logging.INFO) # 잘못된 값일 경우 INFO 레벨로 유지

    logger.info(f"Script arguments received: {args}", extra={'vla_node': _vla_node_main_setup})

    # --- 선택된 목표 결정 ---
    try:
        selected_goal = Goal(args.goal) # selected_action -> selected_goal
        logger.info(f"Selected goal for execution: {selected_goal.name} ('{selected_goal.value}')", extra={'vla_node': _vla_node_main_setup})
    except ValueError:
        logger.warning(f"Invalid goal specified: '{args.goal}'. Defaulting to '{Goal.DESCRIBE_SCENE.value}'.", extra={'vla_node': _vla_node_main_setup})
        selected_goal = Goal.DESCRIBE_SCENE

    # --- VLM (PaliGemma) 및 Processor 로드 --- 
    _vla_node_model_load = VLA_NODE_VLM # 모델 및 토크나이저 로딩은 VLM_INFERENCE 노드로 분류
    try:
        logger.info(f"Loading VLM configuration from: {args.config_path}", extra={'vla_node': _vla_node_model_load})
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        logger.debug(f"Configuration loaded successfully: {config}", extra={'vla_node': _vla_node_model_load})
        
        vlm_config = config.get("vlm", {}) # 설정 파일에서 "vlm" 섹션 가져오기 (없으면 빈 딕셔너리)
        tokenizer_config = config.get("tokenizer", {}) # "tokenizer" 섹션 가져오기
        
        # 모델/토크나이저 경로 결정: 설정 파일에 없으면 기본값(PaliGemma 3B) 사용
        model_name_or_path = vlm_config.get("pretrained_model_name_or_path", "google/paligemma-3b-mix-224")
        tokenizer_name_or_path = tokenizer_config.get("pretrained_model_name_or_path", model_name_or_path)
        logger.info(f"Using VLM model path: {model_name_or_path}", extra={'vla_node': _vla_node_model_load})
        logger.info(f"Using Tokenizer path: {tokenizer_name_or_path}", extra={'vla_node': _vla_node_model_load})

        logger.info(f"Loading HuggingFace AutoProcessor from '{tokenizer_name_or_path}' (cache: {args.model_cache_dir})", extra={'vla_node': _vla_node_model_load})
        processor = AutoProcessor.from_pretrained(tokenizer_name_or_path, cache_dir=args.model_cache_dir)
        
        logger.info(f"Loading HuggingFace PaliGemmaForConditionalGeneration model from '{model_name_or_path}' (cache: {args.model_cache_dir})", extra={'vla_node': _vla_node_model_load})
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32, # Apple MPS(GPU) 환경 호환성을 위해 float32 사용
            cache_dir=args.model_cache_dir,
            # trust_remote_code=True # 일부 모델은 필요할 수 있음
        )
        # 사용 가능한 디바이스 (MPS > CPU 순으로) 설정
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Moving model to device: {device}", extra={'vla_node': _vla_node_model_load})
        model.to(device) # 모델을 해당 디바이스로 이동
        model.eval()     # 모델을 평가 모드로 설정 (Dropout 등 비활성화)
        logger.info(f"Model '{model_name_or_path}' loaded and configured successfully on '{device}'.", extra={'vla_node': _vla_node_model_load})

    except Exception as e:
        logger.error(f"Fatal error during model or processor loading: {e}", exc_info=True, extra={'vla_node': _vla_node_model_load})
        return # 모델 로드 실패 시 스크립트 종료

    # --- 메인 로직 분기: 이미지 파일 처리 또는 웹캠 모드 --- 
    if args.image_path: # --image_path 인자가 제공된 경우: 단일 이미지 파일 처리 모드
        _vla_node_img_proc = VLA_NODE_VISION # 이미지 파일 처리는 VISION 노드
        logger.info(f"Single image processing mode selected. Target image: {args.image_path}", extra={'vla_node': _vla_node_img_proc})
        primary_query_result: bool | None = None  # <--- primary_query_result 초기화
        try:
            logger.debug(f"Opening image file: {args.image_path}", extra={'vla_node': _vla_node_img_proc})
            raw_image = Image.open(args.image_path).convert('RGB')

            # vlm_processed_output = "" # VLM의 최종 처리 결과 - 이 변수는 이후 로직에서 사용되지 않으므로 주석 처리 또는 삭제 가능

            prompt_text = "" # VLM에 전달될 프롬프트 초기화

            if args.goal == Goal.FIND_OBJECT.value and args.use_primary_query:
                # --- 1차 VLM 질의 (find_object 목표이고, 1차 질의 사용 시) ---
                _vla_node_primary_query = VLA_NODE_VLM
                logger.info(f"Primary VLM query process starting for target '{args.target_object}'.", extra={'vla_node': _vla_node_primary_query})
                
                primary_prompt_text = get_primary_vlm_prompt(target_object=args.target_object)
                logger.debug(f"Preparing inputs for PRIMARY VLM (image + prompt: '{primary_prompt_text}')...", extra={'vla_node': _vla_node_primary_query})
                primary_inputs = processor(text=primary_prompt_text, images=raw_image, return_tensors="pt").to(device)
                
                logger.info("Performing PRIMARY VLM inference...", extra={'vla_node': _vla_node_primary_query})
                primary_start_time = time.time()
                with torch.no_grad():
                    primary_generated_ids = model.generate(**primary_inputs, max_new_tokens=10, do_sample=False) # 1차 질의는 짧게 (yes/no)
                primary_end_time = time.time()
                logger.info(f"PRIMARY VLM inference completed in {primary_end_time - primary_start_time:.2f} seconds.", extra={'vla_node': _vla_node_primary_query})
                
                decoded_primary_output = processor.batch_decode(primary_generated_ids, skip_special_tokens=True)[0]
                logger.info(f"PRIMARY VLM raw output: '{decoded_primary_output}'", extra={'vla_node': _vla_node_primary_query})
                
                # 1차 VLM 결과 파싱
                # parse_yes_no_answer는 이미 내부적으로 로깅을 수행함
                primary_query_result = parse_yes_no_answer(decoded_primary_output) 

                if primary_query_result is None:
                    logger.warning(f"Primary query for '{args.target_object}' did not yield a clear yes/no. Proceeding as if object not found or confirmation failed.", extra={'vla_node': _vla_node_primary_query})
                    # primary_query_result가 None으로 유지되면 get_vlm_prompt에서 fallback 로직을 타거나, 혹은 False로 간주할 수 있음.
                    # 여기서는 get_vlm_prompt가 None을 처리하도록 둠.

                # 1차 질의 결과에 따른 2차 (본) 프롬프트 생성
                prompt_text = get_vlm_prompt(selected_goal, args.custom_prompt, args.target_object, primary_query_result)
                logger.info(f"Secondary VLM prompt (conditional on primary query result: {primary_query_result}): '{prompt_text}'", extra={'vla_node': VLA_NODE_LANGUAGE})

            else:
                # --- 단일 VLM 질의 (1차 질의를 사용하지 않거나, find_object가 아닌 다른 목표일 때) ---
                prompt_text = get_vlm_prompt(selected_goal, args.custom_prompt, args.target_object, primary_query_result) # primary_query_result는 None일 것임
            
            _vla_node_vlm_inputs = VLA_NODE_VLM
            logger.debug("Preparing inputs for VLM (image + prompt)...", extra={'vla_node': _vla_node_vlm_inputs})
            inputs = processor(text=prompt_text, images=raw_image, return_tensors="pt").to(device)

            if selected_goal in [Goal.DESCRIBE_SCENE, Goal.IDENTIFY_MAIN_OBJECT, Goal.FIND_OBJECT]: # FIND_OBJECT 추가
                _vla_node_vlm_inference = VLA_NODE_VLM # 실제 VLM 추론 수행 부분
                logger.info("Performing VLM inference on the provided image...", extra={'vla_node': _vla_node_vlm_inference})
                start_time = time.time() # 추론 시작 시간 기록
                with torch.no_grad(): # 기울기 계산 비활성화 (추론 시에는 필요 없음)
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False) # max_length -> max_new_tokens
                end_time = time.time() # 추론 종료 시간 기록
                logger.info(f"VLM inference completed in {end_time - start_time:.2f} seconds.", extra={'vla_node': _vla_node_vlm_inference})
                
                # VLM 출력 후처리: 생성된 텍스트에서 입력 프롬프트 부분을 제거합니다.
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                logger.info(f"VLM raw output (after processor.batch_decode): '{generated_text}'", extra={'vla_node': _vla_node_vlm_inference})

                processed_text = generated_text # 기본적으로 원본 텍스트로 시작

                # 1차 VLM 질의 결과 처리 (find_object + use_primary_query 일 때, 아직 1차 결과가 안 정해졌다면)
                if args.goal == Goal.FIND_OBJECT.value and args.use_primary_query and primary_query_result is None:
                    # 1차 질의의 응답("yes" 또는 "no")은 전체 원본 생성 텍스트에서 파싱 시도
                    parsed_primary_answer = parse_yes_no_answer(generated_text) 
                    if parsed_primary_answer is not None:
                        primary_query_result = parsed_primary_answer # 파싱 성공 시 primary_query_result 업데이트
                        logger.info(f"Primary query answer parsed from raw output: '{'yes' if primary_query_result else 'no'}'", extra={'vla_node': _vla_node_vlm_inference})
                        # 1차 질의의 목적은 정보 획득이므로, processed_text를 여기서 "yes"나 "no"로 바꾸지 않음.
                        # processed_text는 일반적인 프롬프트 제거 로직을 따름 (아래에서 수행)
                    else:
                        logger.warning(f"Primary query answer 'yes'/'no' not clearly found in raw output: '{generated_text}'. Proceeding with full text for potential secondary prompt.", extra={'vla_node': _vla_node_vlm_inference})
                
                # 일반적인 프롬프트 제거 로직 (1차 질의 "yes"/"no" 응답이 아닌 경우 또는 1차 질의 후 2차 질의 응답 처리 시)
                # prompt 변수에는 <image> 가 포함되어 있을 수 있음
                prompt_text_only = prompt_text.replace("<image>", "").strip() # <image> 제외한 순수 텍스트 프롬프트

                # generated_text에서 prompt_text_only 부분을 찾아 제거 시도
                # 모델이 프롬프트를 그대로 반복하고 답변을 생성하는 일반적인 경우를 처리
                # split을 사용하여 prompt_text_only 이후의 내용을 가져옴
                if prompt_text_only in generated_text:
                    # generated_text에서 prompt_text_only가 시작되는 지점부터 그 길이만큼 건너뛴 후의 텍스트를 가져옴
                    # split으로 나누고 마지막 부분을 취하는 것이 더 안전할 수 있음
                    parts = generated_text.split(prompt_text_only, 1)
                    if len(parts) > 1:
                        processed_text = parts[1].strip()
                    else:
                        # prompt_text_only가 포함은 되어있으나, 그 뒤에 텍스트가 없는 경우 (예: 응답이 프롬프트와 정확히 동일)
                        processed_text = "" # 빈 문자열로 처리
                        logger.warning(f"Response was identical to the prompt (text only part): '{prompt_text_only}'. Processed text is empty.", extra={'vla_node': _vla_node_vlm_inference})
                else:
                    # <image> 태그를 제외한 프롬프트가 응답에 그대로 포함되어 있지 않은 경우,
                    # 모델이 프롬프트를 변형했거나, 응답이 매우 짧을 수 있음.
                    # 이 경우, 일단 <image> 태그만 제거하고 기본적인 strip만 수행.
                    logger.warning(f"Could not strip prompt using prompt_text_only ('{prompt_text_only}'). Input prompt might have been altered by the model. Basic cleaning will be applied.", extra={'vla_node': _vla_node_vlm_inference})
                    processed_text = generated_text # 이미 processed_text는 generated_text로 시작했으므로, 여기선 특별한 작업 X
                
                # 최종적으로 남아있을 수 있는 <image> 플레이스홀더 및 양쪽 공백 제거
                processed_text = processed_text.replace("<image>", "").strip()

                logger.info(f"VLM processed output (after attempting prompt removal): '{processed_text}'", extra={'vla_node': _vla_node_vlm_inference})

                # 1차 VLM 질의 결과에 따른 2차 질의 또는 액션 실행 로직 (find_object + use_primary_query)
                execute_action(selected_goal, processed_text, frame=None, output_dir=args.output_image_dir, target_object=args.target_object)
                logger.info("Image-based action executed successfully.", extra={'vla_node': VLA_NODE_ACTION}) # 액션 완료 후 로그

        except FileNotFoundError:
            logger.error(f"Error: Image file not found at path: {args.image_path}", extra={'vla_node': _vla_node_img_proc})
        except Exception as e:
            logger.error(f"An unexpected error occurred during image processing or VLM inference: {e}", exc_info=True, extra={'vla_node': VLA_NODE_GENERAL})
        logger.info("Single image processing finished.", extra={'vla_node': _vla_node_img_proc}) # 이미지 처리 모드 종료
        return # 이미지 파일 처리 후 스크립트 종료
    
    else: # --image_path 인자가 없는 경우: 웹캠 실시간 처리 모드
        _vla_node_webcam = VLA_NODE_VISION # 웹캠 처리는 VISION 노드
        logger.info("Webcam processing mode selected.", extra={'vla_node': _vla_node_webcam})
        cap = cv2.VideoCapture(0) # 기본 웹캠(0번) 열기. TODO: 웹캠 ID를 인자로 받을 수 있도록 개선
        if not cap.isOpened(): # 웹캠 열기 실패 시
            logger.error("Fatal error: Could not open webcam. Please check camera connection and permissions.", extra={'vla_node': _vla_node_webcam})
            return

        logger.info(f"Webcam opened successfully. Goal: {selected_goal.value}. Press ESC to quit window.", extra={'vla_node': _vla_node_webcam})
        
        frame_count = 0 # 처리된 프레임 수 카운터
        inference_interval = 60 # VLM 추론을 수행할 프레임 간격 (예: 60프레임마다 한 번)

        while True: # 웹캠 프레임 실시간 처리 루프
            ret, frame = cap.read() # 웹캠으로부터 한 프레임 읽기
            if not ret: # 프레임 읽기 실패 시 (예: 카메라 연결 끊김)
                logger.error("Failed to capture frame from webcam. Exiting loop.", extra={'vla_node': _vla_node_webcam})
                break
            
            # 현재 프레임을 화면에 표시 (OpenCV 윈도우)
            cv2.imshow(f"Webcam Feed (Goal: {selected_goal.value})", frame)

            if selected_goal in [Goal.DESCRIBE_SCENE, Goal.IDENTIFY_MAIN_OBJECT, Goal.FIND_OBJECT]:
                if frame_count % inference_interval == 0:
                    logger.info(f"Preparing webcam frame {frame_count} for VLM inference...", extra={'vla_node': _vla_node_webcam})
                    try:
                        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        current_prompt_text = "" # 현재 프레임에 대한 프롬프트
                        if args.use_primary_query:
                            # --- 1차 VLM 질의 (웹캠 - 구현 예정) ---
                            _vla_node_primary_query_wc = VLA_NODE_VLM
                            logger.info("Primary VLM query for webcam frame. (Implementation pending)", extra={'vla_node': _vla_node_primary_query_wc})
                            # TODO: 웹캠용 1차 질의 및 2차 프롬프트 생성 로직 (위의 이미지 파일 처리 부분 참고)
                            # current_prompt_text = get_vlm_prompt_for_webcam_conditional(selected_goal, ...)
                            current_prompt_text = get_vlm_prompt(selected_goal, args.custom_prompt, args.target_object) # 임시
                        else:
                            # --- 단일 VLM 질의 (웹캠 - 기존 방식) ---
                            current_prompt_text = get_vlm_prompt(selected_goal, args.custom_prompt, args.target_object)
                        
                        inputs = processor(text=current_prompt_text, images=image_pil, return_tensors="pt").to(device)
                        
                        _vla_node_webcam_inf = VLA_NODE_VLM
                        logger.info("Performing VLM inference on current webcam frame...", extra={'vla_node': _vla_node_webcam_inf})
                        start_time = time.time()
                        with torch.no_grad():
                            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                        end_time = time.time()
                        logger.info(f"VLM inference on webcam frame completed in {end_time - start_time:.2f} seconds.", extra={'vla_node': _vla_node_webcam_inf})

                        decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # VLM 출력 후처리 (current_prompt_text 사용)
                        clean_current_prompt_text = current_prompt_text.replace("<image> ", "")
                        if decoded_output.startswith(current_prompt_text):
                             vlm_processed_output = decoded_output[len(current_prompt_text):].strip()
                        elif decoded_output.startswith(clean_current_prompt_text):
                             vlm_processed_output = decoded_output[len(clean_current_prompt_text):].strip()
                        elif decoded_output.startswith("<image>"):
                             temp_output = decoded_output[len("<image> "):].strip()
                             if temp_output.startswith(clean_current_prompt_text):
                                 vlm_processed_output = temp_output[len(clean_current_prompt_text):].strip()
                             else:
                                 vlm_processed_output = temp_output
                        else:
                             vlm_processed_output = decoded_output
                        
                        logger.info(f"VLM raw output (webcam): '{decoded_output}'", extra={'vla_node': _vla_node_webcam_inf})
                        logger.info(f"VLM processed output (webcam, prompt removed): '{vlm_processed_output}'", extra={'vla_node': _vla_node_webcam_inf})
                        
                        execute_action(selected_goal, vlm_processed_output, frame, args.output_image_dir, args.target_object)

                    except Exception as e:
                        logger.error(f"An error occurred during VLM inference on webcam frame: {e}", exc_info=True, extra={'vla_node': VLA_NODE_VLM})
            
            elif selected_goal == Goal.SAVE_IMAGE:
                if frame_count % (inference_interval * 2) == 0:
                    _vla_node_save = VLA_NODE_ACTION
                    logger.info(f"Executing SAVE_IMAGE goal for webcam frame {frame_count}", extra={'vla_node': _vla_node_save})
                    # SAVE_IMAGE 시에는 VLM 출력이 필수는 아니지만, get_vlm_prompt는 "Describe this image briefly."를 반환함.
                    # 이 출력을 사용할 수도, 안 할 수도 있음. 현재 execute_action은 vlm_output을 받음.
                    # 간단하게 빈 문자열 또는 기본 설명을 위한 VLM 호출을 여기서 수행할 수도 있음.
                    # 여기서는 execute_action에 빈 vlm_output을 전달하여 이미지 저장에만 집중.
                    execute_action(selected_goal, "", frame, args.output_image_dir) # ACTION

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF # 키보드 입력 대기 (1ms). 0xFF 마스크는 일부 시스템 호환성용.
            if key == 27:  # ESC 키가 눌리면 루프 종료
                logger.info("ESC key pressed by user. Exiting webcam loop...", extra={'vla_node': VLA_NODE_GENERAL})
                break
        
        # --- 웹캠 루프 종료 후 처리 --- 
        cap.release() # 웹캠 리소스 해제
        cv2.destroyAllWindows() # 모든 OpenCV 창 닫기
        logger.info("Webcam resources released and windows closed.", extra={'vla_node': _vla_node_webcam})

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main() 함수 호출
    main()
    # main 함수 실행 완료 후 전체 스크립트 종료 로그
    logger.info("VLA inference script finished execution.", extra={'vla_node': VLA_NODE_SETUP}) # 스크립트 종료는 SETUP의 마무리로 간주