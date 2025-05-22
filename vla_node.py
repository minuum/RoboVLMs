# --- 1. 필수 라이브러리 임포트 ---
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
# from robovlms.model.vlm_builder import build_vlm # build_vlm 함수는 직접 사용하지 않음
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import cv2 # OpenCV 추가
import numpy as np # OpenCV 이미지 처리 및 big_vision 마스크 처리를 위해 추가
import os # os.path 등을 위해 추가
import sys # sys.path 수정을 위해 추가
import re # 결과 파싱 시 사용 가능

# --- big_vision 리포지토리 경로 설정 및 sys.path 추가 (스크립트 A로부터 가져옴) ---
# !!! 중요: 사용자님이 big_vision을 클론한 실제 경로로 수정해주세요 !!!
big_vision_path = "/Users/hayubin/big_vision" # 예시 경로, 실제 경로로 수정 필요

big_vision_available = False
reconstruct_masks_fn = None

if os.path.isdir(big_vision_path) and os.path.abspath(big_vision_path) not in [os.path.abspath(p) for p in sys.path]:
    sys.path.append(os.path.abspath(big_vision_path))
    print(f"'{os.path.abspath(big_vision_path)}' 경로를 sys.path에 추가했습니다.")

# big_vision 모듈 로드 시도 (스크립트 A로부터 가져옴)
# (스크립트 B에는 원래 없던 부분이므로, 필요 없다면 이 블록 전체를 주석 처리하거나 삭제 가능)
try:
    # sys.path에 추가된 후 임포트 시도
    import big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
    reconstruct_masks_fn = segeval.get_reconstruct_masks('oi') # 'oi'는 예시, 모델/데이터셋에 따라 다를 수 있음
    print("big_vision의 segeval 모듈 및 reconstruct_masks 함수 로드 성공!")
    big_vision_available = True
except ImportError as e:
    print(f"big_vision 모듈 임포트 실패: {e}")
    print(f"big_vision_path 변수('{big_vision_path}')가 정확한지, 해당 경로에 big_vision 리포지토리가 올바르게 클론되었는지,")
    print("그리고 필요한 의존성 (예: absl-py, ml_collections, einops)이 설치되었는지 확인해주세요.")
    print("세그멘테이션 마스크 재구성은 big_vision 없이는 불가능합니다.")
except Exception as e: # 다른 예외도 처리
    print(f"big_vision 관련 로드 중 예상치 못한 오류: {e}")

if not big_vision_available:
    print("경고: big_vision 모듈을 로드할 수 없어 세그멘테이션 마스크 재구성이 불가능합니다. 바운딩 박스만 추출될 수 있습니다.")

print("라이브러리 임포트 완료.")


# --- 세그멘테이션 결과 파싱 함수 정의 (스크립트 A로부터 가져옴) ---
def parse_segmentation_output(text_output, img_width, img_height, current_prompt_text):
    detections = []
    
    text_to_parse = text_output
    prompt_cleaned = current_prompt_text.strip() # 프롬프트에서 앞뒤 공백 제거
    # 모델 출력에서 프롬프트 부분을 제거 (존재한다면)
    if text_to_parse.startswith(prompt_cleaned):
            text_to_parse = text_to_parse[len(prompt_cleaned):].strip()
    
    # 객체별 데이터 분리 (예: "segment object1 <loc...> <seg...>; segment object2 <loc...> <seg...>")
    # PaliGemma는 프롬프트에 따라 하나의 텍스트 블록으로 여러 객체를 표현할 수 있음
    # 여기서는 프롬프트에서 ';'로 구분했으므로, 출력도 유사한 패턴을 따르거나,
    # 하나의 텍스트 내에서 여러 <loc...><seg...> 레이블 패턴을 찾아야 함.
    # 스크립트 A에서는 segments_data = text_to_parse.split(';') 로 단순 분리했음.
    # PaliGemma의 실제 출력 패턴에 따라 이 부분은 매우 중요하고 조정이 필요함.
    # 예를 들어, 'label1 <loc...> <seg...>; label2 <loc...> <seg...>' 또는
    # 'label1 <loc...> <seg...> label2 <loc...> <seg...>' 등 다양.
    # 여기서는 스크립트 A의 방식을 따르되, 프롬프트에 있는 객체 수만큼 반복하는 것을 고려.
    # 또는 정규표현식으로 모든 '레이블 <loc...> <seg...>' 패턴을 찾는 것이 더 강인할 수 있음.

    # 여기서는 단순화를 위해 ';' 로 구분된 각 세그먼트를 처리한다고 가정 (스크립트 A 방식)
    # 실제로는 모델이 프롬프트의 ';'를 어떻게 해석하고 출력하는지 봐야 함.
    # 만약 모델이 woman에 대한 정보, dog에 대한 정보를 순차적으로 모두 출력한다면
    # 정규표현식으로 '<loc....><seg....> 레이블' 패턴을 찾는 것이 더 나을 수 있음.
    # 아래는 스크립트 A의 로직을 최대한 유지하되, 레이블 추출을 개선한 버전입니다.

    norm_factor = 1023.0  # PaliGemma의 좌표 정규화 상수

    # 정규표현식으로 "text <loc...><loc...><loc...><loc...> [<seg...>...]" 패턴 찾기
    # 레이블은 <loc...> 앞에 올 수도 있고, <seg...> 뒤에 올 수도 있음.
    # 예: "woman <loc0395>... <seg023>" 또는 "<loc0395>... <seg023> woman"
    # 이 정규표현식은 "텍스트 (공백) <loc토큰들> (공백과 <seg토큰들> 선택적)" 또는 "<loc토큰들> (공백과 <seg토큰들> 선택적) (공백) 텍스트"
    # 여기서는 좀 더 간단하게, 각 segment_data_str 내에서 정보를 찾는 로직을 유지
    
    # 스크립트 A의 split(';') 방식 대신, 모델이 응답한 전체 텍스트에서 패턴을 찾아야 할 수 있습니다.
    # 예를 들어, "segment woman <loc...>; segment dog <loc...>" 이 아니라
    # "woman <loc...> <seg...> dog <loc...> <seg...>" 처럼 나올 수 있습니다.
    # 우선 스크립트 A처럼 ';'로 분리된 데이터가 있다고 가정하고 진행합니다.
    # 하지만 실제로는 아래 주석처리된 finditer 방식이 더 일반적일 수 있습니다.
    
    # segments_data = text_to_parse.split(';') # 스크립트 A 방식
    # for segment_data_str in segments_data:

    # 좀 더 강인한 접근: text_to_parse 전체에서 패턴을 반복적으로 찾기
    # 패턴: (선택적 레이블) <loc...> <loc...> <loc...> <loc...> (선택적 <seg...>들) (선택적 레이블)
    # 복잡하므로, 스크립트 A의 split(';')이 프롬프트와 잘 맞는다고 가정하고 일단은 유지하되,
    # 레이블 추출 로직을 개선합니다.
    
    segments_data = text_to_parse.split(';') # 스크립트 A의 분리 방식
    if len(segments_data) == 1 and not "<loc" in segments_data[0] and len(prompt_cleaned.split(';')) > 1 :
        # 프롬프트는 여러개인데, 응답에 loc 토큰이 없고, 세미콜론도 없다면, 아마도 모델이 프롬프트를 그대로 반환한 경우
        # 또는 "찾을 수 없음"과 같은 일반 텍스트 응답일 수 있음.
        # 이럴 경우, 분리하지 않고 단일 텍스트로 처리하거나, 객체 탐색 실패로 간주.
        # 여기서는 일단 분리된 것으로 간주하고 진행. 필요시 이 부분에 대한 로직 추가.
        pass


    for segment_data_str in segments_data:
        segment_data_str = segment_data_str.strip()
        if not segment_data_str: # 빈 문자열이면 건너뛰기
            continue

        loc_tokens_match = re.findall(r"<loc(\d{4})>", segment_data_str)
        seg_tokens_match = re.findall(r"<seg(\d{3})>", segment_data_str)
        
        label_text_candidate = segment_data_str
        # 토큰 부분을 제거하여 레이블 후보 만들기
        for loc_token_str in re.findall(r"<loc\d{4}>", label_text_candidate):
            label_text_candidate = label_text_candidate.replace(loc_token_str, "")
        for seg_token_str in re.findall(r"<seg\d{3}>", label_text_candidate):
            label_text_candidate = label_text_candidate.replace(seg_token_str, "")
        
        label = label_text_candidate.replace("segment", "").strip() # "segment" 키워드 및 공백 제거
        if not label: # 레이블이 추출되지 않으면 "unknown" 또는 다른 기본값 사용 가능
            label = "unknown"


        if len(loc_tokens_match) == 4: # 바운딩 박스 좌표 토큰이 4개인 경우
            y_min_token, x_min_token, y_max_token, x_max_token = map(int, loc_tokens_match)

            # 정규화된 좌표 계산 (0~1 범위)
            norm_y_min = y_min_token / norm_factor
            norm_x_min = x_min_token / norm_factor
            norm_y_max = y_max_token / norm_factor
            norm_x_max = x_max_token / norm_factor

            # 실제 이미지 픽셀 좌표로 변환
            box_scaled_pixels = [
                round(norm_x_min * img_width, 2),
                round(norm_y_min * img_height, 2),
                round(norm_x_max * img_width, 2),
                round(norm_y_max * img_height, 2)
            ]
            
            current_detection = {
                "label": label,
                "original_loc_tokens": [y_min_token, x_min_token, y_max_token, x_max_token],
                "scaled_box_pixels": box_scaled_pixels
            }

            # big_vision을 사용한 마스크 재구성 (스크립트 A 로직)
            if big_vision_available and reconstruct_masks_fn and len(seg_tokens_match) > 0 : # 세그 토큰이 하나라도 있으면
                # PaliGemma는 보통 16개의 세그 토큰을 사용
                if len(seg_tokens_match) == 16:
                    seg_token_values = np.array([int(st) for st in seg_tokens_match], dtype=np.int32)
                    try:
                        reconstructed_mask_array = reconstruct_masks_fn(seg_token_values[np.newaxis, :])
                        current_detection["segmentation_mask_tokens"] = seg_token_values.tolist()
                        # 결과는 (1, H, W) 이므로 [0]으로 첫번째 마스크 가져옴
                        current_detection["reconstructed_mask"] = reconstructed_mask_array[0].tolist() 
                    except Exception as e_mask:
                        print(f"마스크 재구성 중 오류 발생 (Label: {label}): {e_mask}")
                        current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match] # 재구성은 못해도 토큰은 저장
                else:
                    # 16개가 아닌 경우, 일단 토큰만 저장
                    current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match]
                    print(f"경고: Label '{label}'에 대해 {len(seg_tokens_match)}개의 세그멘테이션 토큰이 발견되었습니다. (16개 예상)")

            detections.append(current_detection)
    return detections


def main():
    parser = argparse.ArgumentParser(description="PaliGemma Inference Script with Parsing and Webcam Support") # 설명 수정
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to the inference config JSON file (e.g., RoboVLMs/configs/calvin_finetune/inference_paligemma_mps.json)"
    )
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image. If not provided, webcam will be used.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--model_cache_dir", type=str, default=".vlms_cache", help="Directory to cache downloaded models")
    parser.add_argument("--webcam_id", type=int, default=0, help="ID of the webcam to use (default: 0)")
    # 파싱을 위해 max_new_tokens 인자 추가 가능 (스크립트 A는 256 사용)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens for model generation (default: 256 for segmentation)")


    args = parser.parse_args()

    # --- 1. Load Config or Use Defaults ---
    model_path_from_config = "google/paligemma-3b-mix-224"
    tokenizer_path_from_config = "google/paligemma-3b-mix-224"
    # max_new_tokens_from_config = 128 # 스크립트 B의 기본값
    # 명령줄 인자 또는 설정 파일에서 max_new_tokens를 가져오도록 수정
    max_new_tokens = args.max_new_tokens # 명령줄 인자 우선

    if args.config_path:
        print(f"Loading config from: {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        model_url = config.get("model_url", "google/paligemma-3b-mix-224")
        model_path_from_config = config.get("vlm", {}).get("pretrained_model_name_or_path", model_url)
        tokenizer_path_from_config = config.get("tokenizer", {}).get("pretrained_model_name_or_path", model_url)
        # 설정 파일에 max_new_tokens가 있다면 그것을 사용, 없으면 명령줄 인자 값 유지
        max_new_tokens = config.get("generation_config", {}).get("max_new_tokens", max_new_tokens)
        # 또는 tokenizer 설정의 max_text_len을 참고할 수도 있음 (스크립트 B 방식)
        # max_new_tokens = config.get("tokenizer", {}).get("max_text_len", max_new_tokens) 
    else:
        print("No config path provided, using default model, tokenizer paths, and max_new_tokens.")

    print(f"Effective max_new_tokens: {max_new_tokens}") # 실제 사용될 max_new_tokens 값 확인

    model_save_dir = Path(args.model_cache_dir) / model_path_from_config.split('/')[-1]
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using model path: {model_path_from_config}")
    print(f"Using tokenizer path: {tokenizer_path_from_config}")
    print(f"Models will be cached in/loaded from: {model_save_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model and Processor ---
    try:
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(tokenizer_path_from_config, cache_dir=model_save_dir)

        model_kwargs = {
            "cache_dir": model_save_dir,
            "low_cpu_mem_usage": True,
        }

        # 데이터 타입 설정 (스크립트 A는 bfloat16을 CUDA에 사용했었음)
        if device.type == "cuda":
            # model_kwargs["torch_dtype"] = torch.float16 # 스크립트 B 방식
            model_kwargs["torch_dtype"] = torch.bfloat16 # 스크립트 A 방식 (bfloat16이 더 안정적일 수 있음)
            model_kwargs["device_map"] = "auto" 
        elif device.type == "mps":
            model_kwargs["torch_dtype"] = torch.float32
        else: # CPU
            model_kwargs["torch_dtype"] = torch.float32

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path_from_config,
            **model_kwargs
        )

        if device.type != "cuda": 
            model.to(device)
        
        model.eval()
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return

    # --- 공통 추론 및 파싱 로직을 위한 함수 ---
    def infer_and_parse(current_raw_image, current_prompt):
        img_width, img_height = current_raw_image.size
        print(f"Input image size: ({img_width}, {img_height}) for prompt: '{current_prompt}'")

        inputs_data = processor(text=current_prompt, images=current_raw_image, return_tensors="pt").to(device)
        if 'pixel_values' in inputs_data: # MPS 등 일부 환경에서 float32로 명시적 변환
            inputs_data['pixel_values'] = inputs_data['pixel_values'].to(torch.float32 if device.type == "mps" else model.dtype)


        print("Performing inference...")
        with torch.inference_mode(): # torch.no_grad() 와 유사, 추론 시 사용
            try:
                output_ids = model.generate(**inputs_data, max_new_tokens=max_new_tokens, do_sample=False)
                generated_text_output = processor.decode(output_ids[0], skip_special_tokens=True)
                
                print(f"\n--- Prompt ---\n{current_prompt.strip()}")
                print(f"--- Raw Model Output ---\n{generated_text_output.strip()}")

                # 파싱 함수 호출
                parsed_detections = parse_segmentation_output(generated_text_output, img_width, img_height, current_prompt)
                
                print("\n--- Parsed Detection and Segmentation Results ---")
                if parsed_detections:
                    for det in parsed_detections:
                        print(f"Label: {det.get('label')}")
                        print(f"  Original Loc Tokens: {det.get('original_loc_tokens')}")
                        print(f"  Scaled BBox Pixels: {det.get('scaled_box_pixels')}")
                        if "reconstructed_mask" in det and det['reconstructed_mask'] is not None:
                             mask_shape = (len(det['reconstructed_mask']), len(det['reconstructed_mask'][0]) if det['reconstructed_mask'] and len(det['reconstructed_mask']) > 0 else 0)
                             print(f"  Reconstructed Mask (shape {mask_shape}): Available (display omitted)")
                        elif "segmentation_mask_tokens" in det:
                             print(f"  Segmentation Mask Tokens: {det.get('segmentation_mask_tokens')}")
                        print("-" * 20)
                else:
                    print("No objects parsed from the output.")

            except Exception as e_gen:
                print(f"Error during model generation or parsing: {e_gen}")

    if args.image_path:
        # --- 3a. Prepare Input from Image File ---
        try:
            raw_image_file = Image.open(args.image_path).convert('RGB')
            infer_and_parse(raw_image_file, args.prompt) # 공통 함수 호출
        except FileNotFoundError:
            print(f"Error: Image file not found at {args.image_path}")
            return
        except Exception as e_load:
            print(f"Error loading image: {e_load}")
            return
    else:
        # --- 3b. Prepare Input from Webcam ---
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {args.webcam_id}.")
            return

        print("\nWebcam mode enabled. Press SPACE to capture and infer, ESC to quit.")
        cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Prompt: {args.prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA) # 초록색, 얇게
            cv2.putText(display_frame, "SPACE: Infer | ESC: Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)

            cv2.imshow("Webcam Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                print("\nCapturing frame and performing inference...")
                rgb_frame_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_image_cam = Image.fromarray(rgb_frame_cv)
                infer_and_parse(raw_image_cam, args.prompt) # 공통 함수 호출
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()