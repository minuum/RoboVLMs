# --- 1. 필수 라이브러리 임포트 ---
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import cv2
import numpy as np
import os
import sys
import re

# --- big_vision 관련 변수 초기화 ---
big_vision_available = False
reconstruct_masks_fn = None

def setup_big_vision(custom_big_vision_path):
    global big_vision_available, reconstruct_masks_fn
    if not custom_big_vision_path or not os.path.isdir(custom_big_vision_path):
        if custom_big_vision_path:
            print(f"경고: 제공된 big_vision 경로를 찾을 수 없습니다: {custom_big_vision_path}")
        print("경고: big_vision 경로가 제공되지 않았거나 유효하지 않아 세그멘테이션 마스크 고급 재구성이 불가능합니다.")
        big_vision_available = False
        reconstruct_masks_fn = None
        return

    abs_bv_path = os.path.abspath(custom_big_vision_path)
    if abs_bv_path not in [os.path.abspath(p) for p in sys.path]:
        sys.path.append(abs_bv_path)
        print(f"'{abs_bv_path}' 경로를 sys.path에 추가했습니다.")

    try:
        import big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
        reconstruct_masks_fn = segeval.get_reconstruct_masks('oi')
        print("big_vision의 segeval 모듈 및 reconstruct_masks 함수 로드 성공!")
        big_vision_available = True
    except ImportError as e:
        print(f"big_vision 모듈 임포트 실패: {e}")
        print(f"big_vision_path 변수('{abs_bv_path}')가 정확한지, 해당 경로에 big_vision 리포지토리가 올바르게 클론되었는지,")
        print("그리고 필요한 의존성 (예: absl-py, ml_collections, einops)이 설치되었는지 확인해주세요.")
        big_vision_available = False
        reconstruct_masks_fn = None
    except Exception as e:
        print(f"big_vision 관련 로드 중 예상치 못한 오류: {e}")
        big_vision_available = False
        reconstruct_masks_fn = None

    if not big_vision_available:
        print("경고: big_vision 모듈을 로드할 수 없어 세그멘테이션 마스크 재구성이 불가능합니다.")

# --- 세그멘테이션 결과 파싱 함수 정의 ---
def parse_segmentation_output(text_output, img_width, img_height, current_prompt_text):
    detections = []
    text_to_parse = text_output
    prompt_cleaned = current_prompt_text.strip()
    if text_to_parse.startswith(prompt_cleaned):
            text_to_parse = text_to_parse[len(prompt_cleaned):].strip()
    
    segments_data = text_to_parse.split(';')
    norm_factor = 1023.0

    for segment_data_str in segments_data:
        segment_data_str = segment_data_str.strip()
        if not segment_data_str:
            continue
        loc_tokens_match = re.findall(r"<loc(\d{4})>", segment_data_str)
        seg_tokens_match = re.findall(r"<seg(\d{3})>", segment_data_str)
        label_text_candidate = segment_data_str
        for loc_token_str in re.findall(r"<loc\d{4}>", label_text_candidate):
            label_text_candidate = label_text_candidate.replace(loc_token_str, "")
        for seg_token_str in re.findall(r"<seg\d{3}>", label_text_candidate):
            label_text_candidate = label_text_candidate.replace(seg_token_str, "")
        label = label_text_candidate.replace("segment", "").strip()
        if not label: label = "unknown"
        if len(loc_tokens_match) == 4:
            y_min_token, x_min_token, y_max_token, x_max_token = map(int, loc_tokens_match)
            norm_y_min, norm_x_min, norm_y_max, norm_x_max = y_min_token/norm_factor, x_min_token/norm_factor, y_max_token/norm_factor, x_max_token/norm_factor
            box_scaled_pixels = [round(norm_x_min*img_width,2), round(norm_y_min*img_height,2), round(norm_x_max*img_width,2), round(norm_y_max*img_height,2)]
            current_detection = {"label":label, "original_loc_tokens":[y_min_token,x_min_token,y_max_token,x_max_token], "scaled_box_pixels":box_scaled_pixels}
            if big_vision_available and reconstruct_masks_fn and len(seg_tokens_match)==16:
                seg_token_values = np.array([int(st) for st in seg_tokens_match], dtype=np.int32)
                try:
                    reconstructed_mask_array = reconstruct_masks_fn(seg_token_values[np.newaxis,:])
                    current_detection["segmentation_mask_tokens"] = seg_token_values.tolist()
                    current_detection["reconstructed_mask"] = reconstructed_mask_array[0].tolist()
                except Exception as e_mask:
                    print(f"마스크 재구성 중 오류(Label:{label}):{e_mask}")
                    current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match]
            elif len(seg_tokens_match) > 0: current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match]
            detections.append(current_detection)
    return detections

# --- 로봇 이동 명령 계산 함수 (Twist 메시지 형식 고려하여 수정) ---
def calculate_move_command_for_twist(bbox_pixels, image_width, image_height):
    if bbox_pixels is None or len(bbox_pixels) != 4:
        print("경고: 유효한 바운딩 박스 정보가 없어 이동 명령을 계산할 수 없습니다.")
        return 0.0, 0.0, 0.0 # linear_x, linear_y, angular_z (정지)

    x_min, y_min, x_max, y_max = bbox_pixels

    # --- 튜닝 필요한 상수들 ---
    KP_ALIGN_ANGULAR_Z = 0.005      # 좌우 정렬(회전)을 위한 비례 게인
    ALIGNMENT_TOLERANCE_RATIO = 0.05 # 이미지 너비 대비 중앙 정렬 허용 오차 (5%)
    FORWARD_SPEED_WHEN_ALIGNED = 0.1 # 정렬 후 전진 속도 (m/s)
    MAX_ANGULAR_Z_SPEED = 0.5        # 최대 회전 속도 (rad/s)
    OBJECT_REACHED_HEIGHT_RATIO = 0.7 # 객체 도달 판단 기준 (바운딩 박스 높이/이미지 높이)
    # --- 튜닝 상수 끝 ---

    target_cx = (x_min + x_max) / 2
    image_center_x = image_width / 2
    error_x = target_cx - image_center_x
    bbox_height = y_max - y_min

    linear_x_cmd = 0.0
    linear_y_cmd = 0.0  # 이 예제에서는 직접적인 좌우 이동(strafe)은 사용하지 않고 회전으로 대체
    angular_z_cmd = 0.0

    # 1. 도착 여부 판단
    if (bbox_height / image_height) >= OBJECT_REACHED_HEIGHT_RATIO:
        print(f"객체({(bbox_height / image_height)*100:.1f}% 차지)에 충분히 접근함. 정지.")
        return 0.0, 0.0, 0.0 # 모든 속도 0으로 정지

    # 2. 좌우 정렬 단계 (회전 사용)
    if abs(error_x) > (image_width * ALIGNMENT_TOLERANCE_RATIO):
        angular_z_cmd = -KP_ALIGN_ANGULAR_Z * error_x # error_x에 비례하여 회전 속도 결정
        angular_z_cmd = max(-MAX_ANGULAR_Z_SPEED, min(MAX_ANGULAR_Z_SPEED, angular_z_cmd)) # 최대 속도 제한
        print(f"정렬 시도: error_x={error_x:.2f}, 계산된 angular_z_cmd={angular_z_cmd:.3f}")
    else:
        # 3. 정렬 완료 시 전진 단계
        print("정렬 완료. 전진 시도.")
        linear_x_cmd = FORWARD_SPEED_WHEN_ALIGNED
    
    # 만약 로봇이 전진과 회전을 동시에 할 수 있다면 이대로 반환
    # 그렇지 않다면, 상태를 관리하며 '회전 완료 후 전진' 순서로 명령을 나누어 발행해야 함
    # (예: angular_z_cmd가 0에 가까워질 때까지 회전 명령만 보내고, 그 후 linear_x_cmd 명령 전송)
    # 여기서는 일단 계산된 값을 그대로 반환
    return linear_x_cmd, linear_y_cmd, angular_z_cmd

print("라이브러리 임포트 완료.")

def main():
    parser = argparse.ArgumentParser(description="PaliGemma Inference Script with Parsing, Twist-like Move Command Calc, and Webcam Support")
    parser.add_argument("--config_path",type=str,default=None,help="Path to config JSON")
    parser.add_argument("--image_path",type=str,default=None,help="Path to input image. If None, use webcam.")
    parser.add_argument("--prompt",type=str,required=True,help="Text prompt for the model")
    parser.add_argument("--model_cache_dir",type=str,default=".vlms_cache",help="Directory to cache models")
    parser.add_argument("--webcam_id",type=int,default=0,help="Webcam ID")
    parser.add_argument("--max_new_tokens",type=int,default=256,help="Max new tokens for generation")
    parser.add_argument("--big_vision_path",type=str,default=None,help="Path to cloned big_vision repo (optional)")
    args = parser.parse_args()

    if args.big_vision_path: setup_big_vision(args.big_vision_path)
    else:
        print("경고: --big_vision_path가 제공되지 않아 세그멘테이션 마스크 고급 재구성이 불가능합니다.")
        global big_vision_available
        big_vision_available = False

    model_path_from_config = "google/paligemma-3b-mix-224"
    tokenizer_path_from_config = "google/paligemma-3b-mix-224"
    max_new_tokens = args.max_new_tokens
    if args.config_path:
        print(f"Loading config from: {args.config_path}")
        with open(args.config_path,'r') as f: config = json.load(f)
        model_url = config.get("model_url", model_path_from_config)
        model_path_from_config = config.get("vlm",{}).get("pretrained_model_name_or_path",model_url)
        tokenizer_path_from_config = config.get("tokenizer",{}).get("pretrained_model_name_or_path",model_url)
        max_new_tokens = config.get("generation_config",{}).get("max_new_tokens",max_new_tokens)
    else: print("No config path provided, using default model, tokenizer paths, and max_new_tokens.")
    print(f"Effective max_new_tokens: {max_new_tokens}")
    model_save_dir = Path(args.model_cache_dir)/model_path_from_config.split('/')[-1]
    model_save_dir.mkdir(parents=True,exist_ok=True)
    print(f"Using model path: {model_path_from_config}")
    print(f"Using tokenizer path: {tokenizer_path_from_config}")
    print(f"Models will be cached in/loaded from: {model_save_dir}")
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(tokenizer_path_from_config,cache_dir=model_save_dir)
        model_kwargs = {"cache_dir":model_save_dir,"low_cpu_mem_usage":True}
        if device.type=="cuda": model_kwargs["torch_dtype"]=torch.bfloat16; model_kwargs["device_map"]="auto"
        elif device.type=="mps": model_kwargs["torch_dtype"]=torch.float32
        else: model_kwargs["torch_dtype"]=torch.float32
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path_from_config,**model_kwargs)
        if device.type!="cuda": model.to(device)
        model.eval()
        print("Model and processor loaded successfully.")
    except Exception as e: print(f"Error loading model or processor: {e}"); return

    def infer_parse_and_calculate_move_twist(current_raw_image, current_prompt): # 함수 이름 변경
        img_width,img_height = current_raw_image.size
        print(f"\nInput image size: ({img_width},{img_height}) for prompt: '{current_prompt}'")
        inputs_data = processor(text=current_prompt,images=current_raw_image,return_tensors="pt").to(device)
        if 'pixel_values' in inputs_data: inputs_data['pixel_values'] = inputs_data['pixel_values'].to(torch.float32 if device.type=="mps" else model.dtype)
        print("Performing inference...")
        with torch.inference_mode():
            try:
                output_ids = model.generate(**inputs_data,max_new_tokens=max_new_tokens,do_sample=False)
                generated_text_output = processor.decode(output_ids[0],skip_special_tokens=True)
                print(f"\n--- Prompt ---\n{current_prompt.strip()}")
                print(f"--- Raw Model Output ---\n{generated_text_output.strip()}")
                parsed_detections = parse_segmentation_output(generated_text_output,img_width,img_height,current_prompt)
                print("\n--- Parsed Detection and Segmentation Results ---")
                if parsed_detections:
                    for i,det in enumerate(parsed_detections):
                        print(f"Detection #{i+1}:")
                        print(f"  Label: {det.get('label','N/A')}")
                        print(f"  Original Loc Tokens: {det.get('original_loc_tokens')}")
                        print(f"  Scaled BBox Pixels: {det.get('scaled_box_pixels')}")
                        if "reconstructed_mask" in det and det['reconstructed_mask'] is not None:
                             mask_shape = (len(det['reconstructed_mask']),len(det['reconstructed_mask'][0]) if det['reconstructed_mask'] and len(det['reconstructed_mask'])>0 else 0)
                             print(f"  Reconstructed Mask (shape {mask_shape}): Available (display omitted)")
                        elif "segmentation_mask_tokens" in det: print(f"  Segmentation Mask Tokens: {det.get('segmentation_mask_tokens')}")
                        print("-" * 10)
                    
                    target_detection = parsed_detections[0]
                    if target_detection.get('scaled_box_pixels'):
                        print(f"\n--- Calculating Twist-like Move Command for Label: {target_detection.get('label','N/A')} ---")
                        lx,ly,az = calculate_move_command_for_twist(target_detection['scaled_box_pixels'],img_width,img_height) # 수정된 함수 호출
                        print(f"===> Suggested Twist Command: linear.x={lx:.3f}, linear.y={ly:.3f}, angular.z={az:.3f} <===")
                        print("     (Note: This is a calculated suggestion, not actual robot movement.)")
                    else: print("\nNo valid bounding box in the first detection to calculate move command.")
                else:
                    print("No objects parsed from the output. Cannot calculate move command.")
                    if not "<loc" in generated_text_output: print(f"Model's textual answer (VQA): {generated_text_output.replace(current_prompt.strip(),'').strip()}")
            except Exception as e_gen: print(f"Error during model generation or processing: {e_gen}")

    if args.image_path:
        try:
            raw_image_file = Image.open(args.image_path).convert('RGB')
            infer_parse_and_calculate_move_twist(raw_image_file, args.prompt) # 변경된 함수 호출
        except FileNotFoundError: print(f"Error: Image file not found at {args.image_path}"); return
        except Exception as e_load: print(f"Error loading image: {e_load}"); return
    else: # Webcam mode
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened(): print(f"Error: Could not open webcam with ID {args.webcam_id}."); return
        print("\nWebcam mode enabled. Press SPACE to capture and infer, ESC to quit.")
        cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
        while True:
            ret,frame = cap.read()
            if not ret: print("Error: Failed to capture frame from webcam."); break
            display_frame = frame.copy()
            cv2.putText(display_frame,f"Prompt: {args.prompt}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(display_frame,"SPACE: Infer | ESC: Quit",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow("Webcam Feed",display_frame)
            key = cv2.waitKey(1)&0xFF
            if key==27: break # ESC
            elif key==32: # SPACE
                print("\nCapturing frame, performing inference, and calculating move command...")
                rgb_frame_cv = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                raw_image_cam = Image.fromarray(rgb_frame_cv)
                infer_parse_and_calculate_move_twist(raw_image_cam, args.prompt) # 변경된 함수 호출
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()