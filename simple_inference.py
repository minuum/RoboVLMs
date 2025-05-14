import argparse
import json
from pathlib import Path
import torch
from PIL import Image
# from robovlms.model.vlm_builder import build_vlm # build_vlm 함수는 직접 사용하지 않음
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import cv2 # OpenCV 추가
import numpy as np # OpenCV 이미지 처리를 위해 추가

def main():
    parser = argparse.ArgumentParser(description="Simple PaliGemma Inference Script for MPS with Webcam Support")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the inference config JSON file (e.g., RoboVLMs/configs/calvin_finetune/inference_paligemma_mps.json)"
    )
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image. If not provided, webcam will be used.") # 기본값을 None으로 변경
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--model_cache_dir", type=str, default=".vlms_cache", help="Directory to cache downloaded models")
    parser.add_argument("--webcam_id", type=int, default=0, help="ID of the webcam to use (default: 0)")

    args = parser.parse_args()

    # --- 1. Load Config ---
    print(f"Loading config from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    model_url = config.get("model_url", "google/paligemma-3b-mix-224")
    model_path_from_config = config.get("vlm", {}).get("pretrained_model_name_or_path", model_url)
    tokenizer_path_from_config = config.get("tokenizer", {}).get("pretrained_model_name_or_path", model_url)

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
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path_from_config,
            torch_dtype=torch.float32, # MPS/CUDA 모두 float32 우선 사용
            cache_dir=model_save_dir
        )
        model.to(device)
        model.eval()
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return

    if args.image_path:
        # --- 3a. Prepare Input from Image File ---
        try:
            raw_image = Image.open(args.image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {args.image_path}")
            return
        except Exception as e:
            print(f"Error loading image: {e}")
            return
        
        inputs = processor(text=args.prompt, images=raw_image, return_tensors="pt").to(device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float32)

        # --- 4a. Perform Inference from Image File ---
        print("Performing inference on image file...")
        with torch.no_grad():
            try:
                max_new_tokens = config.get("tokenizer", {}).get("max_text_len", 128)
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
                print(f"\nGenerated Text (from image file):\n{generated_text}")
            except Exception as e:
                print(f"Error during model generation: {e}")
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
            
            # 프레임에 프롬프트 표시 (선택 사항)
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Prompt: {args.prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "SPACE: Infer | ESC: Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


            cv2.imshow("Webcam Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                print("\nCapturing frame and performing inference...")
                # OpenCV BGR 이미지를 PIL RGB 이미지로 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_image = Image.fromarray(rgb_frame)

                inputs = processor(text=args.prompt, images=raw_image, return_tensors="pt").to(device)
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float32)

                # --- 4b. Perform Inference from Webcam Frame ---
                with torch.no_grad():
                    try:
                        max_new_tokens = config.get("tokenizer", {}).get("max_text_len", 128)
                        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
                        print(f"\nGenerated Text (from webcam):\n{generated_text}")
                        
                        # 결과 텍스트를 다음 프레임에 잠시 표시 (예시)
                        # 더 나은 UI를 위해서는 별도의 창이나 오버레이 방식 고려
                        # 여기서는 간단히 콘솔에만 출력합니다.

                    except Exception as e:
                        print(f"Error during model generation: {e}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 