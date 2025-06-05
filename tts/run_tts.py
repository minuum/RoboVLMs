# run_tts.py
# torchaudio 사용 -> PyTorch와 친화적이며 음질도 우수함

from transformers import VitsModel, AutoTokenizer
import torch
import torchaudio
import os

# 1. 장치 설정 (GPU 우선 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 및 토크나이저 로드
model = VitsModel.from_pretrained("Matthijs/mms-tts-kor").to(device)
tokenizer = AutoTokenizer.from_pretrained("Matthijs/mms-tts-kor")

# 3. 입력 텍스트 (다중 문장도 가능)
text = "안녕하세요. 반갑습니다." 

# 4. 텍스트 토크나이징
inputs = tokenizer(text, return_tensors="pt")

inputs = {k: v.to(device).long() for k, v in inputs.items()}

# 5. 음성 합성
with torch.no_grad():
    waveform = model(**inputs).waveform  # shape: [1, samples]

# 6. 출력 정규화 및 torchaudio 형식 변환
waveform = waveform / waveform.abs().max()  # -1.0 ~ 1.0 범위로 정규화
waveform = waveform.squeeze(0).unsqueeze(0).cpu()  # shape: [1, N]

# 7. 파일 저장
output_path = "output.wav"
torchaudio.save(output_path, waveform, sample_rate=model.config.sampling_rate)
print(f"음성 파일 저장 완료: {output_path}")
