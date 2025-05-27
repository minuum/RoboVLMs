#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import soundfile as sf
import io

class WhisperSTTNode(Node):
    def __init__(self):
        super().__init__('whisper_stt_node')
        self.get_logger().info('Whisper STT 노드 시작, /audio/raw 토픽 구독 대기 중...')
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            '/audio/raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Whisper 모델 로드
        self.model_name = "TheoJo/whisper-tiny-kos"
        self.get_logger().info(f"Whisper 모델 로딩: {self.model_name}")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.get_logger().info(f"모델 로딩 완료. 사용 장치: {self.device}")
        except Exception as e:
            self.get_logger().error(f"모델 로딩 중 오류 발생: {e}")
            rclpy.shutdown()

    def listener_callback(self, msg):
        self.get_logger().info('오디오 데이터 수신')
        
        # UInt8MultiArray (bytes) to NumPy array
        audio_bytes = np.array(msg.data, dtype=np.uint8).tobytes()
        
        # Bytes to Audio Tensor
        # arecord의 S16_LE는 16-bit signed little-endian PCM을 의미
        try:
            # soundfile을 사용하여 raw PCM 바이트를 오디오 데이터로 직접 읽음
            # sf.read는 (samples, channels) 형태의 numpy 배열과 샘플레이트를 반환
            # MicPublisher는 16kHz, 모노 (1 채널)로 녹음함
            audio_np, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='int16', channels=1, samplerate=16000, format='RAW', subtype='PCM_16')
            
            # NumPy array to PyTorch tensor
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0) # (1, num_samples) 형태로 변환
            
            # Whisper 모델은 16kHz 샘플링 레이트를 기대함
            if samplerate != 16000:
                self.get_logger().info(f"리샘플링 필요: {samplerate}Hz -> 16000Hz")
                waveform = torchaudio.functional.resample(waveform, orig_freq=samplerate, new_freq=16000)
            
            self.get_logger().info("오디오 데이터 전처리 완료")

            inputs = self.processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            predicted_ids = self.model.generate(inputs)
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            self.get_logger().info(f"STT 인식 결과: {text}")

        except Exception as e:
            self.get_logger().error(f"오디오 처리 또는 STT 변환 중 오류: {e}")

def main(args=None):
    rclpy.init(args=args)
    whisper_stt_node = WhisperSTTNode()
    try:
        rclpy.spin(whisper_stt_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_stt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()