import cv2
import traceback

gst_str = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

print("스크립트 시작")
try:
    print(f"GStreamer 파이프라인: {gst_str}")
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ 카메라 열기 실패 (cap.isOpened() is False)")
    else:
        print("✅ 카메라 열기 성공")
        for i in range(5):
            print(f"프레임 읽기 시도 {i+1}/5...")
            ret, frame = cap.read()
            if ret:
                # 현재 홈 디렉토리에 저장하도록 경로 수정
                file_path = f"/workspace/RoboVLMs/frame_{i}.jpg"
                cv2.imwrite(file_path, frame)
                print(f"✅ {file_path} 저장 완료")
            else:
                print(f"❌ 프레임 {i+1} 읽기 실패 (ret is False)")
                break
        cap.release()
        print("카메라 해제 완료")

except Exception as e:
    print(f"💣 스크립트 실행 중 오류 발생: {e}")
    print("--- Traceback ---")
    traceback.print_exc()
    print("-----------------")

print("스크립트 종료")