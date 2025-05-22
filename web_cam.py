import cv2

def main():
    # 웹캠 열기 (일반적으로 0번 카메라)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    print("웹캠이 성공적으로 열렸습니다. 'q' 키를 누르면 종료됩니다.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break

        # 프레임 보여주기
        cv2.imshow('Webcam Test', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("웹캠 테스트가 종료되었습니다.")

if __name__ == '__main__':
    main()