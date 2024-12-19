import cv2

# 웹캠 연결 (기본 카메라는 인덱스 0으로 설정)
cap = cv2.VideoCapture(0)

# 웹캠 연결 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 프레임을 캡처하여 화면에 출력
while True:
    ret, frame = cap.read()

    # 프레임이 제대로 읽히지 않았을 때
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 화면에 프레임 출력
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 자원 해제
cap.release()
cv2.destroyAllWindows()
