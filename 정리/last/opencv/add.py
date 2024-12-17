import cv2

# 웹캠 연결 (기본적으로 0번 웹캠 사용)
cap = cv2.VideoCapture(0)

# 해상도 설정 (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 비디오 저장 설정 (H.264 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('./mj/output.avi', fourcc, 30.0, (1280, 720))  #저장하고싶은 파일이름으로 변


if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 웹캠으로부터 프레임을 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 비디오 파일에 프레임 저장
    out.write(frame)
    
    # 프레임을 윈도우에 표시
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
