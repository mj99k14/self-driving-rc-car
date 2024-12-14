import Jetson.GPIO as GPIO
import torch
import cv2
import numpy as np
import time

# **1. GPIO 설정**
GPIO.setmode(GPIO.BCM)  # Jetson Nano에서는 BCM 모드 사용

# 서보 모터 설정
servo_pin = 12  # 서보 모터 PWM 핀 (Jetson Nano의 GPIO 번호)
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50Hz PWM
servo.start(7.5)  # 초기값 (90도)

# DC 모터 설정
dir_pin = 5  # IN1
in2_pin = 6  # IN2
pwm_pin = 13  # ENA (속도 제어 핀)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)
GPIO.setup(pwm_pin, GPIO.OUT)
dc_motor = GPIO.PWM(pwm_pin, 1000)  # 1kHz PWM
dc_motor.start(0)  # 초기 속도 (정지 상태)

# **2. 서보 모터 각도 설정 함수**
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)  # 안정성을 위한 딜레이
    servo.ChangeDutyCycle(0)  # 서보 보호를 위해 PWM 끄기

# **3. DC 모터 제어 함수**
def control_dc_motor(direction, speed):
    if direction == "forward":  # 전진
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
    elif direction == "backward":  # 후진
        GPIO.output(dir_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
    elif direction == "stop":  # 정지
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)

    dc_motor.ChangeDutyCycle(speed)  # 속도 설정 (0~100)

# **4. PilotNet 모델 정의**
class PilotNet(torch.nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 36, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(36, 48, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 1 * 18, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5)  # 5개의 범주로 분류
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# **5. 모델 로드**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = PilotNet().to(device)

# 모델 경로 수정
model_path = "/home/kimminjung/Desktop/best_pilotnet_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("모델 로드 성공.")
except FileNotFoundError:
    print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    GPIO.cleanup()
    exit()
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    GPIO.cleanup()
    exit()

# **6. 카테고리 설정**
categories = [30, 60, 90, 120, 150]  # 조향 각도 범주

# **7. 카메라 초기화**
cap = cv2.VideoCapture(0)  # 기본 카메라 장치 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("카메라 초기화 실패.")
    GPIO.cleanup()
    exit()

# **8. RC 카 제어 메인 루프**
try:
    print("RC 카 제어 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # **입력 데이터 전처리**
        frame_resized = cv2.resize(frame, (200, 66))
        frame_normalized = frame_resized / 255.0
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))  # HWC → CHW
        frame_tensor = torch.tensor(frame_transposed, dtype=torch.float32).unsqueeze(0).to(device)

        # **모델 예측**
        outputs = model(frame_tensor)
        predicted_category = torch.argmax(outputs, dim=1).item()
        predicted_angle = categories[predicted_category]
        print(f"예측된 조향 각도: {predicted_angle}")

        # **서보 모터 조정**
        set_servo_angle(predicted_angle)

        # **DC 모터 전진**
        control_dc_motor("forward", 70)

        # **디버깅을 위한 프레임 출력**
        cv2.putText(frame, f"Angle: {predicted_angle} degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('RC Car Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # 약간의 딜레이

except KeyboardInterrupt:
    print("프로그램 종료 중...")
finally:
    try:
        servo.stop()
        dc_motor.stop()
    finally:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("정상적으로 종료되었습니다.")
