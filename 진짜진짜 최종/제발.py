import torch
import cv2
import numpy as np
import time
import Jetson.GPIO as GPIO

# 모델 경로
model_path = "/home/kim/Desktop/best_pilotnet_model.pth"

# **1. GPIO 설정**
GPIO.setmode(GPIO.BOARD)

# 서보 모터 설정
servo_pin = 32  # 서보 모터 PWM 핀
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50Hz PWM
servo.start(7.5)  # 초기값 (90도)

# DC 모터 설정
dir_pin = 29  # IN1
in2_pin = 31  # IN2
pwm_pin = 33  # ENA (속도 제어 핀)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)
GPIO.setup(pwm_pin, GPIO.OUT)
dc_motor = GPIO.PWM(pwm_pin, 1000)  # 1kHz PWM
dc_motor.start(70)  # 항상 70% 속도로 작동

# **2. 서보 모터 각도 설정 함수**
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    print(f"Setting servo angle to {angle}, Duty Cycle: {duty_cycle:.2f}")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)
    servo.ChangeDutyCycle(0)

# **3. DC 모터 제어 함수 (전진, 후진, 정지)**
def control_dc_motor(direction='forward'):
    if direction == 'forward':
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
        dc_motor.ChangeDutyCycle(70)  # 전진 속도 70%
        print("Motor running forward.")
    elif direction == 'reverse':
        GPIO.output(dir_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
        dc_motor.ChangeDutyCycle(70)  # 후진 속도 70%
        print("Motor running reverse.")
    elif direction == 'stop':
        dc_motor.ChangeDutyCycle(0)  # 정지
        print("Motor stopped.")
    else:
        print("Invalid direction input. Please use 'forward', 'reverse', or 'stop'.")

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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# 모델 로드
model = PilotNet()  # 모델 초기화
model.load_state_dict(torch.load(model_path))  # 모델 불러오기
model.eval()  # 추론 모드로 전환

# **5. 카메라 입력 및 예측**
cap = cv2.VideoCapture(0)  # 카메라 캡처
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리: 리사이즈, 정규화, 텐서 변환
    frame_resized = cv2.resize(frame, (224, 224))  # 모델 입력 크기에 맞게 리사이즈
    frame_normalized = frame_resized / 255.0  # 0~1 범위로 정규화
    frame_tensor = torch.tensor(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 224, 224) 형식으로 변환

    # 예측
    with torch.no_grad():
        output = model(frame_tensor)  # 모델을 사용하여 예측
    print("Prediction output:", output)  # 예측 결과 출력

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
