import Jetson.GPIO as GPIO
import torch
import cv2
import numpy as np
import time
import subprocess

# PWM 활성화 명령어
subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
servo_pin = 33
dir_pin = 29
in2_pin = 31
pwm_pin = 32

GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(in2_pin, GPIO.OUT)
GPIO.setup(pwm_pin, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)  # 서보 PWM
servo.start(7.5)  # 초기 각도 (90도)

dc_motor = GPIO.PWM(pwm_pin, 1000)  # DC 모터 PWM
dc_motor.start(0)

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    print(f"Setting servo angle to {angle} degrees (Duty Cycle: {duty_cycle}%)")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.2)

# DC 모터 제어 함수
def control_dc_motor(direction, speed):
    if direction == "forward":
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(dir_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
    elif direction == "stop":
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)
    dc_motor.ChangeDutyCycle(speed)

# PilotNet 모델 정의
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
            torch.nn.Linear(10, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # 반환값 추가

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = PilotNet().to(device)
model_path = "/home/kim/Desktop/best_pilotnet_model.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    GPIO.cleanup()
    exit()

categories = [30, 60, 90, 120, 150]

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("카메라 초기화 실패.")
    GPIO.cleanup()
    exit()

# 메인 루프
try:
    print("RC 카 제어 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # 입력 데이터 전처리
        frame_resized = cv2.resize(frame, (200, 66))
        frame_normalized = frame_resized / 255.0
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        frame_tensor = torch.tensor(frame_transposed, dtype=torch.float32).unsqueeze(0).to(device)

        # 모델 예측
        outputs = model(frame_tensor)
        predicted_category = torch.argmax(outputs, dim=1).item()
        predicted_angle = categories[predicted_category]
        print(f"예측된 조향 각도: {predicted_angle}")

        # 서보 모터 및 DC 모터 제어
        set_servo_angle(predicted_angle)
        control_dc_motor("forward", 50)

        # 화면에 출력
        cv2.putText(frame, f"Angle: {predicted_angle} degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RC Car Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("프로그램 종료 중...")

finally:
    servo.stop()
    dc_motor.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("정상적으로 종료되었습니다.")
