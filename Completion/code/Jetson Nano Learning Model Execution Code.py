import Jetson.GPIO as GPIO
import torch
import cv2
import numpy as np
import time
import threading
import queue
import subprocess

# PWM 활성화 명령어
subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# 큐 생성 (최대 크기 1: 최신 데이터만 유지)
angle_queue = queue.Queue(maxsize=1)
exit_signal = queue.Queue()  # 종료 신호를 위한 큐

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

servo = GPIO.PWM(servo_pin, 50)  # 서보 PWM 50Hz
servo.start(7.5)

dc_motor = GPIO.PWM(pwm_pin, 1000)  # DC 모터 PWM 1kHz
dc_motor.start(0)

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 180.0) * 10
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 제어 함수
def control_dc_motor(direction, speed):
    if direction == "forward":
        GPIO.output(dir_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
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
        return x

# 데이터 전처리 함수
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (200, 66), interpolation=cv2.INTER_LINEAR)
    frame_normalized = frame_resized / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    return torch.tensor(frame_transposed, dtype=torch.float32).unsqueeze(0).to(device), frame_resized

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)
model.load_state_dict(torch.load("/home/kim/Desktop/best_pilotnet_model.pth", map_location=device))
model.eval()

categories = [30, 60, 90, 120, 150]

# 쓰레드 1: DL 예측
def dl_inference():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    with torch.no_grad():  # 그래프 생성 방지
        while exit_signal.empty():
            ret, frame = cap.read()
            if not ret:
                continue

            # 예측 수행
            frame_tensor, frame_resized = preprocess_frame(frame)
            outputs = model(frame_tensor)
            predicted_category = torch.argmax(outputs, dim=1).item()
            predicted_angle = categories[predicted_category]

            # 큐에 최신 데이터 넣기
            if angle_queue.full():
                angle_queue.get()
            angle_queue.put(predicted_angle)

            # 화면 출력
            cv2.putText(frame_resized, f"Angle: {predicted_angle}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Frame", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_signal.put(True)  # 종료 신호
                break

    cap.release()
    cv2.destroyAllWindows()

# 쓰레드 2: 서보 모터 제어
def servo_control():
    while exit_signal.empty():
        if not angle_queue.empty():
            angle = angle_queue.get()
            set_servo_angle(angle)
            control_dc_motor("forward", 70)
        time.sleep(0.05)  # 제어 주기 단축

# 메인 실행
try:
    thread1 = threading.Thread(target=dl_inference)
    thread2 = threading.Thread(target=servo_control)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

except KeyboardInterrupt:
    print("프로그램 종료 중...")

finally:
    try:
        servo.stop()
        dc_motor.stop()
    except:
        pass
    GPIO.cleanup()
    print("정상적으로 종료되었습니다.")
