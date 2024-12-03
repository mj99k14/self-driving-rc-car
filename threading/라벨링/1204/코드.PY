import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard
import threading
from datetime import datetime
import subprocess
import numpy as np

# GPIO 핀에서 PWM 기능을 활성화하는 함수
def enable_pwm_on_pins():
    subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
    subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# 폴더 생성 (출력 파일 저장 경로)
base_output_dir = "./mj"
angle_folders = {
    45: os.path.join(base_output_dir, "45_degrees"),
    90: os.path.join(base_output_dir, "90_degrees"),
    135: os.path.join(base_output_dir, "135_degrees"),
}
for folder in angle_folders.values():
    os.makedirs(folder, exist_ok=True)

# GPIO 설정
servo_pin = 33
dc_motor_pwm_pin = 32  # DC 모터 속도 제어 핀
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

# PWM을 사용할 핀을 초기화하기 전에 활성화
enable_pwm_on_pins()

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
servo.start(0)
dc_motor_pwm.start(0)

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)

# DC 모터 방향과 속도 조절 함수
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 카메라 처리 및 차선 추적을 위한 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            with self.frame_lock:
                self.latest_frame = frame.copy()

            # 차선 추적
            self.process_frame(frame)

            # 화면 출력
            cv2.imshow('Webcam Feed', frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def process_frame(self, frame):
        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 차선 추적을 위한 이진화
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 라인 감지
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 100, minLineLength=50, maxLineGap=200)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 초록색으로 차선 표시

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 모터 제어 쓰레드
class MotorHandler(threading.Thread):
    def __init__(self, camera_handler):
        super().__init__()
        self.servo_angle = 90
        self.camera_handler = camera_handler
        self.running = True
        set_servo_angle(self.servo_angle)

    def run(self):
        while self.running:
            frame = self.camera_handler.get_frame()

            if frame is not None:
                # 차선 추적이 잘 되면, 서보 모터 제어
                self.process_lane_detection(frame)

            time.sleep(0.1)

    def process_lane_detection(self, frame):
        # 차선 추적 (간단한 예시: 차선이 왼쪽에 있으면 왼쪽으로, 오른쪽에 있으면 오른쪽으로 조향)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 100, minLineLength=50, maxLineGap=200)

        if lines is not None:
            # 왼쪽 라인 감지
            left_line = lines[0]  # 예시: 첫 번째 라인을 왼쪽 차선으로 설정
            x1, y1, x2, y2 = left_line[0]
            if x1 < 640:  # 좌측에 차선이 있으면 왼쪽으로 회전
                self.servo_angle -= 5
                set_servo_angle(self.servo_angle)

            # 오른쪽 라인 감지
            right_line = lines[1]  # 예시: 두 번째 라인을 오른쪽 차선으로 설정
            x1, y1, x2, y2 = right_line[0]
            if x2 > 640:  # 우측에 차선이 있으면 오른쪽으로 회전
                self.servo_angle += 5
                set_servo_angle(self.servo_angle)

            print(f"Servo angle: {self.servo_angle}")

    def stop(self):
        self.running = False

# 메인 함수
def main():
    camera_handler = CameraHandler()
    motor_handler = MotorHandler(camera_handler)

    try:
        camera_handler.start()
        motor_handler.start()

        print("Press 'q' to exit.")
        while camera_handler.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("프로그램 종료 중...")

    finally:
        camera_handler.running = False
        motor_handler.stop()
        camera_handler.join()
        motor_handler.join()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        print("정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
