import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard
import threading
from datetime import datetime
import subprocess

# GPIO 핀에서 PWM 기능을 활성화하는 함수
def enable_pwm_on_pins():
    subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
    subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# 저장 폴더 생성 함수
def create_output_folder():
    base_dir = "./mj"  # 기본 폴더 경로
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 현재 시간 기반 폴더명 생성
    folder_path = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)  # 폴더 생성
    print(f"Created output folder: {folder_path}")
    return folder_path

# GPIO 핀 설정
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

# PWM을 사용할 핀을 초기화하기 전에 활성화
enable_pwm_on_pins()

# GPIO 설정
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
    if angle < 30:
        angle = 30
    elif angle > 150:
        angle = 150
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

# 카메라 처리 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self, output_folder):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.frame_count = 0
        self.output_folder = output_folder

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            cv2.imshow('Webcam Feed', frame)

            # 자동으로 사진 저장
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # 매 30 프레임마다 저장 (약 1초 간격)
                self.save_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def save_frame(self, frame):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        output_file = os.path.join(self.output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(output_file, frame)
        print(f"Saved frame: {output_file}")

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 모터 제어 쓰레드
class MotorHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.servo_angle = 90
        set_servo_angle(self.servo_angle)
        self.running = True

    def run(self):
        while self.running:
            if keyboard.is_pressed('0'):
                self.servo_angle = 90
                set_servo_angle(self.servo_angle)
                print("Servo angle reset to 90 degrees.")

            elif keyboard.is_pressed('left'):
                self.servo_angle -= 30
                if self.servo_angle < 30:
                    self.servo_angle = 30
                set_servo_angle(self.servo_angle)
                print(f"Left pressed. Servo angle: {self.servo_angle} degrees")

            elif keyboard.is_pressed('right'):
                self.servo_angle += 30
                if self.servo_angle > 150:
                    self.servo_angle = 150
                set_servo_angle(self.servo_angle)
                print(f"Right pressed. Servo angle: {self.servo_angle} degrees")

            if keyboard.is_pressed('up'):
                set_dc_motor(50, "forward")
                print("DC motor moving forward...")
            elif keyboard.is_pressed('down'):
                set_dc_motor(50, "backward")
                print("DC motor moving backward...")
            else:
                set_dc_motor(0, "forward")

            time.sleep(0.1)

    def stop(self):
        self.running = False

# 메인 함수
def main():
    # 자동 생성된 출력 폴더
    output_folder = create_output_folder()

    # 핸들러 초기화
    camera_handler = CameraHandler(output_folder)
    motor_handler = MotorHandler()

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
