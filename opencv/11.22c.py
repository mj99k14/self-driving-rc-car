import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard
import threading
from datetime import datetime

# 폴더 생성 (출력 파일 저장 경로)
output_dir = "./mj"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# GPIO 핀 설정
servo_pin = 33
dc_motor_pwm_pin = 32  # DC 모터 속도 제어 핀
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

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

# 카메라 처리 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.recording = False
        self.out = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            cv2.imshow('Webcam Feed', frame)

            if self.recording and self.out is not None:
                self.out.write(frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def start_recording(self):
        # datetime을 이용한 파일 이름 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"output_{timestamp}.avi")
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'H264'), 30.0, (1280, 720))
        self.recording = True
        print(f"Recording started: {output_file}")

    def stop_recording(self):
        if self.recording and self.out is not None:
            self.out.release()
        self.recording = False
        print("Recording stopped.")

    def cleanup(self):
        if self.recording and self.out is not None:
            self.out.release()
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
            # 서보 모터 제어
            if keyboard.is_pressed('0'):
                self.servo_angle = 90
                set_servo_angle(self.servo_angle)
                print("Servo angle reset to 90 degrees.")

            elif keyboard.is_pressed('left'):
                self.servo_angle -= 5
                if self.servo_angle < 0:
                    self.servo_angle = 0
                set_servo_angle(self.servo_angle)
                print(f"Left pressed. Servo angle: {self.servo_angle} degrees")

            elif keyboard.is_pressed('right'):
                self.servo_angle += 5
                if self.servo_angle > 180:
                    self.servo_angle = 180
                set_servo_angle(self.servo_angle)
                print(f"Right pressed. Servo angle: {self.servo_angle} degrees")

            # DC 모터 제어
            if keyboard.is_pressed('up'):
                set_dc_motor(50, "forward")
                print("DC motor moving forward...")
            elif keyboard.is_pressed('down'):
                set_dc_motor(50, "backward")
                print("DC motor moving backward...")
            else:
                set_dc_motor(0, "forward")

            time.sleep(0.1)  # CPU 점유율 방지

    def stop(self):
        self.running = False

# 메인 함수
def main():
    camera_handler = CameraHandler()
    motor_handler = MotorHandler()

    try:
        camera_handler.start()
        motor_handler.start()

        print("Press 'r' to start/stop recording. Press 'q' to exit.")
        while camera_handler.running:
            if keyboard.is_pressed('r'):
                if camera_handler.recording:
                    camera_handler.stop_recording()
                else:
                    camera_handler.start_recording()
                time.sleep(0.2)  # 키 입력 중복 방지

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
