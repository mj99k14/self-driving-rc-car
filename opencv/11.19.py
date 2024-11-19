import os
import Jetson.GPIO as GPIO
import time
import cv2
import threading
from pynput import keyboard

# --- 서보 모터 클래스 ---
class ServoMotor:
    def __init__(self, pin, frequency=50):
        self.pin = pin
        self.frequency = frequency
        self.angle = 90
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.frequency)
        self.pwm.start(0)

    def set_angle(self, angle):
        angle = max(0, min(180, angle))  # 0~180도로 제한
        duty_cycle = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
        self.pwm.ChangeDutyCycle(0)
        self.angle = angle

    def stop(self):
        self.pwm.stop()

# --- DC 모터 클래스 ---
class DCMotor:
    def __init__(self, pwm_pin, dir_pin1, dir_pin2, frequency=1000):
        self.pwm_pin = pwm_pin
        self.dir_pin1 = dir_pin1
        self.dir_pin2 = dir_pin2
        GPIO.setup(self.pwm_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin1, GPIO.OUT)
        GPIO.setup(self.dir_pin2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pwm_pin, frequency)
        self.pwm.start(0)

    def set_speed(self, speed, direction="forward"):
        speed = max(0, min(100, speed))  # 0~100%로 제한
        GPIO.output(self.dir_pin1, direction == "forward")
        GPIO.output(self.dir_pin2, direction == "backward")
        self.pwm.ChangeDutyCycle(speed)

    def stop(self):
        self.pwm.ChangeDutyCycle(0)

# --- 카메라 녹화 클래스 ---
class CameraRecorder:
    def __init__(self, output_dir="./mj", resolution=(1280, 720), fps=30):
        self.output_dir = output_dir
        self.resolution = resolution
        self.fps = fps
        self.recording = False
        self.out = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def start_recording(self, filename="output.avi"):
        filepath = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.out = cv2.VideoWriter(filepath, fourcc, self.fps, self.resolution)
        self.recording = True
        print(f"Recording started: {filepath}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.out:
                self.out.release()
            print("Recording stopped.")

    def show_feed(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imshow('Webcam Feed', frame)
            if self.recording and self.out:
                self.out.write(frame)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.recording and self.out:
            self.out.release()
        cv2.destroyAllWindows()

# --- 키보드 핸들러 ---
class KeyboardHandler:
    def __init__(self, servo, dc_motor, camera):
        self.servo = servo
        self.dc_motor = dc_motor
        self.camera = camera
        self.listener = keyboard.Listener(on_press=self.on_press)
        #쓰레드 시작
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'r':  # 녹화 시작/중지
                if self.camera.recording:
                    self.camera.stop_recording()
                else:
                    self.camera.start_recording()

            elif key.char == '0':  # 서보 모터 초기화
                self.servo.set_angle(90)

            elif key.char == 'q':  # 종료
                raise KeyboardInterrupt

        except AttributeError:  # 특수 키 처리
            if key == keyboard.Key.left:
                self.servo.set_angle(self.servo.angle - 5)
            elif key == keyboard.Key.right:
                self.servo.set_angle(self.servo.angle + 5)
            elif key == keyboard.Key.up:
                self.dc_motor.set_speed(50, "forward")
            elif key == keyboard.Key.down:
                self.dc_motor.set_speed(50, "backward")

    def stop(self):
        self.listener.stop()

# --- 메인 함수 ---
def main():
    GPIO.setmode(GPIO.BOARD)

    servo = ServoMotor(pin=33)
    dc_motor = DCMotor(pwm_pin=32, dir_pin1=29, dir_pin2=31)
    camera = CameraRecorder()
#키보드 입력 작업과 메인스레드 독립적으로 분리하기위해서
    keyboard_handler = KeyboardHandler(servo, dc_motor, camera) # 쓰레드 생성




#메인 스레드에서 종료 대기
    try:
        while True:
            camera.show_feed()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'로 종료
                break
    except KeyboardInterrupt:
        print("프로그램 종료 중...")
    finally:
        camera.release()
        servo.stop()
        dc_motor.stop()
        keyboard_handler.stop()
        GPIO.cleanup()
        print("정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
