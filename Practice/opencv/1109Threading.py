import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard
import threading

# 폴더 생성 (출력 파일 저장 경로)
output_dir = "./mj"
output_file = os.path.join(output_dir, "output.avi")

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

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = None
recording = False

# 전역 변수
angle = 90
stop_threads = False

# 카메라 캡처 및 녹화 쓰레드 함수
def camera_capture():
    global out, recording, stop_threads
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 화면에 프레임 출력
        cv2.imshow('Webcam Feed', frame)

        # 녹화 중이면 프레임 저장
        if recording and out is not None:
            out.write(frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    cap.release()
    if recording and out is not None:
        out.release()
    cv2.destroyAllWindows()

# 키보드 입력 및 모터 제어 쓰레드 함수
def motor_control():
    global angle, recording, out, stop_threads
    while not stop_threads:
        # 녹화 시작/중지
        if keyboard.is_pressed('r'):
            if recording:
                recording = False
                if out is not None:
                    out.release()
                print("Recording stopped.")
            else:
                print(f"Recording to: {output_file}")
                out = cv2.VideoWriter(output_file, fourcc, 30.0, (1280, 720))
                recording = True
                print("Recording started.")
            time.sleep(0.2)  # 키 입력 중복 방지

        # 서보 모터 제어
        if keyboard.is_pressed('0'):
            angle = 90
            set_servo_angle(angle)
            print("Servo angle reset to 90 degrees.")

        if keyboard.is_pressed('left'):
            angle -= 5
            if angle < 0:
                angle = 0
            set_servo_angle(angle)
            print(f"Left pressed. Servo angle: {angle} degrees")

        elif keyboard.is_pressed('right'):
            angle += 5
            if angle > 180:
                angle = 180
            set_servo_angle(angle)
            print(f"Right pressed. Servo angle: {angle} degrees")

        # DC 모터 제어 - 전진/후진
        if keyboard.is_pressed('up'):
            set_dc_motor(50, "forward")
            print("DC motor moving forward...")
        elif keyboard.is_pressed('down'):
            set_dc_motor(50, "backward")
            print("DC motor moving backward...")
        else:
            set_dc_motor(0, "forward")

        time.sleep(0.1)  # CPU 사용률을 낮추기 위해 대기 시간 추가

# 쓰레드 생성 및 시작
camera_thread = threading.Thread(target=camera_capture)
motor_thread = threading.Thread(target=motor_control)

camera_thread.start()
motor_thread.start()

try:
    # 메인 스레드가 종료되지 않도록 대기
    camera_thread.join()
    motor_thread.join()
finally:
    # 종료 시 GPIO 정리
    stop_threads = True
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
