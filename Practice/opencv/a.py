import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard
import cv2

# sudo 비밀번호를 변수로 설정
sudo_password = " "

# sudo 비밀번호로 셸 명령어를 실행하는 함수
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# busybox가 설치되어 있는지 확인하고, 없으면 설치
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# devmem 명령어 정의
# devmem은 리눅스 환경에서 메모리 맵에 직접 접근하기 위해 사용하는 명령어
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]
# 각 devmem 명령어 실행
for command in commands:
    run_command(command)

# GPIO 핀 설정
servo_pin = 33  # 서보 모터 PWM핀
dc_motor_pwm_pin = 32  # DC 모터의 속도를 조절 PWM핀
dc_motor_dir_pin1 = 29  # GPIO 5, DC 모터의 방향을 제어 + (OUT1 + && IN1 + )
dc_motor_dir_pin2 = 31  # GPIO 6, DC 모터의 방향을 제어 - (OUT2 - && IN2 - )

GPIO.setmode(GPIO.BOARD) # Jetson Nano 핀 번호 사용하는 것을 명시
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 초당 50(50Hz)번의 PWM 신호가 반복(20ms마다)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 초당 1000(1000Hz)번의 PWM 신호가 반복
servo.start(0)
dc_motor_pwm.start(0)

# 서보 각도를 설정하는 함수
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle) # 회전 각도를 조절
    time.sleep(0.02) # 서보가 위치에 도달할 시간 허용
    servo.ChangeDutyCycle(0) # 진동을 피하기 위해 신호를 꺼줌

# DC motor 방향과 속도 조절
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)  # + 가 HIGH면 정방향으로 회전
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)  # - 가 HIGH면 역방향으로 회전
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 영상 촬영
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recoding = None

# 서보 모터의 각도 초기화 -> 실제 모터도 90도로 세팅해야 함
angle = 90
set_servo_angle(angle)

try:
    print("Use arrow keys to control. Press '0' to reset servo angle to 90.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('frame', frame)
        if keyboard.is_pressed('r'):
            if recoding:
                recoding = False
                out.release()
                print("Recoding stopped")
            else:
                out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recoding = True
                print("Recoding stated")
        
        if recoding:
            out.write(frame)
        
        set_dc_motor(30, "forward")
        # 0 -> 90도로 설정
        if keyboard.is_pressed('0'):
            angle = 90  # Immediately set angle to 90
            set_servo_angle(angle)
            print("Servo angle reset to 90 degrees.")
            time.sleep(0.1)

        # DC모터 제어
        # up -> 속도 50, 전진
        if keyboard.is_pressed('up'):
            set_dc_motor(50, "forward")  # set_dc_motor(속도, 방향("forward" or "backward"))
            print("DC motor moving forward...")
        # down -> 속도 50, 후진
        elif keyboard.is_pressed('down'):
            set_dc_motor(50, "backward")  # set_dc_motor(속도, 방향("forward" or "backward"))
            print("DC motor moving backward...")
            
        else:
            set_dc_motor(0, "forward")  # Stop motor
            print("DC motor stopped.")

        # 서보모터 제어
        # 각도는 0 < angle <180
        if 0 < angle < 180:
            # left -> 현제 각도에서 -1도한 각도로 설정
            if keyboard.is_pressed('left'):
                angle -= 1
                set_servo_angle(angle)
            # right -> 현제 각도에서 +1도한 각도로 설정
            elif keyboard.is_pressed('right'):
                angle += 1
                set_servo_angle(angle)
            print(f"현제 각도 : {angle}도")


        time.sleep(0.05) # CPU 과부하 방지를 위한 짧은 지연, 인간의 입력 속도에 맞춤

finally:
    # PWM 정지 및 GPIO 정리
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    cap.release()
    if recoding:
        out.release()
    cv2.destroyAllWindows
