import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard  # 키보드 입력을 감지하기 위한 모듈

# Set the sudo password as a variable for easy updating
sudo_password = "1212"

# Function to run shell commands with the sudo password
def run_command(command):
    # Form the full command with password input
    full_command = f"echo {sudo_password} | sudo -S {command}"
    # Execute the command in the shell
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

# Set up GPIO pins for servo
servo_pin = 33  # PWM-capable pin for servo motor

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# Configure PWM on servo
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
servo.start(0)

# Function to set servo angle
def set_servo_angle(angle):
    # Calculate duty cycle based on angle (0 to 180 degrees)
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.01)  # 짧은 반응 시간으로 부드럽게 전환
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

# Initialize servo angle to 90 degrees
current_angle = 90
set_servo_angle(current_angle)

# Control the servo using keyboard input
try:
    print("Press 's' to reset to 90 degrees, arrow keys to adjust angle.")
    while True:
        # Reset to 90 degrees when 's' is pressed
        if keyboard.is_pressed('s'):
            if current_angle != 90:  # 현재 각도가 90도가 아닐 때만 실행
                current_angle = 90
                set_servo_angle(current_angle)
                print("Servo angle reset to 90 degrees.")
                time.sleep(0.1)  # 키 반복 실행 방지

        # 왼쪽 화살표 키가 눌려 있는 동안 부드럽게 이동
        elif keyboard.is_pressed('left'):
            while keyboard.is_pressed('left') and current_angle > 0:
                current_angle -= 1  # 부드럽게 1도씩 이동
                set_servo_angle(current_angle)
                print(f"Moving left: {current_angle} degrees.")
                time.sleep(0.01)  # 부드러운 이동을 위해 짧은 대기 시간

        # 오른쪽 화살표 키가 눌려 있는 동안 부드럽게 이동
        elif keyboard.is_pressed('right'):
            while keyboard.is_pressed('right') and current_angle < 180:
                current_angle += 1  # 부드럽게 1도씩 이동
                set_servo_angle(current_angle)
                print(f"Moving right: {current_angle} degrees.")
                time.sleep(0.01)  # 부드러운 이동을 위해 짧은 대기 시간

finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    GPIO.cleanup()
