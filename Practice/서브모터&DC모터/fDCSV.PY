import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard

# Set the sudo password for shell commands
sudo_password = "1212"

# Function to run shell commands with sudo password
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# Check and install busybox if needed
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Run devmem commands for PWM configuration
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]
for command in commands:
    run_command(command)

# Set up GPIO for servo and DC motor
servo_pin = 33  # Servo control pin
dc_motor_pwm_pin = 32  # DC motor speed control
dc_motor_dir_pin1 = 29  # DC motor direction control pin 1
dc_motor_dir_pin2 = 31  # DC motor direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# Configure PWM on pins
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

# Servo control function with moderate response speed
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.02)
    servo.ChangeDutyCycle(0)

# DC motor control function
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)  # Adjusted for forward movement
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)  # Adjusted for backward movement
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(speed)

# Initial settings
current_angle = 90
set_servo_angle(current_angle)

try:
    print("Use arrow keys to control. Press '0' to reset servo angle to 90.")
    while True:
        # Reset servo to 90 degrees when '0' is pressed
        if keyboard.is_pressed('0'):
            current_angle = 90  # Immediately set angle to 90
            set_servo_angle(current_angle)
            print("Servo angle reset to 90 degrees.")
            time.sleep(0.1)

        # Control DC motor based on up/down arrow keys
        if keyboard.is_pressed('up'):
            set_dc_motor(50, "forward")  # Move forward
            print("DC motor moving forward...")
        elif keyboard.is_pressed('down'):
            set_dc_motor(50, "backward")  # Move backward
            print("DC motor moving backward...")
        else:
            set_dc_motor(0, "forward")  # Stop motor
            print("DC motor stopped.")

        # Move servo left/right in real-time if keys are held down
        if keyboard.is_pressed('left') and current_angle > 0:
            current_angle -= 1
            set_servo_angle(current_angle)
            print(f"Servo moving left: {current_angle} degrees.")
        elif keyboard.is_pressed('right') and current_angle < 180:
            current_angle += 1
            set_servo_angle(current_angle)
            print(f"Servo moving right: {current_angle} degrees.")

        time.sleep(0.05)

finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
