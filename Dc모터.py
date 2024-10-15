import Jetson.GPIO as GPIO
import time
import subprocess

# Set the sudo password as a variable for easy updating
sudo_password = "your_password_here"

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

# Set up GPIO pins for servo and DC motor control
servo_pin = 33  # PWM-capable pin for servo motor
dc_motor_pwm_pin = 32  # PWM-capable pin for DC motor speed
dc_motor_dir_pin1 = 29  # Direction control pin 1
dc_motor_dir_pin2 = 31  # Direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# Configure PWM on servo and DC motor pins
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

# Function to set servo angle
def set_servo_angle(angle):
    # Calculate duty cycle based on angle (0 to 180 degrees)
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Allow time for the servo to reach position
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction):
    # Set direction: 'forward' or 'backward'
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    
    # Control speed with PWM (0 to 100%)
    dc_motor_pwm.ChangeDutyCycle(speed)

# Example usage: Rotate servo and control DC motor
try:
    # Rotate servo from 0 to 180 degrees and back to 0
    for angle in range(0, 181, 10):  # Move from 0 to 180 in 10-degree steps
        set_servo_angle(angle)
    for angle in range(180, -1, -10):  # Move from 180 back to 0 in 10-degree steps
        set_servo_angle(angle)
    
    # Run DC motor at 50% speed forward, then backward
    set_dc_motor(50, "forward")
    time.sleep(2)
    set_dc_motor(50, "backward")
    time.sleep(2)
finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
