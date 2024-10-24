import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard

# Set the sudo password as a variable for easy updating
sudo_password = "1212"

# Function to run shell commands with the sudo password
def run_command(command):
    try:
        full_command = f"echo {sudo_password} | sudo -S {command}"
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")

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
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Allow time for the servo to reach position
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

# Function to stop the servo motor (reset the angle)
def stop_servo():
    servo.ChangeDutyCycle(0)  # Stop PWM signal to servo

# Main loop to control the servo motor based on keyboard input
try:
    print("Press the left (←) or right (→) arrow keys to move, and space to stop.")
    while True:
        if keyboard.is_pressed('left'):
            print("Turning left")
            set_servo_angle(45)  # Set angle for turning left
        elif keyboard.is_pressed('right'):
            print("Turning right")
            set_servo_angle(135)  # Set angle for turning right
        elif keyboard.is_pressed('space'):
            print("Stopping the motor")
            stop_servo()  # Stop the motor
        time.sleep(0.1)  # Small delay to avoid CPU overload
finally:
    print("Cleaning up...")
    # Stop all PWM and clean up GPIO
    servo.stop()
    GPIO.cleanup()