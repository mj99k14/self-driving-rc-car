import os
import Jetson.GPIO as GPIO
import time
import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn

# GPIO 설정
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)  # 서보 모터 PWM 주파수
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC 모터 PWM 주파수
servo.start(0)
dc_motor_pwm.start(0)

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)

def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 모델 로드
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)  # 클래스 개수
model.load_state_dict(torch.load('/home/jetson/Desktop/model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_steering_angle(frame):
    input_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 입력 오류")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predicted_class = predict_steering_angle(frame_rgb)

        if predicted_class == 0:  # 좌회전
            set_servo_angle(45)
        elif predicted_class == 1:  # 직진
            set_servo_angle(90)
        elif predicted_class == 2:  # 우회전
            set_servo_angle(135)

        set_dc_motor(50, "forward")  # 계속 전진

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
