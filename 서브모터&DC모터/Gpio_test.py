import Jetson.GPIO as GPIO
import time

# 사용할 GPIO 핀 번호
BUTTON_PIN = 16

# GPIO 모드 설정 및 핀 입력 모드로 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# 버튼 상태 확인 함수
def check_button():
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            print("버튼이 눌렸습니다!")
        else:
            print("버튼이 눌리지 않았습니다.")
        time.sleep(0.5)

try:
    check_button()

except KeyboardInterrupt:
    GPIO.cleanup()  # 종료 시 GPIO 리셋
