import time

# 가상 GPIO 핀 상태 (HIGH: 1, LOW: 0)
gpio_pin_state = {"PIN_18": 0}

# 가상으로 핀 출력 상태 설정하기
def set_gpio_pin(pin, state):
    gpio_pin_state[pin] = state
    if state == 1:
        print(f"{pin} 상태: HIGH")
    else:
        print(f"{pin} 상태: LOW")

# 가상으로 핀 상태 읽기
def get_gpio_pin_state(pin):
    return gpio_pin_state[pin]

# 핀 상태 변경 테스트 함수
def gpio_simulation_test():
    while True:
        user_input = input("핀 상태를 입력하세요 (high/low): ").strip().lower()
        if user_input == "high":
            set_gpio_pin("PIN_18", 1)
        elif user_input == "low":
            set_gpio_pin("PIN_18", 0)
        else:
            print("잘못된 입력입니다. 'high' 또는 'low'를 입력하세요.")

try:
    gpio_simulation_test()

except KeyboardInterrupt:
    print("테스트 종료")
