import threading

#1. rc 자동차 제어 함수
def control_rc_car():

    """
    rc 카의 동작을 제어하는 함수
    키보드 입력을 받아 모터와 서브모터를 제어
    """

#2.open cv웹캠 촬영을 및 저장 함수
def capture_video():
    """
    웹캠에서 실시간 영상을 캡쳐하고 파일로 저장하는 함수
    """

#3.메인 함수
def main():
    """
    메인 실행 함수
    -rc 자동차 제어: 메인 스레드에서 실행
    -웹캠 촬영 및 저장: 별도 스레드에서 병렬 실행
    """

    #3.1 스레드 생성
    Video_thread = threading.Thread(target=capture_video)
    Video_thread.start()

    #3.2 RC 자동차 제어
    control_rc_car()

    #3.3 스레드 종료대기
    Video_thread.join()
    print("program has ended")

#4.프로그램 실행
if __name___ =="__main__":
    main()
