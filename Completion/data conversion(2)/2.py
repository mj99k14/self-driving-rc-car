
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# CSV 파일 경로
csv_path = "C:/Users/USER/Desktop/training_data.csv"

# CSV 파일 존재 여부 확인
if not os.path.exists(csv_path):
    print(f"CSV 파일 경로를 다시 확인하세요: {csv_path}")
    exit()

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # 조향각 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(df['steering_angle'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 데이터 초기화
    data = df.values.tolist()
    current_index = 0

    def load_image_and_angle(index):
        "이미지와 조향각을 로드"
        frame_path, angle = data[index]
        image = cv2.imread(frame_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {frame_path}")
            return None, None
        return image, float(angle)

    # 윈도우 설정
    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)

    while True:
        if current_index < 0 or current_index >= len(data):
            print("더 이상 데이터가 없습니다.")
            break

        image, angle = load_image_and_angle(current_index)
        if image is None:
            current_index += 1  # 이미지 오류 시 다음 데이터로 이동
            continue

        # 이미지와 조향각 표시
        display_image = image.copy()
        cv2.putText(display_image, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image Viewer", display_image)

        # 키 입력 처리
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # 종료
            print("프로그램을 종료합니다.")
            break
        elif key == ord('a'):  # 이전 데이터
            current_index = max(0, current_index - 1)
        elif key == ord('d'):  # 다음 데이터
            current_index = min(len(data) - 1, current_index + 1)

    cv2.destroyAllWindows()

except pd.errors.EmptyDataError:
    print(f"CSV 파일이 비어 있습니다: {csv_path}")
except Exception as e:
    print(f"예기치 않은 오류가 발생했습니다: {e}")



