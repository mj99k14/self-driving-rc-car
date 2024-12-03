import cv2
import numpy as np
import os

# 입력 폴더 경로
input_folder = "C:/Users/USER/Desktop/p/90_degrees"
output_folder = "C:/Users/USER/Desktop/p/90_degrees_filtered"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 처리 함수
def process_image(image_path, output_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    # 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 흰색 줄 검출
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 흰색 줄 두 개 영역 선택
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 흰색 줄 두 개 사이의 영역 마스크 생성
    mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 저장
    cv2.imwrite(output_path, result)

# 폴더 내 모든 이미지 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path)

print(f"모든 이미지가 {output_folder}에 저장되었습니다.")
