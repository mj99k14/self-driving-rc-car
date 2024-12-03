import cv2
import os
import numpy as np

# 입력 및 출력 폴더 설정
input_folder = "C:/Users/USER/Desktop/p/45_degrees"
output_folder = "C:/Users/USER/Desktop/p/45_degrees_processed"
os.makedirs(output_folder, exist_ok=True)

def process_image(image):
    # 1. 이미지를 흑백으로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 흰색 선 강조 (이진화)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 3. 작은 점 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. 선의 연결성 강화
    binary_dilated = cv2.dilate(binary_cleaned, kernel, iterations=1)

    # 5. 오른쪽 배경 제거
    mask = np.zeros_like(binary_dilated)
    height, width = binary_dilated.shape
    points = np.array([[0, 0], [width // 2, 0], [width // 2, height], [0, height]])
    cv2.fillPoly(mask, [points], 255)
    processed = cv2.bitwise_and(binary_dilated, mask)

    # 6. 리사이즈 (256x256)
    resized = cv2.resize(processed, (256, 256))

    return resized


# 폴더의 모든 이미지 처리
for file_name in os.listdir(input_folder):
    if file_name.endswith((".jpg", ".png")):  # 이미지 파일만 처리
        input_path = os.path.join(input_folder, file_name)
        image = cv2.imread(input_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {input_path}")
            continue
        
        # 이미지 처리
        processed_image = process_image(image)
        
        # 처리된 이미지 저장
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, processed_image)

print("모든 이미지 처리가 완료되었습니다!")
