import cv2
import numpy as np
import os

# 입력 및 출력 폴더 경로 설정
input_folder = "C:/Users/USER/Desktop/p/135_degrees"
output_folder = "C:/Users/USER/Desktop/p/135_degrees_processed"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 흰색 필터를 적용할 범위 (HSV 기준)
lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([255, 50, 255], dtype=np.uint8)

# 이미지 처리 함수
def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    
    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 흰색 영역 마스크 생성
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 마스크를 적용하여 흰색 선만 남김
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 새 마스크 생성
    new_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 작은 노이즈 제거
            cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # 최종 영역 적용
    final_result = cv2.bitwise_and(image, image, mask=new_mask)
    
    # 결과 저장
    cv2.imwrite(output_path, final_result)
    print(f"처리 완료: {output_path}")

# 입력 폴더의 모든 이미지 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path)
