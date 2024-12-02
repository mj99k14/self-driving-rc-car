import os
import cv2
import numpy as np

# 입력 및 출력 폴더 경로
input_folder = "C:/Users/USER/Desktop/p/135_degrees"  # 실제 입력 폴더 경로
output_folder = "C:/Users/USER/Desktop/p/135_degrees_output"  # 출력 폴더 경로
os.makedirs(output_folder, exist_ok=True)

def detect_lane(image):
    height, width = image.shape[:2]

    # 1. ROI 설정 (차선이 포함될 가능성이 높은 영역만 남김)
    roi_vertices = np.array([[
        (0, height),                # 왼쪽 아래
        (int(width * 0.4), int(height * 0.6)),  # 중간 왼쪽
        (int(width * 0.6), int(height * 0.6)),  # 중간 오른쪽
        (width, height)             # 오른쪽 아래
    ]], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))  # ROI 영역만 흰색으로 설정
    roi_image = cv2.bitwise_and(image, mask)  # ROI 마스크 적용

    # 2. 흰색 차선 필터링 (HSV 색상 필터링)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)  # HSV 색상 공간으로 변환
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))  # 흰색 차선 필터
    filtered_image = cv2.bitwise_and(image, image, mask=white_mask)

    # 3. 엣지 검출
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 노이즈 제거를 위한 블러링
    edges = cv2.Canny(blurred, 50, 150)  # Canny 엣지 검출

    return edges  # 단순히 엣지만 반환

# 폴더 내 이미지 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # 이미지 읽기
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:  # 이미지 로딩 실패 시 스킵
            print(f"이미지를 열 수 없습니다: {image_path}")
            continue

        # 차선 검출
        edges = detect_lane(image)

        # 결과 저장
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, edges)

print(f"모든 이미지가 처리되어 출력 폴더({output_folder})에 저장되었습니다!")
