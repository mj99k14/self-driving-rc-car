import cv2
import numpy as np
import os

# ROI 설정 함수
def set_roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(image, mask)

# 차선 검출 함수
def process_lane(image, output_path, step_debug=False):
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if step_debug:
        cv2.imwrite(output_path + "_step1_gray.jpg", gray)

    # 2. 블러 처리 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if step_debug:
        cv2.imwrite(output_path + "_step2_blur.jpg", blur)

    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    if step_debug:
        cv2.imwrite(output_path + "_step3_edges.jpg", edges)

    # 4. ROI 설정
    height, width = edges.shape
    vertices = np.array([[
        (0, height),
        (width // 2 - 50, height // 2),
        (width // 2 + 50, height // 2),
        (width, height)
    ]], dtype=np.int32)
    roi = set_roi(edges, vertices)
    if step_debug:
        cv2.imwrite(output_path + "_step4_roi.jpg", roi)

    # 5. 흰색 선만 남기기 위한 색상 필터
    lower_white = 200
    upper_white = 255
    _, binary = cv2.threshold(gray, lower_white, upper_white, cv2.THRESH_BINARY)
    filtered = cv2.bitwise_and(roi, binary)
    if step_debug:
        cv2.imwrite(output_path + "_step5_filtered.jpg", filtered)

    return filtered

# 입력 폴더 경로 리스트
input_folders = [
    "C:/Users/USER/Desktop/p/45_degrees",
    "C:/Users/USER/Desktop/p/90_degrees",
    "C:/Users/USER/Desktop/p/135_degrees"
]

# 출력 폴더 경로 리스트
output_folders = [
    "C:/Users/USER/Desktop/p/45_degrees_filtered",
    "C:/Users/USER/Desktop/p/90_degrees_filtered",
    "C:/Users/USER/Desktop/p/135_degrees_filtered"
]

# 각 폴더 처리
for input_folder, output_folder in zip(input_folders, output_folders):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일 확장자 필터
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 이미지 읽기
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 불러올 수 없습니다: {input_path}")
                continue

            # 차선 검출
            filtered_lane = process_lane(image, output_path.replace(".jpg", ""), step_debug=True)

            # 결과 저장
            cv2.imwrite(output_path, filtered_lane)
            print(f"처리 완료: {output_path}")

print("모든 이미지 처리가 완료되었습니다.")
