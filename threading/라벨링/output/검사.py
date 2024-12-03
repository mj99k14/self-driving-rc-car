import os
import cv2

input_folder = "C:/Users/USER/Desktop/p/45_degrees"  # 실제 입력 폴더 경로
output_folder = "C:/Users/USER/Desktop/p/45_degrees_output"  # 출력 폴더 경로

os.makedirs(output_folder, exist_ok=True)

def detect_lane(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# 입력 폴더 확인
print(f"입력 폴더: {input_folder}")
print(f"입력 파일 목록: {os.listdir(input_folder)}")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        edges = detect_lane(image)

        if edges is None or edges.sum() == 0:
            print(f"차선 검출 실패: {image_path}")
            continue

        output_path = os.path.join(output_folder, filename)
        print(f"저장 경로: {output_path}")
        cv2.imwrite(output_path, edges)

print(f"모든 이미지가 처리되어 {output_folder}에 저장되었습니다!")
