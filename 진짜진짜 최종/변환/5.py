import cv2
import os

# 입력 및 출력 폴더 경로
input_folder = r"C:/Users/USER/Desktop/csv"  # 원본 이미지 폴더
output_folder = r"C:/Users/USER/Desktop/augmented_frames"  # 증강 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 이미지 전처리 함수
def preprocess_image(image_path, output_path, angle):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    # ROI 설정 (상단 30% 제거)
    height = image.shape[0]
    roi = image[int(height * 0.3):, :]

    # 크기 조정 (200x66)
    resized = cv2.resize(roi, (200, 66))

    # 결과 저장 (파일명에 각도 포함)
    output_filename = f"{angle}_cropped_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(output_path, output_filename), resized)

# 폴더 내 모든 이미지 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        # 각도를 파일명에서 추출 (예: '20_image.jpg' -> 각도 20도)
        angle = filename.split('_')[0]  # 파일명 규칙에 따라 각도를 추출
        input_path = os.path.join(input_folder, filename)
        preprocess_image(input_path, output_folder, angle)

print("모든 이미지 전처리가 완료되었습니다.")
