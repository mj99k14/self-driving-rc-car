import os  # os 모듈을 import합니다.
import cv2
import pandas as pd

# 입력 데이터 경로
csv_path = "C:/Users/USER/Desktop/training_data.csv"  # CSV 파일 경로
output_folder = "C:/Users/USER/Desktop/augmented_frames"  # 전처리된 이미지 저장 경로

# 디렉토리 생성 (이미 존재해도 오류 발생 안 함)
os.makedirs(output_folder, exist_ok=True)

# 데이터 로드
df = pd.read_csv(csv_path)

# 경로 정리: 역슬래시를 슬래시로 변환
df['frame_path'] = df['frame_path'].str.replace("\\", "/")

# 전처리 함수 정의
def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    # 상단 30% 자르기
    height = image.shape[0]
    roi = image[int(height * 0.3):, :]
    # 크기 조정
    resized = cv2.resize(roi, (200, 66))
    # 전처리된 이미지 저장
    cv2.imwrite(output_path, resized)

# 모든 이미지 전처리
for idx, row in df.iterrows():
    input_path = row['frame_path']
    output_path = os.path.join(output_folder, f"cropped_{os.path.basename(input_path)}")
    preprocess_image(input_path, output_path)

print("모든 이미지 전처리가 완료되었습니다.")
