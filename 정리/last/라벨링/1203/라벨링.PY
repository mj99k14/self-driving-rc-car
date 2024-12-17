import os
import cv2
import numpy as np

# 데이터 경로 설정
base_dir = "C:/Users/USER/Desktop/new"  # 경로를 올바르게 지정
IMG_SIZE = (64, 64)  # 이미지 크기 설정 (모델 입력 크기)

# 데이터와 라벨 저장 변수
images = []
labels = []

# 각 폴더 이름과 이미지를 읽어 라벨과 연결
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):  # 폴더인지 확인
        # 폴더 이름에서 각도 값을 추출
        angle = int(folder_name.split('_')[0])  # "45_degrees" -> 45
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)  # 크기 변경
                img = img / 255.0  # 픽셀 값 정규화 (0~1 범위)
                images.append(img)
                labels.append(angle)

# NumPy 배열로 변환
images = np.array(images)
labels = np.array(labels)

print(f"이미지 데이터 크기: {images.shape}")
print(f"라벨 데이터 크기: {labels.shape}")
