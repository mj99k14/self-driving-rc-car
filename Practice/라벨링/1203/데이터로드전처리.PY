import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: 데이터 로드 및 전처리 함수
def load_and_preprocess_data(base_dir, img_size):
    images = []
    labels = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):  # 폴더인지 확인
            # 폴더 이름에서 각도 추출 ("45_degrees" -> 45)
            angle = int(folder_name.split('_')[0])
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)  # 이미지 크기 변경
                    img = img / 255.0  # 정규화
                    images.append(img)
                    labels.append(angle)
    return np.array(images), np.array(labels)

# Step 2: 경로 및 설정
base_dir = r"C:\Users\USER\Desktop\new"  # 데이터가 저장된 경로
IMG_SIZE = (64, 64)  # 모델 입력 크기 설정

# 데이터 로드
images, labels = load_and_preprocess_data(base_dir, IMG_SIZE)
print(f"이미지 데이터 크기: {images.shape}")
print(f"라벨 데이터 크기: {labels.shape}")

# Step 3: 데이터셋 나누기
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f"학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_val.shape}")

# Step 4: CNN 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # 각도를 출력하는 노드
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])  # MAE: 예측 각도와 실제 각도의 평균 절대 오차

print(model.summary())

# Step 5: 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Step 6: 모델 저장
model.save('angle_prediction_model.h5')
print("모델이 성공적으로 저장되었습니다!")
