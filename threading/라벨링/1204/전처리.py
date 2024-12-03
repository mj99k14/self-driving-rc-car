import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 이미지가 저장된 폴더 경로
input_folders = [
    "C:/Users/USER/Desktop/new/45_degrees_filtered",  # 45도
    "C:/Users/USER/Desktop/new/90_degrees_filtered",  # 90도
    "C:/Users/USER/Desktop/new/135_degrees_filtered"  # 135도
]

# 라벨 맵
labels = {
    "45_degrees_filtered": 0,
    "90_degrees_filtered": 1,
    "135_degrees_filtered": 2
}

# 이미지와 라벨을 저장할 리스트
images = []
labels_list = []

# 폴더 순회하여 이미지 로드
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # 이미지를 (224, 224) 크기로 리사이즈
            image = cv2.resize(image, (224, 224))

            # 이미지를 정규화 (0~1 사이로)
            image = image / 255.0

            # 이미지와 레이블 저장
            images.append(image)
            labels_list.append(labels[folder.split('/')[-1]])  # 폴더 이름을 레이블로 사용

# 이미지를 numpy 배열로 변환
images = np.array(images)
labels_list = np.array(labels_list)

# 데이터 분할 (훈련/검증 데이터)
X_train, X_val, y_train, y_val = train_test_split(images, labels_list, test_size=0.2, random_state=42)

# 데이터 증강 설정 (선택적)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# CNN 모델 정의
model = Sequential()

# 1번 Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2번 Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3번 Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 과적합 방지

# 출력층 (3개의 클래스: 45도, 90도, 135도)
model.add(Dense(3, activation='softmax'))  # 3개의 클래스로 분류

# 모델 컴파일
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 훈련
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)
# 훈련 완료 후 모델 저장
model.save('my_model.keras')  # .keras 형식으로 모델 저장

# 모델 저장 경로 출력
print("모델이 'my_model.keras' 파일로 저장되었습니다.")
# 모델 평가
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# 모델 저장
model.save('lane_detection_model.h5')
