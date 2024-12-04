import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 준비를 위한 경로 설정
base_dir = "C:/Users/USER/Desktop/new"  # 데이터를 저장할 기본 경로
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# 각도별 폴더 리스트 (45도, 90도, 135도)
angle_folders = [45, 90, 135]

# train과 validation 폴더 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# 각도별로 train과 validation 하위 폴더 생성
for angle in angle_folders:
    os.makedirs(os.path.join(train_dir, f"{angle}_degrees"), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, f"{angle}_degrees"), exist_ok=True)

print("폴더 구조가 성공적으로 생성되었습니다!")

# 모델 학습을 위한 준비: 이미지와 레이블 준비
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 데이터를 훈련과 검증으로 나누기 위한 비율 설정
)

# 데이터 로딩 및 분류
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # 레이블이 정수 형태로 되어 있음을 나타냄
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # 레이블이 정수 형태로 되어 있음을 나타냄
    subset='validation'
)

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3개의 클래스를 분류
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 학습이 완료된 후 모델 저장
model.save('lane_detection_model.h5')
print("모델이 성공적으로 저장되었습니다!")

