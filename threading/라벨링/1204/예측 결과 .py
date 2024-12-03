import os
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 불러오기
model = tf.keras.models.load_model("C:/Users/USER/Desktop/Project캡스톤/lane_detection_model.h5")

# 예측하려는 이미지 경로 설정 (input_folders 변수 정의)
input_folders = [
    "C:/Users/USER/Desktop/new/45_degrees",  # 45도
    "C:/Users/USER/Desktop/new/90_degrees",  # 90도
    "C:/Users/USER/Desktop/new/135_degrees"  # 135도
]

# 이미지를 불러와서 모델에 맞게 전처리하는 함수
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # 모델이 요구하는 크기로 이미지 크기 조정
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 정규화
    return img_array

# 예측 함수
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction

# 예측 결과를 저장할 CSV 파일 경로
output_file = "C:/Users/USER/Desktop/Project캡스톤/output_predictions.csv"

# 예측 결과를 저장할 리스트 초기화
prediction_data = []

# 이미지를 예측한 후 결과 리스트에 추가 (예: 이미지 경로와 예측 값)
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            prediction = predict_image(image_path)
            prediction_data.append([image_path, prediction])

# CSV 파일로 저장
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Prediction"])
    writer.writerows(prediction_data)

print(f"예측 결과를 {output_file}에 저장했습니다.")
