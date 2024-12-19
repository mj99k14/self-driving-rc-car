import os
from PIL import Image
import random

# 언더샘플링 함수
def undersample_data(folder_path, target_count):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    # 초과하는 이미지 삭제
    for img_name in image_files[target_count:]:
        os.remove(os.path.join(folder_path, img_name))

# 클래스별 데이터 언더샘플링
base_path = "C:/Users/USER/Desktop"  # 바탕화면 경로
class_folders = ["30", "60", "90", "120", "150"]  # 클래스 폴더 이름
target_counts = {"30": 600, "60": 600, "90": 600, "120": 600, "150": 600}  # 목표 수량

for class_folder in class_folders:
    folder_path = os.path.join(base_path, class_folder)
    if not os.path.exists(folder_path):
        print(f"폴더가 존재하지 않습니다: {folder_path}")
    else:
        print(f"언더샘플링 실행 중: {folder_path}")
        undersample_data(folder_path, target_counts[class_folder])

print("언더샘플링 완료!")
