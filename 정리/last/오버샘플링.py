import os  # os 모듈 추가
import random
import shutil

# 데이터 경로
base_path = "C:/Users/USER/Desktop"
folders = ["train/90", "val/90", "test/90"]

# 언더샘플링 함수
def undersample(folder_path, target_count):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)

    if current_count <= target_count:
        print(f"{folder_path}: 데이터가 이미 적습니다 ({current_count}/{target_count})")
        return

    # 랜덤 선택 후 삭제
    images_to_delete = random.sample(images, current_count - target_count)
    for img in images_to_delete:
        os.remove(os.path.join(folder_path, img))

    print(f"{folder_path}: 언더샘플링 완료 ({len(os.listdir(folder_path))}/{target_count})")

# 언더샘플링 수행
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    undersample(folder_path, target_count=600)  # 원하는 데이터 수
