import os
import shutil
from sklearn.model_selection import train_test_split

# 각도별 폴더 경로
base_path = "C:/Users/USER/Desktop"
folders = ["30", "60", "90", "120", "150"]

# 데이터를 나눌 비율
train_ratio = 0.7  # 70% 학습용
val_ratio = 0.15   # 15% 검증용
test_ratio = 0.15  # 15% 테스트용

# 결과 폴더 생성
output_dirs = ["train", "val", "test"]
for output_dir in output_dirs:
    for folder in folders:
        os.makedirs(os.path.join(base_path, output_dir, folder), exist_ok=True)

# 각도별로 데이터 나누기
for folder in folders:
    folder_path = os.path.join(base_path, folder)  # 각도 폴더 경로
    if not os.path.exists(folder_path):
        print(f"폴더가 없습니다: {folder_path}")
        continue

    # 이미지 파일 목록 가져오기
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"{folder_path} 폴더에 이미지가 없습니다.")
        continue

    # 데이터 나누기
    train_images, temp_images = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # 파일 이동
    for img in train_images:
        src = os.path.join(folder_path, img)
        dest = os.path.join(base_path, "train", folder, img)
        shutil.move(src, dest)

    for img in val_images:
        src = os.path.join(folder_path, img)
        dest = os.path.join(base_path, "val", folder, img)
        shutil.move(src, dest)

    for img in test_images:
        src = os.path.join(folder_path, img)
        dest = os.path.join(base_path, "test", folder, img)
        shutil.move(src, dest)

    print(f"{folder}: Train {len(train_images)}, Val {len(val_images)}, Test {len(test_images)}")
