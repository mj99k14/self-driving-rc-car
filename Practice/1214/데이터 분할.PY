import os
from sklearn.model_selection import train_test_split

base_path = "C:/Users/USER/Desktop"
folders = ["30", "60", "90", "120", "150"]

# 데이터 분할
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"경고: {folder} 폴더에 이미지가 없습니다.")
        continue

    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.33, random_state=42)

    # 파일 이동
    for img in train_images:
        os.rename(os.path.join(folder_path, img), os.path.join(base_path, "train", folder, img))
    for img in val_images:
        os.rename(os.path.join(folder_path, img), os.path.join(base_path, "val", folder, img))
    for img in test_images:
        os.rename(os.path.join(folder_path, img), os.path.join(base_path, "test", folder, img))

print("데이터 분할 완료!")
