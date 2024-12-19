from torchvision import datasets, transforms
from collections import Counter

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Train 데이터셋 다시 로드
train_dataset = datasets.ImageFolder(root=r"C:\Users\USER\Desktop\train", transform=transform)

# 클래스 분포 출력
train_class_counts = Counter(train_dataset.targets)
print("Train 클래스 분포 (증강 후):")
for class_idx, count in train_class_counts.items():
    print(f"클래스 {train_dataset.classes[class_idx]}: {count}개")

import os
import shutil

# 증강된 데이터 폴더와 기존 데이터 폴더 경로 설정
augmented_dir = r"C:\Users\USER\Desktop\train\30_augmented"
class_30_dir = r"C:\Users\USER\Desktop\train\30"

# 증강된 데이터를 기존 클래스 폴더로 이동
for filename in os.listdir(augmented_dir):
    src_path = os.path.join(augmented_dir, filename)
    dst_path = os.path.join(class_30_dir, filename)
    shutil.move(src_path, dst_path)

# 증강된 데이터 폴더 삭제
os.rmdir(augmented_dir)

print("증강된 데이터를 통합 완료!")
