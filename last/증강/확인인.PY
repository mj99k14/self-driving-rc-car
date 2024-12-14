
from torchvision import datasets, transforms
from collections import Counter

# Train 데이터 경로 설정
train_dir = r"C:\Users\USER\Desktop\train"

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Train 데이터셋 로드
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# 클래스별 데이터 개수 확인
train_class_counts = Counter(train_dataset.targets)
print("Train 클래스 분포:")
for class_idx, count in train_class_counts.items():
    print(f"클래스 {train_dataset.classes[class_idx]}: {count}개")
