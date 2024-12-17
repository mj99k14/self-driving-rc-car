import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 조정
    transforms.ToTensor()          # 텐서로 변환
])

# 데이터 경로
train_path = "C:/Users/USER/Desktop/train"
val_path = "C:/Users/USER/Desktop/val"
test_path = "C:/Users/USER/Desktop/test"

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train 데이터셋 크기: {len(train_dataset)}")
print(f"Validation 데이터셋 크기: {len(val_dataset)}")
print(f"Test 데이터셋 크기: {len(test_dataset)}")
