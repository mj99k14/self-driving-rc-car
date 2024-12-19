import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 전처리 (크기 변경, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 이미지 폴더 경로
data_dir = r'C:\Users\USER\Desktop\new'  # 바탕화면에 있는 new 폴더 경로

# 데이터셋 로드
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# DataLoader로 배치 단위로 데이터 로드
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터셋 크기 및 클래스 확인
print(f"데이터셋 크기: {len(dataset)}")
print(f"클래스: {dataset.classes}")
