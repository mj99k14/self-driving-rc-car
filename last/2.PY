import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# 데이터 경로 설정
train_dir = r"C:\Users\USER\Desktop\train"
val_dir = r"C:\Users\USER\Desktop\val"

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# 데이터 로더 생성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 모델의 출력 클래스 수를 데이터셋의 클래스 수에 맞게 조정
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("모델 설정 완료")
