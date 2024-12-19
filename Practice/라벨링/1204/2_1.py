import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

# 이미지 전처리 (크기 변경, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터셋 경로 설정
data_dir = r'C:\Users\USER\Desktop\new'  # 바탕화면에 있는 new 폴더 경로

# ImageFolder를 사용하여 데이터셋 로드
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# DataLoader로 배치 단위로 데이터 로드
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 사전 학습된 ResNet 모델 불러오기
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 최신 방식으로 가중치 불러오기

# 마지막 레이어를 데이터셋에 맞게 수정 (클래스 개수는 dataset.classes의 길이에 맞게)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# 모델을 훈련 모드로 전환
model.train()

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()  # 분류 문제이므로 CrossEntropyLoss 사용
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10  # 훈련할 에폭 수

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 데이터 및 레이블을 GPU로 이동
        inputs, labels = inputs.to(device), labels.to(device)

        # Gradients 초기화
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 모델 저장
torch.save(model.state_dict(), 'model.pth')
print("훈련된 모델이 저장되었습니다.")
