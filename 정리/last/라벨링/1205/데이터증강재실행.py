# 필요한 라이브러리 임포트
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 데이터 경로 설정
train_data_dir = r'C:\Users\USER\Desktop\new'  # 학습 데이터 경로
test_data_dir = r'C:\Users\USER\Desktop\test'   # 테스트 데이터 경로

# 데이터 증강 (강도를 조정)
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 좌우 반전만 사용
    transforms.RandomRotation(10),     # 회전 각도를 10도로 줄임
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 데이터 전처리 (증강 없음)
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform_test)

# DataLoader 준비
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("데이터 로드 완료")

# 모델 준비 (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 사전 학습된 가중치 사용
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 클래스 개수 설정

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 학습률을 0.0001로 줄임

# 학습 파라미터
num_epochs = 20  # 에폭 수

# 학습 루프
for epoch in range(num_epochs):
    model.train()  # 학습 모드
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Optimizer 초기화
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward Pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # 매 에폭마다 테스트 데이터로 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Epoch [{epoch+1}/{num_epochs}] 테스트 데이터 정확도: {accuracy:.2f}%')

# 모델 저장
torch.save(model.state_dict(), 'model_augmented.pth')
print("모델 학습 및 저장 완료")
