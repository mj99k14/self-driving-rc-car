import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# PilotNet 모델 정의
class PilotNet(nn.Module):
    def __init__(self, num_classes):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가
        self.fc1 = nn.Linear(64 * 9 * 9, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)  # LeakyReLU 추가

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.dropout(x)  # 드롭아웃 적용
        x = x.view(x.size(0), -1)  # Flatten
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),              # 크기 조정
    transforms.RandomRotation(15),             # 15도 이내의 랜덤 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
    transforms.RandomHorizontalFlip(),         # 랜덤 좌우 반전
    transforms.ToTensor()                      # 텐서로 변환
])

# 검증 데이터 전처리
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 검증 데이터는 증강하지 않음
    transforms.ToTensor()
])

# 데이터셋 로드
train_path = "C:/Users/USER/Desktop/train"
val_path = "C:/Users/USER/Desktop/val"
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 장치 설정 (GPU가 있으면 GPU, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 손실 함수, 옵티마이저 정의
num_classes = 5  # 클래스 수 (30, 60, 90, 120, 150)
model = PilotNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # 학습률 감소 및 AdamW 옵티마이저 사용

# 학습 루프
num_epochs = 20  # 에포크 수 증가
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 검증 단계
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# 모델 저장
torch.save(model.state_dict(), "pilotnet_model_improved.pth")
print("모델 저장 완료: pilotnet_model_improved.pth")
