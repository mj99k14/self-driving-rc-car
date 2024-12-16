import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# **1. 데이터셋 정의**
class SteeringDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.categories = [30, 60, 90, 120, 150]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        category = int(row['steering_category'])

        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        image = cv2.resize(image, (200, 66))
        image = image / 255.0  # 정규화
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW (PyTorch 입력 형식)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(category, dtype=torch.long)

# **2. PilotNet 모델 정의**
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 5)  # 5개의 범주로 분류
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# **3. 데이터 로더 설정**
csv_path = "C:/Users/USER/Desktop/training_data.csv"

# 데이터 로드
df = pd.read_csv(csv_path)
df['steering_category'] = df['steering_angle'].apply(
    lambda angle: [30, 60, 90, 120, 150].index(min([30, 60, 90, 120, 150], key=lambda x: abs(x - angle)))
)

train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

train_data.to_csv("train.csv", index=False)
val_data.to_csv("val.csv", index=False)
test_data.to_csv("test.csv", index=False)

train_dataset = SteeringDataset("train.csv")
val_dataset = SteeringDataset("val.csv")
test_dataset = SteeringDataset("test.csv")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **4. 학습 설정**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for training: {device}")
model = PilotNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 학습률 조정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 학습률 감소 스케줄러 추가

# **모델 저장 관련 설정**
best_val_loss = float('inf')  # Validation Loss 초기화
best_model_path = "C:/Users/USER/Desktop/best_pilotnet_model.pth"  # 모델 저장 경로

# **5. 학습 및 검증**
epochs = 30  # 에포크 증가
train_losses = []
val_losses = []

for epoch in range(epochs):
    # **Training**
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # **Validation**
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # 학습 결과 저장
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    # 학습 결과 출력
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")

    # **Best Model 저장**
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # 학습률 감소 스케줄러 호출
    scheduler.step()

print("학습 완료. Best Model이 저장되었습니다.")

# **6. 테스트 손실 계산**
print("테스트 데이터 평가 시작...")
model.eval()  # 평가 모드로 전환
test_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # 정확도 계산
        _, predicted = torch.max(outputs, 1)  # 예측값
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = correct_predictions / total_predictions * 100

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# **7. 학습 결과 시각화**
import matplotlib.pyplot as plt  # 시각화 모듈

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.grid()
plt.show()

# **8. 저장된 모델 로드**
print("저장된 모델 로드 및 테스트...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 새 데이터 예측 예제 (전처리 필요)
# new_image_tensor = <이미지 전처리된 텐서>
# output = model(new_image_tensor)
# print("Predicted category:", torch.argmax(output, dim=1).item())
