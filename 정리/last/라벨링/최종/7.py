import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# **1. 데이터셋 정의**
class SteeringDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
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
csv_path = "C:/Users/USER/Desktop/csv/steering_data.csv"  # 사용자 상황에 맞는 CSV 경로
df = pd.read_csv(csv_path)

# **조향각을 범주화**
df['steering_category'] = df['steering_angle'].apply(
    lambda angle: [30, 60, 90, 120, 150].index(min([30, 60, 90, 120, 150], key=lambda x: abs(x - angle)))
)

# 데이터 분리
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

# DataLoader 생성
train_dataset = SteeringDataset(train_data)
val_dataset = SteeringDataset(val_data)
test_dataset = SteeringDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **4. 학습 설정**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for training: {device}")
model = PilotNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **모델 저장 관련 설정**
best_val_loss = float('inf')  # Validation Loss 초기화
best_model_path = "C:/Users/USER/Desktop/csv/best_pilotnet_model.pth"  # 모델 저장 경로

# **5. 학습 및 검증**
epochs = 50  # 필요한 경우 Epoch 수 조정
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

    # 학습 결과 출력
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # **Best Model 저장**
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss/len(val_loader):.4f}")

print("학습 완료. Best Model이 저장되었습니다.")
