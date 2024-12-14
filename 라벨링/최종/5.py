import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np
import os

# GPU 최적화
torch.backends.cudnn.benchmark = True

# 1. CSV 파일 경로 및 데이터셋 준비
csv_path = r"C:\Users\USER\Desktop\csv\steering_data.csv"

class SteeringDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        angle = row['angle']

        try:
            # 이미지 로드 및 전처리
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # 크기 조정
            img = np.array(img) / 255.0  # 정규화
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
            angle = torch.tensor(angle, dtype=torch.float32)
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            raise e

        return img, angle

if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv(csv_path)
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_dataset = SteeringDataset(train_data)
    val_dataset = SteeringDataset(val_data)
    test_dataset = SteeringDataset(test_data)

    # DataLoader 설정 (num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 2. CNN 모델 정의
    class PilotNet(nn.Module):
        def __init__(self):
            super(PilotNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
            self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
            self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
            self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 21 * 21, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 10)
            self.fc4 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = torch.relu(self.conv5(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # 3. 모델, 손실 함수 및 옵티마이저 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 모델 학습
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for images, angles in train_loader:
                images, angles = images.to(device), angles.to(device)

                # Forward
                outputs = model(images)
                loss = criterion(outputs.squeeze(), angles)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, angles in val_loader:
                    images, angles = images.to(device), angles.to(device)
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), angles)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 학습 실행
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs} 시작...")

        for batch_idx, (images, angles) in enumerate(train_loader):
            images, angles = images.to(device), angles.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs.squeeze(), angles)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # 진행 상황 출력
            if batch_idx % 10 == 0:  # 10번째 배치마다 출력
                print(f"  [배치 {batch_idx}/{len(train_loader)}] 현재 손실: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                images, angles = images.to(device), angles.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), angles)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    # 5. 테스트 데이터 평가
    def evaluate_model(model, test_loader):
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, angles in test_loader:
                images, angles = images.to(device), angles.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), angles)
                test_loss += loss.item() * images.size(0)

        test_loss /= len(test_loader.dataset)
        print(f"테스트 데이터 Loss: {test_loss:.4f}")

    # 테스트 실행
    evaluate_model(model, test_loader)

    # 모델 저장 경로
    save_path = r"C:\Users\USER\Desktop\csv\pilotnet_model.pth"

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"모델이 저장되었습니다: {save_path}")
