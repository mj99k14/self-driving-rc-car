import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# 각도를 레이블로 사용할 데이터셋 클래스
class AngleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 각도별 폴더 순회
        for angle in os.listdir(data_dir):
            angle_path = os.path.join(data_dir, angle)
            if os.path.isdir(angle_path):
                for image_name in os.listdir(angle_path):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        image_path = os.path.join(angle_path, image_name)
                        self.image_paths.append(image_path)
                        
                        # 각도 폴더 이름에서 'degrees' 제거하고 실수로 변환
                        try:
                            angle_value = float(angle.replace('_degrees', ''))  # '_degrees' 제거하고 float 변환
                            self.labels.append(angle_value)
                        except ValueError:
                            print(f"잘못된 폴더 이름: {angle}. 'degrees'가 포함되어야 합니다.")
                            continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')  # 이미지 열기
        if self.transform:
            image = self.transform(image).float()  # 텐서로 변환 후 float32로 변환

        return image, label

# CNN 모델 정의
class AngleCNN(nn.Module):
    def __init__(self):
        super(AngleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 첫 번째 컨볼루션
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 두 번째 컨볼루션
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 세 번째 컨볼루션
        self.pool = nn.MaxPool2d(2, 2)  # 최대 풀링
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # 완전 연결층 1
        self.fc2 = nn.Linear(128, 1)  # 완전 연결층 2 (출력: 1개 각도 값)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 첫 번째 Conv + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv2(x)))  # 두 번째 Conv + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv3(x)))  # 세 번째 Conv + ReLU + MaxPool
        x = x.view(-1, 64 * 16 * 16)  # 데이터를 1차원으로 펼침
        x = torch.relu(self.fc1(x))  # 첫 번째 FC + ReLU
        x = self.fc2(x)  # 두 번째 FC (각도 출력)
        return x

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터셋 로드
data_dir = r'C:\Users\USER\Desktop\new'
dataset = AngleDataset(data_dir=data_dir, transform=transform)

# DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 학습
model = AngleCNN()
criterion = nn.MSELoss()  # 평균 제곱 오차
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()  # GPU 사용 시
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
