import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
import os
from PIL import Image
import numpy as np
from torch import nn, optim

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 45, 90, 135도로 나누어진 폴더들에서 이미지 경로를 읽어옵니다.
        for label, subfolder in enumerate(['45_degrees_processed', '90_degrees_processed', '135_degrees_processed']):
            folder_path = os.path.join(root_dir, subfolder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# 이미지 전처리 과정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 통계값
])

# 데이터셋과 데이터로더 설정
root_dir = 'C:/Users/USER/Desktop/p'  # 데이터가 있는 루트 디렉토리
dataset = CustomDataset(root_dir=root_dir, transform=transform)

# 학습/검증 데이터셋 분리 (예시로 80% 학습, 20% 검증)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 모델 초기화 (예시로 ResNet18을 사용)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3개의 클래스 (45, 90, 135도)

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # GPU 사용 시 모델과 데이터를 GPU로 보내기
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                model = model.cuda()
            
            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 에포크마다 손실 출력
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        
        # 검증
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy: {100 * correct / total}%")

# 모델 훈련 실행
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
# 모델 훈련 후 저장하기
torch.save(model.state_dict(), 'trained_model.pth')
