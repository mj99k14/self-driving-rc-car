import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# 1. 데이터셋 클래스 정의
class LaneDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# 2. 데이터 로드
base_dir = "C:/Users/USER/Desktop/p"
folders = ["45_degrees_augmented", "90_degrees_augmented", "135_degrees_augmented"]
datasets = []

for folder in folders:
    dataset = LaneDataset(
        image_dir=os.path.join(base_dir, folder),
        transform=transforms.ToTensor()
    )
    datasets.append(dataset)

# 데이터 통합 및 DataLoader 생성
final_dataset = ConcatDataset(datasets)
train_loader = DataLoader(final_dataset, batch_size=8, shuffle=True)

# 3. 모델 정의
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 초기화
model = SimpleUNet()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 루프
num_epochs = 5  # 반복 횟수
for epoch in range(num_epochs):
    model.train()
    for batch_idx, images in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)  # 모델에 데이터 입력
        # 여기에 차선에 대한 라벨(마스크) 데이터를 넣어야 하지만 현재는 간단히 진행
        loss = criterion(outputs, images.mean(dim=1, keepdim=True))  # 예제용 Loss 계산
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

# 5. 모델 저장
torch.save(model.state_dict(), "simple_unet.pth")
print("모델 학습 완료 및 저장!")
