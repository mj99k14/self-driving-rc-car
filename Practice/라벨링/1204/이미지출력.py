import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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

# 하나의 배치에서 이미지와 레이블을 가져오기
images, labels = next(iter(train_loader))

# 이미지 출력 (첫 번째 이미지 4개 출력 예시)
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# 4개의 이미지를 출력
for i in range(4):
    image = images[i].numpy().transpose((1, 2, 0))  # CHW -> HWC 변환
    image = np.clip(image, 0, 1)  # 이미지 값 범위 조정
    label = dataset.classes[labels[i]]  # 해당 레이블

    axes[i].imshow(image)
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')  # 축을 끄기

plt.show()

