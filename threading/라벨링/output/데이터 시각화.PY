import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 클래스 정의
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

# 데이터 경로 및 DataLoader 정의
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

# 시각화 코드
for batch_idx, batch in enumerate(train_loader):
    img_tensor = batch[0]  # 첫 번째 이미지
    img = img_tensor.permute(1, 2, 0).numpy()  # 채널 순서 변경 (C, H, W -> H, W, C)
    
    # 이미지 출력
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Batch {batch_idx + 1}: Image 1")
    plt.show()
    break  # 첫 번째 배치만 확인
