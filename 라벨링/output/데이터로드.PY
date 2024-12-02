import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

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
        image = Image.open(img_path).convert("RGB")  # RGB로 변환

        if self.transform:
            image = self.transform(image)

        return image

# 각 폴더에서 데이터셋 로드
base_dir = "C:/Users/USER/Desktop/p"
folders = ["45_degrees_augmented", "90_degrees_augmented", "135_degrees_augmented"]
datasets = []

for folder in folders:
    dataset = LaneDataset(
        image_dir=os.path.join(base_dir, folder),
        transform=transforms.ToTensor()  # 이미지를 Tensor로 변환
    )
    datasets.append(dataset)

# 데이터셋 통합
final_dataset = ConcatDataset(datasets)

# DataLoader 생성
train_loader = DataLoader(final_dataset, batch_size=8, shuffle=True)

# DataLoader에서 데이터 확인
for batch_idx, batch in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Batch Shape: {batch.shape}")  # 배치 크기 출력
    if batch_idx == 1:  # 예제니까 2번째 배치까지만 출력
        break
