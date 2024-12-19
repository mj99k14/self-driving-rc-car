import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# CSV 파일 경로
csv_path = r"C:\Users\USER\Desktop\csv\steering_data.csv"

# 1. 데이터 로드 및 전처리
class SteeringDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        angle = row['angle']

        try:
            # 이미지 열기 및 전처리
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img.resize((224, 224)))  # 크기 조정 (224x224)
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (H, W, C) → (C, H, W) & 정규화
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            raise e

        return img, torch.tensor(angle, dtype=torch.float32)

# 데이터 로드
try:
    df = pd.read_csv(csv_path)
    print(f"CSV 파일이 정상적으로 로드되었습니다: {csv_path}")
    print(f"총 데이터 개수: {len(df)}")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    raise e

# 2. 데이터셋 분리
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"훈련 데이터 개수: {len(train_data)}")
print(f"검증 데이터 개수: {len(val_data)}")
print(f"테스트 데이터 개수: {len(test_data)}")

# 3. PyTorch 데이터셋 및 DataLoader 준비
train_dataset = SteeringDataset(train_data)
val_dataset = SteeringDataset(val_data)
test_dataset = SteeringDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# DataLoader 테스트
print("\n데이터 로더 테스트:")
for images, angles in train_loader:
    print(f"이미지 배치 크기: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"각도 배치 크기: {angles.shape}")    # (batch_size,)
    break  # 첫 번째 배치만 출력
