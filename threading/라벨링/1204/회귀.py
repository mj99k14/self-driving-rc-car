import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

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
            image = self.transform(image)  # 전처리 적용

        return image, label

# 데이터 전처리 (크기 변경, 텐서 변환, 정규화)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 이미지 폴더 경로
data_dir = r'C:\Users\USER\Desktop\new'  # 바탕화면에 있는 new 폴더 경로

# 커스텀 데이터셋 로드
dataset = AngleDataset(data_dir=data_dir, transform=transform)

# DataLoader로 배치 단위로 데이터 로드
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터셋 크기 및 샘플 확인
print(f"데이터셋 크기: {len(dataset)}")
print(f"샘플 레이블 (각도): {dataset.labels[:5]}")
