# 필요한 라이브러리 임포트
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 테스트 데이터 경로
test_data_dir = r'C:\Users\USER\Desktop\test'  # 테스트 데이터 경로 설정

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 데이터셋 로드
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("테스트 데이터 로드 완료")

# 학습된 모델 불러오기
from torchvision import models
import torch.nn as nn

model = models.resnet18()  # 사전 학습된 모델 구조 사용
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))  # 클래스 개수에 맞게 수정
model.load_state_dict(torch.load('model.pth'))  # 저장된 가중치 불러오기
model.eval()  # 평가 모드로 전환

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 모델 성능 평가
correct = 0
total = 0

with torch.no_grad():  # 그래디언트 비활성화
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스를 예측
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f'테스트 데이터 정확도: {accuracy:.2f}%')
