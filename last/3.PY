import torch
from torchvision import datasets, transforms, models

# 테스트 데이터 경로
test_dir = r"C:\Users\USER\Desktop\test"

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 데이터셋 로드
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 설정 (저장된 가중치의 클래스 수를 확인 후 수정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # Pretrained 가중치를 사용하지 않음
num_classes = 5  # 저장된 모델에 맞는 클래스 수를 입력
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 저장된 가중치 로드
model.load_state_dict(torch.load("resnet18_best_model.pth"))
model.eval()

# 테스트 평가
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
