import torch
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트 데이터 경로 및 전처리
test_path = "C:/Users/USER/Desktop/test"
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 테스트 데이터셋 로드
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

# 학습에 사용된 유효한 클래스
valid_classes = ['120', '150', '30', '60', '90']  # 학습 데이터의 클래스
valid_class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}

# 테스트 데이터 필터링
filtered_samples = [
    (path, valid_class_to_idx[test_dataset.classes[label]])
    for path, label in test_dataset.samples
    if test_dataset.classes[label] in valid_classes
]

# 필터링된 데이터셋 생성
test_dataset.samples = filtered_samples
test_dataset.classes = valid_classes
test_dataset.class_to_idx = valid_class_to_idx

# 데이터 로더 생성
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 클래스 정보 출력
print("Filtered Test 클래스:", test_dataset.classes)

# 모델 정의 및 로드
model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(test_dataset.classes))
model.load_state_dict(torch.load("resnet18_best_model.pth"))
model = model.to(device)
model.eval()

# 평가 변수 초기화
num_classes = len(test_dataset.classes)
class_correct = [0] * num_classes
class_total = [0] * num_classes
total_correct = 0
total_samples = 0

# 테스트 평가
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# 테스트 정확도 출력
print(f"Test Accuracy: {100 * total_correct / total_samples:.2f}%")

# 클래스별 정확도 출력
for i in range(num_classes):
    if class_total[i] > 0:
        print(f"Class {test_dataset.classes[i]} Accuracy: {100 * class_correct[i] / class_total[i]:.2f}%")
    else:
        print(f"Class {test_dataset.classes[i]} Accuracy: N/A (no samples)")
