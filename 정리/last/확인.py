import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import Counter

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),              # 크기 조정
    transforms.RandomHorizontalFlip(),          # 좌우 반전 추가
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 밝기 및 대비 조정
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 검증 데이터는 증강 없음
    transforms.ToTensor()
])

# 데이터셋 로드
train_path = "C:/Users/USER/Desktop/train"  # 학습 데이터 경로
val_path = "C:/Users/USER/Desktop/val"      # 검증 데이터 경로
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 클래스 분포 확인
class_counts = [len([y for _, y in train_dataset.samples if y == c]) for c in range(len(train_dataset.classes))]
print("Train 클래스 분포:", class_counts)

# 클래스 가중치 계산
class_weights = [1.0 / count for count in class_counts]
weights = torch.tensor(class_weights).to("cuda" if torch.cuda.is_available() else "cpu")

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained ResNet18 모델 정의
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# ResNet18 전체 가중치 활성화
for param in model.parameters():
    param.requires_grad = True

# 출력 레이어 수정 (클래스 개수에 맞춤)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

# 손실 함수, 옵티마이저, 스케줄러 정의
criterion = nn.CrossEntropyLoss(weight=weights)  # 클래스 가중치 적용
optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # 학습률 설정
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early Stopping 설정
best_loss = float('inf')
patience = 5
wait = 0

# 학습 루프
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 검증 단계
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 학습률 스케줄러 업데이트
    scheduler.step(val_loss / len(val_loader))

    # Early Stopping 체크
    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0  # 성능이 개선되면 대기 카운터 초기화
        torch.save(model.state_dict(), "resnet18_best_model.pth")  # 가장 성능이 좋은 모델 저장
        print(f"Best model saved at epoch {epoch+1}")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered!")
            break

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# 최종 모델 저장
torch.save(model.state_dict(), "resnet18_model_final.pth")
print("최종 모델 저장 완료: resnet18_model_final.pth")
