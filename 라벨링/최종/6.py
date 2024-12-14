import torch
import torch.nn as nn
import os

# 1. PilotNet 클래스 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 21 * 21, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 2. 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)

# 3. 학습된 모델 불러오기 (필요한 경우)
trained_model_path = r"C:\Users\USER\Desktop\csv\pilotnet_model.pth"

try:
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    print("학습된 모델이 성공적으로 불러와졌습니다.")
except Exception as e:
    print(f"학습된 모델 불러오기 실패: {e}")

# 4. 모델 저장
save_path = r"C:\Users\USER\Desktop\csv\pilotnet_model.pth"

try:
    torch.save(model.state_dict(), save_path)
    print(f"모델이 저장되었습니다: {save_path}")
except Exception as e:
    print(f"모델 저장 중 오류 발생: {e}")
