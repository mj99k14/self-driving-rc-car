import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

# CSV 파일 경로
csv_path = "C:/Users/USER/Desktop/training_data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 클래스별 데이터 분리
class_groups = {
    30: df[df['steering_angle'] == 30],
    60: df[df['steering_angle'] == 60],
    90: df[df['steering_angle'] == 90],
    120: df[df['steering_angle'] == 120]
}

# Oversampling 및 Undersampling
balanced_classes = {
    30: resample(class_groups[30], replace=True, n_samples=1000, random_state=42),
    60: class_groups[60].sample(n=1000, random_state=42),
    90: class_groups[90].sample(n=1000, random_state=42),
    120: resample(class_groups[120], replace=True, n_samples=1000, random_state=42)
}

# 최종 데이터셋 병합
df_balanced = pd.concat(balanced_classes.values())

# 균형 잡힌 데이터셋 저장
balanced_csv_path = "C:/Users/USER/Desktop/balanced_training_data.csv"
df_balanced.to_csv(balanced_csv_path, index=False)
print(f"균형 잡힌 데이터셋이 저장되었습니다: {balanced_csv_path}")

# 데이터 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df_balanced['steering_angle'], bins=4, color='purple', alpha=0.7, edgecolor='black')
plt.title("Balanced Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
