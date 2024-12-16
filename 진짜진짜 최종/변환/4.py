import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = csv_path = "C:/Users/USER/Desktop/training_data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 1. 특정 각도(-10, 10) 제거
df_filtered = df[~df['steering_angle'].isin([-10, 10])]

# 2. Oversampling: 가장 많은 데이터 수로 맞춤
max_count = df_filtered['steering_angle'].value_counts().max()

df_oversampled = df_filtered.groupby('steering_angle', group_keys=False).apply(
    lambda x: x.sample(max_count, replace=True, random_state=42)
)

# 3. 결과 저장
balanced_csv_path = "C:/Users/USER/Desktop/oversampled_training_data.csv"
df_oversampled.to_csv(balanced_csv_path, index=False)
print(f"Oversampled 데이터셋이 저장되었습니다: {balanced_csv_path}")

# 4. 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df_oversampled['steering_angle'], bins=len(df_oversampled['steering_angle'].unique()), 
         color='green', alpha=0.7, edgecolor='black')
plt.title("Oversampled Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
