import pandas as pd
import os

# 바탕화면 CSV 폴더 경로
base_path = "C:/Users/USER/Desktop/csv"
output_csv_path = "C:/Users/USER/Desktop/training_data.csv"

# 폴더별 각도를 매칭하기 위한 딕셔너리
angle_mapping = {"30": 30, "60": 60, "90": 90, "120": 120, "150": 150}

# 결과 저장을 위한 리스트
data = []

# 각 폴더를 탐색하며 파일 경로 수집
for folder, angle in angle_mapping.items():
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 선택
                frame_path = os.path.join(folder_path, file)
                data.append({"frame_path": frame_path, "angle": angle})

# DataFrame 생성
df = pd.DataFrame(data)

# CSV 파일 저장
df.to_csv(output_csv_path, index=False)

print(f"CSV 파일이 저장되었습니다: {output_csv_path}")
