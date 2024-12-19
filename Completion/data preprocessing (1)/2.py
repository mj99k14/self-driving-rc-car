import pandas as pd
import os

# 바탕화면 CSV 폴더 경로
base_path = "C:/Users/USER/Desktop/csv"
output_csv_path = "C:/Users/USER/Desktop/training_data.csv"

# CSV 파일 로드
df = pd.read_csv(output_csv_path)

# 경로 수정 (절대 경로로 변경)
df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(base_path, x))

# 존재하지 않는 파일 확인
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

# 결과 출력
print(f"최종 누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 존재합니다.")
