import pandas as pd
import os

# CSV 파일 경로 (바탕화면)
csv_path = "C:/Users/USER/Desktop/training_data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 존재하지 않는 파일 확인
missing_files = []
for path in df['frame_path']:
    if not os.path.exists(path):
        missing_files.append(path)

# 누락된 파일 경로 출력
print(f"누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")

# 누락된 파일을 데이터프레임에서 제거
df = df[~df['frame_path'].isin(missing_files)]

# direction 열 제거 (열 이름이 정확한지 확인)
if 'direction' in df.columns:
    df = df.drop(columns=['direction'])
    print("'direction' 열이 제거되었습니다.")

# 수정된 CSV 저장 경로
updated_csv_path = "C:/Users/USER/Desktop/training_data_cleaned.csv"
df.to_csv(updated_csv_path, index=False)

print(f"수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")
