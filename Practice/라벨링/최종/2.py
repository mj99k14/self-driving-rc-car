import os
import pandas as pd

# CSV 파일 경로
csv_path = r"C:\Users\USER\Desktop\csv\steering_data.csv"

# 누락된 파일 저장 경로
missing_files_path = r"C:\Users\USER\Desktop\csv\missing_files.csv"

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    print(f"CSV 파일이 정상적으로 로드되었습니다: {csv_path}")
    
    # 존재하지 않는 파일 확인
    missing_files = []
    for path in df['frame_path']:
        if not os.path.exists(path):
            missing_files.append(path)

    # 결과 출력
    total_files = len(df)
    missing_count = len(missing_files)

    print(f"총 파일 개수: {total_files}")
    print(f"누락된 파일 개수: {missing_count}")

    if missing_count > 0:
        print("누락된 파일 예시:")
        for file in missing_files[:5]:  # 최대 5개의 누락 파일만 출력
            print(f" - {file}")

        # 누락된 파일을 CSV로 저장
        pd.DataFrame({"missing_files": missing_files}).to_csv(missing_files_path, index=False, encoding='utf-8-sig')
        print(f"누락된 파일 목록이 저장되었습니다: {missing_files_path}")
    else:
        print("모든 파일이 정상적으로 존재합니다.")

except FileNotFoundError:
    print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
