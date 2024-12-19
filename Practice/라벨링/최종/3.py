import os
import pandas as pd
import shutil

# CSV 파일 경로
csv_path = r"C:\Users\USER\Desktop\csv\steering_data.csv"

# 누락된 파일 저장 경로
missing_files_path = r"C:\Users\USER\Desktop\csv\missing_files.csv"

# 복구된 파일이 저장된 폴더 경로 (복구된 파일 위치)
recovered_files_path = r"C:\RecoveredFiles"

# 최종 파일 저장 경로 (원래 파일 위치)
destination_base = r"C:\Users\USER\Desktop\csv"

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    print(f"CSV 파일이 정상적으로 로드되었습니다: {csv_path}")
    
    # 누락된 파일 확인
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

        # 누락된 파일 목록 저장
        pd.DataFrame({"missing_files": missing_files}).to_csv(missing_files_path, index=False, encoding='utf-8-sig')
        print(f"누락된 파일 목록이 저장되었습니다: {missing_files_path}")

        # 복구된 파일을 누락된 파일 경로에 맞게 이동
        print("\n복구된 파일을 적절한 위치로 이동 중...")
        for file_path in missing_files:
            file_name = os.path.basename(file_path)  # 파일 이름 추출
            angle_folder = os.path.basename(os.path.dirname(file_path))  # 각도 폴더 이름 추출
            dest_folder = os.path.join(destination_base, angle_folder)  # 최종 저장 폴더 경로
            os.makedirs(dest_folder, exist_ok=True)  # 폴더가 없으면 생성

            # 복구된 파일 경로
            recovered_file = os.path.join(recovered_files_path, file_name)
            if os.path.exists(recovered_file):
                shutil.move(recovered_file, os.path.join(dest_folder, file_name))
                print(f"복구된 파일 이동: {recovered_file} → {os.path.join(dest_folder, file_name)}")
            else:
                print(f"복구 파일 없음: {recovered_file}")
        
        print("\n모든 복구 파일 이동이 완료되었습니다.")
    else:
        print("모든 파일이 정상적으로 존재합니다.")

except FileNotFoundError:
    print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
