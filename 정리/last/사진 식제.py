import os
import shutil

# 삭제할 폴더 경로 설정
base_dir = "./mj"  # 사진이 저장된 기본 폴더 경로

def delete_all_photos(directory):
    if os.path.exists(directory):
        # 폴더 내부 모든 파일 및 하위 폴더 삭제
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            for dir_ in dirs:
                dir_path = os.path.join(root, dir_)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except Exception as e:
                    print(f"Error deleting folder {dir_path}: {e}")
        print(f"All files and folders in '{directory}' have been deleted.")
    else:
        print(f"The directory '{directory}' does not exist.")

if __name__ == "__main__":
    delete_all_photos(base_dir)
