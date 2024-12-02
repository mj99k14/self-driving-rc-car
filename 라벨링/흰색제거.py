import cv2
import os

# 입력 폴더 경로
input_image_path = "C:/Users/USER/Desktop/p/45_degrees"

# 출력 폴더 경로
output_image_path = "C:/Users/USER/Desktop/p/45_degrees_processed"

# 출력 폴더 생성 (없으면 생성)
os.makedirs(output_image_path, exist_ok=True)

# 폴더 내 모든 파일에 대해 반복
for filename in os.listdir(input_image_path):
    # 파일 경로 생성
    file_path = os.path.join(input_image_path, filename)
    
    # 이미지 읽기
    image = cv2.imread(file_path)
    
    # 이미지가 제대로 읽혔는지 확인
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {file_path}")
        continue

    # 이미지 크기 정보 확인
    height, width, _ = image.shape
    print(f"처리 중: {filename} - 크기: {width}x{height}")

    # 처리 로직 추가 (예: 그레이스케일 변환)
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 출력 파일 저장
    output_file_path = os.path.join(output_image_path, filename)
    cv2.imwrite(output_file_path, processed_image)

print("모든 이미지 처리가 완료되었습니다.")
