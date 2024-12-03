import cv2
import os

# 입력 폴더 경로 (잘라낼 이미지를 포함한 폴더)
input_folder = "C:/Users/USER/Desktop/p/45_degrees"  # 45도 폴더 경로
# 출력 폴더 경로 (잘라낸 이미지를 저장할 폴더)
output_folder = "C:/Users/USER/Desktop/p/45_degrees_cropped"  # 잘라낸 이미지 저장 폴더

# 잘라낼 영역 (x_start, y_start, x_end, y_end)
# 필요에 따라 x_start, y_start, x_end, y_end 값을 수정
x_start, y_start, x_end, y_end = 100, 100, 800, 600

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 입력 폴더 내 모든 파일 처리
for file_name in os.listdir(input_folder):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 처리
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 이미지 읽기
        image = cv2.imread(input_path)

        if image is not None:
            # 관심 영역(ROI) 추출
            cropped_image = image[y_start:y_end, x_start:x_end]

            # 잘라낸 이미지 저장
            cv2.imwrite(output_path, cropped_image)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Failed to read: {input_path}")

print("모든 이미지가 처리되었습니다.")
