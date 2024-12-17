import os
import cv2
import numpy as np

def setup_folder(folder_path):
    """폴더가 없으면 생성"""
    if not os.path.exists(folder_path):
        print(f"'{folder_path}' 폴더가 없습니다. 새로 생성합니다.")
        os.makedirs(folder_path)

def get_all_image_paths(base_dir):
    """모든 이미지 파일 경로 가져오기"""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, file_name))
    return sorted(image_paths)

def draw_arrow(image, angle):
    """화살표를 이미지에 그리는 함수"""
    center = (image.shape[1] // 2, image.shape[0] // 2)  # 이미지 중심점
    length = min(image.shape[0], image.shape[1]) // 3   # 화살표 길이

    # 각도 변환 (시계 방향으로 조정)
    corrected_angle = 90 - angle
    angle_rad = np.radians(corrected_angle)

    # 화살표 끝점 계산 (numpy 사용)
    end_point = (
        int(center[0] + length * np.cos(angle_rad)),
        int(center[1] - length * np.sin(angle_rad))
    )

    # 화살표 그리기
    cv2.arrowedLine(image, center, end_point, (0, 255, 0), 5, tipLength=0.2)
    return image

def review_and_filter_images(image_paths):
    """이미지 검토 및 삭제"""
    current_index = 0
    while current_index < len(image_paths):
        image_path = image_paths[current_index]
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            current_index += 1
            continue

        file_name = os.path.basename(image_path)
        try:
            angle = int(file_name.split("_")[2].split(".")[0])  # 예: image_001_30.jpg → 30
        except (IndexError, ValueError):
            print(f"Invalid file format: {file_name}")
            angle = 0

        # 화살표 및 각도 표시
        display_image = image.copy()
        display_image = draw_arrow(display_image, angle)
        cv2.putText(display_image, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Review Images", display_image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):  # 다음 이미지
            current_index += 1
        elif key == ord('a'):  # 이전 이미지
            current_index = max(current_index - 1, 0)
        elif key == ord('r'):  # 이미지 삭제
            print(f"Removing {image_path}")
            os.remove(image_path)
            del image_paths[current_index]
            if current_index >= len(image_paths):
                current_index -= 1
        elif key == ord('s'):  # 건너뛰기
            print(f"Skipping {image_path}")
            current_index += 1
        elif key == 27:  # ESC 키
            print("Exiting review process…")
            break  # 검토 작업만 종료

    cv2.destroyAllWindows()
    return image_paths

if __name__ == "__main__":
    # 설정할 폴더 경로
    base_dir = r"C:\Users\USER\Desktop\csv"
    setup_folder(base_dir)
    image_paths = get_all_image_paths(base_dir)

    print(f"총 {len(image_paths)}개의 이미지가 발견되었습니다.")
    if image_paths:
        remaining_images = review_and_filter_images(image_paths)
        print(f"남은 이미지 개수: {len(remaining_images)}")
    else:
        print("이미지를 찾을 수 없습니다.")
