import os
from PIL import Image
from torchvision import transforms

# 입력 및 출력 폴더 설정
input_folder = "C:/Users/USER/Desktop/p/90_degrees_output"  # 전처리된 이미지 폴더
output_folder = "C:/Users/USER/Desktop/p/90_degrees_augmented"  # 증강된 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 데이터 증강 변환 정의
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomRotation(10),          # -10도~10도 랜덤 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # 밝기, 대비, 채도 변화
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 랜덤 이동
])

# 이미지 증강 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 이미지 불러오기
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")

        # 여러 증강된 이미지 생성
        for i in range(5):  # 각 이미지당 5개의 증강 데이터 생성
            augmented_image = augmentations(image)
            output_path = os.path.join(output_folder, f"{filename.replace('.jpg', '')}_aug_{i}.jpg")
            augmented_image.save(output_path)

print(f"데이터 증강 완료! 증강된 이미지는 {output_folder}에 저장되었습니다.")
