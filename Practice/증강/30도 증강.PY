from torchvision import transforms
from PIL import Image
import os

# 데이터 증강 설정
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 클래스 30 데이터 경로
class_30_dir = r"C:\Users\USER\Desktop\train\30"
augmented_dir = r"C:\Users\USER\Desktop\train\30_augmented"

os.makedirs(augmented_dir, exist_ok=True)

# 기존 이미지를 불러와 증강
for filename in os.listdir(class_30_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(class_30_dir, filename)
        img = Image.open(img_path)
        
        for i in range(5):  # 각 이미지당 5개의 증강 데이터 생성
            augmented_img = augmentation(img)
            save_path = os.path.join(augmented_dir, f"{os.path.splitext(filename)[0]}_aug{i}.jpg")
            transforms.ToPILImage()(augmented_img).save(save_path)

print("데이터 증강 완료!")
