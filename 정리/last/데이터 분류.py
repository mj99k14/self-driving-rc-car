import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 선별 도구")

        # 이미지 목록 변수
        self.image_folder = ""
        self.image_list = []
        self.current_index = 0

        # GUI 구성
        self.label = tk.Label(root, text="폴더를 선택하세요")
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        # 키보드 이벤트 바인딩
        self.root.bind("<a>", self.keep_image)  # 'a' 키로 사용
        self.root.bind("<d>", self.delete_image)  # 'd' 키로 삭제
        self.root.bind("<s>", self.next_image)  # 's' 키로 다음

        # 폴더 선택 버튼
        self.select_folder_button = tk.Button(root, text="폴더 선택", command=self.select_folder)
        self.select_folder_button.pack(pady=10)

    def select_folder(self):
        # 폴더 선택
        self.image_folder = filedialog.askdirectory(title="이미지 폴더 선택")
        if not self.image_folder:
            return

        # 이미지 파일 목록 불러오기
        self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0

        if not self.image_list:
            self.label.config(text="이미지가 없습니다.")
        else:
            self.label.config(text=f"{len(self.image_list)}개의 이미지 발견")
            self.display_image()

    def display_image(self):
        # 현재 이미지 표시
        if self.current_index < len(self.image_list):
            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            img = Image.open(image_path)
            img = img.resize((500, 500))  # 이미지 크기 조정
            img = ImageTk.PhotoImage(img)

            self.image_label.config(image=img)
            self.image_label.image = img  # 이미지 참조 유지
            self.label.config(text=f"현재 이미지: {self.image_list[self.current_index]} ({self.current_index + 1}/{len(self.image_list)})")
        else:
            self.label.config(text="이미지 검열 완료")
            self.image_label.config(image="")
            self.image_label.image = None

    def keep_image(self, event=None):
        # 사용 이미지는 그대로 유지
        if self.current_index < len(self.image_list):
            self.next_image()

    def delete_image(self, event=None):
        # 이미지 삭제 -> delete 폴더로 이동
        if self.current_index < len(self.image_list):
            delete_folder = os.path.join(self.image_folder, "delete")
            os.makedirs(delete_folder, exist_ok=True)

            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            shutil.move(image_path, os.path.join(delete_folder, self.image_list[self.current_index]))

            self.next_image()

    def next_image(self, event=None):
        # 다음 이미지로 이동
        self.current_index += 1
        self.display_image()

# 프로그램 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelector(root)
    root.mainloop()
