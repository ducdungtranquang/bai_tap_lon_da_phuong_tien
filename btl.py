import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import numpy as np
from skimage import feature
from PIL import Image, ImageTk


from images import danhSachHinhAnh, taoCoSoDuLieu, timKiemTuongTu, tinhLBP

duongDanDanhSachHinhAnh = "./output_images"
soLuongAnhHienThi = 3

def show_image(label, img):
    # Convert the OpenCV image to a format that can be displayed by Tkinter
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    # Display the image on the Label
    label.configure(image=img)
    label.image = img
        
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.master.geometry("400x300")
        self.master.configure(bg="pink") 
        self.create_widgets()
        self.output_folder = None
        self.selected_images = []

    def create_widgets(self):
        self.label = tk.Label(text="Khai pha Bi kip vo lam")
        self.label.pack()

        self.chon_video_btn = tk.Button(self)
        self.chon_video_btn["text"] = "Chon file Video"
        self.chon_video_btn["command"] = self.select_video
        self.chon_video_btn.pack()

        self.chuyen_doi = tk.Button(self)
        self.chuyen_doi["text"] = "Chuyen doi video thanh anh"
        self.chuyen_doi["state"] = "disabled"
        self.chuyen_doi["command"] = self.chuyen_video_to_anh
        self.chuyen_doi.pack()

        self.chon_anh1 = tk.Button(self)
        self.chon_anh1["text"] = "Chon anh 1"
        self.chon_anh1["state"] = "normal"
        self.chon_anh1["command"] = self.select_image1
        self.chon_anh1.pack()

        self.so_sanh_btn = tk.Button(self)
        self.so_sanh_btn["text"] = "So sanh anh"
        self.so_sanh_btn["state"] = "disabled"
        self.so_sanh_btn["command"] = self.compare_images
        self.so_sanh_btn.pack()

        self.image_labels = []
        for _ in range(soLuongAnhHienThi):
            label = tk.Label(self)
            label.pack(side="left", padx=10)
            self.image_labels.append(label)

        self.quit = tk.Button(self, text="Thoat", command=root.destroy)
        self.quit.pack()

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.video_path = file_path
            self.output_folder = "output_images"  # Thay đổi thành thư mục đầu ra bạn muốn
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.chuyen_doi["state"] = "normal"  # Kích hoạt nút chuyển đổi

    def chuyen_video_to_anh(self):
        if not hasattr(self, 'video_path'):
            messagebox.showinfo("Error", "Loi chon video.")
            return

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_path = os.path.join(self.output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(image_path, frame)
            frame_count += 1

        cap.release()
        message = f"Video converted to {frame_count} images in {self.output_folder}"
        messagebox.showinfo("Success", message)
        self.chon_anh1["state"] = "normal"

    def select_image1(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if image_path:
            self.selected_images.append(image_path)
            if len(self.selected_images) == 1:
                self.so_sanh_btn["state"] = "normal"
            self.chon_anh1["state"] = "disabled"

    def compare_images(self):
        if len(self.selected_images) != 1:
            messagebox.showinfo("Error", "Vui long chon anh.")
            return

        image1_path = self.selected_images[0]
        # Tính toán LBP và so sánh hình ảnh ở đây
        danhSach = danhSachHinhAnh(duongDanDanhSachHinhAnh)
        coSoDuLieu = taoCoSoDuLieu(danhSach)
        anhCanTim = cv2.imread(image1_path)
        hist = tinhLBP(anhCanTim)

        # Tìm hình ảnh tương tự
        mangDoTuongTu = timKiemTuongTu(hist, coSoDuLieu)
        doTuongTuSapXep = np.argsort(mangDoTuongTu)[::-1]

        # Lấy ra N hình ảnh tương tự hàng đầu

        nhungHinhTuongTuNhat = doTuongTuSapXep[:soLuongAnhHienThi]

        # Hiển thị hình ảnh tương tự từ danh sách đã sắp xếp
        for i, index in enumerate(nhungHinhTuongTuNhat):
            img = danhSach[index]

            # Chuyển đổi hình ảnh từ OpenCV sang định dạng có thể hiển thị bởi Tkinter
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)


            for i, image_path in enumerate(self.selected_images[:soLuongAnhHienThi]):
                img = cv2.imread(image_path)
                label = self.image_labels[i]
                show_image(label, img)

            # Đợi một khoảng thời gian trước khi hiển thị hình ảnh tiếp theo
            self.update_idletasks()
            self.after(500)

root = tk.Tk()
app = Application(master=root)
app.master.title("Video to Image Converter")
app.master.minsize(800, 600)
app.mainloop()
