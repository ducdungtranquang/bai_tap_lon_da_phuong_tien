import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import os

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
        self.label = tk.Label(text="Video to Image Converter")
        self.label.pack()

        self.select_video_button = tk.Button(self)
        self.select_video_button["text"] = "Select Video File"
        self.select_video_button["command"] = self.select_video
        self.select_video_button.pack()

        self.convert_button = tk.Button(self)
        self.convert_button["text"] = "Convert Video to Images"
        self.convert_button["state"] = "disabled"
        self.convert_button["command"] = self.convert_video_to_images
        self.convert_button.pack()

        self.select_image1_button = tk.Button(self)
        self.select_image1_button["text"] = "Select Image 1"
        self.select_image1_button["state"] = "disabled"
        self.select_image1_button["command"] = self.select_image1
        self.select_image1_button.pack()

        self.select_image2_button = tk.Button(self)
        self.select_image2_button["text"] = "Select Image 2"
        self.select_image2_button["state"] = "disabled"
        self.select_image2_button["command"] = self.select_image2
        self.select_image2_button.pack()

        self.compare_button = tk.Button(self)
        self.compare_button["text"] = "Compare Images"
        self.compare_button["state"] = "disabled"
        self.compare_button["command"] = self.compare_images
        self.compare_button.pack()

        self.quit = tk.Button(self, text="QUIT", command=root.destroy)
        self.quit.pack()

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.video_path = file_path
            self.output_folder = "output_images"  # Thay đổi thành thư mục đầu ra bạn muốn
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.convert_button["state"] = "normal"  # Kích hoạt nút chuyển đổi

    def convert_video_to_images(self):
        if not hasattr(self, 'video_path'):
            messagebox.showinfo("Error", "Please select a video file first.")
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
        self.select_image1_button["state"] = "normal"
        self.select_image2_button["state"] = "normal"

    def select_image1(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if image_path:
            self.selected_images.append(image_path)
            if len(self.selected_images) == 2:
                self.compare_button["state"] = "normal"
            self.select_image1_button["state"] = "disabled"

    def select_image2(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if image_path:
            self.selected_images.append(image_path)
            if len(self.selected_images) == 2:
                self.compare_button["state"] = "normal"
            self.select_image2_button["state"] = "disabled"

    def compare_images(self):
        if len(self.selected_images) != 2:
            messagebox.showinfo("Error", "Please select two images for comparison.")
            return

        image1_path, image2_path = self.selected_images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        
        if self.compare_images_similarity(image1, image2):
            messagebox.showinfo("Image Comparison", "Images are similar.")
        else:
            messagebox.showinfo("Image Comparison", "Images are different.")

    def compare_images_similarity(self, image1, image2):
        # Add your image comparison logic here (e.g., using OpenCV's image processing functions)
        # Return True if images are similar, False if they are different.
        # For a simple example, you can compare the average pixel value difference.
        diff = cv2.absdiff(image1, image2)
        avg_diff = diff.mean()
        return avg_diff < 30  # You can adjust this threshold as needed

root = tk.Tk()
app = Application(master=root)
app.master.title("Video to Image Converter")
app.master.minsize(800, 600)
app.mainloop()
