import cv2
import os
import numpy as np
from skimage import feature
from sklearn.metrics import pairwise_distances


duongDanDanhSachHinhAnh = "./output_images"
duongDanAnhCanTim = "test1.png"

def tinhLBP(image):
    anhXam = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    banKinh = 1
    soDiem = 8 * banKinh

    lbp = feature.local_binary_pattern(anhXam, soDiem, banKinh, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, soDiem + 3), range=(0, soDiem + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def danhSachHinhAnh(folder):
    danhSachHinhAnh = []
    for duongDanGoc, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # Kiểm tra các định dạng hình ảnh phổ biến
                duongDanAnh = os.path.join(duongDanGoc, file)
                docAnh = cv2.imread(duongDanAnh)
                danhSachHinhAnh.append(docAnh)
    return danhSachHinhAnh

def taoCoSoDuLieu(danhSachHInhAnh):
    taoCoSoDuLieu = []
    for img in danhSachHInhAnh:
        hist = tinhLBP(img)
        taoCoSoDuLieu.append(hist)
    return taoCoSoDuLieu

def timKiemTuongTu(anhTim, coSoDuLieu):
    similarities = []
    for hist in coSoDuLieu:
        similarity = cosine_compare(hist, anhTim)
        similarities.append(similarity)
    return similarities

def cosine_compare(vector1, vector2):
    dot_product = sum(vector1[i] * vector2[i] for i in range(len(vector1)))
    norm1 = sum(val * 2 for val in vector1) * 0.5
    norm2 = sum(val * 2 for val in vector2) * 0.5
    similarity = dot_product / (norm1 * norm2)
    return similarity

def main():
    danhSach = danhSachHinhAnh(duongDanDanhSachHinhAnh)
    coSoDuLieu = taoCoSoDuLieu(danhSach)
    anhCanTim = cv2.imread(duongDanAnhCanTim)
    hist = tinhLBP(anhCanTim)

    # Tìm hình ảnh tương tự
    mangDoTuongTu = timKiemTuongTu(hist, coSoDuLieu)
    doTuongTuSapXep = np.argsort(mangDoTuongTu)[::-1]

    # Lấy ra N hình ảnh tương tự hàng đầu
    soLuongHinhAnhLayRa = 3
    nhungHinhTuongTuNhat = doTuongTuSapXep[:soLuongHinhAnhLayRa]

    # Hiển thị hình ảnh tương tự từ danh sách đã sắp xếp
    for i, index in enumerate(nhungHinhTuongTuNhat):
        print(index)
        img = danhSach[index]
        cv2.imshow(f'Hinh anh tuong tu {i+1}', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()