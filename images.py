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

def timKienTuongTu(anhTim, coSoDuLieu):
    khoangCach = pairwise_distances(np.array(coSoDuLieu), anhTim.reshape(1, -1), metric='euclidean')
    return khoangCach

def main():
    coSoDuLieu = taoCoSoDuLieu(danhSachHinhAnh(duongDanDanhSachHinhAnh))
    anhCanTim = cv2.imread(duongDanAnhCanTim)
    hist = tinhLBP(anhCanTim)

    # Tìm hình ảnh tương tự
    mangDoTuongTu = timKienTuongTu(hist, coSoDuLieu)
    doTuongTuSapXep = np.short(mangDoTuongTu, reverse=True).flatten().astype(int)

    # Lấy ra N hình ảnh tương tự hàng đầu
    soLuongAnhLayRa = 3  # Thay đổi giá trị này để lấy ra số lượng hình ảnh tương tự khác nhau
    # Lấy ra chỉ số của top N hình ảnh tương tự
    nhungDoTuongTuLonNhat = doTuongTuSapXep[:soLuongAnhLayRa].flatten()

    # Hiển thị hình ảnh tương tự từ danh sách đã sắp xếp
    for i, index in enumerate(nhungDoTuongTuLonNhat):
        img = danhSachHinhAnh[index+1]
        cv2.imshow(f'Hinh anh tuong tu {i+1}', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()