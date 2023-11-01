import cv2
import os
import numpy as np
from skimage import feature

duong_dan_folder = "./output_images"
duong_dan_and_can_tim = "test1.png"


def tinh_LBP(image):
    anh_xam = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    banKinh = 1
    soDiem = 8 * banKinh

    lbp = feature.local_binary_pattern(anh_xam, soDiem, banKinh)
    hist, _ = np.histogram(lbp.ravel())
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# doc anh trong folder
def danh_sach_hinh_anh_trong_folder(folder):
    danh_sach_hinh_anh_trong_folder = []
    for duong_dan_goc, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # Kiểm tra các định dạng hình ảnh phổ biến
                duong_dan_anh = os.path.join(duong_dan_goc, file)
                doc_anh = cv2.imread(duong_dan_anh)
                danh_sach_hinh_anh_trong_folder.append(doc_anh)
    return danh_sach_hinh_anh_trong_folder

# co so du lieu gom mang cac gia tri diem anh
def tao_co_so_du_lieu(danh_sach_hinh_anh_trong_folder):
    co_so_du_lieu = []
    for img in danh_sach_hinh_anh_trong_folder:
        hist = tinh_LBP(img)
        print(hist)
        print("=")
        co_so_du_lieu.append(hist)
    return co_so_du_lieu

# tinh do tuong tu dua tren cosine
def tinh_do_tuong_tu_trong_folder(anhTim, coSoDuLieu):
    mang_ket_qua_theo_cosine = []
    for hist in coSoDuLieu:
        ketquaTheoCosine = tinh_cosine(hist, anhTim)
        mang_ket_qua_theo_cosine.append(ketquaTheoCosine)
    return mang_ket_qua_theo_cosine

def tinh_cosine(vector1, vector2):
    dot_product = sum(vector1[i] * vector2[i] for i in range(len(vector1)))
    norm1 = sum(val * 2 for val in vector1) * 0.5
    norm2 = sum(val * 2 for val in vector2) * 0.5
    similarity = dot_product / (norm1 * norm2)
    return similarity

def main():
    danhSach = danh_sach_hinh_anh_trong_folder(duong_dan_folder)
    coSoDuLieu = tao_co_so_du_lieu(danhSach)
    anhCanTim = cv2.imread(duong_dan_and_can_tim)
    hist = tinh_LBP(anhCanTim)

    # Tìm hình ảnh tương tự
    mangDoTuongTu = tinh_do_tuong_tu_trong_folder(hist, coSoDuLieu)
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