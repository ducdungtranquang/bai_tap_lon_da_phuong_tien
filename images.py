import cv2
import os
import numpy as np
from skimage import feature
from sklearn.metrics import pairwise_distances

def tinhLBP(image):
    anh_xam = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ban_kinh = 1
    so_diem = 8 * ban_kinh

    lbp = feature.local_binary_pattern(anh_xam, so_diem, ban_kinh, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, so_diem + 3), range=(0, so_diem + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def danhSachHinhAnh(folder):
    danh_sach_hinh_anh = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # Kiểm tra các định dạng hình ảnh phổ biến
                img_path = os.path.join(root, file)
                docAnh = cv2.imread(img_path)
                danh_sach_hinh_anh.append(docAnh)
    return danh_sach_hinh_anh

# Bước 1: Tạo cơ sở dữ liệu của các histogram LBP
def tao_co_so_du_lieu_LBP(danh_sach_hinh_anh):
    co_so_du_lieu = []
    for img in danh_sach_hinh_anh:
        hist = tinhLBP(img)
        co_so_du_lieu.append(hist)
    return co_so_du_lieu

# Bước 2: Tìm khoảng cách giữa các histogram
def tim_kiem_anh_tuong_tu(query_hist, co_so_du_lieu):
    khoang_cach = pairwise_distances(np.array(co_so_du_lieu), query_hist.reshape(1, -1), metric='euclidean')
    return khoang_cach

# Bước 3: Sử dụng chức năng tìm kiếm
def main():
    # Giả sử danh_sach_hinh_anh là danh sách các hình ảnh trong cơ sở dữ liệu của bạn
    danh_sach_hinh_anh = danhSachHinhAnh("./output_images")
    # Tạo cơ sở dữ liệu histogram LBP
    co_so_du_lieu = tao_co_so_du_lieu_LBP(danh_sach_hinh_anh)
    
    # Giả sử query_img là hình ảnh bạn muốn tìm hình ảnh tương tự
    query_img = cv2.imread("test1.png")
    hist = tinhLBP(query_img)

    # Tìm hình ảnh tương tự
    tuong_duong = tim_kiem_anh_tuong_tu(hist, co_so_du_lieu)
    indices_sap_xep = np.argsort(tuong_duong, axis=0)
    indices_sap_xep = indices_sap_xep.flatten().astype(int)

    # Lấy ra N hình ảnh tương tự hàng đầu
    so_luong_top = 5  # Thay đổi giá trị này để lấy ra số lượng hình ảnh tương tự khác nhau
    # Lấy ra chỉ số của top N hình ảnh tương tự
    top_N_indices = indices_sap_xep[1:so_luong_top].flatten()

    # Hiển thị hình ảnh tương tự từ danh sách đã sắp xếp
    for i, index in enumerate(top_N_indices):
        img = danh_sach_hinh_anh[index+1]
        cv2.imshow(f'Hinh anh tuong tu {i+1}', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()