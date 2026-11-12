import joblib

def kiem_tra_file_model():
    print("=== ĐANG ĐỌC NỘI DUNG FILE MODEL ===")
    try:
        mo_hinh_da_luu = joblib.load("ch4_models.joblib")
    except FileNotFoundError:
        print("Không tìm thấy file mô hình!")
        return

    for thang_do, thong_tin in mo_hinh_da_luu.items():
        print(f"- Thang đo: {thang_do}")
        print(f"  + Thuật toán tốt nhất: {thong_tin['ten_mo_hinh']}")
        print(f"  + Danh sách đặc trưng sử dụng: {thong_tin['dac_trung']}")
        print("-" * 50)

if __name__ == "__main__":
    kiem_tra_file_model()