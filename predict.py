import pandas as pd
import joblib
from utils import THANG_DO

def du_bao_con_bo_moi():
    print("=== TẢI MÔ HÌNH VÀ THỰC HIỆN DỰ BÁO CHO KHẨU PHẦN BÒ MỚI ===")
    
    try:
        mo_hinh_da_luu = joblib.load("ch4_models.joblib")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'ch4_models.joblib'. Hãy chạy file 'train.py' trước để huấn luyện mô hình!")
        return

    # Thông số đầu vào khẩu phần ăn của cá thể bò mới
    mau_bo_moi = pd.DataFrame([dict(
        Luong_an_DMI=18.5, 
        Xo_NDF=37.0, 
        Protein_CP=16.2, 
        Beo_EE=3.4, 
        Tinh_bot_starch=23.0, 
        Khoi_luong_BW=560.0, 
        San_luong_sua=26.0
    )])

    print("\nThông số khẩu phần ăn và thể trạng đầu vào:")
    print(mau_bo_moi.to_string(index=False))
    print("\nKết quả dự báo phát thải methane trên 4 thang đo:")
    print("-" * 55)

    for thang_do, thong_tin in mo_hinh_da_luu.items():
        gia_tri_du_doan = thong_tin["model"].predict(mau_bo_moi[thong_tin["dac_trung"]])[0]
        print(f"Thang đo {thang_do:10s} : {gia_tri_du_doan:8.2f} {THANG_DO[thang_do]['nhãn']}")

if __name__ == "__main__":
    du_bao_con_bo_moi()