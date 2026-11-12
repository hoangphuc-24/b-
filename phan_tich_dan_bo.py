import pandas as pd
import joblib
from utils import THANG_DO

def goi_y_dieu_chinh_khau_phan(row):
    """
    Hàm phân tích thông số dinh dưỡng của từng con bò chưa đạt chuẩn
    để tự động đưa ra khuyến nghị điều chỉnh khẩu phần ăn thực tế.
    """
    if row["Trạng_thái"] == "Đạt chuẩn (Phát thải thấp)":
        return "Duy trì khẩu phần ổn định hiện tại"
    
    # Các quy tắc chuyên gia dinh dưỡng thú y:
    khuyen_nghi = []
    
    if row["Xo_NDF"] > 40.0:
        khuyen_nghi.append("Giảm bớt tỷ lệ xơ NDF (thô xanh già), tăng cường thức ăn dễ tiêu hóa")
        
    if row["Tinh_bot_starch"] < 20.0:
        khuyen_nghi.append("Tăng thêm tỷ lệ tinh bột (thức ăn tinh/cám) để tối ưu hóa lên men dạ cỏ")
        
    if row["Beo_EE"] < 3.0:
        khuyen_nghi.append("Bổ sung thêm chất béo (EE) để ức chế quá trình sinh khí methane")
        
    if not khuyen_nghi:
        khuyen_nghi.append("Cân đối lại tổng lượng ăn DMI hoặc xem xét lại thể trạng cá thể")
        
    return "; ".join(khuyen_nghi)

def phan_tich_va_phan_nhom_dan():
    print("=== BẮT ĐẦU PHÂN TÍCH, PHÂN LOẠI VÀ ĐỀ XUẤT KHẨU PHẦN ĐÀN BÒ ===")
    
    try:
        mo_hinh_da_luu = joblib.load("ch4_models.joblib")
    except FileNotFoundError:
        print("Lỗi: Hãy chạy 'train.py' trước để tạo file mô hình!")
        return

    # Dữ liệu danh sách đàn bò (hoặc thay bằng pd.read_csv('danh_sach_dan_bo.csv'))
    du_lieu_dan_bo = pd.DataFrame({
        "ID_Bo": [f"BO_{i:03d}" for i in range(1, 11)],
        "Luong_an_DMI": [16.5, 19.0, 15.0, 21.0, 17.5, 18.0, 22.5, 16.0, 18.5, 20.0],
        "Xo_NDF": [35.0, 42.0, 32.0, 45.0, 36.0, 37.0, 46.0, 34.0, 38.0, 40.0],
        "Protein_CP": [16.0, 15.0, 17.0, 14.5, 16.2, 15.8, 14.0, 16.5, 15.5, 15.2],
        "Beo_EE": [3.2, 3.0, 3.5, 2.8, 3.4, 3.3, 2.7, 3.6, 3.1, 2.9],
        "Tinh_bot_starch": [25.0, 20.0, 28.0, 18.0, 23.0, 24.0, 17.0, 26.0, 22.0, 21.0],
        "Khoi_luong_BW": [520, 580, 500, 610, 550, 560, 630, 510, 540, 590],
        "San_luong_sua": [24.0, 22.0, 25.0, 20.0, 26.0, 24.5, 19.0, 25.5, 23.0, 21.0]
    })

    # Chạy dự báo lượng CH4
    model_ch4 = mo_hinh_da_luu["CH4"]["model"]
    dac_trung_ch4 = mo_hinh_da_luu["CH4"]["dac_trung"]
    du_lieu_dan_bo["CH4_Du_Bao"] = model_ch4.predict(du_lieu_dan_bo[dac_trung_ch4])

    # Phân nhóm đạt / chưa đạt chuẩn dựa trên mức trung bình của đàn
    nguong_tieu_chuẩn = du_lieu_dan_bo["CH4_Du_Bao"].mean()
    
    def phan_nhom(gia_tri_ch4):
        if gia_tri_ch4 <= nguong_tieu_chuẩn:
            return "Đạt chuẩn (Phát thải thấp)"
        else:
            return "Chưa đạt chuẩn (Phát thải cao, cần điều chỉnh)"

    du_lieu_dan_bo["Trạng_thái"] = du_lieu_dan_bo["CH4_Du_Bao"].apply(phan_nhom)

    # Tự động sinh ra khuyến nghị điều chỉnh khẩu phần cho từng con
    du_lieu_dan_bo["Đề_xuất_khẩu_phần"] = du_lieu_dan_bo.apply(goi_y_dieu_chinh_khau_phan, axis=1)

    # Làm tròn số liệu CH4 cho dễ đọc
    du_lieu_dan_bo["CH4_Du_Bao"] = du_lieu_dan_bo["CH4_Du_Bao"].round(2)

    # Xuất kết quả ra file Excel
    ten_file_xuat = "ket_qua_phan_tich_va_dieu_chinh_dan_bo.xlsx"
    try:
        du_lieu_dan_bo.to_excel(ten_file_xuat, index=False)
        print(f"\n[THÀNH CÔNG] Đã xuất báo cáo kèm giải pháp điều chỉnh ra file: '{ten_file_xuat}'")
    except Exception:
        ten_file_xuat = "ket_qua_phan_tich_va_dieu_chinh_dan_bo.csv"
        du_lieu_dan_bo.to_csv(ten_file_xuat, index=False, encoding="utf-8-sig")
        print(f"\n[THÀNH CÔNG] Đã xuất báo cáo ra file CSV: '{ten_file_xuat}'")

    print("\n--- BẢNG TÓM TẮT KẾT QUẢ VÀ ĐỀ XUẤT TRÊN TERMINAL ---")
    print(du_lieu_dan_bo[["ID_Bo", "CH4_Du_Bao", "Trạng_thái", "Đề_xuất_khẩu_phần"]].to_string(index=False))

if __name__ == "__main__":
    phan_tich_va_phan_nhom_dan()