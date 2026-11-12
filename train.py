import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import tai_du_lieu, them_thang_do, THANG_DO, danh_gia_ipcc_co_ban, tinh_chi_so_danh_gia

def lay_danh_sach_mo_hinh():
    return {
        "Hồi quy Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "Rừng ngẫu nhiên (Random Forest)": RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Tăng cường độ dốc (Gradient Boosting)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH AI ===")
    df = them_thang_do(tai_du_lieu())
    mo_hinh_da_luu = {}

    for thang_do, cau_hinh in THANG_DO.items():
        print(f"\n==================================================")
        print(f" ĐANG ĐÁNH GIÁ THANG ĐO: {thang_do} ({cau_hinh['nhãn']})")
        print(f"==================================================")
        
        X = df[cau_hinh["dac_trung"]]
        y = df[thang_do].values
        nhom = df.ma_nghien_cuu
        
        ket_qua_ipcc = danh_gia_ipcc_co_ban(thang_do, df)
        print(f"-> Mức chuẩn cơ sở IPCC Tier 2: R2 = {ket_qua_ipcc['R2']}, Sai số RMSE = {ket_qua_ipcc['RMSE']}")

        danh_sach_ket_qua = []
        r2_tot_nhat = -999
        mo_hinh_tot_nhat = None
        ten_model_tot = ""

        for ten_model, model in lay_danh_sach_mo_hinh().items():
            du_doan = np.zeros(len(df))
            gkf = GroupKFold(n_splits=5)
            for tr, te in gkf.split(X, y, nhom):
                model.fit(X.iloc[tr], y[tr])
                du_doan[te] = model.predict(X.iloc[te])
            
            chi_so = tinh_chi_so_danh_gia(y, du_doan)
            danh_sach_ket_qua.append({"Mô_hình": ten_model, **chi_so})
            
            if chi_so["R2"] > r2_tot_nhat:
                r2_tot_nhat = chi_so["R2"]
                mo_hinh_tot_nhat = model
                ten_model_tot = ten_model

        df_kq = pd.DataFrame(danh_sach_ket_qua).set_index("Mô_hình")
        print(df_kq.to_string())
        print(f"🏆 Mô hình tối ưu nhất cho thang đo '{thang_do}': {ten_model_tot} với hệ số R2 = {r2_tot_nhat}")

        mo_hinh_tot_nhat.fit(X, y)
        mo_hinh_da_luu[thang_do] = {
            "model": mo_hinh_tot_nhat,
            "dac_trung": cau_hinh["dac_trung"],
            "ten_mo_hinh": ten_model_tot
        }

    ten_file_luu = "ch4_models.joblib"
    joblib.dump(mo_hinh_da_luu, ten_file_luu)
    print(f"\n[HOÀN TẤT] Đã huấn luyện và lưu tất cả mô hình thành công vào file: '{ten_file_luu}'")

if __name__ == "__main__":
    main()