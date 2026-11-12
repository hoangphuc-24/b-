import numpy as np
import pandas as pd

TONG_NL_GE = 18.45      # MJ/kg DM (Năng lượng tổng chuẩn IPCC)
MJ_TREN_G = 0.05565     # MJ/g CH4
YM_IPCC = 6.5           # % (Ym tham chiếu baseline)

# Các biến đặc trưng khẩu phần ăn và thể trạng (đã đổi sang tiếng Việt để dễ báo cáo)
DANH_SACH_DAC_TRUNG = ["Xo_NDF", "Protein_CP", "Beo_EE", "Tinh_bot_starch", "Khoi_luong_BW", "San_luong_sua"]

# 4 Thang đo dự báo CH4
THANG_DO = {
    "CH4":       dict(nhãn="g CH4/con/ngày",dac_trung=["Luong_an_DMI"] + DANH_SACH_DAC_TRUNG),
    "yield":     dict(nhãn="g CH4/kg DMI",   dac_trung=DANH_SACH_DAC_TRUNG),
    "Ym":        dict(nhãn="% năng lượng GE",dac_trung=DANH_SACH_DAC_TRUNG),
    "intensity": dict(nhãn="g CH4/kg sữa",   dac_trung=["Luong_an_DMI"] + DANH_SACH_DAC_TRUNG),
}

def tai_du_lieu(n=800, seed=42):
    """Hàm tạo dữ liệu mô phỏng với tên cột bằng tiếng Việt"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ma_nghien_cuu": rng.integers(0, 35, n),
        "Luong_an_DMI": rng.normal(17.5, 2.8, n).clip(8, 28),
        "Xo_NDF": rng.normal(36, 5.5, n).clip(20, 55),
        "Protein_CP": rng.normal(15.8, 1.8, n).clip(10, 22),
        "Beo_EE": rng.normal(3.5, 1.1, n).clip(1, 7),
        "Tinh_bot_starch": rng.normal(24, 5.5, n).clip(8, 40),
        "Khoi_luong_BW": rng.normal(550, 55, n).clip(380, 720),
        "San_luong_sua": rng.normal(24, 6.5, n).clip(5, 45),
    })
    hieu_ung_nc = pd.Series(rng.normal(0, 1.5, 35))[df.ma_nghien_cuu].values
    ch4_goc = (df.Luong_an_DMI * 21.5 * (0.067 - 0.008 * (df.Tinh_bot_starch/25) + 0.005 * ((df.Xo_NDF-35)/35))) * 1000 / 55.65
    df["CH4"] = (ch4_goc + hieu_ung_nc * 5 + rng.normal(0, 12, n)).clip(100, 500)
    return df

def them_thang_do(df):
    """Tính toán 3 thang đo còn lại từ tổng lượng CH4 (g/ngày)"""
    df = df.copy()
    df["yield"] = df.CH4 / df.Luong_an_DMI
    df["Ym"] = df.CH4 * MJ_TREN_G / (df.Luong_an_DMI * TONG_NL_GE) * 100
    df["intensity"] = df.CH4 / df.San_luong_sua
    return df

def ham_ccc(y, p):
    my, mp = y.mean(), p.mean()
    cov = np.mean((y - my) * (p - mp))
    return 2 * cov / (y.var() + p.var() + (my - mp) ** 2)

def tinh_chi_so_danh_gia(y, p):
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "R2": round(r2, 3), 
        "RMSE": round(rmse, 2), 
        "RMSE_%": round(100 * rmse / y.mean(), 2),
        "CCC": round(ham_ccc(y, p), 3), 
        "Sai_so_Bias": round(np.mean(p - y), 2)
    }

def danh_gia_ipcc_co_ban(thang_do_chon, d):
    gday = d.Luong_an_DMI.values * TONG_NL_GE * (YM_IPCC / 100) / MJ_TREN_G
    if thang_do_chon == "CH4": p = gday
    elif thang_do_chon == "yield": p = np.full(len(d), TONG_NL_GE * (YM_IPCC / 100) / MJ_TREN_G)
    elif thang_do_chon == "Ym": p = np.full(len(d), YM_IPCC)
    elif thang_do_chon == "intensity": p = gday / d.San_luong_sua.values
    else: raise ValueError(thang_do_chon)
    return tinh_chi_so_danh_gia(d[thang_do_chon].values, p)