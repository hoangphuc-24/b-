import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# CẤU HÌNH TRANG & GIAO DIỆN HIỆN ĐẠI
# ---------------------------------------------------------
st.set_page_config(
    page_title="EcoDairy - Quản lý Phát thải CH4",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

GE_DMI_MAC_DINH = 18.45      
CH4_ENERGY_DENSITY = 55.65   
YM_CHUAN_IPCC = 6.5          

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 800; color: #0F172A; margin-bottom: 0rem; letter-spacing: -0.5px; }
    .sub-title { font-size: 1rem; color: #64748B; margin-bottom: 1.5rem; }
    .custom-alert-danger {
        background-color: #FEF2F2; border: 1px solid #FCA5A5; border-left: 5px solid #EF4444;
        padding: 14px; border-radius: 8px; color: #991B1B; font-size: 0.95rem; font-weight: 500;
    }
    .custom-alert-success {
        background-color: #F0FDF4; border: 1px solid #86EFAC; border-left: 5px solid #22C55E;
        padding: 14px; border-radius: 8px; color: #166534; font-size: 0.95rem; font-weight: 500;
    }
    .info-box {
        background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 5px solid #3B82F6;
        padding: 14px; border-radius: 8px; color: #1E40AF; font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

def tinh_ch4_ipcc_chuan(dmi, ge=18.45, ym=6.5):
    ch4_kg = (dmi * ge * (ym / 100.0)) / CH4_ENERGY_DENSITY
    return ch4_kg * 1000.0

@st.cache_resource
def load_trained_models(path="ch4_models.joblib"):
    try:
        data = joblib.load(path)
        return data
    except Exception as e:
        return None

models_dict = load_trained_models()

def xac_dinh_giai_doan(thang_tuoi, bw):
    if thang_tuoi <= 15 and bw <= 350:
        return {
            "ten": "Bò tơ / Giai đoạn nuôi lớn",
            "ch4_chuan_min": 80.0,
            "ch4_chuan_max": 140.0,
            "mo_ta": "Dạ cỏ đang phát triển ổn định, lượng thức ăn thu nhận vừa phải."
        }
    elif thang_tuoi > 15 and bw <= 450:
        return {
            "ten": "Bò thịt vỗ béo / Bò cạn sữa",
            "ch4_chuan_min": 150.0,
            "ch4_chuan_max": 250.0,
            "mo_ta": "Hệ vi sinh vật dạ cỏ cân bằng, tiêu hóa tốt, tối ưu năng lượng."
        }
    else:
        return {
            "ten": "Bò sữa đang khai thác (Cao sản)",
            "ch4_chuan_min": 260.0,
            "ch4_chuan_max": 420.0,
            "mo_ta": "Thu nhận vật chất khô lớn (15-22 kg/ngày) phục vụ tổng hợp sữa."
        }

with st.sidebar:
    st.markdown("### 🐄 **EcoDairy AI**")
    st.caption("Đồ án Tốt nghiệp - Quản lý Phát thải Chăn nuôi")
    st.markdown("---")
    if models_dict is not None:
        st.success("✨ Đã kết nối mô hình AI thành công")
        thang_do_chon = st.selectbox("Chọn thang đo huấn luyện:", list(models_dict.keys()))
    else:
        st.info("ℹ️ Đang chạy theo công thức chuẩn IPCC Tier 2")
        thang_do_chon = None

    with st.expander("⚙️ Tham số chuẩn IPCC", expanded=False):
        ym_ipcc = st.number_input("Hệ số Ym chuẩn (% GE)", value=float(YM_CHUAN_IPCC), step=0.1)
        ge_dmi = st.number_input("Năng lượng thô GE (MJ/kg)", value=float(GE_DMI_MAC_DINH), step=0.1)
    
    st.markdown("---")
    st.markdown("📌 **Hướng dẫn:** Sử dụng Tab 1 để phân tích cá thể hoặc Tab 2 để quản lý toàn trại.")

st.markdown('<div class="main-title">Hệ Thống Dự Báo & Quản Lý Phát Thải CH₄ Bò Sữa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng trí tuệ nhân tạo và mô hình IPCC Tier 2 nhằm tối ưu hóa dinh dưỡng và giảm phát thải khí nhà kính.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Dự Báo & Phân Loại Cá Thể", "📊 Quản Lý Đàn Bò Toàn Trại"])

# =========================================================
# TAB 1: DỰ BÁO CÁ THỂ
# =========================================================
with tab1:
    col_input, col_result = st.columns([1, 1.5], gap="large")
    
    with col_input:
        st.markdown("#### 📝 Thông Số Đầu Vào")
        with st.form("form_ca_the_pro"):
            st.caption("1. Độ tuổi & Thể trạng cá thể")
            c_tuoi, c_bw = st.columns(2)
            thang_tuoi = c_tuoi.number_input("Tháng tuổi", min_value=6, max_value=120, value=24, step=1)
            bw = c_bw.number_input("Khối lượng BW (kg)", min_value=200.0, max_value=900.0, value=550.0, step=10.0)
            
            milk = st.number_input("Sản lượng sữa (kg/ngày)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
            
            st.caption("2. Thành phần khẩu phần ăn (% chất khô - DM)")
            dmi = st.slider("Lượng ăn vào - DMI (kg/ngày)", min_value=5.0, max_value=35.0, value=18.0, step=0.5)
            
            c3, c4 = st.columns(2)
            ndf = c3.number_input("Xơ trung tính NDF", min_value=15.0, max_value=65.0, value=34.0, step=0.5)
            cp = c4.number_input("Protein thô CP", min_value=8.0, max_value=25.0, value=16.5, step=0.5)
            
            c5, c6 = st.columns(2)
            ee = c5.number_input("Chất béo EE", min_value=1.0, max_value=10.0, value=4.0, step=0.2)
            starch = c6.number_input("Tinh bột Starch", min_value=5.0, max_value=45.0, value=24.0, step=0.5)
            
            forage = st.slider("Tỷ lệ thô xanh Forage (%)", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
            
            submitted = st.form_submit_button("🚀 Thực Hiện Phân Tích Chuyên Sâu", use_container_width=True, type="primary")

    with col_result:
        st.markdown("#### 📊 Kết Quả Đánh Giá & Giai Đoạn Sinh Trưởng")
        
        input_dict = {
            "DMI": dmi, "BW": bw, "Milk": milk,
            "NDF": ndf, "CP": cp, "EE": ee,
            "Starch": starch, "Forage": forage,
            "Thang_Tuoi": thang_tuoi
        }
        df_input = pd.DataFrame([input_dict])
        
        # Xử lý dự báo linh hoạt từ file joblib hoặc fallback IPCC
        if models_dict is not None and thang_do_chon in models_dict:
            try:
                model_info = models_dict[thang_do_chon]
                if isinstance(model_info, dict):
                    pipeline = model_info.get("model", model_info.get("pipeline"))
                    features = model_info.get("dac_trung", list(input_dict.keys()))
                else:
                    pipeline = model_info
                    features = list(input_dict.keys())

                df_pred_input = df_input.copy()
                for col in features:
                    if col not in df_pred_input.columns:
                        df_pred_input[col] = 0
                
                raw_pred = float(pipeline.predict(df_pred_input[features])[0])
                ch4_pred = raw_pred / 10.0 if raw_pred > 1500 else raw_pred
            except Exception:
                ym_thuc_te = ym_ipcc + (ndf - 34.0) * 0.05 - (ee - 4.0) * 0.35 - (starch - 24.0) * 0.03
                ch4_pred = tinh_ch4_ipcc_chuan(dmi, ge=ge_dmi, ym=max(3.5, min(9.5, ym_thuc_te)))
        else:
            ym_thuc_te = ym_ipcc + (ndf - 34.0) * 0.05 - (ee - 4.0) * 0.35 - (starch - 24.0) * 0.03
            ch4_pred = tinh_ch4_ipcc_chuan(dmi, ge=ge_dmi, ym=max(3.5, min(9.5, ym_thuc_te)))

        giai_doan_info = xac_dinh_giai_doan(thang_tuoi, bw)
        min_nhom = giai_doan_info["ch4_chuan_min"]
        max_nhom = giai_doan_info["ch4_chuan_max"]

        yield_val = ch4_pred / dmi if dmi > 0 else 0
        ym_val = ((ch4_pred / 1000.0) * CH4_ENERGY_DENSITY) / (dmi * ge_dmi) * 100 if dmi > 0 else 0
        intensity_val = ch4_pred / milk if milk > 0 else 0

        st.markdown(f"""
        <div class="info-box">
            🏷️ <strong>Phân loại giai đoạn:</strong> {giai_doan_info['ten']}<br>
            📌 <em>Đặc điểm sinh lý:</em> {giai_doan_info['mo_ta']}<br>
            🎯 <em>Khoảng phát thải chuẩn:</em> <strong>{min_nhom} – {max_nhom} g/ngày</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng CH₄ dự báo", f"{ch4_pred:.1f} g")
        m2.metric("CH₄ Yield", f"{yield_val:.2f}", "g/kg DMI")
        m3.metric("Tỷ lệ Ym", f"{ym_val:.2f}%")
        m4.metric("Cường độ", f"{intensity_val:.2f}" if milk > 0 else "N/A", "g/kg sữa")

        fig = go.Figure(data=[
            go.Bar(
                x=[ch4_pred, max_nhom],
                y=['Dự báo thực tế', 'Ngưỡng tối đa của nhóm'],
                orientation='h',
                marker_color=['#2563EB' if ch4_pred <= max_nhom else '#EF4444', '#94A3B8'],
                text=[f"{ch4_pred:.1f} g", f"{max_nhom} g"],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title=dict(text="So sánh phát thải thực tế với giới hạn của giai đoạn", font=dict(size=13)),
            xaxis_title="Gram CH₄ / ngày",
            height=200,
            margin=dict(l=10, r=10, t=35, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

        if ch4_pred > max_nhom:
            st.markdown(f"""
            <div class="custom-alert-danger">
                ⚠️ <strong>Cảnh báo vượt mức chuẩn:</strong> Lượng phát thải ({ch4_pred:.1f} g/ngày) cao hơn ngưỡng tối đa ({max_nhom} g/ngày) của nhóm. Cần tối ưu khẩu phần.
            </div>
            """, unsafe_allow_html=True)
        elif ch4_pred < min_nhom:
            st.markdown(f"""
            <div class="custom-alert-success">
                💡 <strong>Thấp hơn tiêu chuẩn:</strong> Lượng phát thải thấp hơn mức tối thiểu ({min_nhom} g/ngày). Cần đánh giá lại năng lượng cung cấp.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-alert-success">
                ✅ <strong>Đạt chuẩn sinh lý:</strong> Mức phát thải hoàn toàn nằm trong khung chuẩn tối ưu ({min_nhom} – {max_nhom} g/ngày).
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🤖 Đề xuất khẩu phần tối ưu bằng AI", expanded=True):
            if ch4_pred > max_nhom:
                st.markdown("⚠️ **Khẩu phần hiện tại đang khiến bò phát thải cao hơn chuẩn!** Gợi ý điều chỉnh từ hệ thống AI:")
                
                de_xuat_ndf = max(28.0, ndf - 4.0) if ndf > 36 else ndf
                de_xuat_ee = min(5.5, max(4.5, ee + 1.0)) if ee < 4.5 else ee
                de_xuat_starch = 25.0 if (starch < 22 or starch > 30) else starch
                de_xuat_forage = max(40.0, forage - 10.0) if forage > 58 else forage
                
                st.markdown(f"""
                * 🌾 **Giảm Xơ (NDF):** Khuyến nghị hạ từ **{ndf}%** xuống khoảng **{de_xuat_ndf:.1f}%** để hạn chế lên men sinh axetat.
                * 🥑 **Tăng Chất béo (EE):** Khuyến nghị nâng từ **{ee}%** lên mức tối ưu **{de_xuat_ee:.1f}%** (bổ sung dầu/hạt có dầu) để ức chế tự nhiên vi sinh vật sinh methane.
                * 🌽 **Cân đối Tinh bột (Starch):** Điều chỉnh về mức **{de_xuat_starch:.1f}%** giúp tăng cường tạo propionat có lợi.
                * 🌿 **Điều chỉnh Thô xanh (Forage):** Đưa tỷ lệ về mức **{de_xuat_forage:.1f}%**.
                """)
            else:
                st.markdown("✅ **Khẩu phần hiện tại đã đạt chuẩn!** Các thành phần dinh dưỡng đang hỗ trợ kiểm soát phát thải sinh lý tốt.")

# =========================================================
# TAB 2: QUẢN LÝ TOÀN ĐÀN
# =========================================================
with tab2:
    st.markdown("#### 📂 Quản Lý & Phân Loại Đàn Bò Toàn Trại Theo Giai Đoạn")
    
    col_up, col_dl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Tải lên file dữ liệu đàn bò (.csv hoặc .xlsx)", type=["csv", "xlsx"])
    with col_dl:
        sample_df = pd.DataFrame({
            "Ma_Bo": ["BO_01", "BO_02", "BO_03", "BO_04", "BO_05"],
            "Thang_Tuoi": [10, 20, 36, 14, 48],
            "BW": [280, 500, 620, 320, 650],
            "Milk": [0.0, 0.0, 28.0, 0.0, 32.0],
            "DMI": [5.5, 10.0, 19.0, 6.0, 21.0],
            "NDF": [40.0, 35.0, 32.0, 39.0, 30.0],
            "CP": [15.0, 14.0, 17.0, 15.5, 18.0],
            "EE": [3.0, 3.5, 4.5, 3.2, 5.0],
            "Starch": [18.0, 22.0, 26.0, 19.0, 28.0],
            "Forage": [65.0, 60.0, 50.0, 62.0, 45.0]
        })
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Tải file dữ liệu mẫu đàn",
            data=sample_df.to_csv(index=False).encode('utf-8-sig'),
            file_name="mau_du_lieu_dan_bo.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_file is not None:
        df_herd = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    else:
        df_herd = sample_df.copy()

    if models_dict is not None and thang_do_chon in models_dict:
        try:
            model_info = models_dict[thang_do_chon]
            if isinstance(model_info, dict):
                pipeline = model_info.get("model", model_info.get("pipeline"))
                features = model_info.get("dac_trung", list(df_herd.columns))
            else:
                pipeline = model_info
                features = list(df_herd.columns)

            df_pred_input = df_herd.copy()
            for col in features:
                if col not in df_pred_input.columns:
                    df_pred_input[col] = 0
            raw_preds = pipeline.predict(df_pred_input[features])
            df_herd["CH4_Du_Bao (g/ngay)"] = [val / 10.0 if val > 1500 else val for val in raw_preds]
        except Exception:
            ym_series = ym_ipcc + (df_herd["NDF"] - 34.0) * 0.05 - (df_herd["EE"] - 4.0) * 0.35 - (df_herd["Starch"] - 24.0) * 0.03
            df_herd["CH4_Du_Bao (g/ngay)"] = tinh_ch4_ipcc_chuan(df_herd["DMI"], ge=ge_dmi, ym=ym_series.clip(3.5, 9.5))
    else:
        ym_series = ym_ipcc + (df_herd["NDF"] - 34.0) * 0.05 - (df_herd["EE"] - 4.0) * 0.35 - (df_herd["Starch"] - 24.0) * 0.03
        df_herd["CH4_Du_Bao (g/ngay)"] = tinh_ch4_ipcc_chuan(df_herd["DMI"], ge=ge_dmi, ym=ym_series.clip(3.5, 9.5))

    ds_giai_doan = []
    ds_trang_thai = []
    for idx, row in df_herd.iterrows():
        t = row.get("Thang_Tuoi", 24)
        w = row.get("BW", 500)
        ch4 = row["CH4_Du_Bao (g/ngay)"]
        
        info = xac_dinh_giai_doan(t, w)
        ds_giai_doan.append(info["ten"])
        
        if ch4 > info["ch4_chuan_max"]:
            ds_trang_thai.append("🔴 Vượt chuẩn nhóm")
        elif ch4 < info["ch4_chuan_min"]:
            ds_trang_thai.append("🟡 Thấp hơn chuẩn")
        else:
            ds_trang_thai.append("🟢 Đạt chuẩn")

    df_herd["Giai_Doan"] = ds_giai_doan
    df_herd["Trang_Thai"] = ds_trang_thai

    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Quy mô đàn", f"{len(df_herd)} con")
    s2.metric("Tổng phát thải đàn", f"{df_herd['CH4_Du_Bao (g/ngay)'].sum() / 1000:.2f} kg/ngày")
    s3.metric("Trung bình / con", f"{df_herd['CH4_Du_Bao (g/ngay)'].mean():.1f} g/ngày")
    so_vuot = len(df_herd[df_herd["Trang_Thai"].str.contains("Vượt")])
    s4.metric("Cần can thiệp", f"{so_vuot} con", f"{(so_vuot/len(df_herd)*100):.1f}%", delta_color="inverse")

    c_pie, c_bar = st.columns(2)
    with c_pie:
        fig_pie = px.pie(df_herd, names="Giai_Doan", title="Phân bố đàn bò theo 3 giai đoạn sinh trưởng", hole=0.4)
        fig_pie.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_bar:
        x_col = "Ma_Bo" if "Ma_Bo" in df_herd.columns else df_herd.index
        fig_bar = px.bar(
            df_herd, x=x_col, y="CH4_Du_Bao (g/ngay)", color="Trang_Thai",
            title="Phát thải CH₄ từng cá thể",
            color_discrete_map={"🔴 Vượt chuẩn nhóm": "#EF4444", "🟡 Thấp hơn chuẩn": "#F59E0B", "🟢 Đạt chuẩn": "#10B981"}
        )
        fig_bar.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 📋 Dữ liệu chi tiết đàn bò")
    selected_status = st.multiselect("Lọc theo trạng thái:", options=df_herd["Trang_Thai"].unique(), default=df_herd["Trang_Thai"].unique())
    df_filtered = df_herd[df_herd["Trang_Thai"].isin(selected_status)]

    st.dataframe(df_filtered, use_container_width=True)

    st.download_button(
        label="💾 Tải xuống báo cáo toàn đàn (.CSV)",
        data=df_filtered.to_csv(index=False).encode('utf-8-sig'),
        file_name="bao_cao_phat_thai_theo_giai_doan.csv",
        mime="text/csv",
        type="primary"
    )