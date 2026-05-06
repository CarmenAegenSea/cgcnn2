import streamlit as st
import subprocess
import os
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="光催化材料筛选", layout="wide")
st.title("基于 CGCNN 的光催化材料高通量筛选系统")

# ---- 侧边栏：功能介绍 ----
st.sidebar.header("功能导航")
mode = st.sidebar.radio("选择模式", ["单个 CIF 预测", "批量预测与筛选", "模型性能展示"])

# ---- 公共参数 ----
MODEL_PATH = "./model_best.pth.tar"
UPLOAD_DIR = "./demo_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- 单个 CIF 预测 ----
if mode == "单个 CIF 预测":
    st.header("上传 CIF 文件，预测带隙")
    uploaded_file = st.file_uploader("选择 .cif 文件", type="cif")
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已上传 { uploaded_file.name }")
        if st.button("开始预测"):
            with st.spinner("正在预测..."):
                # 调用  predict.py  对单个文件预测，需要适配 single file 脚本
                result = subprocess.run(
                    ["python", "cgcnn/ predict_single.py ", MODEL_PATH, file_path],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    bandgap = float(result.stdout.strip())
                    st.metric("预测带隙", f"{bandgap:.3f} eV")
                    if 1.6 <= bandgap <= 2.8:
                        st.info("该材料带隙在可见光响应窗口内，具备光催化潜力")
                    else:
                        st.warning("带隙不在理想范围 (1.6-2.8 eV)")
                else:
                    st.error(f"预测失败：{result.stderr}")

# ---- 批量预测与筛选 ----
elif mode == "批量预测与筛选":
    st.header("批量预测 TMCs 候选材料")
    if st.button("运行批量预测"):
        with st.spinner("正在预测，约需 3-5 分钟..."):
            # 运行批量预测
            pred_proc = subprocess.run(
                ["python", "cgcnn/ predict.py ", MODEL_PATH, "./data/tmc_data/cif"],
                capture_output=True, text=True
            )
            if pred_proc.returncode == 0:
                st.success("预测完成！")
                # 运行筛选
                filter_proc = subprocess.run(
                    ["python", "change/ filter_candidates.py "],
                    capture_output=True, text=True
                )
                if filter_proc.returncode == 0:
                    df = pd.read_csv("final_candidates.csv")
                    st.write(f"筛选出 {len(df)} 种候选材料")
                    st.dataframe(df)
                    # 提供下载
                    with open("final_candidates.csv", "rb") as f:
                        st.download_button("下载 CSV", f, "final_candidates.csv")
                else:
                    st.error(f"筛选失败：{filter_proc.stderr}")
            else:
                st.error(f"预测失败：{pred_proc.stderr}")

# ---- 模型性能展示 ----
elif mode == "模型性能展示":
    st.header("模型评估")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("预测 vs 真实值")
        if os.path.exists("bandgap_prediction_scatter.png"):
            st.image("bandgap_prediction_scatter.png", use_column_width=True)
    with col2:
        st.subheader("误差分布")
        if os.path.exists("error_distribution.png"):
            st.image("error_distribution.png", use_column_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("训练收敛曲线")
        if os.path.exists("loss_curve.png"):
            st.image("loss_curve.png", use_column_width=True)
    with col4:
        st.subheader("模型性能指标")
        st.metric("MAE", "0.366 eV")
        st.metric("R²", "0.87")
        st.metric("筛选效率提升", "38,000 倍")