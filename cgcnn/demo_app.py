import streamlit as st
import pandas as pd
import os
import plotly.express as px
from pathlib import Path
from datetime import datetime
import base64
import json
import numpy as np

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="光催化材料高通量筛选系统",
    layout="wide",
    initial_sidebar_state="auto"
)

# 自定义样式
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== 标题 ====================
st.title("基于 CGCNN 的光催化材料高通量筛选系统")
st.markdown("### 中国大学生计算机设计大赛 · 大数据实践赛")
st.markdown("---")

# ==================== 一、模型性能评估 ====================
st.header("一、模型性能评估")

LOG_DIR = "log"

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def display_images_side_by_side(img_paths, captions=None, height=360):
    parts = []
    for i, p in enumerate(img_paths):
        if p and os.path.exists(p):
            b64 = _img_to_base64(p)
            cap = captions[i] if captions and i < len(captions) else ""
            parts.append(f"""
                <div style="flex:1; text-align:center;">
                  <img src="data:image/png;base64,{b64}" style="height:{height}px; width:auto; object-fit:contain; border-radius:6px;"/>
                  <div style="font-size:12px; color:#666">{cap}</div>
                </div>
            """)
        else:
            parts.append(f"""
                <div style="flex:1; display:flex; align-items:center; justify-content:center; height:{height}px; background:#f6f6f6; border-radius:6px;">
                  <div style="color:#888">未找到图片</div>
                </div>
            """)
    html = f"<div style='display:flex; gap:16px; align-items:flex-start'>{''.join(parts)}</div>"
    st.markdown(html, unsafe_allow_html=True)

def load_metrics_from_log(log_path):
    metrics = {"mae": None, "r2": None, "train_size": None, "speedup": None}
    # try common json summary files
    for fname in ("metrics.json", "summary.json", "run_metrics.json"):
        p = os.path.join(log_path, fname)
        data = _read_json(p)
        if data:
            metrics["mae"] = data.get("mae") or data.get("MAE")
            metrics["r2"] = data.get("r2") or data.get("R2")
            metrics["train_size"] = data.get("train_size") or data.get("n_train") or data.get("train_samples")
            metrics["speedup"] = data.get("speedup")
            return metrics
    # fall back: compute from prediction csv if available
    for fname in ("test_results_final.csv", "test_results.csv", "predictions.csv"):
        p = os.path.join(log_path, fname)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = [c.lower() for c in df.columns]
            pred_col = None
            true_col = None
            for c in df.columns:
                lc = c.lower()
                if any(k in lc for k in ("pred", "prediction", "predicted", "y_hat")) and pred_col is None:
                    pred_col = c
                if any(k in lc for k in ("target", "true", "bandgap", "dft")) and true_col is None:
                    true_col = c
            if pred_col and true_col:
                y_true = pd.to_numeric(df[true_col], errors="coerce")
                y_pred = pd.to_numeric(df[pred_col], errors="coerce")
                mask = y_true.notna() & y_pred.notna()
                if mask.sum() > 0:
                    mae_val = float((y_true[mask] - y_pred[mask]).abs().mean())
                    ss_res = float(((y_true[mask] - y_pred[mask])**2).sum())
                    ss_tot = float(((y_true[mask] - y_true[mask].mean())**2).sum())
                    r2_val = 1.0 - ss_res / ss_tot if ss_tot != 0 else None
                    metrics["mae"] = mae_val
                    metrics["r2"] = r2_val
                    metrics["train_size"] = len(df)
                    return metrics
    return metrics

# show metrics from latest log (if any)
latest_metrics = {"mae": None, "r2": None, "train_size": None, "speedup": None}
if os.path.exists(LOG_DIR):
    log_folders = sorted([d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))], reverse=True)
    if log_folders:
        latest_log = os.path.join(LOG_DIR, log_folders[0])
        latest_metrics = load_metrics_from_log(latest_log)

# ---- 关键指标 ----
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    if latest_metrics.get("mae") is not None:
        st.metric("测试集 MAE", f"{latest_metrics['mae']:.3f} eV")
    else:
        st.metric("测试集 MAE", "N/A")
with col_m2:
    if latest_metrics.get("r2") is not None:
        st.metric("决定系数 R²", f"{latest_metrics['r2']:.2f}")
    else:
        st.metric("决定系数 R²", "N/A")
with col_m3:
    ts = latest_metrics.get("train_size")
    st.metric("训练数据量", f"{int(ts)} 条" if ts else "未知")
with col_m4:
    st.metric("筛选效率提升", latest_metrics.get("speedup") or "N/A")

# ---- 核心图表 ----
st.subheader("1.2 训练质量分析")
col_img1, col_img2 = st.columns(2)
with col_img1:
    if os.path.exists("bandgap_prediction_scatter.png"):
        st.image("bandgap_prediction_scatter.png", caption="预测值 vs DFT 计算值", use_container_width=True)
    else:
        st.info("💡 根目录下未找到 bandgap_prediction_scatter.png")

with col_img2:
    if os.path.exists("loss_curve.png"):
        st.image("loss_curve.png", caption="收敛曲线", use_container_width=True)
    else:
        st.info("💡 根目录下未找到 loss_curve.png")

st.markdown("---")

# ==================== 二、高通量筛选结果 ====================
# 原第三章，逻辑调整为从 log 文件夹读取
st.header("二、高通量筛选结果")

LOG_DIR = "log"

if os.path.exists(LOG_DIR):
    log_folders = sorted([d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))], reverse=True)
    
    if log_folders:
        # 日志选择
        log_options = []
        for folder in log_folders:
            try:
                dt = datetime.strptime(folder, "%Y%m%d%H%M")
                log_options.append(f"{folder} ({dt.strftime('%Y-%m-%d %H:%M')})")
            except:
                log_options.append(folder)

        st.sidebar.header("数据源选择")
        selected_log_display = st.sidebar.selectbox("选择实验批次", log_options)
        selected_folder = selected_log_display.split(" (")[0]
        log_path = os.path.join(LOG_DIR, selected_folder)

        # 文件路径定义
        full_pred_path = os.path.join(log_path, "test_results_final.csv")
        candidate_path = os.path.join(log_path, "final_candidates.csv")

        # 1. 展示筛选过的可用材料 (final_candidates)
        st.subheader("2.1 潜力材料库 (筛选后)")
        if os.path.exists(candidate_path):
            df_cand = pd.read_csv(candidate_path)
            st.success(f"从当前批次中筛选出 **{len(df_cand)}** 种高潜力材料 (ZnS多晶型等)")
            
            # 带隙分布可视化
            fig = px.histogram(df_cand, x=df_cand.columns[1], nbins=15, 
                               title="高潜力材料带隙分布", color_discrete_sequence=['#2E8B57'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_cand, use_container_width=True)
            
            with open(candidate_path, "rb") as f:
                st.download_button("下载筛选后的可用材料 (CSV)", data=f, file_name="final_candidates.csv")
        else:
            st.warning("该日志文件夹下未找到 final_candidates.csv")

        st.markdown("---")

        # 2. 展示全部预测结果 (test_results_final)
        st.subheader("2.2 全量预测原始数据")
        if os.path.exists(full_pred_path):
            with st.expander("点击展开全量预测结果查看"):
                df_full = pd.read_csv(full_pred_path)
                st.write(f"共包含 {len(df_full)} 条预测记录")
                st.dataframe(df_full, use_container_width=True)
                
                with open(full_pred_path, "rb") as f:
                    st.download_button("下载原始预测数据 (CSV)", data=f, file_name="test_results_final.csv")
        else:
            st.info("ℹ该日志文件夹下未找到 test_results_final.csv")

    else:
        st.error("log 文件夹内没有子目录，请确认训练输出路径。")
else:
    st.error("未检测到 log 文件夹，请确保 log 目录与本脚本在同一级。")

st.markdown("---")

# ==================== 三、历史训练日志 ====================
# 原第四章
st.header("三、实验可视化回溯")

if os.path.exists(LOG_DIR) and 'log_path' in locals():
    st.markdown(f"当前查看批次：`{selected_folder}`")
    
    col_log1, col_log2 = st.columns(2)
    
    scatter_path = os.path.join(log_path, "bandgap_prediction_scatter.png")
    error_path = os.path.join(log_path, "error_distribution.png")
    loss_path = os.path.join(log_path, "loss_curve.png")

    with col_log1:
        if os.path.exists(scatter_path):
            st.image(scatter_path, caption="该批次：预测值对比图", use_container_width=True)
        else:
            st.info("该日志中未找到散点图")

    with col_log2:
        if os.path.exists(error_path):
            st.image(error_path, caption="该批次：误差分布图", use_container_width=True)
        else:
            st.info("该日志中未找到误差图")

    st.subheader("训练损失收敛情况")
    if os.path.exists(loss_path):
        st.image(loss_path, use_container_width=True)
    else:
        st.info("该日志中未找到 Loss 曲线")
