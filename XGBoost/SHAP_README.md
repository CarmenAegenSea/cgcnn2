**快速开始 — SHAP 分析**

本说明介绍如何在本地对训练好的 XGBoost 模型运行 SHAP 以获得特征重要性图。

依赖（在 `XGBoost/requirements.txt` 中已列出）：
- shap
- matplotlib
- seaborn

基本用法（建议先安装依赖）：

```powershell
pip install -r XGBoost\requirements.txt
python XGBoost\shap_analysis.py --model XGBoost\model_min2.joblib --data XGBoost\clean_catalysis_min2.csv --output-dir XGBoost\shap_output --sample 500
```

快速干跑（不计算 SHAP，仅验证特征化与格式）：

```powershell
python XGBoost\shap_analysis.py --data XGBoost\clean_catalysis_min2.csv --dry-run
```

输出文件（`--output-dir`）：
- `shap_feature_importance.csv` — 每个特征的平均 |SHAP| 值（排序）
- `shap_feature_importance_bar.png` — 条形图（前 30 个特征）
- `shap_beeswarm.png` — SHAP beeswarm / summary plot
- `shap_values.csv` — 每个样本的 SHAP 值矩阵
- `transformed_features.csv` — 传入模型的变换后特征（便于复现）

注意事项：
- 若模型中包含额外预处理（例如 StandardScaler），脚本会尝试使用保存的 pipeline 的预处理步骤进行转换，保证 SHAP 输入与训练时一致。
- SHAP 计算（尤其 kernel explainer）可能很慢。对于大型数据，使用 `--sample` 限制样本数。

如需我代你运行并解读结果，请在本地运行后把 `XGBoost/shap_output` 下的图像或 `shap_feature_importance.csv` 发给我。