# XGBoost: ABX3 (Perovskite) Band Gap Prediction

本目录实现论文复现的 XGBoost 管线（数据特征化、训练、评估、预测）。

快速开始：

- 安装依赖：

```powershell
pip install -r XGBoost/requirements.txt
```

- 训练（示例）：

```powershell
python XGBoost/train_xgboost.py --data path/to/abx3.csv --target band_gap --model_out XGBoost/model.joblib
```

输入 CSV 要求：包含目标列（默认 band_gap），以及元素信息之一：
- 单独列 `A`, `B`, `X`，或
- `formula`（例如 `CsPbI3`），脚本会从式子自动识别 ABX3（X 为计数最高的元素）。

输出：模型文件（`model.joblib`）和评估结果 JSON。更多细节见各脚本。
