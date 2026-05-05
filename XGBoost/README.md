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

数据清洗与重现论文建议：

- 使用 `prepare_data.py` 进行预处理（过滤 ABX3、去重、去异常值）：

```powershell
python XGBoost/prepare_data.py --input data/catalysis/catalysis.csv --output XGBoost/clean_catalysis_abx3.csv --filter-mode abx3
```

- 训练时可启用清洗与目标 Z-score（与论文保持一致）:

```powershell
python XGBoost/train_xgboost.py --data XGBoost/clean_catalysis_abx3.csv --target band_gap --model_out XGBoost/model.joblib --scale_target --clean_data --clean_mode abx3

如果想放宽过滤（仅保留含 3 个不同元素的化合物），使用：

```powershell
python XGBoost/prepare_data.py --input data/catalysis/catalysis.csv --output XGBoost/clean_catalysis_3elem.csv --filter-mode three_elements
python XGBoost/train_xgboost.py --data XGBoost/clean_catalysis_3elem.csv --target band_gap --model_out XGBoost/model_3elem.joblib --scale_target --clean_data --clean_mode three_elements

要保留至少两个元素（例如 TiO2），使用：

```powershell
python XGBoost/prepare_data.py --input data/catalysis/catalysis.csv --output XGBoost/clean_catalysis_min2.csv --filter-mode two_or_more
python XGBoost/train_xgboost.py --data XGBoost/clean_catalysis_min2.csv --target band_gap --model_out XGBoost/model_min2.joblib --scale_target --clean_data --clean_mode two_or_more
```
```
```

更多参数见脚本帮助：`python XGBoost/train_xgboost.py -h`。
