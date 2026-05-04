# Rotation forest46 + REPTree 复现目录

说明：本文件夹包含一个在本地可运行的 Rotation Forest 基线复现实现（默认基学习器使用 sklearn 的决策树）。所有输入、输出、模型与日志均写入本目录下的子目录，便于独立运行与管理。

目录结构简述：
- `rotation_forest.py`：Rotation Forest 实现（回归与分类）。
- `weka_utils.py`：可选的 Weka/ARFF 帮助函数（如需使用 REPTree，请自行放置 `weka.jar` 并确保 `java` 可用）。
- `run_rotation_forest.py`：主运行脚本，默认会在本目录 `data/` 下查找数据，若不存在会生成合成示例数据；训练结果、模型和图保存在 `experiments/` 里。
- `requirements.txt`：Python 依赖。

快速开始：
1. 安装依赖：
```
pip install -r "Rotation forest46+REPTree/requirements.txt"
```
2. 进入目录并运行（Windows PowerShell）：
```
cd "Rotation forest46+REPTree"
python run_rotation_forest.py --n_estimators 46 --K 3
```

说明与扩展：
- 若你希望严格使用 Weka 的 `REPTree`，请将 `weka.jar` 放到本目录下（或通过 `--weka_jar` 指定路径），并确保系统安装了 Java。脚本中已提供 `weka_utils.py` 来帮助生成 ARFF、调用 Weka。当前默认流程在找不到或不使用 Weka 时会回退使用 sklearn。
- 将你自己的数据放入 `Rotation forest46+REPTree/data/`，CSV 要求最后一列为目标列（默认名 `target`，可通过 `--target` 指定）。

输出位置：
- 模型： `experiments/models/`
- 预测/结果/图： `experiments/outputs/`
- 日志： `experiments/logs/`

如果需要，我可以：
- 将脚本改为对接你已有的特征生成（例如 matminer/论文描述符）；
- 完整实现 Weka 的训练/预测流水线并演示（需要 Java + weka.jar 可用）。
