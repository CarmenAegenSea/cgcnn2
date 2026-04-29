# 运行环境配置

+ Python 3.11
+ 配置文件：environment_cgcnn.yml（位于项目根目录）
  
# 配置环境

本项目部分脚本可能使用相对或绝对路径，使用前请检查脚本内的路径设置。建议在项目根目录运行下面的命令。

1. 创建并激活环境（yml 中定义的名称为 `cgcnn`，如需确认请查看 `environment_cgcnn.yml` 中的 `name:` 字段）：

```bash
conda env create -f environment_cgcnn.yml
conda activate cgcnn
```

2. 运行拉取脚本（测试集）：

```bash
python cgcnn/change/pull.py
```

1. 运行拉取脚本（预测集，过渡金属硫族化合物）：

```bash
python cgcnn/change/pull_data.py
```

4. 在 `cgcnn/change/id_prop_data.py` 中设置目标目录后，生成训练/预测所需的 `id_prop.csv`：

```bash
python cgcnn/change/id_prop_data.py
```

5. 训练示例（在项目根目录运行，data 路径为数据目录）：

```bash
python cgcnn/main.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --epochs 200 --batch-size 32 --lr 0.01 data/catalysis/cif
```

预测时需要使用训练集的 mean 和 std：

- 计算 mean 和 std（在项目根目录运行）：

```bash
python -c "import pandas as pd; df = pd.read_csv('cgcnn/data/catalysis/cif/id_prop.csv', header=None); bg = df.iloc[:, 1].values; print(f'mean = {bg.mean():.4f}, std = {bg.std():.4f}')"
```

将得到的 `mean` 和 `std` 填入 `cgcnn/predict_data.py`（或通过脚本参数传入，若脚本支持）。

运行预测：

单模型预测
```bash
python cgcnn/predict_data.py cgcnn/model_best.pth.tar data/catalysis/cif --disable-cuda --batch-size 64
```

集成模型置信度计算
···bash
python cgcnn/change/confidence.py
```

后处理：

```bash
python cgcnn/change/filter_candidates.py
python cgcnn/change/parityPlot.py
```

输出结果会出现在`log`文件夹中。
`final_candidates.csv`是最终筛选出的可用材料

备注：
- 在 Windows PowerShell 中运行时，注意外层与内层引号的使用；若命令出错，请尝试调整引号或在命令行中拆分为更简单的步骤。
- 如果 `environment_cgcnn.yml` 中的环境名与 `cgcnn` 不同，请用 yml 中的名称替换上述 `conda activate` 命令。
  


所有带data的文件或文件夹都是预测用的文件
