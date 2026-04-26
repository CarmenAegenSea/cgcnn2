## 项目文件表

main.py 训练入口脚本
predict_data.py 预测入口脚本
model_best.pth.tar：存储验证精度最高的 CGCNN 模型。
change 辅助脚本文件夹
    -> filter_candidates.py 筛选脚本
    -> id_prop_data.py id名单生成脚本
    -> parityPlot.py 绘制脚本
    -> pull.py 测试集拉取脚本
    -> pull_data.py 过渡金属硫族化合物拉取脚本
data 素材文件夹
    -> catalysis 训练集文件夹
        -> cif 晶胞cif文件文件夹
            -> id_prop.csv cif文件id表
            -> ~~~~.cif cif文件
    -> tmc_data 过渡金属硫族化合物文件夹
        -> cif 晶胞cif文件文件夹
            -> id_prop.csv cif文件id表
            -> ~~~~.cif cif文件
cgcnn cgcnn核心代码库
    -> __init__.py Python包标识
    -> data.py 解析cif文件
    -> model.py cgcnn模型架构


## 临时文件表

data/.../~~~.csv 本次拉取的cif文件id总表
data/.../~~~.json 本次拉取的cif文件id总表
