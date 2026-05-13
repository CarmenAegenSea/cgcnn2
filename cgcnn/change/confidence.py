import pandas as pd
import numpy as np
import os

# 1. 定义文件名（请确保路径正确）
files = [
    r'models\ensemble\seed_7\res_1.csv', 
    r'models\ensemble\seed_42\res_2.csv', 
    r'models\ensemble\seed_2021\res_3.csv'
]

try:
    # 2. 读取并提取关键列
    data_list = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}")
        temp_df = pd.read_csv(f, header=0)
        # 只取 id 和 prediction 列
        data_list.append(temp_df[['id', 'prediction']])

    # 3. 合并数据
    df_ensemble = data_list[0].copy()
    df_ensemble = df_ensemble.rename(columns={'prediction': 'model_1'})
    df_ensemble['model_2'] = data_list[1]['prediction']
    df_ensemble['model_3'] = data_list[2]['prediction']

    # 4. 计算统计量
    # 注意：这里必须使用重命名后的列名 ['model_1', 'model_2', 'model_3']
    model_cols = ['model_1', 'model_2', 'model_3']
    
    # 最终预测值 = 三个模型均值
    df_ensemble['final_prediction'] = df_ensemble[model_cols].mean(axis=1)
    # 置信度指标 = 标准差 (std 越小，分歧越小，置信度越高)
    df_ensemble['uncertainty_std'] = df_ensemble[model_cols].std(axis=1)

    # 5. 筛选候选者 (1.5 - 3.0 eV)
    mask = (df_ensemble['final_prediction'] >= 1.5) & (df_ensemble['final_prediction'] <= 3.0)
    candidates = df_ensemble[mask].copy()

    # 按照不确定度升序排列 (分歧越小越靠前)
    top_10 = candidates.sort_values(by='uncertainty_std').head(10)

    # 6. 保存结果
    df_ensemble.to_csv('tmc_ensemble_final.csv', index=False)
    top_10.to_csv('top_10_confident_final.csv', index=False)

    print("✅ 分析成功！")
    print("\n--- 基于置信度筛选的 Top 10 候选材料 ---")
    # 增加打印 dist_to_ideal 或其他指标可以更直观
    print(top_10[['id', 'final_prediction', 'uncertainty_std']])

except Exception as e:
    print(f"❌ 运行失败: {e}")
    print("提示：请检查文件路径是否正确，以及不同种子生成的 CSV 行数是否一致。")