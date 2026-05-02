import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def main():
    # 1. 加载数据 (请确保路径正确)
    data_path = "tmc_all_materials.csv"  # 替换为你的文件名或相对路径
    df = pd.read_csv(data_path)

    print(">>> 正在转换化学式并提取特征...")
    # 将字符串转换为 Pymatgen Composition 对象
    df['composition'] = df['formula'].apply(lambda x: Composition(x))

    # 2. 特征提取 (Magpie)
    ep = ElementProperty.from_preset(preset_name="magpie")
    X = ep.featurize_dataframe(df, col_id='composition', ignore_errors=True)

    # 移除不需要的列，仅保留特征（按存在性判断）
    drop_cols = [c for c in ['formula', 'band_gap', 'composition'] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # 保证只使用数值特征：移除/转换非数值列（例如 id 或文件名字符串），并填充缺失值
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f">>> Dropping non-numeric feature columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # 强制转换为数值（无法转换的将变为 NaN），并用列均值填充 NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isna().any().any():
        X = X.fillna(X.mean())

    # 重置索引以确保与 y 对齐
    X = X.reset_index(drop=True)
    y = df['band_gap'].reset_index(drop=True).to_numpy().reshape(-1, 1)

    # 3. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 核心步骤：Z-score 标准化 ---
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    print(f">>> 目标值标准化完成：μ={y_scaler.mean_[0]:.4f}, σ={np.sqrt(y_scaler.var_[0]):.4f}")

    # 5. 模型训练
    print(">>> 正在训练随机森林模型...")
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_scaled.ravel())

    # 6. 预测阶段
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

    # 7. 性能评估 (使用还原后的真实物理数值)
    mae = mean_absolute_error(y_test, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
    r2 = r2_score(y_test, y_pred_original)

    print("\n" + "="*40)
    print(f"{'模型评估结果 (物理单位: eV)':^40}")
    print("-" * 40)
    print(f"平均绝对误差 (MAE): {mae:.4f} eV")
    print(f"均方根误差 (RMSE): {rmse:.4f} eV")
    print(f"决定系数 (R2):     {r2:.4f}")
    print("="*40)

    # 8. 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_original, alpha=0.5, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title(f'Random Forest Regression (R2={r2:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('standardized_rf_results.png')

    # 9. （可选）保存模型/标准化器以便未来预测新数据
    try:
        joblib.dump(y_scaler, 'bandgap_scaler.pkl')
        joblib.dump(model, 'rf_model.joblib')
    except Exception:
        pass


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()