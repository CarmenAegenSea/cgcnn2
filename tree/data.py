import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import jobilb  # 用于保存 scaler

# 1. 加载数据 (请确保路径正确)
# 假设你的数据列名为 'formula' 和 'band_gap'
data_path = "tmc_all_materials.csv"  # 替换为你的文件名
df = pd.read_csv(data_path)

print(">>> 正在转换化学式并提取特征...")
# 将字符串转换为 Pymatgen Composition 对象
df['composition'] = df['formula'].apply(lambda x: Composition(x))

# 2. 特征提取 (Magpie)
# 显式设置 impute_nan=True 以避免警告并处理缺失值
ep = ElementProperty.from_preset(preset_name="magpie")
X = ep.featurize_dataframe(df, col_id='composition', ignore_errors=True)

# 移除不需要的列，仅保留特征
# 假设原来的列名是 'formula' 和 'band_gap'
X = X.drop(columns=['formula', 'band_gap', 'composition'])
y = df['band_gap'].values.reshape(-1, 1)

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 核心步骤：Z-score 标准化 ---
# 4. 初始化目标值标准化器
y_scaler = StandardScaler()

# 仅基于训练集拟合 μ 和 σ，并转换训练集
y_train_scaled = y_scaler.fit_transform(y_train)
# 转换测试集（使用训练集的参数）
y_test_scaled = y_scaler.transform(y_test)

print(f">>> 目标值标准化完成：μ={y_scaler.mean_[0]:.4f}, σ={np.sqrt(y_scaler.var_[0]):.4f}")

# 5. 模型训练
print(">>> 正在训练随机森林模型...")
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train_scaled.ravel())

# 6. 预测阶段
y_pred_scaled = model.predict(X_test).reshape(-1, 1)

# --- 核心步骤：逆标准化还原 ---
# 将标准化后的预测值还原回原始物理单位 (eV)
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
plt.show()

# 9. 保存标准化器以便未来预测新数据
# joblib.dump(y_scaler, 'bandgap_scaler.pkl')