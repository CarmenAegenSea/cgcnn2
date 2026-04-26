"""
绘制 CGCNN 带隙预测结果图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

# 设置中文字体（Windows 用 SimHei，Mac 用 PingFang SC）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===============================================================================================
# 配置参数
df = pd.read_csv("data/catalysis/cif/test_results.csv")
y_true = df["target"].values
y_pred = df["prediction"].values

# ===============================================================================================
# 图1：预测，真实值散点图
fig, ax = plt.subplots(figsize=(6, 5))

# 散点
ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=40)

# 对角线 y=x
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y = x')

# 误差线 ±0.3 eV（可选）
ax.fill_between([min_val, max_val],
                [min_val-0.3, max_val-0.3],
                [min_val+0.3, max_val+0.3],
                alpha=0.1, color='gray', label='±0.3 eV')

# 计算指标
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# 标注文本
ax.text(0.05, 0.95, f'MAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('DFT 计算带隙 (eV)', fontsize=12)
ax.set_ylabel('CGCNN 预测带隙 (eV)', fontsize=12)
ax.set_title('CGCNN 带隙预测性能评估', fontsize=14)
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('bandgap_prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 图2：预测误差分布直方图 ====================
errors = y_pred - y_true

fig, ax = plt.subplots(figsize=(6, 4))
n_bins = int(np.sqrt(len(errors)))
ax.hist(errors, bins=n_bins, edgecolor='k', alpha=0.7, density=True)

# 拟合正态分布
mu, std = np.mean(errors), np.std(errors)
x = np.linspace(mu - 3*std, mu + 3*std, 100)

# 使用原始字符串 r"" 避免转义警告
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2,
        label=rf'正态拟合 ($\mu$={mu:.3f}, $\sigma$={std:.3f})')

ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('预测误差 (eV)', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('CGCNN 预测误差分布', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 图3：损失曲线（需解析训练日志） ====================
# 如果你有训练日志，可解析并绘制；此处为模拟数据
epochs = np.arange(1, 201)
train_loss = 2.5 * np.exp(-0.02 * epochs) + 0.1 + 0.05 * np.random.randn(len(epochs))
val_loss = 2.8 * np.exp(-0.018 * epochs) + 0.15 + 0.08 * np.random.randn(len(epochs))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, train_loss, 'b-', linewidth=1.5, label='训练损失')
ax.plot(epochs, val_loss, 'r-', linewidth=1.5, label='验证损失')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MAE)', fontsize=12)
ax.set_title('CGCNN 训练过程损失曲线', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存：bandgap_prediction_scatter.png, error_distribution.png, loss_curve.png")