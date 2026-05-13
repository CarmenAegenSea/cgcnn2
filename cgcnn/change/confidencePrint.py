import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set_style("white") # 改为纯白背景，使黑边更明显
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 1. 读取数据
df = pd.read_csv('tmc_ensemble_final.csv')
sigma_data = df['uncertainty_std'].dropna()

# 2. 计算统计指标
mean_val = sigma_data.mean()
median_val = sigma_data.median()

# 3. 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# 4. 绘制直方图（纵轴为频率）
# 修改处：edgecolor='black' 设置黑边，linewidth=0.8 设置边框粗细
sns.histplot(sigma_data, binwidth=0.05, color='#2c7fb8', alpha=0.7, 
             edgecolor='black', linewidth=0.8,
             label='样本频率', ax=ax1)

ax1.set_ylabel('频率 (Frequency)', fontsize=13)
ax1.set_xlabel('预测标准差 $\sigma$ (单位: eV)', fontsize=13)

# 6. 标注均值和中位数
ax1.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2, 
            label=f'均值: {mean_val:.4f} eV')
ax1.axvline(median_val, color='#2ca02c', linestyle='-', linewidth=2, 
            label=f'中位数: {median_val:.4f} eV')

# 7. 高置信筛选区阴影
ax1.axvspan(0, 0.15, color='gray', alpha=0.1, label='高置信区 ($\sigma < 0.15$)')

# 8. 添加文字标注
y_max = ax1.get_ylim()[1]
ax1.text(mean_val + 0.005, y_max * 0.85, f'平均值: {mean_val:.4f}', color='#d62728', fontweight='bold')
ax1.text(median_val - 0.06, y_max * 0.75, f'中位数: {median_val:.4f}', color='#2ca02c', fontweight='bold')

# 9. 细节美化：给整个图表区域加个外框
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

plt.title('模型预测不确定度 ($\sigma$) 的分布分析', fontsize=16, pad=20)
ax1.legend(loc='upper right', frameon=True)

ax1.set_xlim(0, max(sigma_data.quantile(0.98), 0.5)) 

plt.tight_layout()
plt.savefig('uncertainty_distribution_black_edge.png')
plt.show()