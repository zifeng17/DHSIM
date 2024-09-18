import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
config = {
    "font.family": 'serif',
    "font.size": 14,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False
}
plt.rcParams.update(config)  # 设置中文字体
# 读取Excel文件到DataFrame
errors = pd.read_excel('abs_scada_pressure.xlsx')
print(errors.shape, errors.head(5))  # 输出DataFrame的形状（行数，列数）
print(errors.min())
print(errors.max())
# 确定要显示的列数（最多5列）
num_cols = min(errors.shape[1], 5)
# 计算需要的行数（确保是4的倍数或向上取整到最近的4的倍数）
num_rows = -(-errors.shape[1] // num_cols)  # 使用负负取整，即向上取整到最近的4的倍数
if num_rows == 0:  # 如果所有列都能放在一个子图中，则至少需要一个行
    num_rows = 1

# 创建一个4x5的子图网格（或者更少的列和适当的行数）
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))  # 调整figsize以适应子图

# 遍历每一列并绘制直方图到子图中
for i, ax in enumerate(axs.flatten()):
    if i < errors.shape[1]:  # 确保我们不会超出DataFrame的列数
        col = errors.columns[i]
        print(errors[col])
        ax.hist(errors[col], bins=48, density=True, alpha=0.6, color='g')
        ax.set_title(f'{col}')
        ax.grid(axis='y', alpha=0.75)

        # 移除横坐标的刻度标签
        ax.set_xticks([])
    else:
        # 如果列数不足，隐藏额外的子图
        ax.axis('off')

    # 注意：tight_layout可能在复杂的网格中效果不佳，可以考虑使用constrained_layout=True
# 或者手动调整子图参数（比如subplots_adjust）
plt.tight_layout()  # 尝试使用tight_layout调整子图间距
# plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 如果需要，可以手动调整子图之间的间距

# 保存图片，设置分辨率为600dpi
plt.savefig('histogram_plot.png', dpi=600)

# 显示图形
plt.show()

# __________________________计算汇总统计数据__________________________________
# 计算每一列的平均数
mean_values = errors.mean()

# 计算每一列的最大值
max_values = errors.max()

# 计算每一列的最小值
min_values = errors.min()

# 计算每一列的标准差
std_values = errors.std()

# 将统计量组合成一个新的DataFrame
stats_df = pd.DataFrame({
    'Mean': mean_values,
    'Max': max_values,
    'Min': min_values,
    'Std': std_values
})

print(stats_df)

# 将结果保存为新的Excel文件
stats_df.to_excel('stats_abs_scada_pressure.xlsx')

