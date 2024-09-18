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
df_sim_pres = pd.read_excel('df_sim_scada_pressure.xlsx')
df_ture_pres = pd.read_excel('df_ture_scada_pressure.xlsx')
print(df_sim_pres.shape, df_sim_pres.head(5))  # 输出DataFrame的形状（行数，列数）
print(df_sim_pres.min())
print(df_sim_pres.max())
print(df_ture_pres.shape, df_ture_pres.head(5))  # 输出DataFrame的形状（行数，列数）
print(df_ture_pres.min())
print(df_ture_pres.max())
# 确定要显示的列数（最多5列）
num_cols = min(df_sim_pres.shape[1], 4)
# 计算需要的行数（确保是4的倍数或向上取整到最近的4的倍数）
num_rows = -(-df_sim_pres.shape[1] // num_cols)  # 使用负负取整，即向上取整到最近的4的倍数
if num_rows == 0:  # 如果所有列都能放在一个子图中，则至少需要一个行
    num_rows = 1
# 确保我们只对两个DataFrame中都存在的列进行操作
common_columns = df_sim_pres.columns.intersection(df_ture_pres.columns)
# 创建一个4x5的子图网格（或者更少的列和适当的行数）
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))  # 调整figsize以适应子图
# 遍历公共列，并绘制折线图
for i, col in enumerate(common_columns):
    row = i // num_cols
    col_pos = i % num_cols
    ax = axs[row, col_pos]  # 选择当前子图

    # 绘制模拟数据和真实数据
    line_sim, = ax.plot(df_sim_pres[col], label='sim')
    line_true, = ax.plot(df_ture_pres[col], label='true')

    # 合并handles以便在图例中只显示一次
    if i > 0:
        # 对于非第一个子图，我们不需要在legend中再次添加handles
        # 但我们可以通过传递之前的handles来避免重复的图例条目
        handles = [line_sim, line_true] if i == 0 else []
        labels = ['sim', 'true'] if i == 0 else []
    else:
        # 对于第一个子图，我们添加handles和labels
        handles = [line_sim, line_true]
        labels = ['sim', 'true']

        # 设置子图标题和显示图例
    ax.set_title(col)
    ax.grid(axis='y', alpha=0.75)
    ax.set_xticks([])
    ax.legend(handles=handles, labels=labels)  # 在每个子图上设置图例

# 调整子图之间的间距和边距
plt.tight_layout()

# 保存图片
plt.savefig('comparison_plot.png', dpi=600)
# 显示图表
plt.show()

# __________________________计算汇总统计数据__________________________________
# 计算每一列的平均数
mean_values = df_sim_pres.mean()

# 计算每一列的最大值
max_values = df_sim_pres.max()

# 计算每一列的最小值
min_values = df_sim_pres.min()

# 计算每一列的标准差
std_values = df_sim_pres.std()

# 将统计量组合成一个新的DataFrame
stats_df = pd.DataFrame({
    'Mean': mean_values,
    'Max': max_values,
    'Min': min_values,
    'Std': std_values
})

# print(stats_df)

# 将结果保存为新的Excel文件
stats_df.to_excel('stats_sim_scada_pressure.xlsx')

# 计算每一列的平均数
mean_values = df_ture_pres.mean()

# 计算每一列的最大值
max_values = df_ture_pres.max()

# 计算每一列的最小值
min_values = df_ture_pres.min()

# 计算每一列的标准差
std_values = df_ture_pres.std()

# 将统计量组合成一个新的DataFrame
stats_df = pd.DataFrame({
    'Mean': mean_values,
    'Max': max_values,
    'Min': min_values,
    'Std': std_values
})

# print(stats_df)

# 将结果保存为新的Excel文件
stats_df.to_excel('stats_ture_scada_pressure.xlsx')