import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
def cluster_data_by_user_id(data_frame):
    grouped_data = data_frame.groupby('user_id')
    clustered_data = {}
    for user_id, group in grouped_data:
        clustered_data[user_id] = group
    return clustered_data


def get_most_probable_values(data_frame, start_time, end_time):
    user_data = data_frame[(data_frame['数据采集时间'] >= start_time) &
                           (data_frame['数据采集时间'] <= end_time)]
    query_times = pd.date_range(start=start_time, end=end_time, freq='30min')
    results = []
    for i in range(len(query_times) - 1):
        query_data = user_data[(user_data['数据采集时间'] >= query_times[i].strftime('%Y-%m-%d %H:%M:%S')) &
                               (user_data['数据采集时间'] < query_times[i + 1].strftime('%Y-%m-%d %H:%M:%S'))][
            '瞬时流量']
        query_data_numeric = pd.to_numeric(query_data, errors='coerce').fillna(0)
        if not query_data_numeric.empty:
            kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
            kde.fit(query_data_numeric.values.reshape(-1, 1))
            query_points = np.linspace(query_data_numeric.min(), query_data_numeric.max(), 1000).reshape(-1, 1)
            log_dens = kde.score_samples(query_points)
            most_probable_value = query_points[np.argmax(log_dens)]
            results.append(most_probable_value[0])
        else:
            results.append(0)
    return results


def process_data_by_user_id(data_dict, start_time, end_time):
    processed_data = {}
    for user_id, data_frame in data_dict.items():
        most_probable_values = get_most_probable_values(data_frame, start_time, end_time)
        if None not in most_probable_values:
            processed_data[user_id] = most_probable_values
    return processed_data


@staticmethod
def perform_kmeans_clustering(processed_data, num_clusters):
    data_array = np.array(list(processed_data.values()))
    print(data_array)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_array)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers_dict = {f"Cluster {i + 1}": center for i, center in enumerate(cluster_centers)}
    return cluster_centers_dict


@staticmethod
def plot_line_chart_with_envelope(data_frame, category):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        'axes.unicode_minus': False
    }
    plt.rcParams.update(config)
    grouped_data = data_frame.groupby(data_frame['Timestamp'].dt.time)
    fig, ax = plt.subplots(figsize=(10, 6))
    medians = []
    max_values = []
    min_values = []
    for timestamp, group in grouped_data:
        median_value = group['ModeFactor'].median()
        medians.append(median_value)
        max_value = group['ModeFactor'].max()
        max_values.append(max_value)
        min_value = group['ModeFactor'].min()
        min_values.append(min_value)
        timestamps_in_hours = [times.hour + times.minute / 60 for times in grouped_data.groups.keys()]
        ax.plot(timestamps_in_hours, medians, marker='o', markersize=2, label=f'{category} - 中位数')
        ax.fill_between(timestamps_in_hours, max_values, min_values, alpha=0.3, label=f'{category} - 包络线')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(25))
        ax.set_xlabel('timestamp(h)')
        ax.set_ylabel('factor')
        ax.set_title(f'{category} - 模式因子折线图和包络线（6月份）')
        ax.legend()
        plt.tight_layout()
        plt.show()

#用远传表聚类手抄表模式----------------------------------------------------yw
def cluster_pattern(yuanchuan_demand, start_time, end_time):
    # 每个用户根据数据采集时间排序
    grouped = yuanchuan_demand.groupby('用户编号')
    sorted_data = []
    all_mode_factors = {}
    for user_id, user_data in grouped:
        sorted_user_data = user_data.sort_values(by='数据采集时间')
        sorted_data.append(sorted_user_data)

    # Concatenate all the sorted groups to reconstruct the entire sorted dataset
    sorted_yuanchuan_demand = pd.concat(sorted_data)

    #output_file = "C:/Users/袁伟/Desktop/sorted_yuanchuan_demand.xlsx"
    #sorted_yuanchuan_demand.to_excel(output_file, index=False)

    # 所有远传表按用水类型分组
    grouped_by_type = sorted_yuanchuan_demand.groupby('cus_type_code')

    for user_type, user_type_data in grouped_by_type:
        print("分组类型为：", user_type)
        ['value'].append(user_type_data['瞬时流量'])
        group_data = user_type_data.groupby('用户编号')
        all_cluster_data = []
        for user_id, user_data in group_data:
            all_cluster_data.append(user_data['瞬时流量'].values)

        all_cluster_data_array = np.array(all_cluster_data)
        # print(all_cluster_data_np)
        num_clusters = 1
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(all_cluster_data_array)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_dict = {f"Cluster {i + 1}": center for i, center in enumerate(cluster_centers)}
        # Print cluster labels
        # print(cluster_centers_dict)
        mean_value = np.mean(cluster_centers)
        multipliers = cluster_centers / mean_value
        query_times = pd.date_range(start=start_time, end=end_time, freq='30min', closed='left')

        # print(query_times)
        mode_result_df = pd.DataFrame({'数据采集时间': query_times, 'ModeFactor': multipliers.flatten()})
        # print(mode_result_df)
        all_mode_factors[user_type] = mode_result_df
        # print("所有模式因子为", all_mode_factors)

        # 用水模式绘图-------------------------
        #plt.figure(figsize=(10, 6))
        #plt.rcParams['font.size'] = 15  # 设置更大的默认字体大小

        #plt.plot(mode_result_df['数据采集时间'], mode_result_df['ModeFactor'])
        #plt.title(f'{user_type}的用水模式')  # Set the plot title to the user_type variable
        #plt.xlabel('数据采集时间')
        #plt.ylabel('模式乘子')
        #plt.xticks(rotation=60)

        #plt.grid(True)
        #plt.tight_layout()
        #plt.show()

    # 把模式乘子导入excel文件-------------------
    #writer = pd.ExcelWriter('C:/Users/袁伟/Desktop/mode_factors(3.8).xlsx')
    #for category, mode_factors_df in all_mode_factors.items():
     #   mode_factors_df.to_excel(writer, sheet_name=category, index=False)
    #writer.save()
    return all_mode_factors